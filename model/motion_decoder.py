"""Motion decoder: intention query-based multi-modal trajectory prediction.

For each target agent:
  1. Initialize K=64 intention queries from pre-computed anchor points
  2. 4 layers of (agent_cross_attn + map_cross_attn + FFN)
  3. Per-layer trajectory head (deep supervision)
  4. NMS: 64 modes -> 6 final modes

Each intention query refines its trajectory prediction across decoder layers,
attending to agent tokens and map tokens at each layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import AttentionCaptureDecoderLayer


class TrajectoryHead(nn.Module):
    """MLP head that predicts trajectory + confidence from a query embedding.

    Outputs:
        trajectory: (K, future_len, 2)
        score: (K,) logit for mode selection
    """

    def __init__(self, d_model: int, future_len: int):
        super().__init__()
        self.traj_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, future_len * 2),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.future_len = future_len

    def forward(self, queries: torch.Tensor) -> tuple:
        """
        Args:
            queries: (B, K, d_model)

        Returns:
            trajs: (B, K, future_len, 2)
            scores: (B, K)
        """
        trajs = self.traj_mlp(queries)  # (B, K, future_len*2)
        trajs = trajs.reshape(queries.shape[0], queries.shape[1], self.future_len, 2)
        scores = self.score_mlp(queries).squeeze(-1)  # (B, K)
        return trajs, scores


class MotionDecoder(nn.Module):
    """Intention-query based motion decoder with deep supervision.

    Args:
        d_model: embedding dimension (256)
        nhead: attention heads (8)
        num_layers: decoder layers (4)
        dim_feedforward: FFN hidden dim (1024)
        dropout: dropout rate
        num_intentions: number of intention queries (64)
        num_modes_output: final output modes after NMS (6)
        future_len: prediction horizon in timesteps (80)
        nms_dist_thresh: NMS distance threshold in meters (2.0)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_intentions: int = 64,
        num_modes_output: int = 6,
        future_len: int = 80,
        nms_dist_thresh: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_intentions = num_intentions
        self.num_modes_output = num_modes_output
        self.future_len = future_len
        self.nms_dist_thresh = nms_dist_thresh
        self.num_layers = num_layers

        # Intention query embedding: projects (2,) anchor point to d_model
        self.intention_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Target agent context embedding: projects agent token to query context
        self.target_context = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            AttentionCaptureDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Per-layer trajectory heads (deep supervision)
        self.traj_heads = nn.ModuleList([
            TrajectoryHead(d_model, future_len) for _ in range(num_layers)
        ])

        # Learnable intention anchor points (initialized from k-means later)
        self.register_buffer(
            "intention_points",
            torch.zeros(num_intentions, 2),
        )

    def load_intention_points(self, points: torch.Tensor):
        """Load pre-computed k-means intention points."""
        assert points.shape == (self.num_intentions, 2)
        self.intention_points.copy_(points)

    def _nms_select(self, trajs: torch.Tensor, scores: torch.Tensor) -> tuple:
        """Distance-based NMS to select diverse modes.

        Args:
            trajs: (B, K, future_len, 2) predicted trajectories
            scores: (B, K) confidence scores

        Returns:
            selected_trajs: (B, num_modes_output, future_len, 2)
            selected_scores: (B, num_modes_output)
            selected_indices: (B, num_modes_output) indices into K
        """
        B, K = scores.shape
        device = scores.device
        M = self.num_modes_output

        selected_trajs = torch.zeros(B, M, self.future_len, 2, device=device)
        selected_scores = torch.full((B, M), -1e9, device=device)
        selected_indices = torch.zeros(B, M, dtype=torch.long, device=device)

        for b in range(B):
            endpoints = trajs[b, :, -1, :]  # (K, 2) final positions
            s = scores[b]  # (K,)
            order = torch.argsort(s, descending=True)

            keep = []
            suppressed = torch.zeros(K, dtype=torch.bool, device=device)

            for idx in order:
                idx_val = idx.item()
                if suppressed[idx_val]:
                    continue
                keep.append(idx_val)
                if len(keep) >= M:
                    break

                # Suppress nearby modes
                dists = torch.norm(endpoints - endpoints[idx_val].unsqueeze(0), dim=1)
                suppressed = suppressed | (dists < self.nms_dist_thresh)

            # Fill selected
            for m, k_idx in enumerate(keep):
                selected_trajs[b, m] = trajs[b, k_idx]
                selected_scores[b, m] = scores[b, k_idx]
                selected_indices[b, m] = k_idx

        return selected_trajs, selected_scores, selected_indices

    def forward(
        self,
        target_agent_tokens: torch.Tensor,
        agent_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        agent_mask: torch.Tensor,
        map_mask: torch.Tensor,
        capture_attention: bool = False,
    ) -> dict:
        """Decode trajectories for target agents.

        Args:
            target_agent_tokens: (B, d_model) encoded token for the target agent
            agent_tokens: (B, A, d_model) all encoded agent tokens
            map_tokens: (B, M, d_model) all encoded map tokens
            agent_mask: (B, A) True=valid
            map_mask: (B, M) True=valid
            capture_attention: whether to capture cross-attention weights

        Returns:
            dict with:
                trajectories: (B, num_modes_output, future_len, 2) NMS-selected
                scores: (B, num_modes_output) NMS-selected scores
                all_trajectories: (B, K, future_len, 2) all intention trajectories (last layer)
                all_scores: (B, K) all intention scores (last layer)
                layer_trajectories: list of (B, K, future_len, 2) per layer (deep supervision)
                layer_scores: list of (B, K) per layer
                nms_indices: (B, num_modes_output) which intentions were selected
                decoder_agent_attentions: list of (B, nhead, K, A) per layer
                decoder_map_attentions: list of (B, nhead, K, M) per layer
        """
        B = target_agent_tokens.shape[0]
        device = target_agent_tokens.device

        # Initialize intention queries from anchor points + target context
        intent_embed = self.intention_embed(self.intention_points)  # (K, d_model)
        intent_embed = intent_embed.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)

        target_ctx = self.target_context(target_agent_tokens)  # (B, d_model)
        target_ctx = target_ctx.unsqueeze(1).expand(-1, self.num_intentions, -1)  # (B, K, d_model)

        queries = intent_embed + target_ctx  # (B, K, d_model)

        # Padding masks for cross-attention (True = ignore)
        agent_pad_mask = ~agent_mask  # (B, A)
        map_pad_mask = ~map_mask      # (B, M)

        layer_trajectories = []
        layer_scores = []
        decoder_agent_attns = []
        decoder_map_attns = []

        for i, (layer, traj_head) in enumerate(zip(self.layers, self.traj_heads)):
            queries, agent_attn, map_attn = layer(
                queries,
                agent_tokens,
                map_tokens,
                agent_key_padding_mask=agent_pad_mask,
                map_key_padding_mask=map_pad_mask,
                capture_attention=capture_attention,
            )

            # Deep supervision: predict trajectory at each layer
            trajs, scores = traj_head(queries)
            layer_trajectories.append(trajs)
            layer_scores.append(scores)

            if capture_attention:
                decoder_agent_attns.append(agent_attn)
                decoder_map_attns.append(map_attn)

        # Final layer predictions
        final_trajs = layer_trajectories[-1]  # (B, K, future_len, 2)
        final_scores = layer_scores[-1]       # (B, K)

        # NMS selection
        sel_trajs, sel_scores, sel_indices = self._nms_select(final_trajs, final_scores)

        return {
            "trajectories": sel_trajs,          # (B, M, future_len, 2)
            "scores": sel_scores,               # (B, M)
            "all_trajectories": final_trajs,    # (B, K, future_len, 2)
            "all_scores": final_scores,         # (B, K)
            "layer_trajectories": layer_trajectories,
            "layer_scores": layer_scores,
            "nms_indices": sel_indices,
            "decoder_agent_attentions": decoder_agent_attns,
            "decoder_map_attentions": decoder_map_attns,
        }
