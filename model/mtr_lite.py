"""MTR-Lite: Top-level model for multi-agent trajectory prediction.

Stitches together:
  1. PolylineEncoder (agent) -> agent tokens (A, d_model)
  2. PolylineEncoder (map) -> map tokens (M, d_model)
  3. SceneEncoder: global self-attention over [agent; map] tokens
  4. MotionDecoder: per target agent, intention query decoding with attention capture

~8M parameters, designed for RTX 4090 with batch_size=4 + AMP.
"""

import torch
import torch.nn as nn

from model.attention_hooks import AttentionMaps
from model.motion_decoder import MotionDecoder
from model.polyline_encoder import PolylineEncoder
from model.scene_encoder import SceneEncoder


class MTRLite(nn.Module):
    """MTR-Lite trajectory prediction model.

    Args:
        d_model: embedding dimension (256)
        nhead: attention heads (8)
        num_encoder_layers: scene encoder layers (4)
        num_decoder_layers: motion decoder layers (4)
        dim_feedforward: FFN hidden dim (1024)
        dropout: dropout rate (0.1)
        num_intentions: intention queries per target (64)
        num_modes_output: final modes after NMS (6)
        future_len: prediction horizon timesteps (80)
        agent_feat_dim: per-timestep agent feature dim (29)
        map_feat_dim: per-point map feature dim (9)
        max_agents: max agent tokens (32)
        max_map_polylines: max map tokens (64)
        max_targets: max prediction targets per scene (8)
        nms_dist_thresh: NMS distance threshold (2.0m)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_intentions: int = 64,
        num_modes_output: int = 6,
        future_len: int = 80,
        agent_feat_dim: int = 29,
        map_feat_dim: int = 9,
        max_agents: int = 32,
        max_map_polylines: int = 64,
        max_targets: int = 8,
        nms_dist_thresh: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_agents = max_agents
        self.max_map_polylines = max_map_polylines
        self.max_targets = max_targets
        self.future_len = future_len

        # Polyline encoders (shared architecture, separate weights)
        self.agent_encoder = PolylineEncoder(
            input_dim=agent_feat_dim, d_model=d_model,
        )
        self.map_encoder = PolylineEncoder(
            input_dim=map_feat_dim, d_model=d_model,
        )

        # Scene encoder
        self.scene_encoder = SceneEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Motion decoder
        self.motion_decoder = MotionDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_intentions=num_intentions,
            num_modes_output=num_modes_output,
            future_len=future_len,
            nms_dist_thresh=nms_dist_thresh,
        )

    def forward(self, batch: dict, capture_attention: bool = False) -> dict:
        """Forward pass for multi-agent trajectory prediction.

        Args:
            batch: dict with keys:
                agent_polylines: (B, A, 11, 29)
                agent_valid: (B, A, 11) bool
                agent_mask: (B, A) bool
                map_polylines: (B, M, 20, 9)
                map_valid: (B, M, 20) bool
                map_mask: (B, M) bool
                target_agent_indices: (B, T) int, which agents to predict for
                target_mask: (B, T) bool
            capture_attention: whether to capture attention maps

        Returns:
            dict with predictions and optionally attention maps
        """
        device = batch["agent_polylines"].device
        B = batch["agent_polylines"].shape[0]

        # 1. Encode polylines to tokens
        agent_tokens = self.agent_encoder(
            batch["agent_polylines"], batch["agent_valid"],
        )  # (B, A, d_model)

        map_tokens = self.map_encoder(
            batch["map_polylines"], batch["map_valid"],
        )  # (B, M, d_model)

        # 2. Scene encoder: global self-attention
        scene_out = self.scene_encoder(
            agent_tokens, map_tokens,
            batch["agent_mask"], batch["map_mask"],
            capture_attention=capture_attention,
        )
        encoded_agents = scene_out["agent_tokens"]  # (B, A, d_model)
        encoded_map = scene_out["map_tokens"]        # (B, M, d_model)

        # 3. Motion decoder: per target agent
        target_indices = batch["target_agent_indices"]  # (B, T)
        target_mask = batch["target_mask"]              # (B, T)

        # Collect predictions for all targets
        all_trajectories = []
        all_scores = []
        all_layer_trajectories = []
        all_layer_scores = []
        all_nms_indices = []
        all_decoder_agent_attns = []
        all_decoder_map_attns = []

        # Process each target
        T = target_mask.shape[1]
        for t in range(T):
            # Get valid samples for this target slot
            valid_b = target_mask[:, t]  # (B,) bool
            if not valid_b.any():
                # Pad with zeros
                all_trajectories.append(
                    torch.zeros(B, self.motion_decoder.num_modes_output, self.future_len, 2, device=device)
                )
                all_scores.append(
                    torch.full((B, self.motion_decoder.num_modes_output), -1e9, device=device)
                )
                all_nms_indices.append(
                    torch.zeros(B, self.motion_decoder.num_modes_output, dtype=torch.long, device=device)
                )
                # Pad layer predictions for deep supervision
                layer_trajs = []
                layer_scores = []
                for _ in range(self.motion_decoder.num_layers):
                    layer_trajs.append(
                        torch.zeros(B, self.motion_decoder.num_intentions, self.future_len, 2, device=device)
                    )
                    layer_scores.append(
                        torch.full((B, self.motion_decoder.num_intentions), -1e9, device=device)
                    )
                all_layer_trajectories.append(layer_trajs)
                all_layer_scores.append(layer_scores)
                continue

            # Extract target agent token for each batch element
            # target_indices[:, t] contains the agent slot index for this target
            t_idx = target_indices[:, t].clamp(min=0)  # (B,) clamp for invalid
            target_agent_token = torch.gather(
                encoded_agents,
                dim=1,
                index=t_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.d_model),
            ).squeeze(1)  # (B, d_model)

            decoder_out = self.motion_decoder(
                target_agent_token,
                encoded_agents,
                encoded_map,
                batch["agent_mask"],
                batch["map_mask"],
                capture_attention=capture_attention,
            )

            all_trajectories.append(decoder_out["trajectories"])
            all_scores.append(decoder_out["scores"])
            all_nms_indices.append(decoder_out["nms_indices"])

            if capture_attention:
                all_decoder_agent_attns.append(decoder_out["decoder_agent_attentions"])
                all_decoder_map_attns.append(decoder_out["decoder_map_attentions"])

            # Collect layer predictions for deep supervision
            all_layer_trajectories.append(decoder_out["layer_trajectories"])
            all_layer_scores.append(decoder_out["layer_scores"])

        # Stack across targets
        pred_trajectories = torch.stack(all_trajectories, dim=1)  # (B, T, M_out, future_len, 2)
        pred_scores = torch.stack(all_scores, dim=1)              # (B, T, M_out)
        nms_indices = torch.stack(all_nms_indices, dim=1)          # (B, T, M_out)

        # Stack layer predictions for deep supervision
        # all_layer_trajectories[t][layer] = (B, K, future_len, 2)
        num_layers = self.motion_decoder.num_layers
        stacked_layer_trajs = []
        stacked_layer_scores = []
        for layer_i in range(num_layers):
            layer_trajs = torch.stack(
                [all_layer_trajectories[t][layer_i] for t in range(T)], dim=1
            )  # (B, T, K, future_len, 2)
            layer_scores = torch.stack(
                [all_layer_scores[t][layer_i] for t in range(T)], dim=1
            )  # (B, T, K)
            stacked_layer_trajs.append(layer_trajs)
            stacked_layer_scores.append(layer_scores)

        result = {
            "trajectories": pred_trajectories,     # (B, T, M_out, future_len, 2)
            "scores": pred_scores,                 # (B, T, M_out)
            "nms_indices": nms_indices,            # (B, T, M_out)
            "layer_trajectories": stacked_layer_trajs,  # list of (B, T, K, future_len, 2)
            "layer_scores": stacked_layer_scores,       # list of (B, T, K)
            "target_mask": target_mask,                 # (B, T)
        }

        # Organize attention maps
        if capture_attention:
            result["attention_maps"] = AttentionMaps(
                scene_attentions=scene_out["scene_attentions"],
                decoder_agent_attentions=all_decoder_agent_attns,
                decoder_map_attentions=all_decoder_map_attns,
                nms_indices=nms_indices,
                num_agents=self.max_agents,
                num_map=self.max_map_polylines,
            )

        return result
