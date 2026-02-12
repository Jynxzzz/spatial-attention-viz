"""Scene encoder: global self-attention over agent + map tokens.

Concatenates agent and map tokens, applies N layers of self-attention,
then splits back. Optionally captures per-layer, per-head attention maps.
"""

import torch
import torch.nn as nn

from model.layers import AttentionCaptureEncoderLayer


class SceneEncoder(nn.Module):
    """Global self-attention encoder for scene tokens (agents + map lanes).

    Args:
        d_model: token embedding dimension (256)
        nhead: number of attention heads (8)
        num_layers: number of encoder layers (4)
        dim_feedforward: FFN hidden dimension (1024)
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionCaptureEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.num_layers = num_layers

    def forward(
        self,
        agent_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        agent_mask: torch.Tensor,
        map_mask: torch.Tensor,
        capture_attention: bool = False,
    ) -> dict:
        """
        Args:
            agent_tokens: (B, A, d_model)
            map_tokens: (B, M, d_model)
            agent_mask: (B, A) bool - True=valid
            map_mask: (B, M) bool - True=valid
            capture_attention: whether to return attention maps

        Returns:
            dict with:
                agent_tokens: (B, A, d_model)
                map_tokens: (B, M, d_model)
                scene_attentions: list of (B, nhead, A+M, A+M) if capture else []
        """
        B = agent_tokens.shape[0]
        A = agent_tokens.shape[1]
        M = map_tokens.shape[1]

        # Concatenate agent + map tokens
        tokens = torch.cat([agent_tokens, map_tokens], dim=1)  # (B, A+M, d_model)

        # Combined padding mask: True = ignore (invalid)
        combined_mask = ~torch.cat([agent_mask, map_mask], dim=1)  # (B, A+M)

        scene_attentions = []

        for layer in self.layers:
            tokens, attn_weights = layer(
                tokens,
                src_key_padding_mask=combined_mask,
                capture_attention=capture_attention,
            )
            if capture_attention and attn_weights is not None:
                scene_attentions.append(attn_weights)

        tokens = self.final_norm(tokens)

        # Split back
        agent_out = tokens[:, :A, :]
        map_out = tokens[:, A:, :]

        return {
            "agent_tokens": agent_out,
            "map_tokens": map_out,
            "scene_attentions": scene_attentions,
        }
