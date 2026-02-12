"""Attention extraction and organization utilities.

Provides clean interfaces for extracting, slicing, and organizing attention
weights from the model's forward pass. Used by visualization modules.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class AttentionMaps:
    """Organized container for all captured attention weights from one forward pass.

    Scene encoder attentions:
        scene_attentions[layer]: (B, nhead, A+M, A+M) self-attention

    Decoder attentions (per target agent):
        decoder_agent_attentions[layer]: (B, nhead, K, A) query->agent cross-attention
        decoder_map_attentions[layer]: (B, nhead, K, M) query->map cross-attention

    Dimensions:
        B = batch size
        A = num_agents (32)
        M = num_map_polylines (64)
        K = num_intentions (64)
        nhead = 8
    """
    scene_attentions: list = field(default_factory=list)      # list of (B, nhead, A+M, A+M)
    decoder_agent_attentions: list = field(default_factory=list)  # list of (B, nhead, K, A)
    decoder_map_attentions: list = field(default_factory=list)    # list of (B, nhead, K, M)
    nms_indices: Optional[torch.Tensor] = None                    # (B, M_out) selected mode indices
    num_agents: int = 0
    num_map: int = 0

    def get_scene_agent_to_agent(self, layer: int, batch_idx: int = 0) -> torch.Tensor:
        """Extract agent-to-agent attention from scene encoder.

        Returns: (nhead, A, A) attention from agent tokens to agent tokens.
        """
        attn = self.scene_attentions[layer][batch_idx]  # (nhead, A+M, A+M)
        A = self.num_agents
        return attn[:, :A, :A]  # (nhead, A, A)

    def get_scene_agent_to_map(self, layer: int, batch_idx: int = 0) -> torch.Tensor:
        """Extract agent-to-map attention from scene encoder.

        Returns: (nhead, A, M) how much agent tokens attend to map tokens.
        """
        attn = self.scene_attentions[layer][batch_idx]
        A = self.num_agents
        return attn[:, :A, A:]  # (nhead, A, M)

    def get_scene_map_to_agent(self, layer: int, batch_idx: int = 0) -> torch.Tensor:
        """Extract map-to-agent attention from scene encoder.

        Returns: (nhead, M, A) how much map tokens attend to agent tokens.
        """
        attn = self.scene_attentions[layer][batch_idx]
        A = self.num_agents
        return attn[:, A:, :A]  # (nhead, M, A)

    def get_decoder_agent_attn(
        self, layer: int, mode_idx: int, batch_idx: int = 0,
    ) -> torch.Tensor:
        """Get decoder agent cross-attention for a specific mode/query.

        Returns: (nhead, A) attention weights from intention query to agent tokens.
        """
        attn = self.decoder_agent_attentions[layer][batch_idx]  # (nhead, K, A)
        return attn[:, mode_idx, :]  # (nhead, A)

    def get_decoder_map_attn(
        self, layer: int, mode_idx: int, batch_idx: int = 0,
    ) -> torch.Tensor:
        """Get decoder map cross-attention for a specific mode/query.

        Returns: (nhead, M) attention weights from intention query to map tokens.
        """
        attn = self.decoder_map_attentions[layer][batch_idx]  # (nhead, K, M)
        return attn[:, mode_idx, :]  # (nhead, M)

    def get_winning_mode_attention(
        self, layer: int, winning_nms_idx: int, batch_idx: int = 0,
    ) -> dict:
        """Get attention for the winning (best) mode after NMS.

        Args:
            layer: decoder layer index
            winning_nms_idx: index into NMS-selected modes (0-5)
            batch_idx: batch index

        Returns:
            dict with 'agent_attn' (nhead, A) and 'map_attn' (nhead, M)
        """
        # Map NMS index back to intention query index
        intention_idx = self.nms_indices[batch_idx, winning_nms_idx].item()
        return {
            "agent_attn": self.get_decoder_agent_attn(layer, intention_idx, batch_idx),
            "map_attn": self.get_decoder_map_attn(layer, intention_idx, batch_idx),
        }

    def compute_entropy(self, attn: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy (bits) for analysis.

        Higher entropy = more uniform attention (less focused).

        Args:
            attn: (..., N) attention weights (should sum to ~1 along last dim)

        Returns:
            entropy: (...) in bits
        """
        eps = 1e-8
        attn = attn.clamp(min=eps)
        return -(attn * attn.log2()).sum(dim=-1)

    def aggregate_heads(self, attn: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """Aggregate attention across heads.

        Args:
            attn: (nhead, ...) per-head attention
            method: "mean" or "max"

        Returns:
            aggregated: (...) single attention map
        """
        if method == "mean":
            return attn.mean(dim=0)
        elif method == "max":
            return attn.max(dim=0).values
        else:
            raise ValueError(f"Unknown method: {method}")
