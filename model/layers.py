"""Custom transformer layers with attention weight capture.

These layers extend PyTorch's transformer components to optionally return
per-head attention weights (not averaged), enabling detailed visualization.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionCaptureEncoderLayer(nn.Module):
    """Transformer encoder layer that can capture per-head attention weights.

    Standard: pre-norm self-attention + FFN
    When capture_attention=True, returns attention weights (B, nhead, N, N).
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        capture_attention: bool = False,
    ) -> tuple:
        """
        Args:
            src: (B, N, d_model)
            src_key_padding_mask: (B, N) True=ignore
            capture_attention: whether to return attention weights

        Returns:
            output: (B, N, d_model)
            attn_weights: (B, nhead, N, N) if capture_attention else None
        """
        # Pre-norm self-attention
        x = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        src = src + self.dropout(attn_out)

        # FFN
        src = src + self.ffn(self.norm2(src))

        if capture_attention:
            return src, attn_weights
        return src, None


class AttentionCaptureDecoderLayer(nn.Module):
    """Motion decoder layer with agent cross-attention + map cross-attention + FFN.

    Each layer performs:
    1. Cross-attention: query -> agent tokens (captures agent attention)
    2. Cross-attention: query -> map tokens (captures map attention)
    3. FFN

    When capture_attention=True, returns both cross-attention weight tensors.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()

        # Agent cross-attention
        self.agent_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm_agent = nn.LayerNorm(d_model)

        # Map cross-attention
        self.map_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm_map = nn.LayerNorm(d_model)

        # FFN
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        agent_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        agent_key_padding_mask: torch.Tensor = None,
        map_key_padding_mask: torch.Tensor = None,
        capture_attention: bool = False,
    ) -> tuple:
        """
        Args:
            query: (B, K, d_model) intention queries
            agent_tokens: (B, A, d_model) encoded agent tokens
            map_tokens: (B, M, d_model) encoded map tokens
            agent_key_padding_mask: (B, A) True=ignore
            map_key_padding_mask: (B, M) True=ignore
            capture_attention: whether to return attention weights

        Returns:
            output: (B, K, d_model)
            agent_attn: (B, nhead, K, A) if capture else None
            map_attn: (B, nhead, K, M) if capture else None
        """
        # Agent cross-attention
        q = self.norm_agent(query)
        agent_out, agent_attn = self.agent_cross_attn(
            q, agent_tokens, agent_tokens,
            key_padding_mask=agent_key_padding_mask,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        query = query + self.dropout(agent_out)

        # Map cross-attention
        q = self.norm_map(query)
        map_out, map_attn = self.map_cross_attn(
            q, map_tokens, map_tokens,
            key_padding_mask=map_key_padding_mask,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        query = query + self.dropout(map_out)

        # FFN
        query = query + self.ffn(self.norm_ffn(query))

        if capture_attention:
            return query, agent_attn, map_attn
        return query, None, None
