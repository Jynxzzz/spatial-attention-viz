"""Tests for attention extraction and organization.

Verifies:
1. AttentionMaps slicing methods return correct shapes
2. Entropy computation
3. Head aggregation
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.attention_hooks import AttentionMaps


A = 16
M = 32
K = 64
NHEAD = 8
B = 2


def _make_attn_maps():
    """Create AttentionMaps with synthetic data."""
    # Scene attention: softmax-like distributions
    scene_attn = torch.softmax(torch.randn(B, NHEAD, A + M, A + M), dim=-1)

    # Decoder attentions
    dec_agent = [torch.softmax(torch.randn(B, NHEAD, K, A), dim=-1) for _ in range(4)]
    dec_map = [torch.softmax(torch.randn(B, NHEAD, K, M), dim=-1) for _ in range(4)]

    nms_indices = torch.randint(0, K, (B, 8, 6))

    return AttentionMaps(
        scene_attentions=[scene_attn],
        decoder_agent_attentions=dec_agent,
        decoder_map_attentions=dec_map,
        nms_indices=nms_indices,
        num_agents=A,
        num_map=M,
    )


class TestAttentionMaps:
    def test_scene_slicing(self):
        am = _make_attn_maps()

        a2a = am.get_scene_agent_to_agent(layer=0, batch_idx=0)
        assert a2a.shape == (NHEAD, A, A)

        a2m = am.get_scene_agent_to_map(layer=0, batch_idx=0)
        assert a2m.shape == (NHEAD, A, M)

        m2a = am.get_scene_map_to_agent(layer=0, batch_idx=0)
        assert m2a.shape == (NHEAD, M, A)

    def test_decoder_attn(self):
        am = _make_attn_maps()

        agent_attn = am.get_decoder_agent_attn(layer=0, mode_idx=0, batch_idx=0)
        assert agent_attn.shape == (NHEAD, A)

        map_attn = am.get_decoder_map_attn(layer=2, mode_idx=10, batch_idx=1)
        assert map_attn.shape == (NHEAD, M)

    def test_attention_sums(self):
        am = _make_attn_maps()

        # Scene attention rows should sum to ~1
        scene = am.scene_attentions[0][0]  # (nhead, A+M, A+M)
        row_sums = scene.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01)

        # Decoder agent attention should sum to ~1
        dec_a = am.decoder_agent_attentions[0][0]  # (nhead, K, A)
        row_sums = dec_a.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01)

    def test_entropy(self):
        am = _make_attn_maps()

        # Uniform attention should have max entropy
        uniform = torch.ones(10) / 10
        entropy = am.compute_entropy(uniform)
        expected = -10 * (0.1 * (-torch.tensor(10.0).log2()))
        assert abs(entropy.item() - expected.item()) < 0.01

        # Peaked attention should have low entropy
        peaked = torch.zeros(10)
        peaked[0] = 1.0
        entropy = am.compute_entropy(peaked)
        assert entropy.item() < 0.01

    def test_aggregate_heads(self):
        am = _make_attn_maps()
        attn = torch.randn(NHEAD, A, M)

        mean_agg = am.aggregate_heads(attn, method="mean")
        assert mean_agg.shape == (A, M)

        max_agg = am.aggregate_heads(attn, method="max")
        assert max_agg.shape == (A, M)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
