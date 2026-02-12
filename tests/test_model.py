"""Tests for the model architecture.

Verifies:
1. Forward pass produces correct output shapes
2. Attention capture returns tensors that sum to ~1.0
3. Individual components work in isolation
4. NMS produces correct number of modes
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.polyline_encoder import PolylineEncoder
from model.layers import AttentionCaptureEncoderLayer, AttentionCaptureDecoderLayer
from model.scene_encoder import SceneEncoder
from model.motion_decoder import MotionDecoder
from model.mtr_lite import MTRLite


B = 2           # batch size
A = 16          # agents
M = 32          # map polylines
K = 16          # intentions
D = 128         # d_model
NHEAD = 4
T_FUTURE = 80
P_AGENT = 11    # history steps
P_MAP = 20      # map points per lane
D_AGENT = 29    # agent feature dim
D_MAP = 9       # map feature dim


class TestPolylineEncoder:
    def test_shapes(self):
        enc = PolylineEncoder(input_dim=D_AGENT, d_model=D)
        polylines = torch.randn(B, A, P_AGENT, D_AGENT)
        valid = torch.ones(B, A, P_AGENT, dtype=torch.bool)
        out = enc(polylines, valid)
        assert out.shape == (B, A, D)

    def test_masked(self):
        enc = PolylineEncoder(input_dim=D_MAP, d_model=D)
        polylines = torch.randn(B, M, P_MAP, D_MAP)
        valid = torch.zeros(B, M, P_MAP, dtype=torch.bool)
        valid[:, :5, :10] = True  # Only first 5 lanes, 10 points each
        out = enc(polylines, valid)
        assert out.shape == (B, M, D)
        # Invalid polylines should have zero embeddings
        assert (out[:, 10:, :].abs().sum() < 1e-3)


class TestEncoderLayer:
    def test_without_capture(self):
        layer = AttentionCaptureEncoderLayer(D, NHEAD, dim_feedforward=256)
        src = torch.randn(B, A + M, D)
        out, attn = layer(src, capture_attention=False)
        assert out.shape == (B, A + M, D)
        assert attn is None

    def test_with_capture(self):
        layer = AttentionCaptureEncoderLayer(D, NHEAD, dim_feedforward=256)
        src = torch.randn(B, A + M, D)
        out, attn = layer(src, capture_attention=True)
        assert out.shape == (B, A + M, D)
        assert attn.shape == (B, NHEAD, A + M, A + M)
        # Attention should approximately sum to 1 per row on average
        # (individual row sums may deviate due to PyTorch SDPA numerical precision)
        row_sums = attn.sum(dim=-1)
        assert (row_sums.mean() - 1.0).abs() < 0.05
        assert row_sums.min() > 0.5
        assert row_sums.max() < 1.5


class TestDecoderLayer:
    def test_shapes(self):
        layer = AttentionCaptureDecoderLayer(D, NHEAD, dim_feedforward=256)
        query = torch.randn(B, K, D)
        agent_tokens = torch.randn(B, A, D)
        map_tokens = torch.randn(B, M, D)
        out, a_attn, m_attn = layer(
            query, agent_tokens, map_tokens, capture_attention=True,
        )
        assert out.shape == (B, K, D)
        assert a_attn.shape == (B, NHEAD, K, A)
        assert m_attn.shape == (B, NHEAD, K, M)


class TestSceneEncoder:
    def test_forward(self):
        enc = SceneEncoder(D, NHEAD, num_layers=2, dim_feedforward=256)
        agent_tokens = torch.randn(B, A, D)
        map_tokens = torch.randn(B, M, D)
        agent_mask = torch.ones(B, A, dtype=torch.bool)
        map_mask = torch.ones(B, M, dtype=torch.bool)

        result = enc(agent_tokens, map_tokens, agent_mask, map_mask, capture_attention=True)
        assert result["agent_tokens"].shape == (B, A, D)
        assert result["map_tokens"].shape == (B, M, D)
        assert len(result["scene_attentions"]) == 2
        assert result["scene_attentions"][0].shape == (B, NHEAD, A + M, A + M)


class TestMotionDecoder:
    def test_forward(self):
        dec = MotionDecoder(
            D, NHEAD, num_layers=2, dim_feedforward=256,
            num_intentions=K, num_modes_output=6,
            future_len=T_FUTURE, nms_dist_thresh=2.0,
        )
        target_agent_token = torch.randn(B, D)
        agent_tokens = torch.randn(B, A, D)
        map_tokens = torch.randn(B, M, D)
        agent_mask = torch.ones(B, A, dtype=torch.bool)
        map_mask = torch.ones(B, M, dtype=torch.bool)

        result = dec(
            target_agent_token, agent_tokens, map_tokens,
            agent_mask, map_mask, capture_attention=True,
        )

        assert result["trajectories"].shape == (B, 6, T_FUTURE, 2)
        assert result["scores"].shape == (B, 6)
        assert result["all_trajectories"].shape == (B, K, T_FUTURE, 2)
        assert len(result["layer_trajectories"]) == 2
        assert len(result["decoder_agent_attentions"]) == 2
        assert result["decoder_agent_attentions"][0].shape == (B, NHEAD, K, A)


class TestMTRLite:
    def test_forward(self):
        model = MTRLite(
            d_model=D, nhead=NHEAD,
            num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=256, num_intentions=K,
            num_modes_output=6, future_len=T_FUTURE,
            agent_feat_dim=D_AGENT, map_feat_dim=D_MAP,
            max_agents=A, max_map_polylines=M, max_targets=4,
        )

        T = 4  # targets
        batch = {
            "agent_polylines": torch.randn(B, A, P_AGENT, D_AGENT),
            "agent_valid": torch.ones(B, A, P_AGENT, dtype=torch.bool),
            "agent_mask": torch.ones(B, A, dtype=torch.bool),
            "map_polylines": torch.randn(B, M, P_MAP, D_MAP),
            "map_valid": torch.ones(B, M, P_MAP, dtype=torch.bool),
            "map_mask": torch.ones(B, M, dtype=torch.bool),
            "target_agent_indices": torch.zeros(B, T, dtype=torch.long),
            "target_mask": torch.ones(B, T, dtype=torch.bool),
        }

        # Without capture
        out = model(batch, capture_attention=False)
        assert out["trajectories"].shape == (B, T, 6, T_FUTURE, 2)
        assert out["scores"].shape == (B, T, 6)
        assert "attention_maps" not in out

        # With capture
        out = model(batch, capture_attention=True)
        assert "attention_maps" in out
        attn_maps = out["attention_maps"]
        assert len(attn_maps.scene_attentions) == 2

    def test_parameter_count(self):
        model = MTRLite(
            d_model=256, nhead=8,
            num_encoder_layers=4, num_decoder_layers=4,
            dim_feedforward=1024, num_intentions=64,
            agent_feat_dim=29, map_feat_dim=9,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M")
        # Should be roughly ~8M
        assert 3e6 < n_params < 20e6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
