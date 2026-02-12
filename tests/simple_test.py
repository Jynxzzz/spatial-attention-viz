"""Simple test script without pytest dependency.

Tests:
1. Forward pass produces correct output shapes
2. Attention capture returns tensors that sum to ~1.0
3. Parameter count ≈ 8M
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.polyline_encoder import PolylineEncoder
from model.layers import AttentionCaptureEncoderLayer, AttentionCaptureDecoderLayer
from model.scene_encoder import SceneEncoder
from model.motion_decoder import MotionDecoder
from model.mtr_lite import MTRLite


def test_polyline_encoder():
    print("Testing PolylineEncoder...")
    B, A, P_AGENT, D_AGENT = 2, 16, 11, 29
    D = 128

    enc = PolylineEncoder(input_dim=D_AGENT, d_model=D)
    polylines = torch.randn(B, A, P_AGENT, D_AGENT)
    valid = torch.ones(B, A, P_AGENT, dtype=torch.bool)
    out = enc(polylines, valid)

    assert out.shape == (B, A, D), f"Expected {(B, A, D)}, got {out.shape}"
    print(f"  ✓ Output shape: {out.shape}")

    # Test with masking
    valid_masked = torch.zeros(B, A, P_AGENT, dtype=torch.bool)
    valid_masked[:, :5, :10] = True
    out_masked = enc(polylines, valid_masked)
    assert out_masked.shape == (B, A, D)
    assert out_masked[:, 10:, :].abs().sum() < 1e-3
    print(f"  ✓ Masking works correctly")


def test_encoder_layer():
    print("\nTesting AttentionCaptureEncoderLayer...")
    B, N, D, NHEAD = 2, 48, 128, 4

    layer = AttentionCaptureEncoderLayer(D, NHEAD, dim_feedforward=256)
    src = torch.randn(B, N, D)

    # Without capture
    out, attn = layer(src, capture_attention=False)
    assert out.shape == (B, N, D)
    assert attn is None
    print(f"  ✓ Without capture: output shape {out.shape}")

    # With capture
    out, attn = layer(src, capture_attention=True)
    assert out.shape == (B, N, D)
    assert attn.shape == (B, NHEAD, N, N)

    # Check attention row sums
    row_sums = attn.sum(dim=-1)
    mean_sum = row_sums.mean().item()
    min_sum = row_sums.min().item()
    max_sum = row_sums.max().item()
    print(f"  ✓ With capture: attention shape {attn.shape}")
    print(f"  ✓ Attention row sums: mean={mean_sum:.3f}, min={min_sum:.3f}, max={max_sum:.3f}")
    assert abs(mean_sum - 1.0) < 0.05
    assert min_sum > 0.5 and max_sum < 1.5


def test_decoder_layer():
    print("\nTesting AttentionCaptureDecoderLayer...")
    B, K, A, M, D, NHEAD = 2, 16, 16, 32, 128, 4

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
    print(f"  ✓ Output shape: {out.shape}")
    print(f"  ✓ Agent attention shape: {a_attn.shape}")
    print(f"  ✓ Map attention shape: {m_attn.shape}")


def test_scene_encoder():
    print("\nTesting SceneEncoder...")
    B, A, M, D, NHEAD = 2, 16, 32, 128, 4

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
    print(f"  ✓ Agent output shape: {result['agent_tokens'].shape}")
    print(f"  ✓ Map output shape: {result['map_tokens'].shape}")
    print(f"  ✓ Scene attentions: {len(result['scene_attentions'])} layers")


def test_motion_decoder():
    print("\nTesting MotionDecoder...")
    B, A, M, D, K, NHEAD, T_FUTURE = 2, 16, 32, 128, 16, 4, 80

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
    print(f"  ✓ Trajectories shape: {result['trajectories'].shape}")
    print(f"  ✓ Scores shape: {result['scores'].shape}")
    print(f"  ✓ Deep supervision: {len(result['layer_trajectories'])} layers")
    print(f"  ✓ Decoder attentions: {len(result['decoder_agent_attentions'])} layers")


def test_mtr_lite():
    print("\nTesting MTRLite (full model)...")
    B, A, M, T = 2, 16, 32, 4
    D, NHEAD = 128, 4
    P_AGENT, P_MAP = 11, 20
    D_AGENT, D_MAP = 29, 9
    T_FUTURE = 80
    K = 16

    model = MTRLite(
        d_model=D, nhead=NHEAD,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=256, num_intentions=K,
        num_modes_output=6, future_len=T_FUTURE,
        agent_feat_dim=D_AGENT, map_feat_dim=D_MAP,
        max_agents=A, max_map_polylines=M, max_targets=T,
    )

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
    print(f"  ✓ Without capture: trajectories shape {out['trajectories'].shape}")

    # With capture
    out = model(batch, capture_attention=True)
    assert "attention_maps" in out
    attn_maps = out["attention_maps"]
    assert len(attn_maps.scene_attentions) == 2
    print(f"  ✓ With capture: attention_maps present")
    print(f"  ✓ Scene attentions: {len(attn_maps.scene_attentions)} layers")


def test_parameter_count():
    print("\nTesting parameter count (full-scale model)...")
    model = MTRLite(
        d_model=256, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=1024, num_intentions=64,
        agent_feat_dim=29, map_feat_dim=9,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params / 1e6:.2f}M")
    assert 3e6 < n_params < 20e6
    print(f"  ✓ Parameter count in expected range")


if __name__ == "__main__":
    print("=" * 60)
    print("MTR-Lite Model Architecture Tests")
    print("=" * 60)

    try:
        test_polyline_encoder()
        test_encoder_layer()
        test_decoder_layer()
        test_scene_encoder()
        test_motion_decoder()
        test_mtr_lite()
        test_parameter_count()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
