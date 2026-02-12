#!/usr/bin/env python3
"""Integration test: Data pipeline → Model → Visualization"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

def test_data_to_model():
    """Test: Can model consume data pipeline output?"""
    print("\n" + "="*60)
    print("Integration Test: Data Pipeline → Model")
    print("="*60)

    # Import data pipeline
    print("\n[1/5] Loading data pipeline...")
    from data.polyline_dataset import PolylineDataset
    from data.collate import mtr_collate_fn
    from torch.utils.data import DataLoader

    dataset = PolylineDataset(
        scene_list_path="/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt",
        split="train",
        data_fraction=0.001,  # Use tiny fraction for quick test
        history_len=11,
        future_len=80,
        max_agents=32,
        max_map_polylines=64,
        max_targets=8,
    )

    loader = DataLoader(dataset, batch_size=2, collate_fn=mtr_collate_fn, num_workers=0)
    print(f"  ✓ Dataset created: {len(dataset)} samples")

    # Get one batch
    print("\n[2/5] Loading one batch...")
    batch = next(iter(loader))
    print(f"  ✓ Batch loaded")
    print(f"    - Keys: {list(batch.keys())[:5]}...")
    print(f"    - agent_polylines: {batch['agent_polylines'].shape}")
    print(f"    - map_polylines: {batch['map_polylines'].shape}")
    print(f"    - target_future: {batch['target_future'].shape}")

    # Import model
    print("\n[3/5] Loading model...")
    from model.mtr_lite import MTRLite

    # Create dummy intention points for testing
    intent_points = torch.randn(64, 2) * 20  # Random points in [-40, 40] range

    model = MTRLite(
        agent_feat_dim=29,
        map_feat_dim=9,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_intentions=64,
        num_modes_output=6,
        future_len=80,
        intention_points=intent_points,
    )
    print(f"  ✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # Forward pass WITHOUT attention capture
    print("\n[4/5] Testing forward pass (no attention)...")
    model.eval()
    with torch.no_grad():
        output = model(batch, capture_attention=False)

    print(f"  ✓ Forward pass successful")
    print(f"    - trajectories: {output['trajectories'].shape}")  # (B, T_targets, K, 80, 2)
    print(f"    - scores: {output['scores'].shape}")  # (B, T_targets, K)

    # Forward pass WITH attention capture
    print("\n[5/5] Testing forward pass (with attention)...")
    with torch.no_grad():
        output = model(batch, capture_attention=True)

    print(f"  ✓ Attention capture successful")
    if 'attention_maps' in output and output['attention_maps'] is not None:
        attn = output['attention_maps']
        print(f"    - Scene encoder layers: {len(attn.scene_attentions)}")
        print(f"    - Decoder layers: {len(attn.decoder_agent_attentions)}")
        print(f"    - Scene attn shape: {attn.scene_attentions[0].shape}")
        if len(attn.decoder_agent_attentions) > 0 and len(attn.decoder_agent_attentions[0]) > 0:
            print(f"    - Decoder agent attn (layer 0): {attn.decoder_agent_attentions[0][0].shape}")
            print(f"    - Decoder map attn (layer 0): {attn.decoder_map_attentions[0][0].shape}")

    return batch, output

def test_model_to_viz():
    """Test: Can visualization consume model output?"""
    print("\n" + "="*60)
    print("Integration Test: Model → Visualization")
    print("="*60)

    print("\n[1/3] Running model to get attention data...")
    batch, output = test_data_to_model()

    print("\n[2/3] Testing visualization modules...")
    # Extract attention for first sample in batch
    if 'attention_maps' not in output or output['attention_maps'] is None:
        print("  ✗ No attention data captured")
        return False

    attn = output['attention_maps']
    print(f"  ✓ Attention data available")

    # Try to render space attention (will use dummy scene data for now)
    print("\n[3/3] Testing visualization imports...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Test if viz modules are importable
        from visualization import space_attention_bev
        from visualization import time_attention_diagram
        from visualization import lane_token_activation

        print(f"  ✓ All visualization modules importable")

        # Create dummy scene for visualization
        scene_encoder_attn = attn.scene_attentions[-1][0]  # Last layer, first sample: (nhead, N, N)

        print(f"  ✓ Can access attention weights: {scene_encoder_attn.shape}")

        # Note: Full visualization test requires actual scene data from pkl
        # This will be tested in generate_example_figures.py with trained model

    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    print("\n" + "="*80)
    print(" MTR-Lite Integration Test: Data → Model → Visualization")
    print("="*80)

    try:
        success = test_model_to_viz()

        print("\n" + "="*80)
        if success:
            print("✓ ALL INTEGRATION TESTS PASSED!")
            print("\nNext steps:")
            print("1. Generate intention points (K-means on GT endpoints)")
            print("2. Implement training infrastructure (losses, metrics, PL module)")
            print("3. Train model for 60 epochs (~50 hours)")
            print("4. Generate paper figures with trained model")
        else:
            print("✗ Some tests failed. Check errors above.")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Integration test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
