#!/usr/bin/env python3
"""Quick verification that data pipeline is ready.

Runs in <10 seconds. Use this for quick sanity checks.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("MTR-Lite Data Pipeline Quick Verification")
    print("=" * 60)

    errors = []

    # Test 1: Imports
    print("\n[1/6] Testing imports...")
    try:
        from data import (
            AGENT_FEAT_DIM, MAP_FEAT_DIM,
            PolylineDataset, mtr_collate_fn, TokenBookkeeper,
            extract_all_agents, extract_map_polylines,
        )
        assert AGENT_FEAT_DIM == 29, f"AGENT_FEAT_DIM = {AGENT_FEAT_DIM}, expected 29"
        assert MAP_FEAT_DIM == 9, f"MAP_FEAT_DIM = {MAP_FEAT_DIM}, expected 9"
        print("  ✓ All imports successful")
        print(f"  ✓ AGENT_FEAT_DIM = {AGENT_FEAT_DIM}")
        print(f"  ✓ MAP_FEAT_DIM = {MAP_FEAT_DIM}")
    except Exception as e:
        errors.append(f"Import error: {e}")
        print(f"  ✗ Import failed: {e}")
        return 1

    # Test 2: Token Bookkeeper
    print("\n[2/6] Testing token bookkeeper...")
    try:
        import numpy as np
        bk = TokenBookkeeper(
            num_agents=32, num_map=64,
            agent_obj_ids=[0, 1, 2],
            lane_ids=["lane_100", "lane_101"],
            agent_mask=np.ones(32, dtype=bool),
            map_mask=np.ones(64, dtype=bool),
        )
        assert bk.total_tokens == 96, f"Total tokens = {bk.total_tokens}, expected 96"
        assert bk.is_agent_token(0), "Token 0 should be agent"
        assert bk.is_agent_token(31), "Token 31 should be agent"
        assert bk.is_map_token(32), "Token 32 should be map"
        assert bk.is_map_token(95), "Token 95 should be map"
        print("  ✓ Token bookkeeper works correctly")
        print(f"  ✓ Total tokens: {bk.total_tokens}")
    except Exception as e:
        errors.append(f"Token bookkeeper error: {e}")
        print(f"  ✗ Token bookkeeper failed: {e}")

    # Test 3: Dataset initialization
    print("\n[3/6] Testing dataset initialization...")
    try:
        scene_list = "/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt"
        if not os.path.exists(scene_list):
            raise FileNotFoundError(f"Scene list not found: {scene_list}")

        ds = PolylineDataset(
            scene_list_path=scene_list,
            split="train",
            data_fraction=0.0001,  # Tiny fraction for speed
            history_len=11,
            future_len=80,
            max_agents=32,
            max_map_polylines=64,
        )
        print(f"  ✓ Dataset created with {len(ds)} samples")
    except Exception as e:
        errors.append(f"Dataset initialization error: {e}")
        print(f"  ✗ Dataset initialization failed: {e}")
        return 1

    # Test 4: Load one sample
    print("\n[4/6] Testing sample loading...")
    try:
        sample = None
        for i in range(min(10, len(ds))):
            sample = ds[i]
            if sample is not None:
                break

        if sample is None:
            raise ValueError("No valid samples found in first 10")

        assert "agent_polylines" in sample, "Missing agent_polylines"
        assert "map_polylines" in sample, "Missing map_polylines"
        assert sample["agent_polylines"].shape == (32, 11, 29), \
            f"Wrong agent shape: {sample['agent_polylines'].shape}"
        assert sample["map_polylines"].shape == (64, 20, 9), \
            f"Wrong map shape: {sample['map_polylines'].shape}"
        print("  ✓ Sample loaded successfully")
        print(f"  ✓ Agent polylines: {sample['agent_polylines'].shape}")
        print(f"  ✓ Map polylines: {sample['map_polylines'].shape}")
        print(f"  ✓ Valid targets: {sample['target_mask'].sum().item()}")
    except Exception as e:
        errors.append(f"Sample loading error: {e}")
        print(f"  ✗ Sample loading failed: {e}")

    # Test 5: Collation
    print("\n[5/6] Testing collation...")
    try:
        import torch
        batch = [sample, sample]  # Duplicate
        result = mtr_collate_fn(batch)

        assert result["agent_polylines"].shape[0] == 2, "Wrong batch size"
        assert result["agent_polylines"].shape == (2, 32, 11, 29), \
            f"Wrong batch shape: {result['agent_polylines'].shape}"
        print("  ✓ Collation works correctly")
        print(f"  ✓ Batched shape: {result['agent_polylines'].shape}")
    except Exception as e:
        errors.append(f"Collation error: {e}")
        print(f"  ✗ Collation failed: {e}")

    # Test 6: File structure
    print("\n[6/6] Checking file structure...")
    required_files = [
        "data/agent_features.py",
        "data/map_features.py",
        "data/token_bookkeeper.py",
        "data/polyline_dataset.py",
        "data/collate.py",
        "data/intent_points.py",
        "data/__init__.py",
        "scripts/generate_intent_points.py",
        "tests/test_data_pipeline.py",
    ]

    missing = []
    for f in required_files:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f
        )
        if not os.path.exists(path):
            missing.append(f)

    if missing:
        errors.append(f"Missing files: {missing}")
        print(f"  ✗ Missing files: {missing}")
    else:
        print(f"  ✓ All {len(required_files)} required files present")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"✗ VERIFICATION FAILED with {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("✓ ALL CHECKS PASSED!")
        print("\nData pipeline is ready for training.")
        print("\nNext steps:")
        print("1. Generate intention points:")
        print("   python scripts/generate_intent_points.py \\")
        print("       --scene-list scene_list_123k_signal_ssd.txt \\")
        print("       --output /mnt/hdd12t/models/mtr_lite/intent_points_64.npy \\")
        print("       --k 64 --max-scenes 10000")
        print("\n2. Implement model architecture (Agent B)")
        print("\n3. Set up training infrastructure (Agent D)")
        return 0

if __name__ == "__main__":
    sys.exit(main())
