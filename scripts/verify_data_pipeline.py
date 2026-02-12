"""Quick verification script for the data pipeline.

Tests:
1. Loading a pkl file
2. Extracting agent features
3. Extracting map features
4. Token bookkeeper
5. Full dataset sample
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.agent_features import AGENT_FEAT_DIM, extract_all_agents, extract_agent_future
from data.map_features import MAP_FEAT_DIM, extract_map_polylines
from data.token_bookkeeper import TokenBookkeeper
from data.polyline_dataset import PolylineDataset
from data.collate import mtr_collate_fn

SCENE_LIST = "/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt"


def verify_pkl_structure():
    """Test 1: Load and inspect a pkl file."""
    print("=" * 60)
    print("Test 1: Loading pkl file structure")
    print("=" * 60)

    with open(SCENE_LIST, "r") as f:
        scenes = [l.strip() for l in f if l.strip()]

    for path in scenes[:5]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                scene = pickle.load(f)

            print(f"✓ Loaded: {os.path.basename(path)}")
            print(f"  Keys: {list(scene.keys())}")
            print(f"  Num objects: {len(scene['objects'])}")
            print(f"  AV index: {scene['av_idx']}")
            print(f"  Num frames: {len(scene['objects'][0]['position'])}")
            print(f"  Lane graph keys: {list(scene['lane_graph'].keys())}")
            print(f"  Num lanes: {len(scene['lane_graph']['lanes'])}")
            return scene, path

    raise FileNotFoundError("No valid scene files found")


def verify_agent_features(scene):
    """Test 2: Extract agent features."""
    print("\n" + "=" * 60)
    print("Test 2: Agent feature extraction")
    print("=" * 60)

    anchor = 10
    history_len = 11
    av_idx = scene["av_idx"]
    ego_obj = scene["objects"][av_idx]

    if not ego_obj["valid"][anchor]:
        print("✗ Ego invalid at anchor")
        return False

    ego_pos = (
        float(ego_obj["position"][anchor]["x"]),
        float(ego_obj["position"][anchor]["y"]),
    )
    h = ego_obj["heading"][anchor]
    ego_heading = float(h[0]) if isinstance(h, (list, tuple)) else float(h)

    print(f"  Ego position: {ego_pos}")
    print(f"  Ego heading: {ego_heading:.2f} deg")

    result = extract_all_agents(
        scene, anchor, history_len, ego_pos, ego_heading,
        max_agents=32, neighbor_distance=50.0,
    )

    print(f"✓ Agent polylines shape: {result['agent_polylines'].shape}")
    print(f"  Expected: (32, 11, {AGENT_FEAT_DIM})")
    print(f"✓ Valid agents: {result['agent_mask'].sum()}")
    print(f"✓ Target agents: {result['target_mask'].sum()}")

    assert result['agent_polylines'].shape == (32, 11, AGENT_FEAT_DIM)
    assert result['agent_mask'][0] == True  # Ego should be first

    return ego_pos, ego_heading


def verify_map_features(scene, ego_pos, ego_heading):
    """Test 3: Extract map features."""
    print("\n" + "=" * 60)
    print("Test 3: Map feature extraction")
    print("=" * 60)

    result = extract_map_polylines(
        scene, ego_pos, ego_heading,
        max_polylines=64, points_per_lane=20,
    )

    print(f"✓ Map polylines shape: {result['map_polylines'].shape}")
    print(f"  Expected: (64, 20, {MAP_FEAT_DIM})")
    print(f"✓ Valid lanes: {result['map_mask'].sum()}")
    print(f"✓ Ego lane ID: {result['ego_lane_id']}")
    print(f"✓ Lane IDs count: {len(result['lane_ids'])}")

    assert result['map_polylines'].shape == (64, 20, MAP_FEAT_DIM)
    assert result['map_mask'].sum() > 0

    return result


def verify_token_bookkeeper():
    """Test 4: Token bookkeeper."""
    print("\n" + "=" * 60)
    print("Test 4: Token bookkeeper")
    print("=" * 60)

    bk = TokenBookkeeper(
        num_agents=32,
        num_map=64,
        agent_obj_ids=[0, 1, 2, 3],
        lane_ids=["lane_100", "lane_101", "lane_102"],
        agent_mask=np.ones(32, dtype=bool),
        map_mask=np.ones(64, dtype=bool),
    )

    print(f"✓ Total tokens: {bk.total_tokens}")
    print(f"✓ Agent range: {bk.agent_range}")
    print(f"✓ Map range: {bk.map_range}")

    # Test token mappings
    assert bk.is_agent_token(0)
    assert bk.is_agent_token(31)
    assert not bk.is_agent_token(32)
    assert bk.is_map_token(32)
    assert bk.is_map_token(95)

    print(f"✓ Token 5 is: {bk.describe_token(5)}")
    print(f"✓ Token 40 is: {bk.describe_token(40)}")

    print("✓ All token bookkeeper tests passed")


def verify_dataset():
    """Test 5: Full dataset sample."""
    print("\n" + "=" * 60)
    print("Test 5: Dataset sample")
    print("=" * 60)

    ds = PolylineDataset(
        scene_list_path=SCENE_LIST,
        split="train",
        data_fraction=0.001,  # Very small for quick test
        history_len=11,
        future_len=80,
        max_agents=32,
        max_map_polylines=64,
        map_points_per_lane=20,
        max_targets=8,
    )

    print(f"✓ Dataset length: {len(ds)}")
    assert len(ds) > 0

    sample = None
    for i in range(min(10, len(ds))):
        sample = ds[i]
        if sample is not None:
            break

    if sample is None:
        print("✗ No valid samples found")
        return False

    print(f"✓ Sample keys: {list(sample.keys())}")
    print(f"✓ Agent polylines: {sample['agent_polylines'].shape}")
    print(f"✓ Map polylines: {sample['map_polylines'].shape}")
    print(f"✓ Target future: {sample['target_future'].shape}")
    print(f"✓ Target indices: {sample['target_agent_indices']}")
    print(f"✓ Valid targets: {sample['target_mask'].sum().item()}")

    assert sample['agent_polylines'].shape == (32, 11, AGENT_FEAT_DIM)
    assert sample['map_polylines'].shape == (64, 20, MAP_FEAT_DIM)
    assert sample['target_future'].shape[1] == 80
    assert sample['target_future'].shape[2] == 2

    return sample


def verify_collation(sample):
    """Test 6: Collation."""
    print("\n" + "=" * 60)
    print("Test 6: Batch collation")
    print("=" * 60)

    batch = [sample, sample]  # Duplicate for batch
    result = mtr_collate_fn(batch)

    print(f"✓ Batched agent polylines: {result['agent_polylines'].shape}")
    print(f"✓ Batched map polylines: {result['map_polylines'].shape}")
    print(f"✓ Agent IDs type: {type(result['agent_ids'])}, len={len(result['agent_ids'])}")

    assert result['agent_polylines'].shape[0] == 2
    assert result['map_polylines'].shape[0] == 2
    assert len(result['agent_ids']) == 2

    print("✓ Collation test passed")


def main():
    print("\n" + "=" * 60)
    print("MTR-Lite Data Pipeline Verification")
    print("=" * 60)

    try:
        # Test 1: Load pkl
        scene, path = verify_pkl_structure()

        # Test 2: Agent features
        ego_pos, ego_heading = verify_agent_features(scene)

        # Test 3: Map features
        map_data = verify_map_features(scene, ego_pos, ego_heading)

        # Test 4: Token bookkeeper
        verify_token_bookkeeper()

        # Test 5: Dataset
        sample = verify_dataset()

        # Test 6: Collation
        if sample:
            verify_collation(sample)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nData pipeline is ready for training.")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
