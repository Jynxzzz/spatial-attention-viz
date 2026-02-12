"""Tests for the data pipeline.

Verifies:
1. Agent feature shapes and dimensions
2. Map feature shapes and dimensions
3. Dataset __getitem__ returns correct tensor shapes
4. Collation handles variable-length metadata
5. Token bookkeeper correctness
"""

import os
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.agent_features import AGENT_FEAT_DIM, extract_agent_polyline, extract_all_agents
from data.map_features import MAP_FEAT_DIM, extract_map_polylines, resample_polyline
from data.token_bookkeeper import TokenBookkeeper
from data.collate import mtr_collate_fn


# Path to a test scene pkl
SCENE_LIST = "/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt"
ANCHOR_FRAME = 10
HISTORY_LEN = 11
FUTURE_LEN = 80
MAX_AGENTS = 32
MAX_MAP = 64
MAP_PTS = 20


def _load_test_scene():
    """Load first available scene for testing."""
    with open(SCENE_LIST, "r") as f:
        scenes = [l.strip() for l in f if l.strip()]
    for path in scenes[:10]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f), path
    pytest.skip("No scene files available")


def _get_ego_pose(scene):
    ego_obj = scene["objects"][scene["av_idx"]]
    pos = ego_obj["position"][ANCHOR_FRAME]
    ego_pos = (float(pos["x"]), float(pos["y"]))
    h = ego_obj["heading"][ANCHOR_FRAME]
    ego_heading = float(h[0]) if isinstance(h, (list, tuple)) else float(h)
    return ego_pos, ego_heading


class TestResamplePolyline:
    def test_basic(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
        result = resample_polyline(pts, num_points=7)
        assert result.shape == (7, 2)
        np.testing.assert_allclose(result[0], [0, 0], atol=1e-5)
        np.testing.assert_allclose(result[-1], [3, 0], atol=1e-5)

    def test_single_point(self):
        pts = np.array([[1, 2]], dtype=np.float32)
        result = resample_polyline(pts, num_points=5)
        assert result.shape == (5, 2)


class TestAgentFeatures:
    def test_feature_dim(self):
        assert AGENT_FEAT_DIM == 29

    def test_extract_single_agent(self):
        scene, _ = _load_test_scene()
        ego_pos, ego_heading = _get_ego_pose(scene)
        ego_obj = scene["objects"][scene["av_idx"]]

        feats, valid = extract_agent_polyline(
            ego_obj, ANCHOR_FRAME, HISTORY_LEN, ego_pos, ego_heading, is_ego=True,
        )
        assert feats.shape == (HISTORY_LEN, AGENT_FEAT_DIM)
        assert valid.shape == (HISTORY_LEN,)
        assert valid.sum() > 0

    def test_extract_all_agents(self):
        scene, _ = _load_test_scene()
        ego_pos, ego_heading = _get_ego_pose(scene)

        result = extract_all_agents(
            scene, ANCHOR_FRAME, HISTORY_LEN, ego_pos, ego_heading,
            max_agents=MAX_AGENTS, neighbor_distance=50.0,
        )
        assert result["agent_polylines"].shape == (MAX_AGENTS, HISTORY_LEN, AGENT_FEAT_DIM)
        assert result["agent_valid"].shape == (MAX_AGENTS, HISTORY_LEN)
        assert result["agent_mask"].shape == (MAX_AGENTS,)
        assert result["agent_mask"].sum() > 0
        # Ego should be first
        assert result["agent_mask"][0] == True


class TestMapFeatures:
    def test_feature_dim(self):
        assert MAP_FEAT_DIM == 9

    def test_extract_map(self):
        scene, _ = _load_test_scene()
        ego_pos, ego_heading = _get_ego_pose(scene)

        result = extract_map_polylines(
            scene, ego_pos, ego_heading,
            max_polylines=MAX_MAP, points_per_lane=MAP_PTS,
        )
        assert result["map_polylines"].shape == (MAX_MAP, MAP_PTS, MAP_FEAT_DIM)
        assert result["map_valid"].shape == (MAX_MAP, MAP_PTS)
        assert result["map_mask"].shape == (MAX_MAP,)
        assert result["map_mask"].sum() > 0
        assert result["lane_centerlines_bev"].shape == (MAX_MAP, MAP_PTS, 2)


class TestTokenBookkeeper:
    def test_basic(self):
        bk = TokenBookkeeper(
            num_agents=32, num_map=64,
            agent_obj_ids=[0, 1, 2],
            lane_ids=[100, 101, 102],
            agent_mask=np.ones(32, dtype=bool),
            map_mask=np.ones(64, dtype=bool),
        )
        assert bk.total_tokens == 96
        assert bk.is_agent_token(0)
        assert bk.is_agent_token(31)
        assert not bk.is_agent_token(32)
        assert bk.is_map_token(32)
        assert bk.is_map_token(95)
        assert bk.token_to_agent_idx(5) == 5
        assert bk.token_to_map_idx(40) == 8


class TestCollation:
    def test_collate(self):
        batch = [
            {
                "agent_polylines": torch.randn(MAX_AGENTS, HISTORY_LEN, AGENT_FEAT_DIM),
                "agent_mask": torch.ones(MAX_AGENTS, dtype=torch.bool),
                "agent_ids": [0, 1, 2],
                "lane_ids": [100, 101],
            },
            {
                "agent_polylines": torch.randn(MAX_AGENTS, HISTORY_LEN, AGENT_FEAT_DIM),
                "agent_mask": torch.ones(MAX_AGENTS, dtype=torch.bool),
                "agent_ids": [0, 3],
                "lane_ids": [200],
            },
        ]
        result = mtr_collate_fn(batch)
        assert result["agent_polylines"].shape == (2, MAX_AGENTS, HISTORY_LEN, AGENT_FEAT_DIM)
        assert len(result["agent_ids"]) == 2
        assert result["agent_ids"][0] == [0, 1, 2]

    def test_collate_with_none(self):
        batch = [None, {"x": torch.tensor([1.0])}, None]
        result = mtr_collate_fn(batch)
        assert result is not None
        assert result["x"].shape == (1, 1)  # (batch=1, feat=1)


class TestDataset:
    def test_getitem(self):
        if not os.path.exists(SCENE_LIST):
            pytest.skip("Scene list not available")

        from data.polyline_dataset import PolylineDataset
        ds = PolylineDataset(
            scene_list_path=SCENE_LIST,
            split="train",
            data_fraction=0.001,
            history_len=HISTORY_LEN,
            future_len=FUTURE_LEN,
            max_agents=MAX_AGENTS,
            max_map_polylines=MAX_MAP,
            map_points_per_lane=MAP_PTS,
        )
        assert len(ds) > 0

        sample = ds[0]
        if sample is not None:
            assert sample["agent_polylines"].shape == (MAX_AGENTS, HISTORY_LEN, AGENT_FEAT_DIM)
            assert sample["map_polylines"].shape == (MAX_MAP, MAP_PTS, MAP_FEAT_DIM)
            assert sample["target_future"].shape[1] == FUTURE_LEN
            assert sample["target_future"].shape[2] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
