"""MTR-style polyline dataset for trajectory prediction.

Loads Scenario Dreamer's preprocessed Waymo pkl files and extracts:
- Agent polylines (A=32, 11 steps, 29-dim) for all nearby agents
- Map polylines (M=64, 20 points, 9-dim) from waterflow graph
- Ground-truth future trajectories for target agents
- Token bookkeeping metadata for attention visualization

Multi-agent prediction: predicts futures for up to max_targets agents per scene.
"""

import math
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from data.agent_features import (
    AGENT_FEAT_DIM,
    extract_agent_future,
    extract_all_agents,
)
from data.map_features import MAP_FEAT_DIM, extract_map_polylines


class PolylineDataset(Dataset):
    """MTR-Lite dataset producing multi-agent polyline samples.

    Each sample contains:
        - agent_polylines: (A, 11, 29) history features for all agents
        - map_polylines: (M, 20, 9) map lane features
        - target_future: (T, 80, 2) GT future trajectories for target agents
        - target_future_valid: (T, 80) validity masks
        - target_agent_indices: (T,) which agent slots are targets
        - Various masks and metadata for bookkeeping
    """

    def __init__(
        self,
        scene_list_path: str,
        split: str = "train",
        val_ratio: float = 0.15,
        data_fraction: float = 1.0,
        history_len: int = 11,
        future_len: int = 80,
        max_agents: int = 32,
        max_map_polylines: int = 64,
        map_points_per_lane: int = 20,
        max_targets: int = 8,
        neighbor_distance: float = 50.0,
        anchor_frames: list = None,
        augment: bool = False,
        augment_rotation: bool = False,
        seed: int = 42,
    ):
        with open(scene_list_path, "r") as f:
            all_scenes = [line.strip() for line in f if line.strip()]

        all_scenes = [p for p in all_scenes if os.path.exists(p)]

        # Deterministic train/val split
        rng = random.Random(seed)
        indices = list(range(len(all_scenes)))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_scenes) * val_ratio))

        if split == "val":
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        # Apply data fraction (for faster experiments)
        if data_fraction < 1.0:
            n_keep = max(1, int(len(selected) * data_fraction))
            selected = selected[:n_keep]

        self.scenes = [all_scenes[i] for i in selected]
        self.split = split
        self.history_len = history_len
        self.future_len = future_len
        self.max_agents = max_agents
        self.max_map_polylines = max_map_polylines
        self.map_points_per_lane = map_points_per_lane
        self.max_targets = max_targets
        self.neighbor_distance = neighbor_distance
        self.anchor_frames = anchor_frames or [10]
        self.augment = augment and (split == "train")
        self.augment_rotation = augment_rotation
        self.seed = seed

        # One sample per scene (with random/fixed anchor)
        self.samples = [(i, None) for i in range(len(self.scenes))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_idx, fixed_anchor = self.samples[idx]

        try:
            with open(self.scenes[scene_idx], "rb") as f:
                scene = pickle.load(f)
        except Exception:
            return None

        objects = scene["objects"]
        av_idx = scene["av_idx"]
        n_frames = len(objects[0]["position"])

        # Pick anchor frame
        valid_anchors = [
            a for a in self.anchor_frames
            if a >= self.history_len - 1 and a + self.future_len < n_frames
        ]
        if not valid_anchors:
            return None

        if fixed_anchor is not None and fixed_anchor in valid_anchors:
            anchor = fixed_anchor
        elif self.split == "train":
            anchor = random.choice(valid_anchors)
        else:
            anchor = valid_anchors[len(valid_anchors) // 2]

        # Get ego pose at anchor for BEV transform
        ego_obj = objects[av_idx]
        if not ego_obj["valid"][anchor]:
            return None

        ego_pos = (
            float(ego_obj["position"][anchor]["x"]),
            float(ego_obj["position"][anchor]["y"]),
        )
        heading_raw = ego_obj["heading"][anchor]
        ego_heading = float(heading_raw[0]) if isinstance(heading_raw, (list, tuple)) else float(heading_raw)

        # Extract agent polylines
        agent_data = extract_all_agents(
            scene, anchor, self.history_len, ego_pos, ego_heading,
            max_agents=self.max_agents,
            neighbor_distance=self.neighbor_distance,
        )

        # Extract map polylines
        map_data = extract_map_polylines(
            scene, ego_pos, ego_heading,
            max_polylines=self.max_map_polylines,
            points_per_lane=self.map_points_per_lane,
        )

        # Select target agents (those with valid futures)
        target_candidates = []
        for a_slot in range(self.max_agents):
            if not agent_data["agent_mask"][a_slot]:
                continue
            if not agent_data["target_mask"][a_slot]:
                continue
            target_candidates.append(a_slot)

        if not target_candidates:
            return None

        # Limit targets
        if len(target_candidates) > self.max_targets:
            # Always include ego (slot 0), then sample rest
            if 0 in target_candidates:
                target_candidates.remove(0)
                selected_targets = [0] + random.sample(
                    target_candidates, self.max_targets - 1
                )
            else:
                selected_targets = random.sample(target_candidates, self.max_targets)
        else:
            selected_targets = target_candidates

        n_targets = len(selected_targets)

        # Extract GT futures for targets
        target_future = np.zeros((self.max_targets, self.future_len, 2), dtype=np.float32)
        target_future_valid = np.zeros((self.max_targets, self.future_len), dtype=bool)
        target_indices = np.full(self.max_targets, -1, dtype=np.int64)
        target_agent_types = np.full(self.max_targets, -1, dtype=np.int64)
        target_mask = np.zeros(self.max_targets, dtype=bool)

        for t_idx, a_slot in enumerate(selected_targets):
            obj_idx = agent_data["agent_ids"][a_slot]
            obj = objects[obj_idx]

            future, valid = extract_agent_future(
                obj, anchor, self.future_len, ego_pos, ego_heading,
            )
            target_future[t_idx] = future
            target_future_valid[t_idx] = valid
            target_indices[t_idx] = a_slot
            target_agent_types[t_idx] = agent_data["agent_types"][a_slot]
            target_mask[t_idx] = True

        # Data augmentation: random rotation
        if self.augment and self.augment_rotation:
            angle = random.uniform(-math.pi, math.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

            # Rotate agent positions (first 2 dims), prev_pos (next 2), vel, accel, heading
            ap = agent_data["agent_polylines"]
            ap[:, :, 0:2] = (ap[:, :, 0:2].reshape(-1, 2) @ R.T).reshape(self.max_agents, self.history_len, 2)
            ap[:, :, 2:4] = (ap[:, :, 2:4].reshape(-1, 2) @ R.T).reshape(self.max_agents, self.history_len, 2)
            ap[:, :, 4:6] = (ap[:, :, 4:6].reshape(-1, 2) @ R.T).reshape(self.max_agents, self.history_len, 2)
            ap[:, :, 6:8] = (ap[:, :, 6:8].reshape(-1, 2) @ R.T).reshape(self.max_agents, self.history_len, 2)
            # heading sin/cos rotation
            h_sin = ap[:, :, 8].copy()
            h_cos = ap[:, :, 9].copy()
            ap[:, :, 8] = h_sin * c + h_cos * s
            ap[:, :, 9] = -h_sin * s + h_cos * c

            # Rotate map polylines (pos and prev_point and direction)
            mp = map_data["map_polylines"]
            mp[:, :, 0:2] = (mp[:, :, 0:2].reshape(-1, 2) @ R.T).reshape(self.max_map_polylines, self.map_points_per_lane, 2)
            mp[:, :, 2:4] = (mp[:, :, 2:4].reshape(-1, 2) @ R.T).reshape(self.max_map_polylines, self.map_points_per_lane, 2)
            mp[:, :, 7:9] = (mp[:, :, 7:9].reshape(-1, 2) @ R.T).reshape(self.max_map_polylines, self.map_points_per_lane, 2)

            # Rotate futures
            target_future[:, :, :2] = (target_future[:, :, :2].reshape(-1, 2) @ R.T).reshape(self.max_targets, self.future_len, 2)

            # Rotate centerlines_bev
            cl = map_data["lane_centerlines_bev"]
            map_data["lane_centerlines_bev"] = (cl.reshape(-1, 2) @ R.T).reshape(cl.shape)

        sample = {
            # Agent tokens
            "agent_polylines": torch.from_numpy(agent_data["agent_polylines"]),          # (A, 11, 29)
            "agent_valid": torch.from_numpy(agent_data["agent_valid"]),                  # (A, 11)
            "agent_mask": torch.from_numpy(agent_data["agent_mask"]),                    # (A,)
            # Map tokens
            "map_polylines": torch.from_numpy(map_data["map_polylines"]),                # (M, 20, 9)
            "map_valid": torch.from_numpy(map_data["map_valid"]),                        # (M, 20)
            "map_mask": torch.from_numpy(map_data["map_mask"]),                          # (M,)
            # Target futures
            "target_future": torch.from_numpy(target_future),                            # (T, 80, 2)
            "target_future_valid": torch.from_numpy(target_future_valid),                # (T, 80)
            "target_agent_indices": torch.from_numpy(target_indices),                    # (T,)
            "target_agent_types": torch.from_numpy(target_agent_types),                  # (T,) 0=veh,1=ped,2=cyc
            "target_mask": torch.from_numpy(target_mask),                                # (T,)
            # Spatial metadata for visualization
            "lane_centerlines_bev": torch.from_numpy(map_data["lane_centerlines_bev"]),  # (M, 20, 2)
            # Metadata (non-tensor, handled by collate)
            "agent_ids": agent_data["agent_ids"],
            "lane_ids": map_data["lane_ids"],
            "scene_path": self.scenes[scene_idx],
            "ego_pos": ego_pos,
            "ego_heading": ego_heading,
            "anchor_frame": anchor,
        }

        return sample
