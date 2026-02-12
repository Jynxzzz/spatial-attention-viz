"""Attention extraction: runs forward pass with capture and organizes results.

Provides a clean interface for:
1. Running a scene through the model with attention capture
2. Organizing the captured attention maps with bookkeeper
3. Preparing data for the three visualization types
"""

import pickle

import numpy as np
import torch

from data.agent_features import extract_agent_future, extract_all_agents
from data.map_features import extract_map_polylines
from data.token_bookkeeper import TokenBookkeeper
from model.attention_hooks import AttentionMaps


def extract_scene_attention(
    model,
    scene_path: str,
    anchor_frame: int = 10,
    history_len: int = 11,
    future_len: int = 80,
    max_agents: int = 32,
    max_map_polylines: int = 64,
    map_points_per_lane: int = 20,
    device: str = "cuda",
) -> dict:
    """Run a single scene through the model and extract all attention maps.

    Returns:
        dict with:
            attention_maps: AttentionMaps object
            bookkeeper: TokenBookkeeper for token<->entity mapping
            predictions: model output dict
            scene_data: raw scene dict
            ego_pos: (x, y) ego position
            ego_heading: ego heading degrees
            agent_data: extracted agent features
            map_data: extracted map features
    """
    # Load scene
    with open(scene_path, "rb") as f:
        scene = pickle.load(f)

    objects = scene["objects"]
    av_idx = scene["av_idx"]

    # Get ego pose
    ego_obj = objects[av_idx]
    ego_pos = (
        float(ego_obj["position"][anchor_frame]["x"]),
        float(ego_obj["position"][anchor_frame]["y"]),
    )
    heading_raw = ego_obj["heading"][anchor_frame]
    ego_heading = float(heading_raw[0]) if isinstance(heading_raw, (list, tuple)) else float(heading_raw)

    # Extract features
    agent_data = extract_all_agents(
        scene, anchor_frame, history_len, ego_pos, ego_heading,
        max_agents=max_agents, neighbor_distance=50.0,
    )
    map_data = extract_map_polylines(
        scene, ego_pos, ego_heading,
        max_polylines=max_map_polylines,
        points_per_lane=map_points_per_lane,
    )

    # Select targets
    target_indices = np.full(8, -1, dtype=np.int64)
    target_mask = np.zeros(8, dtype=bool)
    target_future = np.zeros((8, future_len, 2), dtype=np.float32)
    target_future_valid = np.zeros((8, future_len), dtype=bool)

    target_candidates = [
        i for i in range(max_agents)
        if agent_data["agent_mask"][i] and agent_data["target_mask"][i]
    ]

    for t_idx, a_slot in enumerate(target_candidates[:8]):
        obj_idx = agent_data["agent_ids"][a_slot]
        obj = objects[obj_idx]
        future, valid = extract_agent_future(obj, anchor_frame, future_len, ego_pos, ego_heading)
        target_future[t_idx] = future
        target_future_valid[t_idx] = valid
        target_indices[t_idx] = a_slot
        target_mask[t_idx] = True

    # Build batch (B=1)
    batch = {
        "agent_polylines": torch.from_numpy(agent_data["agent_polylines"]).unsqueeze(0).to(device),
        "agent_valid": torch.from_numpy(agent_data["agent_valid"]).unsqueeze(0).to(device),
        "agent_mask": torch.from_numpy(agent_data["agent_mask"]).unsqueeze(0).to(device),
        "map_polylines": torch.from_numpy(map_data["map_polylines"]).unsqueeze(0).to(device),
        "map_valid": torch.from_numpy(map_data["map_valid"]).unsqueeze(0).to(device),
        "map_mask": torch.from_numpy(map_data["map_mask"]).unsqueeze(0).to(device),
        "target_agent_indices": torch.from_numpy(target_indices).unsqueeze(0).to(device),
        "target_mask": torch.from_numpy(target_mask).unsqueeze(0).to(device),
        "target_future": torch.from_numpy(target_future).unsqueeze(0).to(device),
        "target_future_valid": torch.from_numpy(target_future_valid).unsqueeze(0).to(device),
    }

    # Forward pass with attention capture
    model.eval()
    with torch.no_grad():
        output = model(batch, capture_attention=True)

    # Build bookkeeper
    bookkeeper = TokenBookkeeper.from_batch_sample(
        agent_obj_ids=agent_data["agent_ids"],
        lane_ids=map_data["lane_ids"],
        agent_mask=agent_data["agent_mask"],
        map_mask=map_data["map_mask"].astype(bool),
        max_agents=max_agents,
        max_map=max_map_polylines,
        target_agent_indices=target_candidates[:8],
    )

    return {
        "attention_maps": output.get("attention_maps"),
        "bookkeeper": bookkeeper,
        "predictions": output,
        "scene_data": scene,
        "ego_pos": ego_pos,
        "ego_heading": ego_heading,
        "agent_data": agent_data,
        "map_data": map_data,
        "batch": batch,
    }
