"""Qualitative analysis: auto-select interesting scenes for visualization.

Selects scenes based on criteria:
1. Intersection scenarios (many lane branches, traffic lights)
2. Lane change scenarios (lateral motion in GT)
3. Highway scenarios (high velocity, long lanes)
4. Complex interaction (many nearby agents)
5. Failure cases (high prediction error)
"""

import math
import os
import pickle
import random

import numpy as np
import torch

from data.agent_features import extract_all_agents
from data.map_features import extract_map_polylines


def analyze_scene_properties(scene_path: str, anchor_frame: int = 10) -> dict:
    """Analyze a scene and compute properties for selection.

    Returns dict with:
        n_agents: number of valid agents
        n_lanes: number of lanes in local graph
        has_traffic_light: whether traffic lights are present
        ego_speed: ego speed at anchor (m/s)
        lateral_motion: max lateral displacement in GT future
        lane_branches: number of unique successor chains
    """
    with open(scene_path, "rb") as f:
        scene = pickle.load(f)

    objects = scene["objects"]
    av_idx = scene["av_idx"]
    ego_obj = objects[av_idx]

    if not ego_obj["valid"][anchor_frame]:
        return None

    # Ego speed
    vel = ego_obj["velocity"][anchor_frame]
    ego_speed = math.sqrt(float(vel["x"]) ** 2 + float(vel["y"]) ** 2)

    # Count valid agents at anchor
    n_agents = sum(
        1 for obj in objects
        if anchor_frame < len(obj["valid"]) and obj["valid"][anchor_frame]
    )

    # Lane count
    n_lanes = len(scene["lane_graph"]["lanes"])

    # Traffic lights
    tl = scene.get("traffic_lights", [])
    has_tl = bool(tl and any(len(frame) > 0 for frame in tl if isinstance(frame, list)))

    # Lane branching (successor count)
    suc = scene["lane_graph"]["suc_pairs"]
    branch_count = sum(1 for v in suc.values() if len(v) > 1)

    # Lateral motion in GT future
    ego_pos = (float(ego_obj["position"][anchor_frame]["x"]),
               float(ego_obj["position"][anchor_frame]["y"]))
    heading_raw = ego_obj["heading"][anchor_frame]
    ego_heading = float(heading_raw[0]) if isinstance(heading_raw, (list, tuple)) else float(heading_raw)

    heading_rad = math.radians(ego_heading)
    adjusted = heading_rad - np.pi / 2
    c, s = np.cos(-adjusted), np.sin(-adjusted)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    ego_xy = np.array(ego_pos, dtype=np.float64)

    max_lateral = 0.0
    for t in range(anchor_frame + 1, min(anchor_frame + 81, len(ego_obj["valid"]))):
        if not ego_obj["valid"][t]:
            continue
        pos = np.array([float(ego_obj["position"][t]["x"]),
                        float(ego_obj["position"][t]["y"])], dtype=np.float64)
        bev = ((pos - ego_xy) @ R.T).astype(np.float32)
        max_lateral = max(max_lateral, abs(bev[1]))

    return {
        "scene_path": scene_path,
        "n_agents": n_agents,
        "n_lanes": n_lanes,
        "has_traffic_light": has_tl,
        "ego_speed": ego_speed,
        "lateral_motion": max_lateral,
        "lane_branches": branch_count,
    }


def select_interesting_scenes(
    scene_list_path: str,
    n_scenes: int = 20,
    max_scan: int = 2000,
    seed: int = 42,
) -> dict:
    """Scan scenes and select diverse interesting scenarios.

    Returns dict with category -> list of (scene_path, properties) tuples:
        'intersection': high traffic lights + many branches
        'lane_change': high lateral motion
        'highway': high speed
        'crowded': many agents
    """
    with open(scene_list_path, "r") as f:
        all_scenes = [line.strip() for line in f if line.strip()]
    all_scenes = [p for p in all_scenes if os.path.exists(p)]

    rng = random.Random(seed)
    rng.shuffle(all_scenes)
    scan_scenes = all_scenes[:max_scan]

    properties = []
    for path in scan_scenes:
        try:
            props = analyze_scene_properties(path)
            if props is not None:
                properties.append(props)
        except Exception:
            continue

    if not properties:
        return {}

    # Sort by each criterion and pick top scenes
    per_category = n_scenes // 4

    categories = {}

    # Intersection: traffic lights + many lane branches
    intersection = sorted(
        [p for p in properties if p["has_traffic_light"]],
        key=lambda x: x["lane_branches"], reverse=True,
    )
    categories["intersection"] = intersection[:per_category]

    # Lane change: high lateral motion
    lane_change = sorted(properties, key=lambda x: x["lateral_motion"], reverse=True)
    categories["lane_change"] = lane_change[:per_category]

    # Highway: high speed
    highway = sorted(properties, key=lambda x: x["ego_speed"], reverse=True)
    categories["highway"] = highway[:per_category]

    # Crowded: many agents
    crowded = sorted(properties, key=lambda x: x["n_agents"], reverse=True)
    categories["crowded"] = crowded[:per_category]

    total = sum(len(v) for v in categories.values())
    print(f"Selected {total} interesting scenes across {len(categories)} categories:")
    for cat, scenes in categories.items():
        print(f"  {cat}: {len(scenes)} scenes")
        if scenes:
            print(f"    Example: {scenes[0]['scene_path']}")

    return categories


def select_failure_cases(
    model,
    scene_list_path: str,
    config_path: str,
    n_cases: int = 10,
    device: str = "cuda",
) -> list:
    """Find scenes where the model performs worst (highest minFDE).

    Returns list of (scene_path, minFDE, minADE) tuples sorted by error.
    """
    import yaml
    from data.polyline_dataset import PolylineDataset
    from data.collate import mtr_collate_fn
    from torch.utils.data import DataLoader
    from training.metrics import compute_min_ade_fde

    cfg = yaml.safe_load(open(config_path))
    data_cfg = cfg["data"]

    val_dataset = PolylineDataset(
        scene_list_path=data_cfg["scene_list"],
        split="val",
        data_fraction=0.1,  # Quick scan
        history_len=data_cfg["history_len"],
        future_len=data_cfg["future_len"],
        max_agents=data_cfg["max_agents"],
        max_map_polylines=data_cfg["max_map_polylines"],
        map_points_per_lane=data_cfg["map_points_per_lane"],
        augment=False,
        seed=cfg.get("seed", 42),
    )

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=mtr_collate_fn)

    errors = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            output = model(batch, capture_attention=False)

            # Check first target
            if batch["target_mask"][0, 0]:
                pred = output["trajectories"][0, 0]  # (M, future_len, 2)
                gt = batch["target_future"][0, 0]
                valid = batch["target_future_valid"][0, 0]

                metrics = compute_min_ade_fde(pred, gt, valid)
                scene_path = batch["scene_path"][0] if "scene_path" in batch else "unknown"
                errors.append((scene_path, metrics["min_fde"], metrics["min_ade"]))

    errors.sort(key=lambda x: x[1], reverse=True)
    return errors[:n_cases]
