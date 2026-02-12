"""Intention point generation via k-means clustering on GT endpoints.

Precomputes K=64 cluster centers from training set ground-truth trajectory
endpoints. These serve as initial anchor queries for the motion decoder.
"""

import os
import pickle
import random

import numpy as np


def collect_gt_endpoints(
    scene_list_path: str,
    anchor_frame: int = 10,
    future_len: int = 80,
    max_scenes: int = 10000,
    neighbor_distance: float = 50.0,
    seed: int = 42,
) -> np.ndarray:
    """Collect GT trajectory endpoints from training scenes.

    For each scene, extracts the endpoint (position at t=anchor+future_len)
    for the ego vehicle and nearby agents, all in BEV coordinates.

    Returns:
        endpoints: (N, 2) array of endpoint positions in BEV
    """
    import math

    with open(scene_list_path, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    scenes = [p for p in scenes if os.path.exists(p)]
    rng = random.Random(seed)
    rng.shuffle(scenes)
    scenes = scenes[:max_scenes]

    endpoints = []

    for path in scenes:
        try:
            with open(path, "rb") as f:
                scene = pickle.load(f)
        except Exception:
            continue

        av_idx = scene["av_idx"]
        objects = scene["objects"]

        # Ego pose at anchor for BEV
        ego_obj = objects[av_idx]
        if not ego_obj["valid"][anchor_frame]:
            continue
        ego_pos = (
            float(ego_obj["position"][anchor_frame]["x"]),
            float(ego_obj["position"][anchor_frame]["y"]),
        )
        heading_raw = ego_obj["heading"][anchor_frame]
        ego_heading = float(heading_raw[0]) if isinstance(heading_raw, (list, tuple)) else float(heading_raw)

        heading_rad = math.radians(ego_heading)
        adjusted = heading_rad - np.pi / 2
        c, s = np.cos(-adjusted), np.sin(-adjusted)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        ego_xy = np.array(ego_pos, dtype=np.float64)

        # Check each agent
        for obj in objects:
            if not obj["valid"][anchor_frame]:
                continue

            # Distance check
            pos_a = np.array(
                [float(obj["position"][anchor_frame]["x"]),
                 float(obj["position"][anchor_frame]["y"])],
                dtype=np.float64,
            )
            if np.linalg.norm(pos_a - ego_xy) > neighbor_distance:
                continue

            # Check endpoint validity
            end_frame = anchor_frame + future_len
            if end_frame >= len(obj["valid"]) or not obj["valid"][end_frame]:
                continue

            # Endpoint in BEV
            pos_end = np.array(
                [float(obj["position"][end_frame]["x"]),
                 float(obj["position"][end_frame]["y"])],
                dtype=np.float64,
            )
            bev = ((pos_end - ego_xy) @ R.T).astype(np.float32)
            endpoints.append(bev)

    return np.array(endpoints, dtype=np.float32) if endpoints else np.zeros((0, 2), dtype=np.float32)


def kmeans_intent_points(
    endpoints: np.ndarray,
    k: int = 64,
    max_iter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Simple k-means clustering to find intention point anchors.

    Args:
        endpoints: (N, 2) array of GT endpoint positions
        k: number of cluster centers
        max_iter: maximum iterations

    Returns:
        centers: (K, 2) array of cluster center positions
    """
    if len(endpoints) < k:
        # Not enough data: use endpoints + zeros
        centers = np.zeros((k, 2), dtype=np.float32)
        centers[:len(endpoints)] = endpoints
        return centers

    rng = np.random.RandomState(seed)

    # K-means++ initialization
    centers = np.zeros((k, 2), dtype=np.float32)
    idx = rng.randint(len(endpoints))
    centers[0] = endpoints[idx]

    for i in range(1, k):
        dists = np.min(
            np.linalg.norm(endpoints[:, None, :] - centers[None, :i, :], axis=2),
            axis=1,
        )
        probs = dists ** 2
        probs = probs / probs.sum()
        idx = rng.choice(len(endpoints), p=probs)
        centers[i] = endpoints[idx]

    # Lloyd's algorithm
    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(endpoints[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Update
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = endpoints[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]

        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return centers


def generate_and_save(
    scene_list_path: str,
    output_path: str,
    k: int = 64,
    future_len: int = 80,
    max_scenes: int = 10000,
):
    """Generate intention points and save to file."""
    print(f"Collecting GT endpoints from up to {max_scenes} scenes...")
    endpoints = collect_gt_endpoints(
        scene_list_path, future_len=future_len, max_scenes=max_scenes,
    )
    print(f"Collected {len(endpoints)} endpoints")

    print(f"Running k-means with K={k}...")
    centers = kmeans_intent_points(endpoints, k=k)
    print(f"Generated {len(centers)} intention points")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, centers)
    print(f"Saved to {output_path}")

    return centers
