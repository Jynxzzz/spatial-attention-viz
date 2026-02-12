"""Compare attention patterns across different scene types.

Categorizes validation scenes by type (dense traffic, sparse, with pedestrians,
with cyclists, highway-like, intersection-like) and aggregates attention
statistics per category to reveal how the model adapts its attention strategy.

Usage:
    python scripts/analysis_scene_type_attention.py
"""

import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=39-val/minADE@6=2.687.ckpt"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "mtr_lite.yaml"
N_SCENES = 200
NUM_AGENTS = 32
NUM_MAP = 64
AGENT_TYPES = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]


# ---------------------------------------------------------------------------
# Scene categorization
# ---------------------------------------------------------------------------

def categorize_scene(scene, agent_data):
    """Categorize a scene into one or more types.

    Args:
        scene: raw scene dict loaded from pkl
        agent_data: dict from extract_all_agents with agent_polylines, agent_mask, etc.

    Returns:
        set of category strings this scene belongs to
    """
    categories = set()

    objects = scene["objects"]
    anchor_frame = 10  # standard anchor frame

    # Count valid agents at anchor frame
    n_valid = int(agent_data["agent_mask"].sum())

    # Dense vs sparse
    if n_valid > 20:
        categories.add("Dense traffic")
    if n_valid <= 5:
        categories.add("Sparse")

    # Agent types from raw scene pkl
    has_pedestrian = False
    has_cyclist = False
    for obj in objects:
        if anchor_frame < len(obj.get("valid", [])) and obj["valid"][anchor_frame]:
            obj_type = obj.get("type", "unknown").lower()
            if obj_type == "pedestrian":
                has_pedestrian = True
            elif obj_type == "cyclist":
                has_cyclist = True

    if has_pedestrian:
        categories.add("With pedestrians")
    if has_cyclist:
        categories.add("With cyclists")

    # Ego speed from agent_polylines[0, -1, 4:6] (vel_bev at last history step)
    ego_vel_bev = agent_data["agent_polylines"][0, -1, 4:6]  # (2,)
    ego_speed = float(np.linalg.norm(ego_vel_bev))

    # Highway-like: ego speed > 10 m/s
    if ego_speed > 10.0:
        categories.add("Highway-like")

    # Intersection-like: ego speed < 3 m/s AND > 15 agents
    if ego_speed < 3.0 and n_valid > 15:
        categories.add("Intersection-like")

    return categories


# ---------------------------------------------------------------------------
# Attention metrics for a single scene
# ---------------------------------------------------------------------------

def compute_scene_metrics(result):
    """Compute attention metrics for one scene.

    Args:
        result: dict returned by extract_scene_attention

    Returns:
        dict with scalar metrics, or None if attention maps not available
    """
    attn_maps = result["attention_maps"]
    agent_data = result["agent_data"]
    map_data = result["map_data"]

    if attn_maps is None:
        return None

    agent_mask = agent_data["agent_mask"]  # (A,) bool
    map_mask = map_data["map_mask"].astype(bool)  # (M,) bool
    n_valid_agents = int(agent_mask.sum())

    # Last encoder layer, ego attention row (row 0), averaged over heads
    # scene_attentions[-1] shape: (B=1, nhead, A+M, A+M)
    scene_attn = attn_maps.scene_attentions[-1][0]  # (nhead, A+M, A+M) tensor
    ego_attn_heads = scene_attn[:, 0, :]  # (nhead, A+M) - ego row
    ego_attn = ego_attn_heads.mean(dim=0).cpu().numpy()  # (A+M,)

    agent_attn = ego_attn[:NUM_AGENTS]  # (A,)
    map_attn = ego_attn[NUM_AGENTS:NUM_AGENTS + NUM_MAP]  # (M,)

    # Mask out invalid tokens
    agent_attn_valid = agent_attn.copy()
    agent_attn_valid[~agent_mask] = 0.0
    map_attn_valid = map_attn.copy()
    map_attn_valid[~map_mask] = 0.0

    total_attn = agent_attn_valid.sum() + map_attn_valid.sum()
    if total_attn < 1e-12:
        return None

    # Agent vs map fraction
    agent_fraction = float(agent_attn_valid.sum() / total_attn)
    map_fraction = float(map_attn_valid.sum() / total_attn)

    # Entropy of ego's attention distribution (over valid tokens only)
    valid_mask_combined = np.concatenate([agent_mask, map_mask])
    valid_attn = ego_attn[valid_mask_combined]
    # Re-normalize for entropy computation
    valid_attn_sum = valid_attn.sum()
    if valid_attn_sum > 1e-12:
        valid_attn_normed = valid_attn / valid_attn_sum
    else:
        valid_attn_normed = valid_attn
    eps = 1e-12
    valid_attn_clamped = np.clip(valid_attn_normed, eps, None)
    entropy = float(-np.sum(valid_attn_clamped * np.log2(valid_attn_clamped)))

    # Agent positions: BEV position at last history step (indices 0:2)
    agent_positions = agent_data["agent_polylines"][:, -1, 0:2]  # (A, 2)
    ego_pos_bev = agent_positions[0]  # should be ~(0,0) in BEV

    # Distances of agents from ego
    distances = np.linalg.norm(agent_positions - ego_pos_bev, axis=1)  # (A,)

    # Top-5 most attended agents: mean distance
    valid_agent_indices = np.where(agent_mask)[0]
    if len(valid_agent_indices) >= 5:
        # Sort valid agents by attention descending
        valid_agent_attn = agent_attn[valid_agent_indices]
        top5_rel = np.argsort(valid_agent_attn)[-5:][::-1]
        top5_abs = valid_agent_indices[top5_rel]
        top5_dist = float(distances[top5_abs].mean())
    elif len(valid_agent_indices) > 0:
        top5_dist = float(distances[valid_agent_indices].mean())
    else:
        top5_dist = 0.0

    # Near-5 vs Far-5 attention
    if len(valid_agent_indices) >= 10:
        sorted_by_dist = valid_agent_indices[np.argsort(distances[valid_agent_indices])]
        # Exclude ego (index 0) from near/far computation if it is in sorted list
        # Actually ego is always nearest, keep it to match the scene's natural structure
        near5 = sorted_by_dist[:5]
        far5 = sorted_by_dist[-5:]
        near5_attn = float(agent_attn[near5].mean())
        far5_attn = float(agent_attn[far5].mean())
    elif len(valid_agent_indices) >= 2:
        sorted_by_dist = valid_agent_indices[np.argsort(distances[valid_agent_indices])]
        half = len(sorted_by_dist) // 2
        near_half = sorted_by_dist[:max(half, 1)]
        far_half = sorted_by_dist[max(half, 1):]
        near5_attn = float(agent_attn[near_half].mean())
        far5_attn = float(agent_attn[far_half].mean()) if len(far_half) > 0 else 0.0
    else:
        near5_attn = float(agent_attn[valid_agent_indices].mean()) if len(valid_agent_indices) > 0 else 0.0
        far5_attn = 0.0

    return {
        "agent_fraction": agent_fraction,
        "map_fraction": map_fraction,
        "entropy": entropy,
        "top5_dist": top5_dist,
        "near5_attn": near5_attn,
        "far5_attn": far5_attn,
        "n_valid_agents": n_valid_agents,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Scene Type Attention Pattern Analysis")
    print("=" * 80)

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    # Load model on CPU
    print(f"\nLoading model from: {CHECKPOINT}")
    module = MTRLiteModule.load_from_checkpoint(CHECKPOINT, map_location="cpu")
    module.eval()
    model = module.model
    print("Model loaded successfully.")

    # Build validation scene list (same split as training)
    with open(data_cfg["scene_list"]) as f:
        all_scenes = [line.strip() for line in f if line.strip()]
    all_scenes = [p for p in all_scenes if os.path.exists(p)]

    rng = random.Random(cfg.get("seed", 42))
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * data_cfg.get("val_ratio", 0.15)))

    # Apply data_fraction
    data_frac = data_cfg.get("data_fraction", 1.0)
    if data_frac < 1.0:
        n_keep = max(1, int(n_val * data_frac))
        val_indices = indices[:n_keep]
    else:
        val_indices = indices[:n_val]

    val_scenes = [all_scenes[i] for i in val_indices]
    n_to_scan = min(N_SCENES, len(val_scenes))
    print(f"Total val scenes: {len(val_scenes)}, scanning: {n_to_scan}")

    # Accumulators per category
    category_metrics = defaultdict(list)
    scene_count = 0
    error_count = 0
    t_start = time.time()

    for idx in range(n_to_scan):
        scene_path = val_scenes[idx]

        try:
            result = extract_scene_attention(
                model=model,
                scene_path=scene_path,
                device="cpu",
            )
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"  [WARN] Scene {idx}: extraction failed: {e}")
            continue

        # Categorize
        categories = categorize_scene(result["scene_data"], result["agent_data"])
        if not categories:
            # Scene does not fit any special category -- still count as processed
            scene_count += 1
            continue

        # Compute metrics
        metrics = compute_scene_metrics(result)
        if metrics is None:
            scene_count += 1
            continue

        for cat in categories:
            category_metrics[cat].append(metrics)

        scene_count += 1

        # Progress
        if (scene_count % 20 == 0) or (scene_count == n_to_scan):
            elapsed = time.time() - t_start
            rate = elapsed / scene_count
            eta = rate * (n_to_scan - scene_count)
            print(f"  Processed {scene_count}/{n_to_scan} scenes "
                  f"({elapsed:.1f}s elapsed, {rate:.2f}s/scene, ETA {eta:.0f}s)")

    elapsed_total = time.time() - t_start
    print(f"\nDone. Processed {scene_count} scenes in {elapsed_total:.1f}s "
          f"({error_count} errors skipped)")

    # ---------------------------------------------------------------------------
    # Aggregate and print table
    # ---------------------------------------------------------------------------
    # Desired column order
    category_order = [
        "Dense traffic",
        "Sparse",
        "With pedestrians",
        "With cyclists",
        "Highway-like",
        "Intersection-like",
    ]

    print("\n" + "=" * 105)
    header = (
        f"{'Scene Type':<20s} | {'N':>3s} | {'Agent%':>7s} | {'Map%':>7s} | "
        f"{'Entropy':>7s} | {'Top5 Dist':>9s} | {'Near5 Attn':>10s} | {'Far5 Attn':>9s}"
    )
    print(header)
    print("-" * 105)

    for cat in category_order:
        if cat not in category_metrics or len(category_metrics[cat]) == 0:
            continue
        mlist = category_metrics[cat]
        n = len(mlist)
        mean_agent = np.mean([m["agent_fraction"] for m in mlist]) * 100
        mean_map = np.mean([m["map_fraction"] for m in mlist]) * 100
        mean_entropy = np.mean([m["entropy"] for m in mlist])
        mean_top5d = np.mean([m["top5_dist"] for m in mlist])
        mean_near5 = np.mean([m["near5_attn"] for m in mlist])
        mean_far5 = np.mean([m["far5_attn"] for m in mlist])

        row = (
            f"{cat:<20s} | {n:>3d} | {mean_agent:>6.1f}% | {mean_map:>6.1f}% | "
            f"{mean_entropy:>7.2f} | {mean_top5d:>8.1f}m | {mean_near5:>10.4f} | {mean_far5:>9.4f}"
        )
        print(row)

    print("-" * 105)

    # Also print any categories not in the predefined order
    extra_cats = set(category_metrics.keys()) - set(category_order)
    for cat in sorted(extra_cats):
        mlist = category_metrics[cat]
        n = len(mlist)
        mean_agent = np.mean([m["agent_fraction"] for m in mlist]) * 100
        mean_map = np.mean([m["map_fraction"] for m in mlist]) * 100
        mean_entropy = np.mean([m["entropy"] for m in mlist])
        mean_top5d = np.mean([m["top5_dist"] for m in mlist])
        mean_near5 = np.mean([m["near5_attn"] for m in mlist])
        mean_far5 = np.mean([m["far5_attn"] for m in mlist])

        row = (
            f"{cat:<20s} | {n:>3d} | {mean_agent:>6.1f}% | {mean_map:>6.1f}% | "
            f"{mean_entropy:>7.2f} | {mean_top5d:>8.1f}m | {mean_near5:>10.4f} | {mean_far5:>9.4f}"
        )
        print(row)

    print("=" * 105)

    # Print category overlap info
    print("\nCategory overlap (scenes can belong to multiple categories):")
    for cat in category_order:
        if cat in category_metrics:
            print(f"  {cat}: {len(category_metrics[cat])} scenes")

    # Print detailed breakdown per category
    print("\n" + "=" * 80)
    print("Detailed Statistics per Category")
    print("=" * 80)
    for cat in category_order:
        if cat not in category_metrics or len(category_metrics[cat]) == 0:
            continue
        mlist = category_metrics[cat]
        n = len(mlist)
        print(f"\n--- {cat} ({n} scenes) ---")
        agent_fracs = [m["agent_fraction"] * 100 for m in mlist]
        map_fracs = [m["map_fraction"] * 100 for m in mlist]
        entropies = [m["entropy"] for m in mlist]
        top5_dists = [m["top5_dist"] for m in mlist]
        near5_attns = [m["near5_attn"] for m in mlist]
        far5_attns = [m["far5_attn"] for m in mlist]
        n_agents_list = [m["n_valid_agents"] for m in mlist]

        print(f"  Avg valid agents:    {np.mean(n_agents_list):.1f} (std={np.std(n_agents_list):.1f})")
        print(f"  Agent attn fraction: {np.mean(agent_fracs):.1f}% (std={np.std(agent_fracs):.1f}%)")
        print(f"  Map attn fraction:   {np.mean(map_fracs):.1f}% (std={np.std(map_fracs):.1f}%)")
        print(f"  Entropy (bits):      {np.mean(entropies):.2f} (std={np.std(entropies):.2f})")
        print(f"  Top-5 dist (m):      {np.mean(top5_dists):.1f} (std={np.std(top5_dists):.1f})")
        print(f"  Near-5 attn:         {np.mean(near5_attns):.4f} (std={np.std(near5_attns):.4f})")
        print(f"  Far-5 attn:          {np.mean(far5_attns):.4f} (std={np.std(far5_attns):.4f})")
        print(f"  Near/Far ratio:      {np.mean(near5_attns)/max(np.mean(far5_attns), 1e-8):.2f}x")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
