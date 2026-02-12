"""Failure diagnosis through attention: what does the model look at when it fails?

Compares attention patterns between SUCCESS cases (low ADE) and FAILURE cases
(high ADE) to understand whether failures correlate with attention pathology.

Key questions:
1. Do failures have different attention entropy? (too diffuse? too focused?)
2. Do failures attend to the wrong agents? (miss the critical one?)
3. Do failures rely too much on map vs agents?
4. Is there a "missed agent" pattern — a nearby agent that should have been
   attended to but wasn't?
"""

import sys
import os
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention


def compute_entropy(probs):
    """Shannon entropy in bits."""
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log2(p))


def analyze_scene(model, scene_path, device="cpu"):
    """Run one scene, return per-target analysis."""
    result = extract_scene_attention(
        model=model, scene_path=scene_path,
        anchor_frame=10, device=device,
    )

    attn_maps = result["attention_maps"]
    agent_data = result["agent_data"]
    map_data = result["map_data"]
    preds = result["predictions"]
    batch = result["batch"]
    scene = result["scene_data"]

    if attn_maps is None or not attn_maps.scene_attentions:
        return []

    # Scene encoder last layer attention
    last_attn = attn_maps.scene_attentions[-1][0]  # (H, N, N)
    avg_attn = last_attn.mean(dim=0).cpu().numpy()  # (N, N)

    pred_trajs = preds["trajectories"][0].cpu().numpy()   # (T, K, 80, 2)
    pred_scores = preds["scores"][0].cpu().numpy()         # (T, K)
    target_future = batch["target_future"][0].cpu().numpy()       # (T, 80, 2)
    target_future_valid = batch["target_future_valid"][0].cpu().numpy()  # (T, 80)
    target_mask = batch["target_mask"][0].cpu().numpy()            # (T,)
    target_indices = batch["target_agent_indices"][0].cpu().numpy()  # (T,)

    results = []

    for t_idx in range(8):
        if not target_mask[t_idx]:
            continue

        agent_slot = int(target_indices[t_idx])
        gt = target_future[t_idx]
        gt_valid = target_future_valid[t_idx].astype(bool)

        if gt_valid.sum() < 20:
            continue

        # Compute minADE@6
        trajs = pred_trajs[t_idx]   # (K, 80, 2)
        scores = pred_scores[t_idx]  # (K,)
        top6 = scores.argsort()[-6:]
        min_ade = float("inf")
        best_mode = 0
        for k in top6:
            diff = trajs[k] - gt
            dist = np.sqrt((diff ** 2).sum(axis=1))
            ade = dist[gt_valid].mean()
            if ade < min_ade:
                min_ade = ade
                best_mode = k

        if min_ade >= 999:
            continue

        # Get this target's attention row
        ego_attn = avg_attn[agent_slot]  # (96,)
        agent_attn = ego_attn[:32]
        map_attn = ego_attn[32:96]

        # Entropy
        entropy = compute_entropy(ego_attn)

        # Agent vs map split
        agent_sum = agent_attn.sum()
        map_sum = map_attn.sum()
        total = agent_sum + map_sum + 1e-8
        agent_pct = agent_sum / total

        # Self-attention (attention to own token)
        self_attn = ego_attn[agent_slot]

        # Max attention to any single token
        max_attn = ego_attn.max()

        # Distance to top-attended agent
        target_pos = agent_data["agent_polylines"][agent_slot, -1, 0:2]
        top_agent = np.argmax(agent_attn)
        if agent_data["agent_mask"][top_agent]:
            top_agent_pos = agent_data["agent_polylines"][top_agent, -1, 0:2]
            dist_to_top = np.linalg.norm(target_pos - top_agent_pos)
        else:
            dist_to_top = 0.0

        # Find the agent closest to GT endpoint that target should attend to
        gt_endpoint = gt[gt_valid][-1] if gt_valid.any() else gt[-1]
        closest_to_gt_dist = float("inf")
        closest_to_gt_slot = -1
        closest_to_gt_attn = 0.0
        for s in range(32):
            if not agent_data["agent_mask"][s] or s == agent_slot:
                continue
            other_pos = agent_data["agent_polylines"][s, -1, 0:2]
            d = np.linalg.norm(other_pos - gt_endpoint)
            if d < closest_to_gt_dist:
                closest_to_gt_dist = d
                closest_to_gt_slot = s
                closest_to_gt_attn = agent_attn[s]

        # Agent type
        obj_idx = agent_data["agent_ids"][agent_slot]
        agent_type = scene["objects"][obj_idx].get("type", "unknown")

        # Ego speed
        vel = agent_data["agent_polylines"][agent_slot, -1, 4:6]
        speed = np.linalg.norm(vel)

        # Number of valid nearby agents
        n_nearby = 0
        for s in range(32):
            if agent_data["agent_mask"][s] and s != agent_slot:
                d = np.linalg.norm(agent_data["agent_polylines"][s, -1, 0:2] - target_pos)
                if d < 15:
                    n_nearby += 1

        results.append({
            "minADE": min_ade,
            "entropy": entropy,
            "agent_pct": agent_pct,
            "self_attn": self_attn,
            "max_attn": max_attn,
            "dist_to_top_agent": dist_to_top,
            "closest_gt_agent_attn": closest_to_gt_attn,
            "closest_gt_agent_dist": closest_to_gt_dist,
            "agent_type": agent_type,
            "speed": speed,
            "n_nearby": n_nearby,
        })

    return results


def main():
    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=39-val/minADE@6=2.687.ckpt"
    config_path = Path(__file__).resolve().parent.parent / "configs" / "mtr_lite.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading model...")
    module = MTRLiteModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    model = module.model

    scene_list_path = cfg["data"]["scene_list"]
    with open(scene_list_path) as f:
        all_scenes = [l.strip() for l in f if l.strip()]
    import random
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * 0.15))
    val_indices = indices[:n_val]
    val_scenes = [all_scenes[i] for i in val_indices if os.path.exists(all_scenes[i])]

    n_scenes = 150
    all_results = []

    print(f"Analyzing {n_scenes} scenes...")
    for s_idx in range(min(n_scenes, len(val_scenes))):
        try:
            results = analyze_scene(model, val_scenes[s_idx], device="cpu")
            all_results.extend(results)
        except Exception as e:
            pass

        if (s_idx + 1) % 25 == 0:
            print(f"  [{s_idx+1}/{n_scenes}] {len(all_results)} targets so far")

    print(f"\nTotal targets analyzed: {len(all_results)}")

    if not all_results:
        print("No results!")
        return

    # Split into success / failure by ADE percentiles
    ades = np.array([r["minADE"] for r in all_results])
    p25 = np.percentile(ades, 25)
    p75 = np.percentile(ades, 75)

    success = [r for r in all_results if r["minADE"] <= p25]
    failure = [r for r in all_results if r["minADE"] >= p75]

    print(f"\nSuccess (bottom 25% ADE, ≤{p25:.2f}m): n={len(success)}")
    print(f"Failure (top 25% ADE, ≥{p75:.2f}m): n={len(failure)}")

    # Compare metrics
    def stats(group, key):
        vals = [float(r[key]) for r in group if not isinstance(r[key], str)]
        if not vals:
            return 0.0, 0.0
        return np.mean(vals), np.std(vals)

    print(f"\n{'='*70}")
    print(f"{'Metric':>30s} | {'Success':>18s} | {'Failure':>18s}")
    print(f"{'-'*70}")

    metrics = [
        ("minADE (m)", "minADE"),
        ("Entropy (bits)", "entropy"),
        ("Agent attn %", "agent_pct"),
        ("Self-attention", "self_attn"),
        ("Max single-token attn", "max_attn"),
        ("Dist to top-attended (m)", "dist_to_top_agent"),
        ("Attn to GT-nearest agent", "closest_gt_agent_attn"),
        ("Dist GT-nearest agent (m)", "closest_gt_agent_dist"),
        ("Speed (m/s)", "speed"),
        ("Nearby agents (<15m)", "n_nearby"),
    ]

    for label, key in metrics:
        sm, ss = stats(success, key)
        fm, fs = stats(failure, key)
        if key == "agent_pct":
            print(f"{label:>30s} | {sm*100:>7.1f}% ± {ss*100:>5.1f}% | {fm*100:>7.1f}% ± {fs*100:>5.1f}%")
        else:
            print(f"{label:>30s} | {sm:>7.3f} ± {ss:>6.3f} | {fm:>7.3f} ± {fs:>6.3f}")

    # Type breakdown
    print(f"\n{'='*70}")
    print("Agent type distribution:")
    for group, label in [(success, "Success"), (failure, "Failure")]:
        types = {}
        for r in group:
            t = r["agent_type"]
            types[t] = types.get(t, 0) + 1
        total = len(group)
        type_str = ", ".join(f"{t}: {n} ({n/total*100:.0f}%)" for t, n in sorted(types.items()))
        print(f"  {label}: {type_str}")

    # Key insight
    print(f"\n{'='*70}")
    print("KEY INSIGHTS:")

    sm_entropy, _ = stats(success, "entropy")
    fm_entropy, _ = stats(failure, "entropy")
    if fm_entropy > sm_entropy:
        print(f"  1. Failures have HIGHER entropy ({fm_entropy:.2f} vs {sm_entropy:.2f} bits)")
        print(f"     -> Model's attention is too DIFFUSE when it fails (can't decide what to focus on)")
    else:
        print(f"  1. Failures have LOWER entropy ({fm_entropy:.2f} vs {sm_entropy:.2f} bits)")
        print(f"     -> Model's attention is too FOCUSED when it fails (tunnel vision)")

    sm_apct, _ = stats(success, "agent_pct")
    fm_apct, _ = stats(failure, "agent_pct")
    print(f"  2. Agent attention: success={sm_apct*100:.1f}% vs failure={fm_apct*100:.1f}%")

    sm_gt, _ = stats(success, "closest_gt_agent_attn")
    fm_gt, _ = stats(failure, "closest_gt_agent_attn")
    print(f"  3. Attention to GT-path agent: success={sm_gt:.4f} vs failure={fm_gt:.4f}")
    if fm_gt < sm_gt:
        print(f"     -> Failures attend LESS to the agent near the ground truth path")
        print(f"     -> The model literally 'misses' the relevant agent!")

    sm_spd, _ = stats(success, "speed")
    fm_spd, _ = stats(failure, "speed")
    print(f"  4. Target speed: success={sm_spd:.1f}m/s vs failure={fm_spd:.1f}m/s")
    if fm_spd > sm_spd:
        print(f"     -> Faster agents are harder to predict (expected)")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
