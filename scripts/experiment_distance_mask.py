"""Quick experiment: does distance-aware attention masking improve predictions?

Injects a distance-decay bias into the scene encoder's self-attention at
INFERENCE time (no retraining). Tests whether suppressing far-away token
attention improves trajectory prediction.

bias[i][j] = -alpha * distance(token_i, token_j)

This is added to attention logits before softmax, so positive alpha
suppresses attention to distant tokens.
"""

import sys
import os
import math
import pickle
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from data.agent_features import extract_all_agents, extract_agent_future
from data.map_features import extract_map_polylines


def build_batch(scene_path, anchor_frame=10, history_len=11, future_len=80,
                max_agents=32, max_map=64, device="cpu"):
    """Build a single-sample batch from a scene file."""
    with open(scene_path, "rb") as f:
        scene = pickle.load(f)

    objects = scene["objects"]
    av_idx = scene["av_idx"]
    ego_obj = objects[av_idx]

    n_frames = len(ego_obj["position"])
    actual_future = min(future_len, n_frames - 1 - anchor_frame)
    if actual_future < 10:
        return None, None, None

    ego_pos = (
        float(ego_obj["position"][anchor_frame]["x"]),
        float(ego_obj["position"][anchor_frame]["y"]),
    )
    heading_raw = ego_obj["heading"][anchor_frame]
    ego_heading = float(heading_raw[0]) if isinstance(heading_raw, (list, tuple)) else float(heading_raw)

    agent_data = extract_all_agents(scene, anchor_frame, history_len, ego_pos, ego_heading,
                                     max_agents=max_agents, neighbor_distance=50.0)
    map_data = extract_map_polylines(scene, ego_pos, ego_heading,
                                      max_polylines=max_map, points_per_lane=20)

    # Targets
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

    if not target_mask.any():
        return None, None, None

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

    # Compute token positions for distance matrix
    # Agent positions: last valid history step, BEV coords (first 2 dims)
    agent_pos = agent_data["agent_polylines"][:, -1, 0:2]  # (A, 2)
    # Map positions: mean of lane centerline points
    map_pos = map_data["lane_centerlines_bev"].mean(axis=1)  # (M, 2)
    # Combined token positions
    token_pos = np.concatenate([agent_pos, map_pos], axis=0)  # (A+M, 2)

    return batch, token_pos, (target_future, target_future_valid, target_mask)


def compute_distance_bias(token_pos, alpha, max_agents=32, max_map=64):
    """Create distance-decay attention bias matrix.

    bias[i][j] = -alpha * ||pos_i - pos_j||

    Returns (N, N) tensor where N = max_agents + max_map.
    """
    N = max_agents + max_map
    pos = torch.from_numpy(token_pos).float()  # (N, 2)
    # Pairwise L2 distance
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 2)
    dist = torch.norm(diff, dim=-1)  # (N, N)
    bias = -alpha * dist
    return bias


@contextmanager
def distance_mask_context(model, distance_bias):
    """Context manager that injects distance bias into scene encoder attention.

    Monkey-patches each encoder layer's self_attn forward to include
    the distance bias as attn_mask.
    """
    original_forwards = []
    encoder = model.scene_encoder

    for layer in encoder.layers:
        original_forward = layer.self_attn.forward
        original_forwards.append(original_forward)

        # Create a closure that captures the bias
        bias = distance_bias.to(next(layer.parameters()).device)

        def make_patched_forward(orig_fwd, attn_bias):
            def patched_forward(query, key, value, key_padding_mask=None,
                              need_weights=True, attn_mask=None,
                              average_attn_weights=True, is_causal=False):
                # Inject our distance bias as attn_mask
                if attn_mask is None:
                    attn_mask = attn_bias
                else:
                    attn_mask = attn_mask + attn_bias
                return orig_fwd(
                    query, key, value,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights,
                )
            return patched_forward

        layer.self_attn.forward = make_patched_forward(original_forward, bias)

    try:
        yield
    finally:
        # Restore original forwards
        for layer, orig in zip(encoder.layers, original_forwards):
            layer.self_attn.forward = orig


def evaluate_scene(model, batch, gt_data, use_mask=False, distance_bias=None):
    """Run inference on one scene and compute minADE@6."""
    target_future, target_future_valid, target_mask = gt_data

    model.eval()
    with torch.no_grad():
        if use_mask and distance_bias is not None:
            with distance_mask_context(model, distance_bias):
                output = model(batch, capture_attention=False)
        else:
            output = model(batch, capture_attention=False)

    pred_trajs = output["trajectories"][0].cpu().numpy()  # (T, K, 80, 2)
    pred_scores = output["scores"][0].cpu().numpy()        # (T, K)

    ades = []
    for t_idx in range(8):
        if not target_mask[t_idx]:
            continue
        gt = target_future[t_idx]         # (80, 2)
        gt_valid = target_future_valid[t_idx]  # (80,)
        if gt_valid.sum() < 10:
            continue

        trajs = pred_trajs[t_idx]  # (K, 80, 2)
        scores = pred_scores[t_idx]  # (K,)

        # minADE@6: min over top-6 modes
        top6 = scores.argsort()[-6:]
        min_ade = float("inf")
        for k in top6:
            diff = trajs[k] - gt
            dist = np.sqrt((diff ** 2).sum(axis=1))
            ade = dist[gt_valid].mean()
            if ade < min_ade:
                min_ade = ade
        if min_ade < float("inf"):
            ades.append(min_ade)

    return ades


def main():
    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=39-val/minADE@6=2.687.ckpt"
    config_path = Path(__file__).resolve().parent.parent / "configs" / "mtr_lite.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading model on CPU...")
    module = MTRLiteModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    model = module.model

    # Get val scenes
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

    # Test alpha values
    alpha_values = [0.0, 0.05, 0.1, 0.2]
    n_scenes = 100  # quick test

    print(f"Evaluating {n_scenes} val scenes with {len(alpha_values)} alpha values...")
    print(f"alpha=0.0 is baseline (no mask)\n")

    results = {a: [] for a in alpha_values}

    for s_idx in range(n_scenes):
        if s_idx >= len(val_scenes):
            break

        batch, token_pos, gt_data = build_batch(val_scenes[s_idx], device="cpu")
        if batch is None:
            continue

        for alpha in alpha_values:
            if alpha == 0.0:
                ades = evaluate_scene(model, batch, gt_data, use_mask=False)
            else:
                dist_bias = compute_distance_bias(token_pos, alpha)
                ades = evaluate_scene(model, batch, gt_data, use_mask=True,
                                     distance_bias=dist_bias)
            results[alpha].extend(ades)

        if (s_idx + 1) % 25 == 0:
            print(f"  [{s_idx+1}/{n_scenes}] Progress:")
            for alpha in alpha_values:
                if results[alpha]:
                    mean_ade = np.mean(results[alpha])
                    print(f"    alpha={alpha:.2f}: minADE@6={mean_ade:.3f}m (n={len(results[alpha])})")
            print()

    # Final results
    print("=" * 60)
    print("FINAL RESULTS: Distance Mask Ablation")
    print("=" * 60)
    baseline = np.mean(results[0.0]) if results[0.0] else 0
    print(f"\n{'Alpha':>8s} {'minADE@6':>10s} {'Delta':>10s} {'Delta%':>8s} {'Samples':>8s}")
    print("-" * 50)
    for alpha in alpha_values:
        if not results[alpha]:
            continue
        mean_ade = np.mean(results[alpha])
        delta = mean_ade - baseline
        delta_pct = delta / baseline * 100 if baseline > 0 else 0
        sign = "+" if delta >= 0 else ""
        label = "(baseline)" if alpha == 0.0 else ""
        print(f"{alpha:>8.2f} {mean_ade:>10.3f}m {sign}{delta:>9.3f}m {sign}{delta_pct:>7.1f}% {len(results[alpha]):>8d} {label}")

    print(f"\nInterpretation:")
    # Find best alpha
    best_alpha = min(alpha_values, key=lambda a: np.mean(results[a]) if results[a] else 999)
    best_ade = np.mean(results[best_alpha])
    if best_alpha > 0 and best_ade < baseline:
        improvement = baseline - best_ade
        print(f"  Best: alpha={best_alpha:.2f}, minADE improved by {improvement:.3f}m ({improvement/baseline*100:.1f}%)")
        print(f"  -> Distance masking HELPS. Far-agent attention was indeed wasteful.")
    else:
        print(f"  No improvement from distance masking.")
        print(f"  -> Model already handles distance implicitly; far attention is softmax residual.")
    print(f"\nBoth results are valid paper material!")


if __name__ == "__main__":
    main()
