"""Analyze attention entropy across encoder layers for quantitative paper statistics.

Processes ~100 validation scenes through the trained model, extracts attention
maps from each of the 4 encoder layers, and computes Shannon entropy, agent vs
map attention fractions, and max attention weight for the ego token (index 0).

Expected finding: entropy decreases from early to later layers as attention
becomes more focused/peaked, a well-known pattern in transformer networks.

Run with: PYTHONUNBUFFERED=1 python scripts/analysis_entropy.py
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention

# --- Configuration ---
CHECKPOINT = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=39-val/minADE@6=2.687.ckpt"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "mtr_lite.yaml"
NUM_SCENES = 100
DEVICE = "cpu"
NUM_LAYERS = 4
NUM_AGENTS = 32
NUM_MAP = 64
NUM_TOKENS = NUM_AGENTS + NUM_MAP  # 96
EGO_TOKEN = 0


def compute_entropy_bits(p: np.ndarray) -> float:
    """Compute Shannon entropy in bits: H = -sum(p * log2(p)).

    Args:
        p: probability distribution (should sum to ~1)

    Returns:
        entropy in bits
    """
    eps = 1e-10
    p = np.clip(p, eps, None)
    return -np.sum(p * np.log2(p))


def main():
    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Load scene list and build val split
    scene_list_path = cfg["data"]["scene_list"]
    print(f"Loading scene list from: {scene_list_path}")
    with open(scene_list_path) as f:
        all_scenes = [line.strip() for line in f if line.strip()]
    print(f"Total scenes in list: {len(all_scenes)}")

    # Val split: random 15% with seed=42
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * 0.15))
    val_indices = indices[:n_val]
    val_scenes = [all_scenes[i] for i in val_indices if os.path.exists(all_scenes[i])]
    print(f"Val scenes available: {len(val_scenes)}")

    # Load model
    print(f"\nLoading model from: {CHECKPOINT}")
    module = MTRLiteModule.load_from_checkpoint(CHECKPOINT, map_location=DEVICE)
    module.eval()
    model = module.model
    print("Model loaded successfully.\n")

    # Storage for per-layer, per-scene metrics
    # Shape: [num_layers] -> list of values across scenes
    entropy_per_layer = [[] for _ in range(NUM_LAYERS)]
    agent_frac_per_layer = [[] for _ in range(NUM_LAYERS)]
    max_weight_per_layer = [[] for _ in range(NUM_LAYERS)]

    scenes_to_process = min(NUM_SCENES, len(val_scenes))
    print(f"Processing {scenes_to_process} validation scenes...\n")

    success_count = 0
    for scene_idx in range(scenes_to_process):
        scene_path = val_scenes[scene_idx]
        scene_name = os.path.basename(scene_path)

        try:
            result = extract_scene_attention(
                model=model,
                scene_path=scene_path,
                anchor_frame=10,
                history_len=cfg["data"]["history_len"],
                future_len=cfg["data"]["future_len"],
                max_agents=cfg["model"]["max_agents"],
                max_map_polylines=cfg["model"]["max_map_polylines"],
                map_points_per_lane=cfg["model"]["map_points_per_lane"],
                device=DEVICE,
            )

            attn_maps = result["attention_maps"]
            if attn_maps is None:
                print(f"  [{scene_idx+1:3d}] {scene_name} - SKIP (no attention maps)")
                continue

            scene_attentions = attn_maps.scene_attentions  # list of 4 tensors

            for layer_idx in range(NUM_LAYERS):
                # Shape: (B, nhead, N, N) -> take batch 0
                layer_attn = scene_attentions[layer_idx][0]  # (nhead, N, N)

                # Average over heads first
                avg_attn = layer_attn.mean(dim=0)  # (N, N)

                # Ego row: attention distribution of ego token over all tokens
                ego_row = avg_attn[EGO_TOKEN].cpu().numpy()  # (N,)

                # Ensure it sums to ~1 (softmax output should)
                ego_row = ego_row / (ego_row.sum() + 1e-10)

                # Shannon entropy in bits
                entropy = compute_entropy_bits(ego_row)
                entropy_per_layer[layer_idx].append(entropy)

                # Agent attention fraction: sum of attention to agent tokens (0-31)
                agent_frac = ego_row[:NUM_AGENTS].sum()
                agent_frac_per_layer[layer_idx].append(agent_frac * 100.0)

                # Max attention weight
                max_w = ego_row.max()
                max_weight_per_layer[layer_idx].append(max_w)

            success_count += 1
            if (scene_idx + 1) % 10 == 0 or scene_idx == 0:
                print(f"  [{scene_idx+1:3d}/{scenes_to_process}] {scene_name} - OK "
                      f"(L0 entropy={entropy_per_layer[0][-1]:.2f} bits)")

        except Exception as e:
            print(f"  [{scene_idx+1:3d}] {scene_name} - FAILED: {e}")
            continue

    print(f"\nSuccessfully processed: {success_count}/{scenes_to_process} scenes\n")

    if success_count == 0:
        print("ERROR: No scenes processed successfully. Cannot compute statistics.")
        return

    # --- Compute aggregate statistics ---
    print("=" * 78)
    print(f"ATTENTION ENTROPY ANALYSIS  (N = {success_count} scenes)")
    print("=" * 78)
    print()
    print(f"{'Layer':>5s} | {'Entropy (bits)':>16s} | {'Agent Attn %':>14s} | "
          f"{'Map Attn %':>12s} | {'Max Weight':>12s}")
    print("-" * 5 + "-+-" + "-" * 16 + "-+-" + "-" * 14 + "-+-"
          + "-" * 12 + "-+-" + "-" * 12)

    for layer_idx in range(NUM_LAYERS):
        ent = np.array(entropy_per_layer[layer_idx])
        afr = np.array(agent_frac_per_layer[layer_idx])
        mfr = 100.0 - afr  # map fraction
        mxw = np.array(max_weight_per_layer[layer_idx])

        print(f"  {layer_idx:3d}   | {ent.mean():6.2f} +/- {ent.std():4.2f} | "
              f"{afr.mean():5.1f} +/- {afr.std():4.1f} | "
              f"{mfr.mean():5.1f} +/- {mfr.std():4.1f} | "
              f"{mxw.mean():8.4f}")

    print()

    # --- Summary statistics ---
    ent_first = np.array(entropy_per_layer[0])
    ent_last = np.array(entropy_per_layer[NUM_LAYERS - 1])
    delta_ent = ent_first - ent_last

    print("--- Summary ---")
    print(f"Entropy drop (Layer 0 -> Layer {NUM_LAYERS-1}): "
          f"{delta_ent.mean():.2f} +/- {delta_ent.std():.2f} bits")
    print(f"Max theoretical entropy (uniform over {NUM_TOKENS} tokens): "
          f"{np.log2(NUM_TOKENS):.2f} bits")
    print()

    # Check monotonicity
    layer_means = [np.mean(entropy_per_layer[i]) for i in range(NUM_LAYERS)]
    is_monotone_decreasing = all(
        layer_means[i] >= layer_means[i + 1] for i in range(NUM_LAYERS - 1)
    )
    print(f"Entropy monotonically decreasing across layers: {is_monotone_decreasing}")
    print(f"Layer mean entropies: {['%.2f' % m for m in layer_means]}")
    print()

    # Per-layer max weight trend
    maxw_means = [np.mean(max_weight_per_layer[i]) for i in range(NUM_LAYERS)]
    print(f"Max weight trend (should increase): {['%.4f' % m for m in maxw_means]}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
