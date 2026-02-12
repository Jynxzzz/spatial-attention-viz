"""Quick demo: generate attention visualization from real model + real scene.

Proves that the trained model produces meaningful, interpretable attention patterns.
Runs on CPU to avoid interfering with training.
"""

import sys
import os
import yaml
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention
from visualization.space_attention_bev import render_space_attention_bev
from visualization.lane_token_activation import render_lane_activation_map


def main():
    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=39-val/minADE@6=2.687.ckpt"
    config_path = Path(__file__).resolve().parent.parent / "configs" / "mtr_lite.yaml"
    output_dir = "/tmp/attention_demo"
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading model...")
    module = MTRLiteModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    model = module.model

    # Find a good scene (pick a few from val set)
    scene_list_path = cfg["data"]["scene_list"]
    with open(scene_list_path) as f:
        all_scenes = [l.strip() for l in f if l.strip()]

    # Use same split logic as dataset
    import random
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * 0.15))
    val_indices = indices[:n_val]
    val_scenes = [all_scenes[i] for i in val_indices if os.path.exists(all_scenes[i])]

    print(f"Val scenes available: {len(val_scenes)}")

    # Try a few scenes to find one with multiple agent types
    for scene_idx in range(min(20, len(val_scenes))):
        scene_path = val_scenes[scene_idx]
        try:
            with open(scene_path, "rb") as f:
                scene = pickle.load(f)
            n_obj = len(scene["objects"])
            types = set()
            for obj in scene["objects"]:
                if obj.get("valid", [False])[10]:
                    types.add(obj.get("type", "unknown"))
            print(f"  Scene {scene_idx}: {n_obj} objects, types={types}")
            if len(types) >= 2 and "vehicle" in types:
                break
        except Exception:
            continue

    print(f"\nUsing scene: {scene_path}")
    print("Extracting attention...")

    result = extract_scene_attention(
        model=model,
        scene_path=scene_path,
        device="cpu",
    )

    attn_maps = result["attention_maps"]
    agent_data = result["agent_data"]
    map_data = result["map_data"]

    if attn_maps is None:
        print("ERROR: No attention maps captured. Check model capture_attention flag.")
        return

    print(f"Attention maps captured: {type(attn_maps)}")

    # Get scene encoder attention for ego (token 0)
    # Scene attention shape: (B, H, N, N) where N = A+M = 96
    scene_attn = attn_maps.scene_attentions  # list of 4 layers
    print(f"Scene encoder layers: {len(scene_attn)}")
    for i, sa in enumerate(scene_attn):
        print(f"  Layer {i}: shape={sa.shape}, mean={sa.mean():.4f}")

    # Use last encoder layer, ego token (index 0), averaged across heads
    last_layer_attn = scene_attn[-1][0]  # (H, N, N), batch=0
    ego_attn = last_layer_attn.mean(dim=0)[0]  # (N,) averaged over heads, ego row
    agent_attn = ego_attn[:32].cpu().numpy()  # attention to agents
    map_attn = ego_attn[32:96].cpu().numpy()  # attention to map tokens

    print(f"\nEgo attention stats:")
    print(f"  Agent attention: min={agent_attn.min():.4f}, max={agent_attn.max():.4f}, sum={agent_attn.sum():.4f}")
    print(f"  Map attention:   min={map_attn.min():.4f}, max={map_attn.max():.4f}, sum={map_attn.sum():.4f}")

    # Top attended agents
    valid_agents = agent_data["agent_mask"]
    for i in np.argsort(-agent_attn)[:5]:
        if valid_agents[i]:
            # Get agent type from polyline features (type_onehot at indices 17:22)
            type_oh = agent_data["agent_polylines"][i, -1, 17:22]  # last valid step
            type_names = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]
            type_name = type_names[np.argmax(type_oh)] if type_oh.sum() > 0 else "unknown"
            pos = agent_data["agent_polylines"][i, -1, 0:2]
            print(f"  Agent {i} ({type_name}): attn={agent_attn[i]:.4f}, pos=({pos[0]:.1f}, {pos[1]:.1f})")

    # Get agent positions at anchor frame (last history step)
    agent_positions = agent_data["agent_polylines"][:, -1, 0:2]  # (A, 2)

    # Get predictions
    preds = result["predictions"]
    pred_trajs = preds["trajectories"][0, 0].cpu().numpy()  # (M, 80, 2), first target
    pred_scores = preds["scores"][0, 0].cpu().numpy()        # (M,)
    best_mode = pred_scores.argmax()

    # Get target future (ego)
    target_future = result["batch"]["target_future"][0, 0].cpu().numpy()  # (80, 2)
    target_valid = result["batch"]["target_future_valid"][0, 0].cpu().numpy()

    # Get ego history
    ego_history = agent_data["agent_polylines"][0, :, 0:2]  # (11, 2)

    print(f"\nPrediction: best mode {best_mode}, score={pred_scores[best_mode]:.3f}")
    print(f"  Pred endpoint: ({pred_trajs[best_mode, -1, 0]:.1f}, {pred_trajs[best_mode, -1, 1]:.1f})")
    if target_valid[-1]:
        print(f"  GT endpoint:   ({target_future[-1, 0]:.1f}, {target_future[-1, 1]:.1f})")

    # === Render Space-Attention BEV Heatmap ===
    print("\nRendering space-attention BEV heatmap...")
    fig1 = render_space_attention_bev(
        agent_positions_bev=agent_positions,
        agent_attention=agent_attn,
        agent_mask=valid_agents,
        lane_centerlines_bev=map_data["lane_centerlines_bev"],
        map_attention=map_attn,
        map_mask=map_data["map_mask"],
        target_history_bev=ego_history,
        target_future_bev=target_future,
        pred_trajectories_bev=pred_trajs[pred_scores.argsort()[-3:]],  # top 3 modes
        all_lane_points=map_data["lane_centerlines_bev"],
        title="Scene Encoder Attention (Last Layer, Ego Agent)",
    )
    fig1.savefig(f"{output_dir}/space_attention_bev.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {output_dir}/space_attention_bev.png")

    # === Render Lane Activation Map ===
    print("Rendering lane activation map...")
    # Use decoder map attention for winning mode
    if hasattr(attn_maps, 'decoder_map_attentions') and attn_maps.decoder_map_attentions:
        dec_map_attn = attn_maps.decoder_map_attentions[-1][0]  # last layer, batch 0
        # (H, K0, M) -> average over heads, pick winning mode
        dec_map_attn_avg = dec_map_attn.mean(dim=0)  # (K0, M)
        # Use the mode with highest score
        winning_query = pred_scores.argmax()
        lane_attn = dec_map_attn_avg[winning_query].cpu().numpy()  # (M,)
    else:
        lane_attn = map_attn  # fallback to encoder attention

    fig2 = render_lane_activation_map(
        lane_centerlines_bev=map_data["lane_centerlines_bev"],
        lane_attention=lane_attn,
        lane_mask=map_data["map_mask"],
        target_history_bev=ego_history,
        target_future_bev=target_future,
        pred_trajectories_bev=pred_trajs[pred_scores.argsort()[-3:]],
        title="Lane Token Activation (Decoder, Winning Mode)",
    )
    fig2.savefig(f"{output_dir}/lane_activation_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {output_dir}/lane_activation_map.png")

    # === Attention by agent type summary ===
    print("\n=== Attention by Agent Type ===")
    type_attn = {}
    for i in range(32):
        if not valid_agents[i]:
            continue
        type_oh = agent_data["agent_polylines"][i, -1, 17:22]
        type_names = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]
        type_name = type_names[np.argmax(type_oh)] if type_oh.sum() > 0 else "unknown"
        if type_name not in type_attn:
            type_attn[type_name] = []
        type_attn[type_name].append(agent_attn[i])

    for tname, attns in sorted(type_attn.items()):
        print(f"  {tname:12s}: n={len(attns):3d}, mean_attn={np.mean(attns):.4f}, max={np.max(attns):.4f}")

    print(f"\nAll figures saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
