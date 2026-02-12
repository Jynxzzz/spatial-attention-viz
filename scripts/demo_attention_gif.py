"""Generate attention evolution GIFs.

For each scene, creates an animated GIF showing how attention evolves:
1. Across encoder layers (Layer 0 → 1 → 2 → 3): broad → focused
2. Encoder vs Decoder attention comparison

Each frame is a BEV attention overlay at a specific layer.
"""

import sys
import os
import pickle
import random
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from pathlib import Path
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention

TYPE_COLORS = {"vehicle": "#4285F4", "pedestrian": "#EA8600", "cyclist": "#34A853",
               "other": "#9E9E9E", "unknown": "#9E9E9E"}
TYPE_MARKERS = {"vehicle": "s", "pedestrian": "o", "cyclist": "D",
                "other": "v", "unknown": "v"}
TYPE_NAMES = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]


def get_agent_type(agent_polylines, idx):
    type_oh = agent_polylines[idx, -1, 12:17]
    if type_oh.sum() > 0:
        return TYPE_NAMES[np.argmax(type_oh)]
    return "unknown"


def render_attention_frame(
    agent_data, map_data, agent_attn, map_attn,
    ego_history, target_future, target_valid,
    pred_trajs, pred_scores,
    layer_label="", scene_label="",
    bev_range=50.0, resolution=0.5,
    global_vmax=None,
):
    """Render one frame of the attention GIF. Returns PIL Image."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    grid_size = int(2 * bev_range / resolution)
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    def bev_to_grid(xy):
        gx = int((xy[0] + bev_range) / resolution)
        gy = int((xy[1] + bev_range) / resolution)
        return np.clip(gx, 0, grid_size - 1), np.clip(gy, 0, grid_size - 1)

    agent_positions = agent_data["agent_polylines"][:, -1, 0:2]
    valid_agents = agent_data["agent_mask"]
    lane_pts = map_data["lane_centerlines_bev"]
    map_mask = map_data["map_mask"]

    # Agent Gaussian splats
    sigma_px = 4.0 / resolution
    for i in range(len(agent_positions)):
        if not valid_agents[i] or agent_attn[i] < 1e-6:
            continue
        pos = agent_positions[i]
        cx, cy = bev_to_grid(pos)
        r = int(3 * sigma_px)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    heatmap[gy, gx] += agent_attn[i] * np.exp(-(dx**2 + dy**2) / (2 * sigma_px**2))

    # Lane painting
    lane_w = int(2.0 / resolution)
    for i in range(len(lane_pts)):
        if not map_mask[i] or map_attn[i] < 1e-6:
            continue
        for p in range(lane_pts.shape[1] - 1):
            p1, p2 = lane_pts[i, p], lane_pts[i, p + 1]
            g1x, g1y = bev_to_grid(p1)
            g2x, g2y = bev_to_grid(p2)
            n_steps = max(abs(g2x - g1x), abs(g2y - g1y), 1)
            for s in range(n_steps + 1):
                t = s / n_steps
                gx = int(g1x + t * (g2x - g1x))
                gy = int(g1y + t * (g2y - g1y))
                for dw in range(-lane_w // 2, lane_w // 2 + 1):
                    for dh in range(-lane_w // 2, lane_w // 2 + 1):
                        nx, ny = gx + dw, gy + dh
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            heatmap[ny, nx] += map_attn[i] * 0.3

    # Smooth
    heatmap = gaussian_filter(heatmap, sigma=2.5)

    # Normalize (use global_vmax for consistent scale across frames)
    if global_vmax is not None and global_vmax > 0:
        heatmap = np.clip(heatmap / global_vmax, 0, 1)
    elif heatmap.max() > 0:
        p95 = np.percentile(heatmap[heatmap > 0], 95) if (heatmap > 0).any() else 1.0
        heatmap = np.clip(heatmap / max(p95, 1e-8), 0, 1)

    # Background lanes
    for i in range(len(lane_pts)):
        if map_mask[i]:
            ax.plot(lane_pts[i, :, 0], lane_pts[i, :, 1], '-', color='#D0D0D0',
                    linewidth=1.0, alpha=0.5, zorder=1)

    # Attention-weighted lanes
    max_map_attn = map_attn[map_mask].max() if map_mask.any() and map_attn[map_mask].max() > 0 else 1.0
    for i in range(len(lane_pts)):
        if not map_mask[i] or map_attn[i] < 0.003:
            continue
        normed = min(map_attn[i] / max_map_attn, 1.0)
        color = plt.cm.YlOrRd(normed * 0.8 + 0.1)
        width = 1.0 + normed * 4.0
        ax.plot(lane_pts[i, :, 0], lane_pts[i, :, 1], '-', color=color,
                linewidth=width, alpha=0.7, zorder=2)

    # Heatmap overlay
    extent = [-bev_range, bev_range, -bev_range, bev_range]
    ax.imshow(heatmap, extent=extent, origin='lower', cmap='magma', alpha=0.55,
              vmin=0, vmax=1, zorder=3)

    # Agent markers
    for i in range(len(agent_positions)):
        if not valid_agents[i]:
            continue
        pos = agent_positions[i]
        if abs(pos[0]) > bev_range or abs(pos[1]) > bev_range:
            continue
        atype = get_agent_type(agent_data["agent_polylines"], i)
        color = TYPE_COLORS.get(atype, "#9E9E9E")
        marker = TYPE_MARKERS.get(atype, "v")
        size = 8 if i == 0 else 6
        edge = 'white' if i == 0 else 'black'
        ew = 1.5 if i == 0 else 0.5
        ax.plot(pos[0], pos[1], marker, color=color, markersize=size,
                markeredgecolor=edge, markeredgewidth=ew, zorder=6)

    # Ego
    ax.plot(0, 0, '^', color='#0D47A1', markersize=14, markeredgecolor='white',
            markeredgewidth=1.5, zorder=8)

    # History
    if ego_history is not None:
        ax.plot(ego_history[:, 0], ego_history[:, 1], '-o', color='#1E88E5',
                linewidth=2.5, markersize=3, zorder=7)

    # GT future
    if target_future is not None and target_valid is not None:
        vf = target_future[target_valid]
        if len(vf) > 0:
            ax.plot(vf[:, 0], vf[:, 1], '--', color='#2E7D32', linewidth=2.5, zorder=7)

    # Predictions (top 3)
    if pred_trajs is not None and pred_scores is not None:
        top_k = min(3, len(pred_scores))
        top_idx = pred_scores.argsort()[-top_k:][::-1]
        for rank, k in enumerate(top_idx):
            alpha = 0.9 if rank == 0 else 0.4
            lw = 2.5 if rank == 0 else 1.5
            ax.plot(pred_trajs[k, :, 0], pred_trajs[k, :, 1], '-',
                    color='#E53935', linewidth=lw, alpha=alpha, zorder=7)

    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_xlabel('Lateral (m)', fontsize=12)
    ax.set_ylabel('Forward (m)', fontsize=12)
    ax.set_aspect('equal')

    # Compute entropy for this layer
    full_attn = np.concatenate([agent_attn[valid_agents], map_attn[map_mask]])
    full_attn = full_attn / (full_attn.sum() + 1e-10)
    entropy = -np.sum(full_attn * np.log2(full_attn + 1e-10))
    agent_sum = agent_attn[valid_agents].sum()
    map_sum = map_attn[map_mask].sum()

    title = (f"{scene_label}\n"
             f"{layer_label}  |  "
             f"Entropy: {entropy:.2f} bits  |  "
             f"Agent: {agent_sum:.1%}  Map: {map_sum:.1%}")
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Legend
    patches = [mpatches.Patch(color=TYPE_COLORS[t], label=t.capitalize())
               for t in ["vehicle", "pedestrian", "cyclist"]]
    patches += [
        plt.Line2D([0], [0], color='#1E88E5', linewidth=2, label='History'),
        plt.Line2D([0], [0], color='#2E7D32', linewidth=2, linestyle='--', label='Ground Truth'),
        plt.Line2D([0], [0], color='#E53935', linewidth=2, label='Prediction'),
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=8, framealpha=0.9)

    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_layer_gif(result, scene_label, output_path, bev_range=50.0):
    """Create GIF showing encoder layer 0→1→2→3 attention evolution."""
    attn_maps = result["attention_maps"]
    agent_data = result["agent_data"]
    map_data = result["map_data"]
    preds = result["predictions"]

    pred_trajs = preds["trajectories"][0, 0].cpu().numpy()
    pred_scores = preds["scores"][0, 0].cpu().numpy()
    target_future = result["batch"]["target_future"][0, 0].cpu().numpy()
    target_valid = result["batch"]["target_future_valid"][0, 0].cpu().numpy().astype(bool)
    ego_history = agent_data["agent_polylines"][0, :, 0:2]
    valid_agents = agent_data["agent_mask"]
    map_mask = map_data["map_mask"]

    # Pre-compute all layers to find global vmax for consistent colorscale
    layer_data = []
    raw_heatmaps = []
    for layer_idx in range(4):
        scene_attn = attn_maps.scene_attentions[layer_idx][0]  # (H, N, N)
        ego_attn = scene_attn.mean(dim=0)[0].cpu().numpy()
        agent_attn = ego_attn[:32]
        map_attn = ego_attn[32:96]
        layer_data.append((agent_attn, map_attn))

    # Also add decoder attention as bonus frames if available
    decoder_frames = []
    if hasattr(attn_maps, 'decoder_agent_attentions') and attn_maps.decoder_agent_attentions:
        for dec_layer in range(len(attn_maps.decoder_agent_attentions)):
            dec_agent = attn_maps.decoder_agent_attentions[dec_layer][0]  # (1, H, K0, A)
            dec_map = attn_maps.decoder_map_attentions[dec_layer][0]     # (1, H, K0, M)
            # Squeeze batch dim, average over heads, then average over all intention queries
            da = dec_agent.squeeze(0).mean(dim=0).mean(dim=0).cpu().numpy()  # (A,)
            dm = dec_map.squeeze(0).mean(dim=0).mean(dim=0).cpu().numpy()    # (M,)
            decoder_frames.append((da, dm))

    # Generate encoder frames
    frames = []
    for layer_idx, (agent_attn, map_attn) in enumerate(layer_data):
        print(f"    Encoder Layer {layer_idx}...")
        frame = render_attention_frame(
            agent_data=agent_data, map_data=map_data,
            agent_attn=agent_attn, map_attn=map_attn,
            ego_history=ego_history,
            target_future=target_future, target_valid=target_valid,
            pred_trajs=pred_trajs, pred_scores=pred_scores,
            layer_label=f"Encoder Layer {layer_idx}/3",
            scene_label=scene_label,
            bev_range=bev_range,
        )
        frames.append(frame)

    # Generate decoder frames
    n_dec = len(decoder_frames)
    # Use at most 4 evenly-spaced decoder layers to keep GIF concise
    if n_dec > 4:
        dec_indices = [int(i * (n_dec - 1) / 3) for i in range(4)]
    else:
        dec_indices = list(range(n_dec))
    for step, dec_idx in enumerate(dec_indices):
        da, dm = decoder_frames[dec_idx]
        print(f"    Decoder Layer {dec_idx}/{n_dec-1}...")
        frame = render_attention_frame(
            agent_data=agent_data, map_data=map_data,
            agent_attn=da, map_attn=dm,
            ego_history=ego_history,
            target_future=target_future, target_valid=target_valid,
            pred_trajs=pred_trajs, pred_scores=pred_scores,
            layer_label=f"Decoder Layer {dec_idx}/{n_dec-1} (All Queries Avg)",
            scene_label=scene_label,
            bev_range=bev_range,
        )
        frames.append(frame)

    # Save GIF (1.5s per frame, loop)
    if frames:
        # Add pause on last frame (repeat it)
        frames_with_pause = frames + [frames[-1]] * 2
        frames_with_pause[0].save(
            output_path,
            save_all=True,
            append_images=frames_with_pause[1:],
            duration=1500,  # 1.5s per frame
            loop=0,
        )
        print(f"    GIF saved: {output_path} ({len(frames)} frames)")

    return frames


def main():
    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=39-val/minADE@6=2.687.ckpt"
    config_path = Path(__file__).resolve().parent.parent / "configs" / "mtr_lite.yaml"
    output_dir = "/tmp/attention_demo/gifs"
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading model on CPU...")
    module = MTRLiteModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    model = module.model

    # Get val scenes
    with open(cfg["data"]["scene_list"]) as f:
        all_scenes = [l.strip() for l in f if l.strip()]
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * 0.15))
    val_scenes = [all_scenes[i] for i in indices[:n_val] if os.path.exists(all_scenes[i])]

    # Pick diverse scenes (same as before)
    scene_picks = {}
    for idx in range(min(100, len(val_scenes))):
        path = val_scenes[idx]
        try:
            with open(path, "rb") as f:
                scene = pickle.load(f)
            types = set()
            n_valid = 0
            for obj in scene["objects"]:
                if obj.get("valid", [False])[10]:
                    types.add(obj.get("type", "unknown").lower())
                    n_valid += 1
            if "pedestrian" in types and "with_pedestrian" not in scene_picks:
                scene_picks["with_pedestrian"] = (idx, f"Intersection with Pedestrians ({n_valid} agents)")
            elif "cyclist" in types and "with_cyclist" not in scene_picks:
                scene_picks["with_cyclist"] = (idx, f"Scene with Cyclists ({n_valid} agents)")
            elif n_valid > 20 and "dense" not in scene_picks:
                scene_picks["dense"] = (idx, f"Dense Traffic ({n_valid} agents)")
            elif 3 < n_valid < 10 and "general" not in scene_picks:
                scene_picks["general"] = (idx, f"General Driving ({n_valid} agents)")
            if len(scene_picks) >= 4:
                break
        except Exception:
            continue

    print(f"Generating GIFs for {len(scene_picks)} scenes...")

    for cat, (scene_idx, desc) in scene_picks.items():
        scene_path = val_scenes[scene_idx]
        print(f"\n[{cat}] {desc}")

        try:
            result = extract_scene_attention(model=model, scene_path=scene_path, device="cpu")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if result["attention_maps"] is None:
            print("  No attention maps!")
            continue

        gif_path = f"{output_dir}/attention_evolution_{cat}.gif"
        make_layer_gif(result, scene_label=desc, output_path=gif_path)

    print(f"\nAll GIFs saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
