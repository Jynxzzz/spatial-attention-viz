"""Multi-scene attention visualization demo.

Generates attention BEV overlays for multiple scenes with:
- Agent bounding boxes colored by type (vehicle=blue, pedestrian=orange, cyclist=green)
- Lane centerlines with attention-weighted coloring
- Gaussian-splatted attention heatmap
- History, GT future, predicted trajectories
- Per-scene attention stats
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention

# Agent type colors
TYPE_COLORS = {
    "vehicle": "#4285F4",       # blue
    "pedestrian": "#EA8600",    # orange
    "cyclist": "#34A853",       # green
    "other": "#9E9E9E",        # gray
    "unknown": "#9E9E9E",
}
TYPE_MARKERS = {
    "vehicle": "s",       # square
    "pedestrian": "o",    # circle
    "cyclist": "D",       # diamond
    "other": "v",
    "unknown": "v",
}
TYPE_NAMES = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]


def get_agent_type(agent_polylines, agent_idx):
    """Get agent type from polyline features (one-hot at indices 12:17)."""
    type_oh = agent_polylines[agent_idx, -1, 12:17]
    if type_oh.sum() > 0:
        return TYPE_NAMES[np.argmax(type_oh)]
    return "unknown"


def render_rich_bev(
    agent_data, map_data, agent_attn, map_attn,
    ego_history, target_future, target_valid,
    pred_trajs, pred_scores,
    title="", bev_range=50.0, resolution=0.5,
    ax=None,
):
    """Render a rich BEV with attention overlay and typed agents."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = ax.figure

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

    # === Build heatmap: agent Gaussian splats ===
    for i in range(len(agent_positions)):
        if not valid_agents[i] or agent_attn[i] < 1e-6:
            continue
        pos = agent_positions[i]
        # Gaussian splat with sigma proportional to attention
        sigma_px = 4.0 / resolution
        cx, cy = bev_to_grid(pos)
        for dy in range(int(-3*sigma_px), int(3*sigma_px)+1):
            for dx in range(int(-3*sigma_px), int(3*sigma_px)+1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    dist2 = dx**2 + dy**2
                    heatmap[gy, gx] += agent_attn[i] * np.exp(-dist2 / (2 * sigma_px**2))

    # === Build heatmap: lane painting ===
    lane_width_px = int(2.0 / resolution)
    for i in range(len(lane_pts)):
        if not map_mask[i] or map_attn[i] < 1e-6:
            continue
        for p in range(lane_pts.shape[1] - 1):
            p1, p2 = lane_pts[i, p], lane_pts[i, p+1]
            g1x, g1y = bev_to_grid(p1)
            g2x, g2y = bev_to_grid(p2)
            # Simple line painting
            n_steps = max(abs(g2x - g1x), abs(g2y - g1y), 1)
            for s in range(n_steps + 1):
                t = s / n_steps
                gx = int(g1x + t * (g2x - g1x))
                gy = int(g1y + t * (g2y - g1y))
                for dw in range(-lane_width_px//2, lane_width_px//2 + 1):
                    for dh in range(-lane_width_px//2, lane_width_px//2 + 1):
                        nx, ny = gx + dw, gy + dh
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            heatmap[ny, nx] += map_attn[i] * 0.3

    # Smooth and normalize
    heatmap = gaussian_filter(heatmap, sigma=2.5)
    if heatmap.max() > 0:
        p95 = np.percentile(heatmap[heatmap > 0], 95) if (heatmap > 0).any() else 1.0
        heatmap = np.clip(heatmap / max(p95, 1e-8), 0, 1)

    # === Background: all lanes in light gray ===
    for i in range(len(lane_pts)):
        if map_mask[i]:
            pts = lane_pts[i]
            ax.plot(pts[:, 0], pts[:, 1], '-', color='#D0D0D0', linewidth=1.0, alpha=0.5, zorder=1)

    # === Attention-weighted lanes on top ===
    for i in range(len(lane_pts)):
        if not map_mask[i] or map_attn[i] < 0.005:
            continue
        pts = lane_pts[i]
        # Color from blue (low) to red (high)
        normed = min(map_attn[i] / max(map_attn[map_mask].max(), 1e-8), 1.0)
        color = plt.cm.YlOrRd(normed * 0.8 + 0.1)
        width = 1.0 + normed * 4.0
        ax.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=width, alpha=0.7, zorder=2)

    # === Heatmap overlay ===
    extent = [-bev_range, bev_range, -bev_range, bev_range]
    ax.imshow(heatmap, extent=extent, origin='lower', cmap='magma', alpha=0.55,
              vmin=0, vmax=1, zorder=3)

    # === Agent markers colored by type ===
    for i in range(len(agent_positions)):
        if not valid_agents[i]:
            continue
        pos = agent_positions[i]
        if abs(pos[0]) > bev_range or abs(pos[1]) > bev_range:
            continue
        atype = get_agent_type(agent_data["agent_polylines"], i)
        color = TYPE_COLORS.get(atype, "#9E9E9E")
        marker = TYPE_MARKERS.get(atype, "v")
        size = 8 if i == 0 else 6  # ego bigger
        edge = 'white' if i == 0 else 'black'
        ew = 1.5 if i == 0 else 0.5
        ax.plot(pos[0], pos[1], marker, color=color, markersize=size,
                markeredgecolor=edge, markeredgewidth=ew, zorder=6)

    # === Ego marker (triangle) ===
    ax.plot(0, 0, '^', color='#0D47A1', markersize=14, markeredgecolor='white',
            markeredgewidth=1.5, zorder=8)

    # === History trajectory ===
    if ego_history is not None:
        ax.plot(ego_history[:, 0], ego_history[:, 1], '-o', color='#1E88E5',
                linewidth=2.5, markersize=3, label='History', zorder=7)

    # === GT future ===
    if target_future is not None and target_valid is not None:
        valid_future = target_future[target_valid]
        if len(valid_future) > 0:
            ax.plot(valid_future[:, 0], valid_future[:, 1], '--', color='#2E7D32',
                    linewidth=2.5, label='Ground Truth', zorder=7)

    # === Predictions (top 3 modes) ===
    if pred_trajs is not None and pred_scores is not None:
        top_k = min(3, len(pred_scores))
        top_indices = pred_scores.argsort()[-top_k:][::-1]
        for rank, k in enumerate(top_indices):
            alpha = 0.9 if rank == 0 else 0.4
            lw = 2.5 if rank == 0 else 1.5
            label = f'Pred (p={pred_scores[k]:.2f})' if rank == 0 else None
            ax.plot(pred_trajs[k, :, 0], pred_trajs[k, :, 1], '-',
                    color='#E53935', linewidth=lw, alpha=alpha, label=label, zorder=7)

    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_xlabel('Lateral (m)', fontsize=11)
    ax.set_ylabel('Forward (m)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.8)

    # Type legend
    type_patches = [mpatches.Patch(color=TYPE_COLORS[t], label=t.capitalize())
                    for t in ["vehicle", "pedestrian", "cyclist"]]
    ax.legend(handles=type_patches + ax.get_legend_handles_labels()[0][:3],
              loc='upper left', fontsize=7, framealpha=0.8)

    return fig


def find_diverse_scenes(val_scenes, n=6):
    """Find scenes with different characteristics."""
    selected = []
    categories = {
        "with_pedestrian": None,
        "with_cyclist": None,
        "many_vehicles": None,
        "few_agents": None,
        "intersection": None,
        "general": None,
    }

    for idx in range(min(100, len(val_scenes))):
        path = val_scenes[idx]
        try:
            with open(path, "rb") as f:
                scene = pickle.load(f)
        except Exception:
            continue

        objects = scene["objects"]
        n_valid = sum(1 for o in objects if o.get("valid", [False])[10])
        types = set()
        for obj in objects:
            if obj.get("valid", [False])[10]:
                types.add(obj.get("type", "unknown").lower())

        has_ped = "pedestrian" in types
        has_cyc = "cyclist" in types
        n_lanes = len(scene.get("lane_graph", {}).get("lane_ids", []))

        if has_ped and categories["with_pedestrian"] is None:
            categories["with_pedestrian"] = (idx, f"Scene with pedestrians ({n_valid} agents)")
        elif has_cyc and categories["with_cyclist"] is None:
            categories["with_cyclist"] = (idx, f"Scene with cyclists ({n_valid} agents)")
        elif n_valid > 20 and categories["many_vehicles"] is None:
            categories["many_vehicles"] = (idx, f"Dense traffic ({n_valid} agents)")
        elif n_valid < 8 and categories["few_agents"] is None:
            categories["few_agents"] = (idx, f"Sparse scene ({n_valid} agents)")
        elif n_lanes > 50 and categories["intersection"] is None:
            categories["intersection"] = (idx, f"Complex intersection ({n_lanes} lanes)")
        elif categories["general"] is None:
            categories["general"] = (idx, f"General scenario ({n_valid} agents)")

        if all(v is not None for v in categories.values()):
            break

    return {k: v for k, v in categories.items() if v is not None}


def main():
    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=39-val/minADE@6=2.687.ckpt"
    config_path = Path(__file__).resolve().parent.parent / "configs" / "mtr_lite.yaml"
    output_dir = "/tmp/attention_demo"
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
    print(f"Val scenes: {len(val_scenes)}")

    # Find diverse scenes
    print("Scanning for diverse scenes...")
    scene_picks = find_diverse_scenes(val_scenes)
    print(f"Found {len(scene_picks)} scene types:")
    for cat, (idx, desc) in scene_picks.items():
        print(f"  {cat}: scene {idx} - {desc}")

    # Generate attention visualizations
    n_scenes = len(scene_picks)
    cols = min(3, n_scenes)
    rows = (n_scenes + cols - 1) // cols

    fig_all, axes = plt.subplots(rows, cols, figsize=(10*cols, 10*rows))
    if n_scenes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for panel_idx, (cat, (scene_idx, desc)) in enumerate(scene_picks.items()):
        scene_path = val_scenes[scene_idx]
        print(f"\n[{panel_idx+1}/{n_scenes}] Processing: {desc}")

        try:
            result = extract_scene_attention(model=model, scene_path=scene_path, device="cpu")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        attn_maps = result["attention_maps"]
        agent_data = result["agent_data"]
        map_data = result["map_data"]
        preds = result["predictions"]

        if attn_maps is None:
            print("  No attention maps!")
            continue

        # Get attention from last encoder layer, ego token
        scene_attn = attn_maps.scene_attentions[-1][0]  # (H, N, N)
        ego_attn = scene_attn.mean(dim=0)[0].cpu().numpy()  # (N,)
        agent_attn = ego_attn[:32]
        map_attn_vals = ego_attn[32:96]

        # Predictions
        pred_trajs = preds["trajectories"][0, 0].cpu().numpy()
        pred_scores = preds["scores"][0, 0].cpu().numpy()
        target_future = result["batch"]["target_future"][0, 0].cpu().numpy()
        target_valid = result["batch"]["target_future_valid"][0, 0].cpu().numpy().astype(bool)
        ego_history = agent_data["agent_polylines"][0, :, 0:2]

        # Count agents by type
        type_counts = {}
        for i in range(32):
            if agent_data["agent_mask"][i]:
                atype = get_agent_type(agent_data["agent_polylines"], i)
                type_counts[atype] = type_counts.get(atype, 0) + 1
        type_str = ", ".join(f"{v} {k}s" for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))

        title = f"{desc}\n({type_str})"

        # Render
        render_rich_bev(
            agent_data=agent_data,
            map_data=map_data,
            agent_attn=agent_attn,
            map_attn=map_attn_vals,
            ego_history=ego_history,
            target_future=target_future,
            target_valid=target_valid,
            pred_trajs=pred_trajs,
            pred_scores=pred_scores,
            title=title,
            ax=axes[panel_idx],
        )

        # Also save individual figure
        fig_single = plt.figure(figsize=(12, 12))
        ax_single = fig_single.add_subplot(111)
        render_rich_bev(
            agent_data=agent_data,
            map_data=map_data,
            agent_attn=agent_attn,
            map_attn=map_attn_vals,
            ego_history=ego_history,
            target_future=target_future,
            target_valid=target_valid,
            pred_trajs=pred_trajs,
            pred_scores=pred_scores,
            title=title,
            ax=ax_single,
        )
        fname = f"{output_dir}/scene_{cat}.png"
        fig_single.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig_single)
        print(f"  Saved: {fname}")

        # Print attention stats
        print(f"  Agent attn sum: {agent_attn[agent_data['agent_mask']].sum():.3f}")
        print(f"  Map attn sum:   {map_attn_vals[map_data['map_mask']].sum():.3f}")
        best_mode = pred_scores.argmax()
        print(f"  Best prediction: mode {best_mode}, score={pred_scores[best_mode]:.3f}")

    # Hide unused axes
    for i in range(n_scenes, len(axes)):
        axes[i].set_visible(False)

    fig_all.suptitle("Attention Visualization Across Scene Types", fontsize=16, fontweight='bold', y=1.01)
    fig_all.tight_layout()
    composite_path = f"{output_dir}/composite_all_scenes.png"
    fig_all.savefig(composite_path, dpi=120, bbox_inches='tight')
    plt.close(fig_all)
    print(f"\nComposite figure saved: {composite_path}")
    print("Done!")


if __name__ == "__main__":
    main()
