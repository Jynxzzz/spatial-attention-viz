"""Counterfactual attention analysis: remove a key vehicle and compare attention maps.

Produces a publication-quality 2-panel (or 4-panel) figure showing how the
model's spatial attention redistribu tes when a highly-attended vehicle is
removed from the scene.

Pipeline:
  1. Load MTR-Lite model on CPU
  2. Scan validation scenes for an intersection with many agents
  3. Run inference + capture attention on the original scene
  4. Identify the most-attended non-ego vehicle
  5. Remove it using SceneEditor and re-run inference
  6. Generate a side-by-side figure comparing attention maps
  7. Save to paper/figures/fig_counterfactual_case_study.pdf

Usage:
    cd /home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper
    python scripts/generate_counterfactual_figure.py
"""

import copy
import os
import pickle
import random
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter

# Setup project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention
from experiments.scene_editor import SceneEditor

# ===========================================================================
# Publication style
# ===========================================================================
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
}

# Agent type colors and markers
TYPE_COLORS = {
    "vehicle": "#4285F4",
    "pedestrian": "#EA8600",
    "cyclist": "#34A853",
    "other": "#9E9E9E",
    "unknown": "#9E9E9E",
}
TYPE_MARKERS = {
    "vehicle": "s",
    "pedestrian": "o",
    "cyclist": "D",
    "other": "v",
    "unknown": "v",
}
TYPE_NAMES = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]

# Trajectory colors
COLOR_HISTORY = "#1565C0"
COLOR_GT = "#2E7D32"
COLOR_PRED = "#C62828"
COLOR_EGO = "#0D47A1"
COLOR_LANE_BG = "#BDBDBD"
COLOR_REMOVED = "#FF6F00"  # Orange for removed-agent marker


def get_agent_type(agent_polylines, agent_idx):
    """Get agent type from polyline features (one-hot at indices 12:17)."""
    type_oh = agent_polylines[agent_idx, -1, 12:17]
    if type_oh.sum() > 0:
        return TYPE_NAMES[int(np.argmax(type_oh))]
    return "unknown"


def extract_vis_data(result):
    """Extract visualization data from an attention extraction result."""
    attn_maps = result["attention_maps"]
    agent_data = result["agent_data"]
    map_data = result["map_data"]
    preds = result["predictions"]

    # Scene encoder attention: last layer, ego token (token 0), head-averaged
    scene_attn = attn_maps.scene_attentions[-1][0]  # (H, N, N)
    ego_attn = scene_attn.mean(dim=0)[0].cpu().numpy()  # (N,)
    agent_attn = ego_attn[:32]
    map_attn = ego_attn[32:96]

    # Predictions
    pred_trajs = preds["trajectories"][0, 0].cpu().numpy()  # (K, 80, 2)
    pred_scores = preds["scores"][0, 0].cpu().numpy()       # (K,)
    target_future = result["batch"]["target_future"][0, 0].cpu().numpy()
    target_valid = result["batch"]["target_future_valid"][0, 0].cpu().numpy().astype(bool)
    ego_history = agent_data["agent_polylines"][0, :, 0:2]

    return {
        "agent_data": agent_data,
        "map_data": map_data,
        "agent_attn": agent_attn,
        "map_attn": map_attn,
        "ego_history": ego_history,
        "target_future": target_future,
        "target_valid": target_valid,
        "pred_trajs": pred_trajs,
        "pred_scores": pred_scores,
    }


def render_counterfactual_bev(
    agent_data, map_data, agent_attn, map_attn,
    ego_history, target_future, target_valid,
    pred_trajs, pred_scores,
    title="", bev_range=50.0, resolution=0.5,
    ax=None, show_colorbar=True, panel_label=None,
    removed_agent_pos=None, removed_agent_label=None,
    highlight_agent_idx=None,
):
    """Render publication-quality BEV with attention overlay.

    Same as render_paper_bev but with support for:
    - Marking the position of a removed agent (orange X)
    - Highlighting the most-attended agent (red circle)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
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

    # --- Build heatmap: agent Gaussian splats ---
    sigma_px = 4.0 / resolution
    for i in range(len(agent_positions)):
        if not valid_agents[i] or agent_attn[i] < 1e-6:
            continue
        pos = agent_positions[i]
        cx, cy = bev_to_grid(pos)
        span = int(3 * sigma_px)
        for dy in range(-span, span + 1):
            for dx in range(-span, span + 1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    dist2 = dx**2 + dy**2
                    heatmap[gy, gx] += agent_attn[i] * np.exp(-dist2 / (2 * sigma_px**2))

    # --- Build heatmap: lane painting ---
    lane_width_px = int(2.0 / resolution)
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
                for dw in range(-lane_width_px // 2, lane_width_px // 2 + 1):
                    for dh in range(-lane_width_px // 2, lane_width_px // 2 + 1):
                        nx_, ny_ = gx + dw, gy + dh
                        if 0 <= nx_ < grid_size and 0 <= ny_ < grid_size:
                            heatmap[ny_, nx_] += map_attn[i] * 0.3

    # Smooth and normalize
    heatmap = gaussian_filter(heatmap, sigma=2.5)
    if heatmap.max() > 0:
        p95 = np.percentile(heatmap[heatmap > 0], 95) if (heatmap > 0).any() else 1.0
        heatmap = np.clip(heatmap / max(p95, 1e-8), 0, 1)

    # --- Background lanes ---
    for i in range(len(lane_pts)):
        if map_mask[i]:
            pts = lane_pts[i]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=COLOR_LANE_BG,
                    linewidth=0.6, alpha=0.5, zorder=1)

    # --- Attention-weighted lanes ---
    max_map_attn = map_attn[map_mask].max() if map_mask.any() and map_attn[map_mask].max() > 0 else 1.0
    for i in range(len(lane_pts)):
        if not map_mask[i] or map_attn[i] < 0.005:
            continue
        pts = lane_pts[i]
        normed = min(map_attn[i] / max_map_attn, 1.0)
        color = plt.cm.YlOrRd(normed * 0.8 + 0.1)
        width = 0.6 + normed * 2.5
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=width,
                alpha=0.7, zorder=2)

    # --- Heatmap overlay ---
    extent = [-bev_range, bev_range, -bev_range, bev_range]
    im = ax.imshow(heatmap, extent=extent, origin="lower", cmap="magma",
                   alpha=0.55, vmin=0, vmax=1, zorder=3)

    # --- Agent markers colored by type ---
    for i in range(len(agent_positions)):
        if not valid_agents[i]:
            continue
        pos = agent_positions[i]
        if abs(pos[0]) > bev_range or abs(pos[1]) > bev_range:
            continue
        atype = get_agent_type(agent_data["agent_polylines"], i)
        color = TYPE_COLORS.get(atype, "#9E9E9E")
        marker = TYPE_MARKERS.get(atype, "v")
        size = 5 if i == 0 else 4
        edge = "white" if i == 0 else "black"
        ew = 0.8 if i == 0 else 0.3

        # Highlight the most-attended agent with a bold ring and label
        if highlight_agent_idx is not None and i == highlight_agent_idx:
            ax.plot(pos[0], pos[1], "o", color="none", markersize=13,
                    markeredgecolor="#E53935", markeredgewidth=2.5, zorder=9)
            # Position annotation to avoid the upper-right legend:
            # Compact offset toward lower-left (away from legend corner)
            hx, hy = pos[0], pos[1]
            hdx = -10 if hx > -bev_range + 15 else 8
            hdy = -8
            if hy < 0:
                hdy = 8
            ax.annotate(
                "Highest\nattention",
                xy=(hx, hy),
                xytext=(hx + hdx, hy + hdy),
                fontsize=7, fontweight="bold", color="#E53935",
                arrowprops=dict(arrowstyle="-|>", color="#E53935",
                                lw=1.5, mutation_scale=10),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#E53935", alpha=0.95, linewidth=1.0),
                zorder=11,
            )

        ax.plot(pos[0], pos[1], marker, color=color, markersize=size,
                markeredgecolor=edge, markeredgewidth=ew, zorder=6)

    # --- Ego marker ---
    ax.plot(0, 0, "^", color=COLOR_EGO, markersize=8, markeredgecolor="white",
            markeredgewidth=0.8, zorder=8)

    # --- History trajectory ---
    if ego_history is not None:
        ax.plot(ego_history[:, 0], ego_history[:, 1], "-o", color=COLOR_HISTORY,
                linewidth=1.5, markersize=2, label="History", zorder=7)

    # --- GT future ---
    if target_future is not None and target_valid is not None:
        valid_future = target_future[target_valid]
        if len(valid_future) > 0:
            ax.plot(valid_future[:, 0], valid_future[:, 1], "--", color=COLOR_GT,
                    linewidth=1.5, label="Ground truth", zorder=7)

    # --- Predictions (top 3 modes) ---
    if pred_trajs is not None and pred_scores is not None:
        top_k = min(3, len(pred_scores))
        top_indices = pred_scores.argsort()[-top_k:][::-1]
        for rank, k in enumerate(top_indices):
            alpha = 0.9 if rank == 0 else 0.4
            lw = 1.8 if rank == 0 else 1.0
            label = "Prediction" if rank == 0 else None
            ax.plot(pred_trajs[k, :, 0], pred_trajs[k, :, 1], "-",
                    color=COLOR_PRED, linewidth=lw, alpha=alpha, label=label,
                    zorder=7)

    # --- Mark removed agent position with orange X ---
    if removed_agent_pos is not None:
        ax.plot(removed_agent_pos[0], removed_agent_pos[1], "X",
                color=COLOR_REMOVED, markersize=14, markeredgecolor="white",
                markeredgewidth=1.2, zorder=10)
        if removed_agent_label:
            # Smart positioning: short offset toward lower-left to avoid
            # the legend (upper-right corner). Keep arrows compact.
            rx, ry = removed_agent_pos[0], removed_agent_pos[1]
            dx = -10 if rx > -bev_range + 15 else 8
            dy = -8
            if ry < 0:
                dy = 8
            ax.annotate(
                removed_agent_label,
                xy=(rx, ry),
                xytext=(rx + dx, ry + dy),
                fontsize=8, fontweight="bold", color=COLOR_REMOVED,
                arrowprops=dict(arrowstyle="-|>", color=COLOR_REMOVED,
                                lw=1.5, mutation_scale=10),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=COLOR_REMOVED, alpha=0.95, linewidth=1.0),
                zorder=11,
            )

    # --- Axis formatting ---
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Longitudinal (m)")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    if title:
        ax.set_title(title, fontweight="bold", pad=4)

    if panel_label:
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.8))

    # Legend
    present_types = set()
    for i in range(len(agent_positions)):
        if valid_agents[i]:
            present_types.add(get_agent_type(agent_data["agent_polylines"], i))

    legend_handles = []
    for tname in ["vehicle", "pedestrian", "cyclist"]:
        if tname in present_types:
            legend_handles.append(
                mpatches.Patch(color=TYPE_COLORS[tname], label=tname.capitalize()))

    for handle, label in zip(*ax.get_legend_handles_labels()):
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.85, edgecolor="none", handlelength=1.5,
              handletextpad=0.4, borderpad=0.3)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=25)
        cbar.set_label("Attention weight", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    return fig, im


def extract_scene_attention_from_dict(model, scene_dict, device="cpu",
                                       anchor_frame=10, history_len=11,
                                       future_len=80, max_agents=32,
                                       max_map_polylines=64, map_points_per_lane=20):
    """Like extract_scene_attention but takes a scene dict instead of a file path.

    We save to a temp file and call extract_scene_attention.
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
        pickle.dump(scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        result = extract_scene_attention(
            model=model,
            scene_path=tmp_path,
            anchor_frame=anchor_frame,
            history_len=history_len,
            future_len=future_len,
            max_agents=max_agents,
            max_map_polylines=max_map_polylines,
            map_points_per_lane=map_points_per_lane,
            device=device,
        )
    finally:
        os.unlink(tmp_path)

    return result


def find_intersection_scene(val_scenes, model, max_scan=300):
    """Scan validation scenes to find the best intersection scene for counterfactual.

    Ranks candidate scenes by: attention weight of most-attended non-ego vehicle.
    Requires:
    - >= 8 valid agents (but prefers 10-20 range for visual clarity)
    - >= 30 lanes (intersection-like)
    - Good trajectory visibility (history >= 3m, future >= 15 frames)
    - Most-attended non-ego vehicle has attn weight >= 0.03
    - Removed vehicle within 35m of ego (visible in BEV)
    """
    candidates = []  # (score, scene_path, result, best_idx)

    for idx in range(min(max_scan, len(val_scenes))):
        scene_path = val_scenes[idx]
        if not os.path.exists(scene_path):
            continue

        print(f"  [{idx:3d}] Trying: {os.path.basename(scene_path)}")

        try:
            result = extract_scene_attention(
                model=model, scene_path=scene_path, device="cpu")
        except Exception as e:
            print(f"        Failed: {e}")
            continue

        if result["attention_maps"] is None:
            print(f"        No attention maps")
            continue

        agent_data = result["agent_data"]
        map_data = result["map_data"]
        n_agents = int(agent_data["agent_mask"].sum())
        n_lanes = int(map_data["map_mask"].sum())

        # Check scene complexity
        if n_agents < 8:
            print(f"        Too few agents ({n_agents})")
            continue
        if n_lanes < 30:
            print(f"        Too few lanes ({n_lanes}), not intersection-like")
            continue

        # Check trajectory visibility
        ego_hist = agent_data["agent_polylines"][0, :, 0:2]
        ego_valid = agent_data["agent_valid"][0]
        valid_hist = ego_hist[ego_valid]
        if len(valid_hist) < 5:
            print(f"        History too short")
            continue
        hist_range = np.linalg.norm(valid_hist.max(axis=0) - valid_hist.min(axis=0))
        if hist_range < 3.0:
            print(f"        History extent too small ({hist_range:.1f}m)")
            continue

        target_future = result["batch"]["target_future"][0, 0].cpu().numpy()
        target_valid_mask = result["batch"]["target_future_valid"][0, 0].cpu().numpy().astype(bool)
        valid_future = target_future[target_valid_mask]
        if len(valid_future) < 15:
            print(f"        Future too short")
            continue

        # Check that at least one non-ego agent has meaningful attention
        vis = extract_vis_data(result)
        agent_attn = vis["agent_attn"]
        valid_mask = agent_data["agent_mask"]

        # Find most-attended non-ego vehicle (must be within 35m of ego)
        best_attn_idx = None
        best_attn_val = 0.0
        for i in range(1, 32):  # skip ego (idx 0)
            if not valid_mask[i]:
                continue
            atype = get_agent_type(agent_data["agent_polylines"], i)
            if atype != "vehicle":
                continue
            pos = agent_data["agent_polylines"][i, -1, 0:2]
            dist = np.linalg.norm(pos)
            if dist > 35.0:
                continue  # too far, would be at edge of BEV
            if agent_attn[i] > best_attn_val:
                best_attn_val = agent_attn[i]
                best_attn_idx = i

        if best_attn_idx is None or best_attn_val < 0.03:
            print(f"        No well-attended nearby non-ego vehicle (best={best_attn_val:.4f})")
            continue

        # Compute a composite score: higher attention + moderate agent count preferred
        agent_count_bonus = 1.0 if 10 <= n_agents <= 22 else 0.7
        score = best_attn_val * agent_count_bonus

        print(f"        CANDIDATE: {n_agents} agents, {n_lanes} lanes, "
              f"best vehicle attn = {best_attn_val:.4f} (slot {best_attn_idx}), "
              f"score = {score:.4f}")
        candidates.append((score, scene_path, result, best_attn_idx))

        # Once we have 5+ good candidates, pick the best
        if len(candidates) >= 5:
            break

    if not candidates:
        return None, None, None

    # Pick the highest-scoring candidate
    candidates.sort(key=lambda x: -x[0])
    best = candidates[0]
    print(f"\n  SELECTED best candidate (score={best[0]:.4f})")
    return best[1], best[2], best[3]


def compute_attention_delta_stats(vis_orig, vis_mod, agent_mask_orig):
    """Compute statistics about how attention redistributed."""
    orig_agent_attn = vis_orig["agent_attn"]
    mod_agent_attn = vis_mod["agent_attn"]
    orig_map_attn = vis_orig["map_attn"]
    mod_map_attn = vis_mod["map_attn"]

    # Total attention on agents vs map
    valid_orig = agent_mask_orig
    orig_agent_total = orig_agent_attn[valid_orig].sum()
    mod_agent_total = mod_agent_attn[vis_mod["agent_data"]["agent_mask"]].sum()
    orig_map_total = orig_map_attn[vis_orig["map_data"]["map_mask"]].sum()
    mod_map_total = mod_map_attn[vis_mod["map_data"]["map_mask"]].sum()

    # Entropy of attention distribution (agent portion only)
    def entropy(w):
        w = w[w > 1e-8]
        if len(w) == 0:
            return 0.0
        w = w / w.sum()
        return -np.sum(w * np.log2(w))

    orig_entropy = entropy(orig_agent_attn[valid_orig])
    mod_entropy = entropy(mod_agent_attn[vis_mod["agent_data"]["agent_mask"]])

    return {
        "orig_agent_attn_total": float(orig_agent_total),
        "mod_agent_attn_total": float(mod_agent_total),
        "orig_map_attn_total": float(orig_map_total),
        "mod_map_attn_total": float(mod_map_total),
        "orig_agent_entropy": float(orig_entropy),
        "mod_agent_entropy": float(mod_entropy),
    }


def main():
    t0 = time.time()
    plt.rcParams.update(PAPER_RC)

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------
    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/last.ckpt"
    config_path = PROJECT_ROOT / "configs" / "mtr_lite.yaml"
    output_dir = PROJECT_ROOT / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------------
    # Load model on CPU
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Counterfactual Attention Analysis")
    print("=" * 70)
    print(f"\nLoading MTR-Lite model on CPU...")
    print(f"  Checkpoint: {checkpoint}")

    module = MTRLiteModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    model = module.model
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ---------------------------------------------------------------
    # Get validation scenes
    # ---------------------------------------------------------------
    with open(cfg["data"]["scene_list"]) as f:
        all_scenes = [l.strip() for l in f if l.strip()]
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * 0.15))
    val_scenes = [all_scenes[i] for i in indices[:n_val] if os.path.exists(all_scenes[i])]
    print(f"  Validation set: {len(val_scenes)} scenes")

    # ---------------------------------------------------------------
    # Step 1: Find a good intersection scene
    # ---------------------------------------------------------------
    print("\nSearching for intersection scene with good properties...")
    scene_path, result_orig, most_attended_idx = find_intersection_scene(
        val_scenes, model, max_scan=300)

    if scene_path is None:
        print("ERROR: Could not find a suitable intersection scene. Exiting.")
        sys.exit(1)

    vis_orig = extract_vis_data(result_orig)
    agent_data_orig = vis_orig["agent_data"]
    scene_data_orig = result_orig["scene_data"]

    # Get removed agent info
    removed_agent_bev_pos = agent_data_orig["agent_polylines"][most_attended_idx, -1, 0:2]
    removed_agent_obj_idx = agent_data_orig["agent_ids"][most_attended_idx]
    removed_agent_attn = vis_orig["agent_attn"][most_attended_idx]
    removed_agent_type = get_agent_type(agent_data_orig["agent_polylines"], most_attended_idx)

    print(f"\n  Most-attended non-ego vehicle:")
    print(f"    Agent slot: {most_attended_idx}")
    print(f"    Object index in scene: {removed_agent_obj_idx}")
    print(f"    BEV position: ({removed_agent_bev_pos[0]:.1f}, {removed_agent_bev_pos[1]:.1f}) m")
    print(f"    Attention weight: {removed_agent_attn:.4f}")
    print(f"    Type: {removed_agent_type}")

    # ---------------------------------------------------------------
    # Step 2: Remove the agent and re-run
    # ---------------------------------------------------------------
    print(f"\nRemoving agent (obj_idx={removed_agent_obj_idx}) from scene...")
    editor = SceneEditor(scene_data_orig)
    editor.remove_agent(removed_agent_obj_idx)
    modified_scene = editor.get_scene()

    print(f"  Original agents: {len(scene_data_orig['objects'])}")
    print(f"  Modified agents: {len(modified_scene['objects'])}")

    print("  Running inference on modified scene...")
    result_mod = extract_scene_attention_from_dict(model, modified_scene, device="cpu")

    if result_mod["attention_maps"] is None:
        print("ERROR: No attention maps from modified scene. Exiting.")
        sys.exit(1)

    vis_mod = extract_vis_data(result_mod)
    print("  Modified scene inference complete.")

    # ---------------------------------------------------------------
    # Step 3: Compute statistics
    # ---------------------------------------------------------------
    stats = compute_attention_delta_stats(vis_orig, vis_mod, agent_data_orig["agent_mask"])
    print(f"\n  Attention redistribution statistics:")
    print(f"    Original agent attn total:  {stats['orig_agent_attn_total']:.4f}")
    print(f"    Modified agent attn total:  {stats['mod_agent_attn_total']:.4f}")
    print(f"    Original map attn total:    {stats['orig_map_attn_total']:.4f}")
    print(f"    Modified map attn total:    {stats['mod_map_attn_total']:.4f}")
    print(f"    Original agent entropy:     {stats['orig_agent_entropy']:.3f} bits")
    print(f"    Modified agent entropy:     {stats['mod_agent_entropy']:.3f} bits")

    # ---------------------------------------------------------------
    # Step 4: Generate figure
    # ---------------------------------------------------------------
    import matplotlib.gridspec as gridspec

    n_agents_orig = int(agent_data_orig["agent_mask"].sum())
    n_lanes = int(vis_orig["map_data"]["map_mask"].sum())

    print(f"\nRendering 2-panel counterfactual figure...")
    fig = plt.figure(figsize=(9.5, 4.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.12)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_mod = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])

    # --- Panel (a): Original scene ---
    _, im_orig = render_counterfactual_bev(
        agent_data=vis_orig["agent_data"],
        map_data=vis_orig["map_data"],
        agent_attn=vis_orig["agent_attn"],
        map_attn=vis_orig["map_attn"],
        ego_history=vis_orig["ego_history"],
        target_future=vis_orig["target_future"],
        target_valid=vis_orig["target_valid"],
        pred_trajs=vis_orig["pred_trajs"],
        pred_scores=vis_orig["pred_scores"],
        title=f"Original Scene ({n_agents_orig} agents)",
        ax=ax_orig,
        show_colorbar=False,
        panel_label="(a)",
        highlight_agent_idx=most_attended_idx,
    )

    # --- Panel (b): Modified scene (agent removed) ---
    _, im_mod = render_counterfactual_bev(
        agent_data=vis_mod["agent_data"],
        map_data=vis_mod["map_data"],
        agent_attn=vis_mod["agent_attn"],
        map_attn=vis_mod["map_attn"],
        ego_history=vis_mod["ego_history"],
        target_future=vis_mod["target_future"],
        target_valid=vis_mod["target_valid"],
        pred_trajs=vis_mod["pred_trajs"],
        pred_scores=vis_mod["pred_scores"],
        title=f"Vehicle Removed ({n_agents_orig - 1} agents)",
        ax=ax_mod,
        show_colorbar=False,
        panel_label="(b)",
        removed_agent_pos=removed_agent_bev_pos,
        removed_agent_label="Removed",
    )

    # Remove y-label on right panel
    ax_mod.set_ylabel("")
    ax_mod.tick_params(labelleft=False)

    # Shared colorbar
    cbar = fig.colorbar(im_orig, cax=cbar_ax)
    cbar.set_label("Attention weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Add summary text annotation below
    delta_entropy = stats["mod_agent_entropy"] - stats["orig_agent_entropy"]
    delta_map = stats["mod_map_attn_total"] - stats["orig_map_attn_total"]
    direction_e = "+" if delta_entropy > 0 else ""
    direction_m = "+" if delta_map > 0 else ""
    dist_to_ego = np.linalg.norm(removed_agent_bev_pos)
    summary = (
        f"Removed vehicle: {dist_to_ego:.0f} m from ego    "
        r"$\mathbf{w}_{\mathrm{orig}}$"
        f" = {removed_agent_attn:.3f}    "
        r"$\Delta H_{\mathrm{agent}}$"
        f" = {direction_e}{delta_entropy:.2f} bits    "
        r"$\Delta$"
        f"map attn = {direction_m}{delta_map:.3f}"
    )
    fig.text(0.48, 0.01, summary, ha="center", fontsize=8.5,
             color="#333333",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5",
                       edgecolor="#BBBBBB", alpha=0.95))

    # Save
    out_path = output_dir / "fig_counterfactual_case_study.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DONE -- Counterfactual attention figure generated")
    print(f"  Output: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
