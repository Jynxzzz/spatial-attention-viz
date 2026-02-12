"""Generate publication-quality BEV spatial attention overlay figures for paper.

Produces:
  - 3 individual scene PDF figures (fig_bev_attention_scene1..3.pdf)
  - 1 composite 3-panel PDF figure (fig_spatial_attention_composite.pdf)

Uses the best MTR-Lite checkpoint (epoch 44, minADE@6=2.670) on CPU.
Scenes are selected for diversity: intersection, general driving, pedestrian/cyclist.

Usage:
    cd /home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper
    conda run -n yolov8 python scripts/generate_paper_bev_figures.py
"""

import os
import pickle
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

# Setup project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention

# ===========================================================================
# Publication style configuration
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

# Agent type styling
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


def get_agent_type(agent_polylines, agent_idx):
    """Get agent type from polyline features (one-hot at indices 12:17)."""
    type_oh = agent_polylines[agent_idx, -1, 12:17]
    if type_oh.sum() > 0:
        return TYPE_NAMES[int(np.argmax(type_oh))]
    return "unknown"


def render_paper_bev(
    agent_data, map_data, agent_attn, map_attn,
    ego_history, target_future, target_valid,
    pred_trajs, pred_scores,
    title="", bev_range=50.0, resolution=0.5,
    ax=None, show_colorbar=True, panel_label=None,
):
    """Render a publication-quality BEV with attention overlay and typed agents.

    Args:
        agent_data: dict from extract_all_agents
        map_data: dict from extract_map_polylines
        agent_attn: (A,) per-agent attention weights
        map_attn: (M,) per-map-polyline attention weights
        ego_history: (11, 2) ego history positions
        target_future: (80, 2) GT future
        target_valid: (80,) bool
        pred_trajs: (K, 80, 2) predicted trajectories
        pred_scores: (K,) predicted scores
        title: figure title string
        bev_range: spatial extent in meters
        resolution: grid resolution in meters/pixel
        ax: matplotlib axes (optional)
        show_colorbar: whether to show colorbar
        panel_label: e.g. "(a)" for panel labeling
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))
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

    # --- Background: all lanes in light gray ---
    for i in range(len(lane_pts)):
        if map_mask[i]:
            pts = lane_pts[i]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=COLOR_LANE_BG,
                    linewidth=0.6, alpha=0.5, zorder=1)

    # --- Attention-weighted lanes on top ---
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

    # --- Axis formatting ---
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Longitudinal (m)")
    ax.set_aspect("equal")

    # Fewer ticks for cleaner look
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    # Title
    if title:
        ax.set_title(title, fontweight="bold", pad=4)

    # Panel label
    if panel_label:
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.8))

    # Legend: only show agent types that are actually present in the scene
    present_types = set()
    for i in range(len(agent_positions)):
        if valid_agents[i]:
            present_types.add(get_agent_type(agent_data["agent_polylines"], i))

    legend_handles = []
    for tname in ["vehicle", "pedestrian", "cyclist"]:
        if tname in present_types:
            legend_handles.append(
                mpatches.Patch(color=TYPE_COLORS[tname], label=tname.capitalize()))
    # Get line handles from plotted data (History, Ground truth, Prediction)
    for handle, label in zip(*ax.get_legend_handles_labels()):
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.85, edgecolor="none", handlelength=1.5,
              handletextpad=0.4, borderpad=0.3)

    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=25)
        cbar.set_label("Attention weight", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    return fig


def classify_scene(scene, agent_data, map_data):
    """Classify a scene for diversity selection.

    Returns: (category_string, description_string)
    """
    objects = scene["objects"]
    n_valid = sum(1 for o in objects if o.get("valid", [False])[10] if len(o.get("valid", [])) > 10)

    types = set()
    type_counts = {}
    for i in range(32):
        if agent_data["agent_mask"][i]:
            atype = get_agent_type(agent_data["agent_polylines"], i)
            types.add(atype)
            type_counts[atype] = type_counts.get(atype, 0) + 1

    n_lanes = map_data["map_mask"].sum()
    has_ped = "pedestrian" in types
    has_cyc = "cyclist" in types
    n_agents = agent_data["agent_mask"].sum()

    type_str = ", ".join(f"{v} {k}s" for k, v in
                         sorted(type_counts.items(), key=lambda x: -x[1]))

    if has_ped:
        return "pedestrian", f"Scene with pedestrians ({type_str})"
    elif has_cyc:
        return "cyclist", f"Scene with cyclist ({type_str})"
    elif n_lanes > 45:
        return "intersection", f"Complex intersection ({int(n_lanes)} lanes, {type_str})"
    elif n_agents > 15:
        return "dense", f"Dense traffic ({type_str})"
    elif n_agents < 6:
        return "sparse", f"Sparse scene ({type_str})"
    else:
        return "general", f"General driving ({type_str})"


def _scene_has_good_trajectories(result):
    """Check that a scene has clearly visible history + GT + prediction trajectories.

    Requires:
    - At least 5 valid history frames with >= 5m spatial extent
    - At least 20 valid future frames with >= 8m spatial extent
    This ensures trajectories are visually prominent in the figure.
    """
    agent_data = result["agent_data"]

    # Check ego history has enough spatial extent to be visible
    ego_hist = agent_data["agent_polylines"][0, :, 0:2]  # (11, 2)
    ego_valid = agent_data["agent_valid"][0]  # (11,)
    valid_hist = ego_hist[ego_valid]
    if len(valid_hist) < 5:
        return False
    hist_range = np.linalg.norm(valid_hist.max(axis=0) - valid_hist.min(axis=0))
    if hist_range < 5.0:
        return False  # History too short to see clearly

    # Check GT future has enough extent to be clearly visible
    target_future = result["batch"]["target_future"][0, 0].cpu().numpy()
    target_valid = result["batch"]["target_future_valid"][0, 0].cpu().numpy().astype(bool)
    valid_future = target_future[target_valid]
    if len(valid_future) < 20:
        return False
    future_range = np.linalg.norm(valid_future.max(axis=0) - valid_future.min(axis=0))
    if future_range < 8.0:
        return False  # Future too short to see clearly

    return True


def find_diverse_scenes(val_scenes, model, desired_indices=None, max_scan=300):
    """Scan validation scenes and select 3 diverse ones with unique categories.

    Strategy:
    - We want exactly 3 scenes, each with a DIFFERENT category
    - Preferred triad: (intersection, pedestrian, general/dense/cyclist)
    - Every selected scene must have visible trajectories (history + GT + pred)
    - First try the requested indices, then scan further if needed

    Returns: list of (scene_path, result_dict, category, description, val_idx)
    """
    if desired_indices is None:
        desired_indices = [0, 10, 30]

    # Pool of all valid candidate results, keyed by category
    pool = {}  # cat -> list of (path, result, cat, desc, idx)

    def try_scene(idx):
        """Try loading a scene, return (cat, entry) or None."""
        if idx >= len(val_scenes):
            return None
        scene_path = val_scenes[idx]
        try:
            result = extract_scene_attention(
                model=model, scene_path=scene_path, device="cpu")
        except Exception as e:
            print(f"    Scene {idx}: extraction failed ({e})")
            return None

        if result["attention_maps"] is None:
            print(f"    Scene {idx}: no attention maps")
            return None

        if not _scene_has_good_trajectories(result):
            cat, desc = classify_scene(
                result["scene_data"], result["agent_data"], result["map_data"])
            print(f"    Scene {idx}: {cat} -- poor trajectory visibility, skipping")
            return None

        cat, desc = classify_scene(
            result["scene_data"], result["agent_data"], result["map_data"])
        entry = (scene_path, result, cat, desc, idx)
        return cat, entry

    # Priority order: desired indices first, then sequential scan
    scan_order = list(desired_indices) + [
        i for i in range(max_scan) if i not in desired_indices
    ]

    for idx in scan_order:
        if idx >= len(val_scenes):
            continue

        print(f"  Trying scene index {idx}: {os.path.basename(val_scenes[idx])}")
        out = try_scene(idx)
        if out is None:
            continue

        cat, entry = out
        print(f"    Category: {cat} -- {entry[3]}")

        if cat not in pool:
            pool[cat] = entry

        # Check if we can assemble 3 distinct categories
        if len(pool) >= 3:
            break

    # Assemble final selection: prefer (intersection, pedestrian, X)
    # where X is general > cyclist > dense > sparse > anything else
    priority_triads = [
        ("intersection", "pedestrian", "general"),
        ("intersection", "pedestrian", "cyclist"),
        ("intersection", "pedestrian", "dense"),
        ("intersection", "general", "cyclist"),
        ("pedestrian", "general", "dense"),
    ]

    selected = None
    for triad in priority_triads:
        if all(t in pool for t in triad):
            selected = [pool[t] for t in triad]
            print(f"  Selected triad: {triad}")
            break

    # Fallback: just pick the first 3 distinct categories in pool
    if selected is None:
        cats = list(pool.keys())[:3]
        selected = [pool[c] for c in cats]
        print(f"  Fallback triad: {cats}")

    # If still fewer than 3, pad with duplicates (shouldn't happen with 300 scan)
    if len(selected) < 3:
        print(f"  WARNING: only found {len(selected)} distinct categories")
        # Add remaining from scan
        seen_idx = set(e[4] for e in selected)
        for idx in scan_order:
            if idx >= len(val_scenes) or idx in seen_idx:
                continue
            out = try_scene(idx)
            if out is not None:
                selected.append(out[1])
                seen_idx.add(idx)
            if len(selected) >= 3:
                break

    return selected[:3]


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
    target_future = result["batch"]["target_future"][0, 0].cpu().numpy()  # (80, 2)
    target_valid = result["batch"]["target_future_valid"][0, 0].cpu().numpy().astype(bool)
    ego_history = agent_data["agent_polylines"][0, :, 0:2]  # (11, 2)

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


def main():
    t0 = time.time()

    # Paths
    project_root = Path(__file__).resolve().parent.parent
    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=44-val/minADE@6=2.670.ckpt"
    config_path = project_root / "configs" / "mtr_lite.yaml"
    output_dir = project_root / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Loading MTR-Lite model on CPU...")
    print(f"  Checkpoint: {checkpoint}")
    module = MTRLiteModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    model = module.model
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ---------------------------------------------------------------
    # Get validation scenes (same deterministic split as training)
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
    # Find 3 diverse scenes
    # ---------------------------------------------------------------
    print("\nSearching for diverse scenes...")
    scene_results = find_diverse_scenes(val_scenes, model, desired_indices=[0, 5, 10, 20, 30, 50])
    print(f"\nSelected {len(scene_results)} scenes:")
    for i, (path, _, cat, desc, idx) in enumerate(scene_results):
        print(f"  Scene {i+1}: idx={idx}, cat={cat}, {desc}")

    # ---------------------------------------------------------------
    # Apply publication matplotlib style
    # ---------------------------------------------------------------
    plt.rcParams.update(PAPER_RC)

    # ---------------------------------------------------------------
    # Generate individual figures
    # ---------------------------------------------------------------
    panel_labels = ["(a)", "(b)", "(c)"]
    vis_data_list = []

    for i, (path, result, cat, desc, idx) in enumerate(scene_results):
        print(f"\nRendering scene {i+1}/3: {desc}")
        t_scene = time.time()

        vis = extract_vis_data(result)
        vis_data_list.append((vis, cat, desc))

        # Count agents by type
        type_counts = {}
        for j in range(32):
            if vis["agent_data"]["agent_mask"][j]:
                atype = get_agent_type(vis["agent_data"]["agent_polylines"], j)
                type_counts[atype] = type_counts.get(atype, 0) + 1

        n_agents = sum(type_counts.values())
        n_lanes = int(vis["map_data"]["map_mask"].sum())

        # Make a clean title for the paper
        if cat == "intersection":
            title = f"Intersection ({n_agents} agents, {n_lanes} lanes)"
        elif cat == "pedestrian":
            title = f"Mixed traffic w/ pedestrians ({n_agents} agents)"
        elif cat == "cyclist":
            title = f"Mixed traffic w/ cyclists ({n_agents} agents)"
        elif cat == "dense":
            title = f"Dense traffic ({n_agents} agents)"
        elif cat == "sparse":
            title = f"Low-density scenario ({n_agents} agents)"
        else:
            title = f"Urban driving ({n_agents} agents)"

        # Individual figure
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        render_paper_bev(
            agent_data=vis["agent_data"],
            map_data=vis["map_data"],
            agent_attn=vis["agent_attn"],
            map_attn=vis["map_attn"],
            ego_history=vis["ego_history"],
            target_future=vis["target_future"],
            target_valid=vis["target_valid"],
            pred_trajs=vis["pred_trajs"],
            pred_scores=vis["pred_scores"],
            title=title,
            ax=ax,
            show_colorbar=True,
            panel_label=panel_labels[i],
        )

        fname = output_dir / f"fig_bev_attention_scene{i+1}.pdf"
        fig.savefig(fname, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}  ({time.time() - t_scene:.1f}s)")

        # Print attention stats
        valid_agent_attn = vis["agent_attn"][vis["agent_data"]["agent_mask"]]
        valid_map_attn = vis["map_attn"][vis["map_data"]["map_mask"]]
        best_mode = vis["pred_scores"].argmax()
        print(f"  Agent attn: mean={valid_agent_attn.mean():.4f}, "
              f"max={valid_agent_attn.max():.4f}")
        print(f"  Map attn:   mean={valid_map_attn.mean():.4f}, "
              f"max={valid_map_attn.max():.4f}")
        print(f"  Best mode: {best_mode}, score={vis['pred_scores'][best_mode]:.3f}")

    # ---------------------------------------------------------------
    # Generate composite 3-panel figure with shared colorbar
    # ---------------------------------------------------------------
    print("\nRendering composite 3-panel figure...")
    t_comp = time.time()

    import matplotlib.gridspec as gridspec

    n_panels = len(vis_data_list)
    # Use gridspec: 3 equal BEV panels + narrow colorbar panel
    fig_comp = plt.figure(figsize=(4.2 * n_panels + 0.6, 4.6))
    gs = gridspec.GridSpec(
        1, n_panels + 1,
        width_ratios=[1] * n_panels + [0.04],
        wspace=0.08,
    )
    axes = [fig_comp.add_subplot(gs[0, i]) for i in range(n_panels)]
    cbar_ax = fig_comp.add_subplot(gs[0, n_panels])

    last_im = None  # store the imshow handle for the colorbar

    for i, (vis, cat, desc) in enumerate(vis_data_list):
        type_counts = {}
        for j in range(32):
            if vis["agent_data"]["agent_mask"][j]:
                atype = get_agent_type(vis["agent_data"]["agent_polylines"], j)
                type_counts[atype] = type_counts.get(atype, 0) + 1
        n_agents = sum(type_counts.values())
        n_lanes = int(vis["map_data"]["map_mask"].sum())

        if cat == "intersection":
            title = f"Intersection ({n_agents} agents)"
        elif cat == "pedestrian":
            title = f"Mixed traffic w/ pedestrians"
        elif cat == "cyclist":
            title = f"Mixed traffic w/ cyclists"
        elif cat == "dense":
            title = f"Dense traffic ({n_agents} agents)"
        elif cat == "sparse":
            title = f"Low-density ({n_agents} agents)"
        else:
            title = f"Urban driving ({n_agents} agents)"

        render_paper_bev(
            agent_data=vis["agent_data"],
            map_data=vis["map_data"],
            agent_attn=vis["agent_attn"],
            map_attn=vis["map_attn"],
            ego_history=vis["ego_history"],
            target_future=vis["target_future"],
            target_valid=vis["target_valid"],
            pred_trajs=vis["pred_trajs"],
            pred_scores=vis["pred_scores"],
            title=title,
            ax=axes[i],
            show_colorbar=False,  # We use a shared colorbar
            panel_label=panel_labels[i],
        )

        # Remove y-label on middle/right panels for cleaner look
        if i > 0:
            axes[i].set_ylabel("")
            axes[i].tick_params(labelleft=False)

        # Grab the imshow artist for the colorbar
        for child in axes[i].get_children():
            if hasattr(child, "get_cmap") and hasattr(child, "get_clim"):
                last_im = child

    # Add shared colorbar to dedicated axis
    if last_im is not None:
        cbar = fig_comp.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Attention weight", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    comp_path = output_dir / "fig_spatial_attention_composite.pdf"
    fig_comp.savefig(comp_path, format="pdf", bbox_inches="tight")
    plt.close(fig_comp)
    print(f"  Saved: {comp_path}  ({time.time() - t_comp:.1f}s)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("DONE -- Publication BEV attention figures generated")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output directory: {output_dir}")
    print("  Files:")
    for f in sorted(output_dir.glob("fig_*.pdf")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
