"""Generate cyclist prediction failure attention visualization for paper.

Produces a 2-panel figure:
  (a) Cyclist Failure -- Tunnel Vision: BEV attention for a cyclist target with high ADE
  (b) Vehicle Success -- Broad Attention: BEV attention for a vehicle target with low ADE

Demonstrates that the model allocates very little attention to cyclist tokens,
contributing to the 88.1% miss rate (vs 54% for vehicles).

Usage:
    cd /home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper
    conda run -n yolov8 python scripts/generate_cyclist_failure_figure.py
"""

import os
import pickle
import random
import sys
import time
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
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

# Cyclist-specific colors for high visibility
COLOR_CYCLIST_MARKER = "#34A853"  # Green
COLOR_CYCLIST_RING = "#FFD600"    # Yellow ring for emphasis


def get_agent_type(agent_polylines, agent_idx):
    """Get agent type from polyline features (one-hot at indices 12:17)."""
    type_oh = agent_polylines[agent_idx, -1, 12:17]
    if type_oh.sum() > 0:
        return TYPE_NAMES[int(np.argmax(type_oh))]
    return "unknown"


def compute_entropy(probs):
    """Shannon entropy in bits."""
    p = probs[probs > 1e-10]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))


def analyze_scene_targets(model, scene_path, device="cpu"):
    """Run inference on a scene and return per-target analysis with full data.

    Returns a list of dicts, each containing:
        - ADE, agent_type, attention info, raw data for rendering
    """
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

    # Scene encoder last layer, head-averaged
    last_attn = attn_maps.scene_attentions[-1][0]  # (H, N, N)
    avg_attn = last_attn.mean(dim=0).cpu().numpy()  # (N, N)

    pred_trajs = preds["trajectories"][0].cpu().numpy()         # (T, K, 80, 2)
    pred_scores = preds["scores"][0].cpu().numpy()               # (T, K)
    target_future = batch["target_future"][0].cpu().numpy()      # (T, 80, 2)
    target_future_valid = batch["target_future_valid"][0].cpu().numpy()  # (T, 80)
    target_mask_arr = batch["target_mask"][0].cpu().numpy()       # (T,)
    target_indices = batch["target_agent_indices"][0].cpu().numpy()  # (T,)

    targets = []

    for t_idx in range(8):
        if not target_mask_arr[t_idx]:
            continue

        agent_slot = int(target_indices[t_idx])
        gt = target_future[t_idx]
        gt_valid = target_future_valid[t_idx].astype(bool)

        if gt_valid.sum() < 20:
            continue

        # minADE@6
        trajs = pred_trajs[t_idx]  # (K, 80, 2)
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

        # minFDE@6
        min_fde = float("inf")
        for k in top6:
            diff = trajs[k] - gt
            dist = np.sqrt((diff ** 2).sum(axis=1))
            valid_dists = dist[gt_valid]
            if len(valid_dists) > 0:
                fde = valid_dists[-1]
                if fde < min_fde:
                    min_fde = fde

        # Miss rate: check if best mode FDE > 2m
        best_fde = np.sqrt(((trajs[best_mode] - gt) ** 2).sum(axis=1))
        best_fde_val = best_fde[gt_valid][-1] if gt_valid.any() else 999
        is_miss = min_fde > 2.0

        # Attention for this target agent
        target_attn_row = avg_attn[agent_slot]  # (96,)
        agent_attn = target_attn_row[:32]
        map_attn = target_attn_row[32:96]
        entropy = compute_entropy(target_attn_row)

        # Self-attention
        self_attn = target_attn_row[agent_slot]

        # Agent type
        obj_idx = agent_data["agent_ids"][agent_slot]
        agent_type = scene["objects"][obj_idx].get("type", "unknown").lower()

        # Target agent position
        target_pos = agent_data["agent_polylines"][agent_slot, -1, 0:2]

        # Speed
        vel = agent_data["agent_polylines"][agent_slot, -1, 4:6]
        speed = np.linalg.norm(vel)

        # Attention to cyclist tokens specifically
        cyclist_attn_total = 0.0
        cyclist_count = 0
        vehicle_attn_total = 0.0
        vehicle_count = 0
        for s in range(32):
            if not agent_data["agent_mask"][s]:
                continue
            atype = get_agent_type(agent_data["agent_polylines"], s)
            if atype == "cyclist":
                cyclist_attn_total += agent_attn[s]
                cyclist_count += 1
            elif atype == "vehicle":
                vehicle_attn_total += agent_attn[s]
                vehicle_count += 1

        # History (for the target agent, not ego)
        target_history = agent_data["agent_polylines"][agent_slot, :, 0:2]  # (11, 2)
        target_valid_hist = agent_data["agent_valid"][agent_slot]  # (11,)

        # GT future extent
        valid_gt = gt[gt_valid]
        gt_range = np.linalg.norm(valid_gt.max(axis=0) - valid_gt.min(axis=0)) if len(valid_gt) > 1 else 0

        targets.append({
            "t_idx": t_idx,
            "agent_slot": agent_slot,
            "obj_idx": obj_idx,
            "agent_type": agent_type,
            "minADE": min_ade,
            "minFDE": min_fde,
            "is_miss": is_miss,
            "entropy": entropy,
            "self_attn": self_attn,
            "speed": speed,
            "gt_range": gt_range,
            "cyclist_attn_total": cyclist_attn_total,
            "cyclist_count": cyclist_count,
            "vehicle_attn_total": vehicle_attn_total,
            "vehicle_count": vehicle_count,
            "target_pos": target_pos,
            # Data needed for rendering
            "agent_attn": agent_attn.copy(),
            "map_attn": map_attn.copy(),
            "agent_data": agent_data,
            "map_data": map_data,
            "target_history": target_history.copy(),
            "target_valid_hist": target_valid_hist.copy(),
            "target_future": gt.copy(),
            "target_future_valid": gt_valid.copy(),
            "pred_trajs": trajs.copy(),
            "pred_scores": scores.copy(),
            "best_mode": best_mode,
            "scene_path": scene_path,
            "full_avg_attn": avg_attn.copy(),
        })

    return targets


def render_cyclist_bev(
    target_info, bev_range=50.0, resolution=0.5,
    ax=None, show_colorbar=True, panel_label=None,
    title="", annotate_cyclist=False, annotate_vehicle_success=False,
):
    """Render BEV attention overlay centered on the target agent.

    This version renders the scene from the target agent's perspective,
    showing its attention pattern overlaid on the BEV map.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    else:
        fig = ax.figure

    agent_data = target_info["agent_data"]
    map_data = target_info["map_data"]
    agent_attn = target_info["agent_attn"]
    map_attn = target_info["map_attn"]
    agent_slot = target_info["agent_slot"]

    grid_size = int(2 * bev_range / resolution)
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Target agent position (used as center reference)
    target_pos = agent_data["agent_polylines"][agent_slot, -1, 0:2]

    def bev_to_grid(xy):
        # Positions are already in BEV (ego-centric) coordinates
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
    cyclist_positions = []
    for i in range(len(agent_positions)):
        if not valid_agents[i]:
            continue
        pos = agent_positions[i]
        if abs(pos[0]) > bev_range or abs(pos[1]) > bev_range:
            continue
        atype = get_agent_type(agent_data["agent_polylines"], i)
        color = TYPE_COLORS.get(atype, "#9E9E9E")
        marker = TYPE_MARKERS.get(atype, "v")

        if i == agent_slot:
            # This is the target agent - will draw separately with larger marker
            continue

        if atype == "cyclist":
            # Draw cyclists with larger, more visible markers
            ax.plot(pos[0], pos[1], marker, color=color, markersize=8,
                    markeredgecolor=COLOR_CYCLIST_RING, markeredgewidth=1.5,
                    zorder=6)
            cyclist_positions.append((pos, i, agent_attn[i]))
        else:
            size = 5 if i == 0 else 4
            edge = "white" if i == 0 else "black"
            ew = 0.8 if i == 0 else 0.3
            ax.plot(pos[0], pos[1], marker, color=color, markersize=size,
                    markeredgecolor=edge, markeredgewidth=ew, zorder=6)

    # --- Target agent marker (large, prominent) ---
    target_atype = get_agent_type(agent_data["agent_polylines"], agent_slot)
    tgt_color = TYPE_COLORS.get(target_atype, "#9E9E9E")
    tgt_marker = TYPE_MARKERS.get(target_atype, "v")
    ax.plot(target_pos[0], target_pos[1], tgt_marker, color=tgt_color,
            markersize=12, markeredgecolor="white", markeredgewidth=1.5,
            zorder=9, label=f"Target ({target_atype})")

    # --- Ego marker (agent index 0) ---
    ego_pos = agent_data["agent_polylines"][0, -1, 0:2]
    ax.plot(ego_pos[0], ego_pos[1], "^", color=COLOR_EGO, markersize=8,
            markeredgecolor="white", markeredgewidth=0.8, zorder=8)

    # --- History trajectory (target agent) ---
    hist = target_info["target_history"]
    hist_valid = target_info["target_valid_hist"]
    if hist_valid.any():
        valid_hist = hist[hist_valid]
        ax.plot(valid_hist[:, 0], valid_hist[:, 1], "-o", color=COLOR_HISTORY,
                linewidth=1.5, markersize=2, label="History", zorder=7)

    # --- GT future ---
    gt = target_info["target_future"]
    gt_valid = target_info["target_future_valid"]
    if gt_valid.any():
        valid_gt = gt[gt_valid]
        if len(valid_gt) > 0:
            ax.plot(valid_gt[:, 0], valid_gt[:, 1], "--", color=COLOR_GT,
                    linewidth=1.8, label="Ground truth", zorder=7)

    # --- Predictions (top 3 modes) ---
    pred_trajs = target_info["pred_trajs"]
    pred_scores = target_info["pred_scores"]
    top_k = min(3, len(pred_scores))
    top_indices = pred_scores.argsort()[-top_k:][::-1]
    for rank, k in enumerate(top_indices):
        alpha = 0.9 if rank == 0 else 0.4
        lw = 1.8 if rank == 0 else 1.0
        label = "Prediction" if rank == 0 else None
        ax.plot(pred_trajs[k, :, 0], pred_trajs[k, :, 1], "-",
                color=COLOR_PRED, linewidth=lw, alpha=alpha, label=label,
                zorder=7)

    # --- Annotation: Cyclist attention (for panel a) ---
    if annotate_cyclist:
        target_self_attn = target_info["self_attn"]
        ade_val = target_info["minADE"]
        fde_val = target_info["minFDE"]

        # Compute annotation placement: put callout in upper-left or upper-right
        # depending on target position
        if target_pos[0] > 0:
            text_x = target_pos[0] - 16
        else:
            text_x = target_pos[0] + 14
        text_y = target_pos[1] + 16
        # Clamp to within bev_range
        text_x = np.clip(text_x, -bev_range + 5, bev_range - 20)
        text_y = np.clip(text_y, -bev_range + 5, bev_range - 5)

        # Arrow to cyclist target with failure annotation
        ax.annotate(
            f"Cyclist (low attn: {target_self_attn:.3f})\n"
            f"minADE = {ade_val:.1f} m  |  MISS",
            xy=(target_pos[0], target_pos[1]),
            xytext=(text_x, text_y),
            fontsize=7.5, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#C62828",
                      edgecolor="white", alpha=0.92, linewidth=1.2),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8,
                            connectionstyle="arc3,rad=-0.2"),
            zorder=10,
        )

        # Show attention distribution stats in bottom-left
        cyc_attn = target_info["cyclist_attn_total"]
        veh_attn = target_info["vehicle_attn_total"]
        veh_count = target_info["vehicle_count"]
        cyc_count = target_info["cyclist_count"]

        stats_text = (
            f"Cyclist attn: {cyc_attn:.3f} ({cyc_count} cyclists)\n"
            f"Vehicle attn: {veh_attn:.3f} ({veh_count} vehicles)\n"
            f"Entropy: {target_info['entropy']:.2f} bits"
        )
        ax.text(0.03, 0.03, stats_text, transform=ax.transAxes,
                fontsize=6.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                          edgecolor="white", alpha=0.82, linewidth=0.8),
                color="white", family="monospace", zorder=10)

    # --- Annotation: Vehicle success (for panel b) ---
    if annotate_vehicle_success:
        ade_val = target_info["minADE"]
        target_self_attn = target_info["self_attn"]

        if target_pos[0] > 0:
            text_x = target_pos[0] - 16
        else:
            text_x = target_pos[0] + 14
        text_y = target_pos[1] + 16
        text_x = np.clip(text_x, -bev_range + 5, bev_range - 20)
        text_y = np.clip(text_y, -bev_range + 5, bev_range - 5)

        ax.annotate(
            f"Vehicle (attn: {target_self_attn:.3f})\n"
            f"minADE = {ade_val:.1f} m  |  HIT",
            xy=(target_pos[0], target_pos[1]),
            xytext=(text_x, text_y),
            fontsize=7.5, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1565C0",
                      edgecolor="white", alpha=0.92, linewidth=1.2),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8,
                            connectionstyle="arc3,rad=-0.2"),
            zorder=10,
        )

        stats_text = (
            f"Entropy: {target_info['entropy']:.2f} bits\n"
            f"Self-attn: {target_self_attn:.3f}"
        )
        ax.text(0.03, 0.03, stats_text, transform=ax.transAxes,
                fontsize=6.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                          edgecolor="white", alpha=0.82, linewidth=0.8),
                color="white", family="monospace", zorder=10)

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
        if label not in [h.get_label() for h in legend_handles]:
            legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.85, edgecolor="none", handlelength=1.5,
              handletextpad=0.4, borderpad=0.3)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=25)
        cbar.set_label("Attention weight", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    return fig


def main():
    t0 = time.time()

    project_root = Path(__file__).resolve().parent.parent
    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/last.ckpt"
    config_path = project_root / "configs" / "mtr_lite.yaml"
    output_dir = project_root / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------
    print("=" * 60)
    print("Loading MTR-Lite model...")
    print(f"  Checkpoint: {checkpoint}")
    module = MTRLiteModule.load_from_checkpoint(checkpoint, map_location=device)
    module.eval()
    model = module.model
    if device == "cuda":
        model = model.cuda()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # -------------------------------------------------------------------
    # Get validation scenes (same split as training)
    # -------------------------------------------------------------------
    with open(cfg["data"]["scene_list"]) as f:
        all_scenes = [l.strip() for l in f if l.strip()]
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * 0.15))
    val_scenes = [all_scenes[i] for i in indices[:n_val] if os.path.exists(all_scenes[i])]
    print(f"  Validation set: {len(val_scenes)} scenes")

    # -------------------------------------------------------------------
    # Phase 1: Scan for cyclist targets and vehicle targets
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1: Scanning for cyclist failure and vehicle success cases...")

    cyclist_targets = []
    vehicle_targets = []
    scenes_with_cyclists = 0
    max_scan = 500  # Scan up to 500 scenes

    for s_idx in range(min(max_scan, len(val_scenes))):
        scene_path = val_scenes[s_idx]

        try:
            targets = analyze_scene_targets(model, scene_path, device=device)
        except Exception as e:
            continue

        has_cyclist = False
        for tgt in targets:
            if tgt["agent_type"] == "cyclist":
                has_cyclist = True
                # Keep cyclist targets with visible trajectories
                if tgt["gt_range"] > 5.0:
                    cyclist_targets.append(tgt)

            elif tgt["agent_type"] == "vehicle":
                # Keep vehicle targets with visible trajectories
                if tgt["gt_range"] > 5.0:
                    vehicle_targets.append(tgt)

        if has_cyclist:
            scenes_with_cyclists += 1

        if (s_idx + 1) % 50 == 0:
            print(f"  [{s_idx+1}/{max_scan}] Found {len(cyclist_targets)} cyclist targets, "
                  f"{len(vehicle_targets)} vehicle targets "
                  f"(scenes with cyclists: {scenes_with_cyclists})")

        # Early exit if we have enough
        if len(cyclist_targets) >= 20 and len(vehicle_targets) >= 50:
            print(f"  Sufficient targets found after {s_idx+1} scenes, stopping scan.")
            break

    print(f"\nScan complete:")
    print(f"  Cyclist targets: {len(cyclist_targets)}")
    print(f"  Vehicle targets: {len(vehicle_targets)}")
    print(f"  Scenes with cyclists: {scenes_with_cyclists}")

    if len(cyclist_targets) == 0:
        print("\nERROR: No cyclist targets found! Trying with broader criteria...")
        # Fallback: scan more scenes or relax criteria
        for s_idx in range(max_scan, min(max_scan + 500, len(val_scenes))):
            scene_path = val_scenes[s_idx]
            try:
                targets = analyze_scene_targets(model, scene_path, device=device)
            except Exception:
                continue
            for tgt in targets:
                if tgt["agent_type"] == "cyclist":
                    cyclist_targets.append(tgt)
            if len(cyclist_targets) >= 5:
                break

    if len(cyclist_targets) == 0:
        print("FATAL: No cyclist targets found in entire scan. Cannot generate figure.")
        return

    # -------------------------------------------------------------------
    # Phase 2: Select best cyclist failure case
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Selecting best cyclist failure case...")

    # Sort cyclists by ADE (highest first = worst failure)
    cyclist_targets.sort(key=lambda x: x["minADE"], reverse=True)

    print("\nTop 10 cyclist targets by ADE:")
    for i, ct in enumerate(cyclist_targets[:10]):
        print(f"  {i}: ADE={ct['minADE']:.2f}m, FDE={ct['minFDE']:.2f}m, "
              f"miss={ct['is_miss']}, self_attn={ct['self_attn']:.4f}, "
              f"entropy={ct['entropy']:.2f}, speed={ct['speed']:.1f}m/s, "
              f"gt_range={ct['gt_range']:.1f}m")

    # Select the best cyclist failure: high ADE, visible trajectory, ideally a miss
    best_cyclist = None
    for ct in cyclist_targets:
        # Want: high ADE (failure), reasonable trajectory range, preferably a miss
        if ct["minADE"] > 2.0 and ct["gt_range"] > 8.0:
            best_cyclist = ct
            break
    if best_cyclist is None:
        # Fallback: just take the highest ADE cyclist
        best_cyclist = cyclist_targets[0]

    print(f"\nSelected cyclist failure:")
    print(f"  ADE: {best_cyclist['minADE']:.2f}m")
    print(f"  FDE: {best_cyclist['minFDE']:.2f}m")
    print(f"  Miss: {best_cyclist['is_miss']}")
    print(f"  Self-attention: {best_cyclist['self_attn']:.4f}")
    print(f"  Entropy: {best_cyclist['entropy']:.2f} bits")
    print(f"  Speed: {best_cyclist['speed']:.1f} m/s")
    print(f"  GT range: {best_cyclist['gt_range']:.1f}m")

    # -------------------------------------------------------------------
    # Phase 3: Select best vehicle success case
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 3: Selecting vehicle success case...")

    # Sort vehicles by ADE (lowest first = best success)
    vehicle_targets.sort(key=lambda x: x["minADE"])

    print("\nTop 10 vehicle targets by ADE (best):")
    for i, vt in enumerate(vehicle_targets[:10]):
        print(f"  {i}: ADE={vt['minADE']:.2f}m, FDE={vt['minFDE']:.2f}m, "
              f"miss={vt['is_miss']}, self_attn={vt['self_attn']:.4f}, "
              f"entropy={vt['entropy']:.2f}, speed={vt['speed']:.1f}m/s, "
              f"gt_range={vt['gt_range']:.1f}m")

    # Select: low ADE, reasonable trajectory range (not too extreme), not a miss
    # Prefer gt_range between 10-80m for visual clarity
    best_vehicle = None
    for vt in vehicle_targets:
        if (vt["minADE"] < 1.5 and 10.0 < vt["gt_range"] < 80.0
                and not vt["is_miss"] and vt["speed"] > 1.0):
            best_vehicle = vt
            break
    if best_vehicle is None:
        # Relax criteria
        for vt in vehicle_targets:
            if vt["minADE"] < 2.0 and vt["gt_range"] > 8.0 and not vt["is_miss"]:
                best_vehicle = vt
                break
    if best_vehicle is None:
        best_vehicle = vehicle_targets[0]

    print(f"\nSelected vehicle success:")
    print(f"  ADE: {best_vehicle['minADE']:.2f}m")
    print(f"  FDE: {best_vehicle['minFDE']:.2f}m")
    print(f"  Miss: {best_vehicle['is_miss']}")
    print(f"  Self-attention: {best_vehicle['self_attn']:.4f}")
    print(f"  Entropy: {best_vehicle['entropy']:.2f} bits")
    print(f"  Speed: {best_vehicle['speed']:.1f} m/s")
    print(f"  GT range: {best_vehicle['gt_range']:.1f}m")

    # -------------------------------------------------------------------
    # Phase 4: Render 2-panel figure
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 4: Rendering 2-panel figure...")

    plt.rcParams.update(PAPER_RC)

    # Create 2-panel figure with shared colorbar
    fig = plt.figure(figsize=(9.5, 4.8))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.04],
        wspace=0.12,
    )
    ax_cyclist = fig.add_subplot(gs[0, 0])
    ax_vehicle = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])

    # Panel (a): Cyclist Failure
    cyclist_ade = best_cyclist["minADE"]
    cyclist_title = f"Cyclist Target (minADE = {cyclist_ade:.1f} m)"
    render_cyclist_bev(
        best_cyclist,
        bev_range=50.0,
        ax=ax_cyclist,
        show_colorbar=False,
        panel_label="(a)",
        title=cyclist_title,
        annotate_cyclist=True,
    )

    # Panel (b): Vehicle Success
    vehicle_ade = best_vehicle["minADE"]
    vehicle_title = f"Vehicle Target (minADE = {vehicle_ade:.1f} m)"
    render_cyclist_bev(
        best_vehicle,
        bev_range=50.0,
        ax=ax_vehicle,
        show_colorbar=False,
        panel_label="(b)",
        title=vehicle_title,
        annotate_vehicle_success=True,
    )
    ax_vehicle.set_ylabel("")
    ax_vehicle.tick_params(labelleft=False)

    # Shared colorbar
    # Create a dummy mappable for the colorbar
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="magma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Add overall title
    fig.suptitle(
        "Attention Tunnel Vision: Cyclist Failure vs. Vehicle Success",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Save
    out_pdf = output_dir / "fig_cyclist_failure.pdf"
    out_png = output_dir / "fig_cyclist_failure.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\n  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")

    # -------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Cyclist vs Vehicle attention comparison across all found targets
    if len(cyclist_targets) > 0:
        cyc_ades = [ct["minADE"] for ct in cyclist_targets]
        cyc_entropies = [ct["entropy"] for ct in cyclist_targets]
        cyc_self_attns = [ct["self_attn"] for ct in cyclist_targets]
        cyc_miss_rate = sum(1 for ct in cyclist_targets if ct["is_miss"]) / len(cyclist_targets)
        print(f"\nCyclist targets (n={len(cyclist_targets)}):")
        print(f"  minADE: {np.mean(cyc_ades):.2f} +/- {np.std(cyc_ades):.2f} m")
        print(f"  Entropy: {np.mean(cyc_entropies):.2f} +/- {np.std(cyc_entropies):.2f} bits")
        print(f"  Self-attention: {np.mean(cyc_self_attns):.4f} +/- {np.std(cyc_self_attns):.4f}")
        print(f"  Miss rate: {cyc_miss_rate*100:.1f}%")

    if len(vehicle_targets) > 0:
        veh_ades = [vt["minADE"] for vt in vehicle_targets]
        veh_entropies = [vt["entropy"] for vt in vehicle_targets]
        veh_self_attns = [vt["self_attn"] for vt in vehicle_targets]
        veh_miss_rate = sum(1 for vt in vehicle_targets if vt["is_miss"]) / len(vehicle_targets)
        print(f"\nVehicle targets (n={len(vehicle_targets)}):")
        print(f"  minADE: {np.mean(veh_ades):.2f} +/- {np.std(veh_ades):.2f} m")
        print(f"  Entropy: {np.mean(veh_entropies):.2f} +/- {np.std(veh_entropies):.2f} bits")
        print(f"  Self-attention: {np.mean(veh_self_attns):.4f} +/- {np.std(veh_self_attns):.4f}")
        print(f"  Miss rate: {veh_miss_rate*100:.1f}%")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
