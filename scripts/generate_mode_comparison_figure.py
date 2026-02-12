"""Generate multi-modal attention comparison figure for the paper.

Shows that different prediction modes (intention queries) attend to different
scene elements, proving the model has learned "intention disentanglement."

Each panel shows one NMS-selected mode's BEV attention heatmap alongside its
predicted trajectory. A common colormap scale is used across panels so the
comparison is fair.

Key improvements over naive approach:
  - Finds scenes where the ego is actively driving (long trajectories)
  - Scores scenes by pairwise endpoint distance (not angle)
  - Uses last decoder layer attention only for sharper mode differences
  - Auto-zooms BEV range to fit the action
  - Highlights per-mode attention differences with stronger lane coloring

Output: paper/figures/fig_mode_attention_comparison.pdf

Usage:
    cd /home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper
    python scripts/generate_mode_comparison_figure.py
"""

import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from training.lightning_module import MTRLiteModule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/last.ckpt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "mtr_lite.yaml"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

DEVICE = "cpu"

# Colors (matches existing paper figures)
TYPE_COLORS = {
    "vehicle": "#4285F4",
    "pedestrian": "#EA8600",
    "cyclist": "#34A853",
    "other": "#9E9E9E",
    "unknown": "#9E9E9E",
}
TYPE_NAMES = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]
COLOR_HISTORY = "#1565C0"
COLOR_GT = "#2E7D32"
COLOR_EGO = "#0D47A1"
COLOR_LANE_BG = "#BDBDBD"

# Per-mode trajectory colors (bright, distinguishable, colorblind-safe)
MODE_COLORS = [
    "#D32F2F",  # deep red
    "#1565C0",  # deep blue
    "#2E7D32",  # deep green
    "#F57C00",  # orange
    "#7B1FA2",  # purple
    "#00838F",  # teal
]

# ---------------------------------------------------------------------------
# Publication matplotlib style
# ---------------------------------------------------------------------------
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 12,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_agent_type(agent_polylines, agent_idx):
    """Get agent type from one-hot at feature indices 12:17."""
    type_oh = agent_polylines[agent_idx, -1, 12:17]
    if type_oh.sum() > 0:
        return TYPE_NAMES[int(np.argmax(type_oh))]
    return "unknown"


def load_model_and_config():
    """Load MTR-Lite from Lightning checkpoint."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    module = MTRLiteModule.load_from_checkpoint(
        CHECKPOINT_PATH,
        model_cfg=cfg["model"],
        training_cfg=cfg["training"],
        loss_cfg=cfg["loss"],
        map_location=DEVICE,
    )
    module.eval()
    module.to(DEVICE)
    model = module.model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded ({n_params:,} params)")
    return model, cfg


def get_val_scenes(cfg):
    """Deterministic validation split."""
    with open(cfg["data"]["scene_list"]) as f:
        all_scenes = [l.strip() for l in f if l.strip()]
    all_scenes = [p for p in all_scenes if os.path.exists(p)]
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * cfg["data"]["val_ratio"]))
    return [all_scenes[i] for i in indices[:n_val]]


# ---------------------------------------------------------------------------
# Scene extraction
# ---------------------------------------------------------------------------
def extract_scene_with_modes(model, scene_path, cfg, device="cpu"):
    """Run one scene through the model and return everything needed."""
    from data.agent_features import extract_agent_future, extract_all_agents
    from data.map_features import extract_map_polylines

    try:
        with open(scene_path, "rb") as f:
            scene = pickle.load(f)
    except Exception:
        return None

    objects = scene["objects"]
    av_idx = scene["av_idx"]
    anchor_frame = 10
    history_len = cfg["data"]["history_len"]
    future_len = cfg["data"]["future_len"]
    max_agents = cfg["data"]["max_agents"]
    max_map = cfg["data"]["max_map_polylines"]
    map_pts = cfg["data"]["map_points_per_lane"]

    ego_obj = objects[av_idx]
    if not ego_obj["valid"][anchor_frame]:
        return None

    ego_pos = (
        float(ego_obj["position"][anchor_frame]["x"]),
        float(ego_obj["position"][anchor_frame]["y"]),
    )
    heading_raw = ego_obj["heading"][anchor_frame]
    ego_heading = (float(heading_raw[0])
                   if isinstance(heading_raw, (list, tuple))
                   else float(heading_raw))

    agent_data = extract_all_agents(
        scene, anchor_frame, history_len, ego_pos, ego_heading,
        max_agents=max_agents,
        neighbor_distance=cfg["data"]["neighbor_distance"],
    )
    map_data = extract_map_polylines(
        scene, ego_pos, ego_heading,
        max_polylines=max_map, points_per_lane=map_pts,
    )

    max_targets = cfg["model"]["max_targets"]
    target_indices = np.full(max_targets, -1, dtype=np.int64)
    target_mask = np.zeros(max_targets, dtype=bool)
    target_future = np.zeros((max_targets, future_len, 2), dtype=np.float32)
    target_future_valid = np.zeros((max_targets, future_len), dtype=bool)

    target_candidates = [
        i for i in range(max_agents)
        if agent_data["agent_mask"][i] and agent_data["target_mask"][i]
    ]
    if not target_candidates:
        return None

    for t_idx, a_slot in enumerate(target_candidates[:max_targets]):
        obj_idx = agent_data["agent_ids"][a_slot]
        future, valid = extract_agent_future(
            objects[obj_idx], anchor_frame, future_len, ego_pos, ego_heading)
        target_future[t_idx] = future
        target_future_valid[t_idx] = valid
        target_indices[t_idx] = a_slot
        target_mask[t_idx] = True

    batch = {
        "agent_polylines": torch.from_numpy(
            agent_data["agent_polylines"]).unsqueeze(0).to(device),
        "agent_valid": torch.from_numpy(
            agent_data["agent_valid"]).unsqueeze(0).to(device),
        "agent_mask": torch.from_numpy(
            agent_data["agent_mask"]).unsqueeze(0).to(device),
        "map_polylines": torch.from_numpy(
            map_data["map_polylines"]).unsqueeze(0).to(device),
        "map_valid": torch.from_numpy(
            map_data["map_valid"]).unsqueeze(0).to(device),
        "map_mask": torch.from_numpy(
            map_data["map_mask"]).unsqueeze(0).to(device),
        "target_agent_indices": torch.from_numpy(
            target_indices).unsqueeze(0).to(device),
        "target_mask": torch.from_numpy(
            target_mask).unsqueeze(0).to(device),
        "target_future": torch.from_numpy(
            target_future).unsqueeze(0).to(device),
        "target_future_valid": torch.from_numpy(
            target_future_valid).unsqueeze(0).to(device),
    }

    with torch.no_grad():
        output = model(batch, capture_attention=True)

    attn_maps = output.get("attention_maps")
    if attn_maps is None:
        return None

    return {
        "output": output,
        "attn_maps": attn_maps,
        "agent_data": agent_data,
        "map_data": map_data,
        "batch": batch,
        "scene": scene,
    }


# ---------------------------------------------------------------------------
# Mode diversity scoring -- focuses on trajectory spread + length
# ---------------------------------------------------------------------------
def compute_mode_diversity(result):
    """Score how well a scene shows multi-modal behavior.

    We want scenes where at least 3 NMS modes reach distinct spatial regions,
    each at least ~8m from origin and ~8m from each other. The ideal scene
    is an intersection where modes branch into straight, left, and right.

    Returns (score, n_lanes, n_agents, max_pairwise_dist, mean_traj_len,
             n_distinct).
    """
    output = result["output"]
    trajs = output["trajectories"][0, 0].cpu().numpy()   # (6, 80, 2)
    scores = output["scores"][0, 0].cpu().numpy()          # (6,)

    valid = scores > -100
    if valid.sum() < 2:
        return 0.0, 0, 0, 0.0, 0.0, 0

    valid_trajs = trajs[valid]
    endpoints = valid_trajs[:, -1, :]

    # Trajectory arc lengths
    arc_lengths = []
    for t in valid_trajs:
        diffs = np.diff(t, axis=0)
        arc_lengths.append(np.sqrt((diffs ** 2).sum(axis=1)).sum())
    mean_arc = np.mean(arc_lengths)

    # Endpoint distances from origin
    endpoint_dists = np.linalg.norm(endpoints, axis=1)

    # Count "far" modes (endpoint > 8m from origin)
    far_mask = endpoint_dists > 8.0
    n_far = int(far_mask.sum())

    # Among far modes, count how many go in distinct ANGULAR directions.
    # Two modes are "angularly distinct" if the angle between their endpoint
    # vectors differs by >25 degrees. This prevents highway scenes (all forward)
    # from scoring high.
    far_indices = np.where(far_mask)[0]
    far_angles = np.arctan2(endpoints[far_mask, 0], endpoints[far_mask, 1])

    n_distinct = 0
    if len(far_angles) > 0:
        selected_angles = [far_angles[0]]
        n_distinct = 1
        for a in far_angles[1:]:
            # Check angular distance to all already-selected angles
            if all(abs(((a - sa + math.pi) % (2 * math.pi)) - math.pi) > math.radians(25)
                   for sa in selected_angles):
                selected_angles.append(a)
                n_distinct += 1

    # Pairwise endpoint distances
    n = len(endpoints)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(endpoints[i] - endpoints[j]))
    max_dist = max(dists) if dists else 0.0

    n_agents = int(result["agent_data"]["agent_mask"].sum())
    n_lanes = int(result["map_data"]["map_mask"].sum())

    # Score: n_distinct is the PRIMARY signal
    # We want n_distinct >= 3 (straight + left + right)
    score = (
        n_distinct * 100.0        # 3 distinct modes -> 300 base
        + max_dist * 1.0           # bonus for spread
        + n_lanes * 0.05
        + n_agents * 0.1
    )

    # Penalty: trajectories must be visible
    if mean_arc < 8.0:
        score *= 0.1
    elif mean_arc < 12.0:
        score *= 0.4

    # Penalty: if fewer than 2 modes reach far from origin
    if n_far < 2:
        score *= 0.1

    return score, n_lanes, n_agents, max_dist, mean_arc, n_distinct


def classify_mode_direction(trajectory):
    """Human-readable label from trajectory shape."""
    endpoint = trajectory[-1]
    x, y = endpoint[0], endpoint[1]

    # Also consider midpoint to distinguish S-curves
    mid = trajectory[len(trajectory) // 2]
    mx, my = mid[0], mid[1]

    dist = np.sqrt(x**2 + y**2)
    if dist < 3.0:
        return "Slow/Stop"

    angle_deg = math.degrees(math.atan2(x, y))  # from +y (forward)

    if abs(angle_deg) < 20:
        return "Straight"
    elif 20 <= angle_deg < 55:
        return "Bear Right"
    elif angle_deg >= 55:
        return "Right Turn"
    elif -55 < angle_deg <= -20:
        return "Bear Left"
    elif angle_deg <= -55:
        return "Left Turn"
    return "Straight"


# ---------------------------------------------------------------------------
# Per-mode attention extraction
# ---------------------------------------------------------------------------
def extract_per_mode_attention(result, target_idx=0, use_last_layer_only=True):
    """Extract per-mode attention for all NMS-selected modes.

    When use_last_layer_only=True (default), uses only the final decoder
    layer's cross-attention. This produces sharper, more mode-specific
    patterns than accumulating across all layers (which tends to wash out
    differences since early layers are less mode-specific).
    """
    output = result["output"]
    attn_maps = result["attn_maps"]

    nms_indices = output["nms_indices"][0, target_idx]
    pred_trajs = output["trajectories"][0, target_idx].cpu().numpy()
    pred_scores = output["scores"][0, target_idx].cpu().numpy()

    dec_agent_attns = attn_maps.decoder_agent_attentions[target_idx]
    dec_map_attns = attn_maps.decoder_map_attentions[target_idx]
    n_layers = len(dec_agent_attns)

    A = result["batch"]["agent_mask"].shape[1]
    M = result["batch"]["map_mask"].shape[1]

    if use_last_layer_only:
        layer_range = [n_layers - 1]
    else:
        layer_range = list(range(n_layers))

    modes = []
    for m_idx in range(len(nms_indices)):
        intention_idx = nms_indices[m_idx].item()
        score = pred_scores[m_idx]
        traj = pred_trajs[m_idx]

        if score < -100:
            continue

        agent_attn = np.zeros(A, dtype=np.float32)
        map_attn = np.zeros(M, dtype=np.float32)

        for li in layer_range:
            a = dec_agent_attns[li][0, :, intention_idx, :]  # (nhead, A)
            agent_attn += a.cpu().numpy().mean(axis=0)

            mp = dec_map_attns[li][0, :, intention_idx, :]   # (nhead, M)
            map_attn += mp.cpu().numpy().mean(axis=0)

        label = classify_mode_direction(traj)

        modes.append({
            "agent_attn": agent_attn,
            "map_attn": map_attn,
            "trajectory": traj,
            "score": float(score),
            "intention_idx": intention_idx,
            "label": label,
            "mode_idx": m_idx,
        })

    modes.sort(key=lambda d: -d["score"])
    return modes


def select_distinct_modes(modes, max_panels=3):
    """Select modes that maximize angular spread for the most compelling figure.

    Strategy: pick the combination of max_panels modes (from available) that
    maximizes the minimum pairwise angular difference. This ensures each panel
    shows a truly different direction.
    """
    if len(modes) <= max_panels:
        return modes

    from itertools import combinations

    # Filter to modes with endpoints >5m from origin (visible trajectories)
    far_modes = [m for m in modes
                 if np.linalg.norm(m["trajectory"][-1]) > 5.0]
    if len(far_modes) < max_panels:
        far_modes = modes  # fallback

    # Compute endpoint angle for each mode
    def endpoint_angle(m):
        ep = m["trajectory"][-1]
        return math.atan2(ep[0], ep[1])

    best_combo = None
    best_min_angle = -1

    for combo in combinations(range(len(far_modes)), max_panels):
        angles = [endpoint_angle(far_modes[i]) for i in combo]
        # Compute min pairwise angular distance
        min_ang = float("inf")
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                diff = abs(((angles[i] - angles[j] + math.pi)
                            % (2 * math.pi)) - math.pi)
                min_ang = min(min_ang, diff)
        if min_ang > best_min_angle:
            best_min_angle = min_ang
            best_combo = combo

    if best_combo is None:
        return modes[:max_panels]

    selected = [far_modes[i] for i in best_combo]

    # Sort by angle (left-to-right in the figure) for natural panel ordering
    selected.sort(key=lambda m: endpoint_angle(m))

    return selected


# ---------------------------------------------------------------------------
# Heatmap construction
# ---------------------------------------------------------------------------
def build_attention_heatmap(agent_data, map_data, agent_attn, map_attn,
                            bev_range=50.0, resolution=0.5):
    """Build 2D spatial heatmap from token attention weights. Returns raw (unnormalized)."""
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
        span = int(3 * sigma_px)
        for dy in range(-span, span + 1):
            for dx in range(-span, span + 1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    d2 = dx**2 + dy**2
                    heatmap[gy, gx] += agent_attn[i] * np.exp(-d2 / (2 * sigma_px**2))

    # Lane painting
    lane_w = int(2.0 / resolution)
    for i in range(len(lane_pts)):
        if not map_mask[i] or map_attn[i] < 1e-6:
            continue
        for p in range(lane_pts.shape[1] - 1):
            p1, p2 = lane_pts[i, p], lane_pts[i, p + 1]
            g1x, g1y = bev_to_grid(p1)
            g2x, g2y = bev_to_grid(p2)
            ns = max(abs(g2x - g1x), abs(g2y - g1y), 1)
            for s in range(ns + 1):
                t = s / ns
                gx = int(g1x + t * (g2x - g1x))
                gy = int(g1y + t * (g2y - g1y))
                for dw in range(-lane_w // 2, lane_w // 2 + 1):
                    for dh in range(-lane_w // 2, lane_w // 2 + 1):
                        nx_, ny_ = gx + dw, gy + dh
                        if 0 <= nx_ < grid_size and 0 <= ny_ < grid_size:
                            heatmap[ny_, nx_] += map_attn[i] * 0.3

    heatmap = gaussian_filter(heatmap, sigma=2.5)
    return heatmap


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------
def render_mode_panel(
    ax, agent_data, map_data, agent_attn, map_attn,
    mode_traj, mode_label, mode_color, mode_score,
    ego_history, target_future, target_valid,
    other_trajs=None,
    bev_range=50.0, resolution=0.5,
    global_vmax=1.0,
    panel_label=None,
    show_y_label=True,
    mode_rank=0,
):
    """Render one BEV panel for a single prediction mode."""
    heatmap = build_attention_heatmap(
        agent_data, map_data, agent_attn, map_attn,
        bev_range=bev_range, resolution=resolution,
    )

    heatmap_norm = np.clip(heatmap / max(global_vmax, 1e-8), 0, 1)

    lane_pts = map_data["lane_centerlines_bev"]
    map_mask_arr = map_data["map_mask"]
    agent_positions = agent_data["agent_polylines"][:, -1, 0:2]
    valid_agents = agent_data["agent_mask"]

    # Background lanes
    for i in range(len(lane_pts)):
        if map_mask_arr[i]:
            pts = lane_pts[i]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=COLOR_LANE_BG,
                    linewidth=0.6, alpha=0.4, zorder=1)

    # Attention-weighted lanes (use PER-MODE normalization for vivid coloring)
    valid_map_attn = map_attn[map_mask_arr.astype(bool)]
    local_max = valid_map_attn.max() if len(valid_map_attn) > 0 and valid_map_attn.max() > 0 else 1.0
    for i in range(len(lane_pts)):
        if not map_mask_arr[i] or map_attn[i] < 0.003:
            continue
        pts = lane_pts[i]
        normed = min(map_attn[i] / local_max, 1.0)
        color = plt.cm.YlOrRd(normed * 0.85 + 0.1)
        width = 0.6 + normed * 3.0
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=width,
                alpha=0.75, zorder=2)

    # Heatmap overlay (global normalization for fair comparison)
    extent = [-bev_range, bev_range, -bev_range, bev_range]
    im = ax.imshow(heatmap_norm, extent=extent, origin="lower", cmap="magma",
                   alpha=0.50, vmin=0, vmax=1, zorder=3)

    # Agent markers
    for i in range(len(agent_positions)):
        if not valid_agents[i]:
            continue
        pos = agent_positions[i]
        if abs(pos[0]) > bev_range or abs(pos[1]) > bev_range:
            continue
        atype = get_agent_type(agent_data["agent_polylines"], i)
        c = TYPE_COLORS.get(atype, "#9E9E9E")
        sz = 5 if i == 0 else 3.5
        edge = "white" if i == 0 else "black"
        ew = 0.8 if i == 0 else 0.3
        ax.plot(pos[0], pos[1], "s", color=c, markersize=sz,
                markeredgecolor=edge, markeredgewidth=ew, zorder=6)

    # Ego marker
    ax.plot(0, 0, "^", color=COLOR_EGO, markersize=9,
            markeredgecolor="white", markeredgewidth=0.8, zorder=8)

    # History
    if ego_history is not None:
        valid_hist = ego_history[~np.all(ego_history == 0, axis=1)]
        if len(valid_hist) > 1:
            ax.plot(valid_hist[:, 0], valid_hist[:, 1], "-o",
                    color=COLOR_HISTORY, linewidth=1.8, markersize=2.5,
                    zorder=7, label="History")

    # GT future (with white outline for visibility over heatmap)
    if target_future is not None and target_valid is not None:
        vf = target_future[target_valid]
        if len(vf) > 0:
            ax.plot(vf[:, 0], vf[:, 1], "--", color="white",
                    linewidth=3.0, alpha=0.6, zorder=6)
            ax.plot(vf[:, 0], vf[:, 1], "--", color=COLOR_GT,
                    linewidth=2.0, zorder=7, label="Ground truth")

    # Other modes (ghost, slightly more visible)
    if other_trajs is not None:
        for ot in other_trajs:
            ax.plot(ot[:, 0], ot[:, 1], "-", color="#BBBBBB",
                    linewidth=1.0, alpha=0.45, zorder=6)

    # This mode's trajectory -- white border for contrast, then colored
    ax.plot(mode_traj[:, 0], mode_traj[:, 1], "-", color="white",
            linewidth=4.5, alpha=0.7, zorder=8)
    ax.plot(mode_traj[:, 0], mode_traj[:, 1], "-", color=mode_color,
            linewidth=3.0, alpha=0.95, zorder=9)

    # Endpoint
    ax.plot(mode_traj[-1, 0], mode_traj[-1, 1], "o", color=mode_color,
            markersize=9, markeredgecolor="white", markeredgewidth=1.2,
            zorder=10)

    # Direction arrow at trajectory midpoint
    mid_idx = len(mode_traj) // 2
    if mid_idx > 0 and mid_idx < len(mode_traj) - 1:
        dx = mode_traj[mid_idx + 1, 0] - mode_traj[mid_idx - 1, 0]
        dy = mode_traj[mid_idx + 1, 1] - mode_traj[mid_idx - 1, 1]
        arrow_len = np.sqrt(dx**2 + dy**2)
        if arrow_len > 0.5:
            ax.annotate(
                "", xy=(mode_traj[mid_idx, 0] + dx * 0.3,
                        mode_traj[mid_idx, 1] + dy * 0.3),
                xytext=(mode_traj[mid_idx, 0], mode_traj[mid_idx, 1]),
                arrowprops=dict(arrowstyle="-|>", color=mode_color,
                                lw=2.0, mutation_scale=14),
                zorder=10,
            )

    # Axes
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_xlabel("Lateral (m)")
    if show_y_label:
        ax.set_ylabel("Longitudinal (m)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    # Title: clean "Mode k: Direction"
    ax.set_title(f"Mode {mode_rank+1}: {mode_label}",
                 fontweight="bold", pad=6)

    # Panel label
    if panel_label:
        ax.text(0.03, 0.96, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
                zorder=20)

    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    plt.rcParams.update(PAPER_RC)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model_and_config()
    val_scenes = get_val_scenes(cfg)
    print(f"Validation scenes: {len(val_scenes)}")

    # -----------------------------------------------------------------
    # Search for a scene with large trajectory spread
    # -----------------------------------------------------------------
    print("\nSearching for scene with diverse, long prediction modes...")
    best_result = None
    best_score = -1
    best_info = None

    max_scan = min(800, len(val_scenes))
    for idx in range(max_scan):
        if idx % 100 == 0:
            print(f"  Scanning scene {idx}/{max_scan} (best so far: "
                  f"score={best_score:.1f})...")

        result = extract_scene_with_modes(model, val_scenes[idx], cfg, device=DEVICE)
        if result is None:
            continue

        score, n_lanes, n_agents, max_dist, mean_arc, n_distinct = \
            compute_mode_diversity(result)

        if score > best_score:
            best_score = score
            best_result = result
            best_info = {
                "idx": idx,
                "n_lanes": n_lanes,
                "n_agents": n_agents,
                "max_dist": max_dist,
                "mean_arc": mean_arc,
                "score": score,
                "n_distinct": n_distinct,
            }
            if score > 20:
                print(f"    New best: idx={idx}, score={score:.1f}, "
                      f"distinct={n_distinct}, maxDist={max_dist:.1f}m, "
                      f"arcLen={mean_arc:.1f}m, "
                      f"lanes={n_lanes}, agents={n_agents}")

        # Early stop: 3+ truly distinct directional modes
        if n_distinct >= 3 and max_dist > 15 and mean_arc > 12:
            print(f"    Excellent scene at idx={idx} with {n_distinct} "
                  f"distinct modes, stopping.")
            break

    if best_result is None:
        print("ERROR: No valid scene found.")
        return

    print(f"\nSelected scene index {best_info['idx']}:")
    print(f"  Score: {best_info['score']:.1f}")
    print(f"  Distinct directional modes: {best_info['n_distinct']}")
    print(f"  Max pairwise dist: {best_info['max_dist']:.1f}m")
    print(f"  Mean traj arc length: {best_info['mean_arc']:.1f}m")
    print(f"  Lanes: {best_info['n_lanes']}, Agents: {best_info['n_agents']}")

    # -----------------------------------------------------------------
    # Extract per-mode attention (last decoder layer only for sharpness)
    # -----------------------------------------------------------------
    modes = extract_per_mode_attention(best_result, target_idx=0,
                                       use_last_layer_only=True)
    print(f"\nNMS-selected modes ({len(modes)} valid):")
    for m in modes:
        ep = m["trajectory"][-1]
        arc = np.sqrt(np.diff(m["trajectory"], axis=0)**2).sum()
        print(f"  Mode {m['mode_idx']}: {m['label']:12s}  "
              f"score={m['score']:.3f}  endpoint=({ep[0]:+.1f}, {ep[1]:+.1f})  "
              f"arcLen={arc:.1f}m  intention={m['intention_idx']}")

    # Select 3 modes with maximum angular diversity
    selected_modes = select_distinct_modes(modes, max_panels=3)

    # If all modes have the same label, try to differentiate them
    labels = [m["label"] for m in selected_modes]
    if len(set(labels)) < len(labels):
        for i, m in enumerate(selected_modes):
            ep = m["trajectory"][-1]
            dist = np.sqrt(ep[0]**2 + ep[1]**2)
            if labels.count(m["label"]) > 1:
                if dist < 5:
                    m["label"] = "Slow"
                else:
                    angle = math.degrees(math.atan2(ep[0], ep[1]))
                    m["label"] = f"{m['label']} ({angle:+.0f} deg)"

    n_panels = len(selected_modes)
    print(f"\nSelected {n_panels} modes for figure:")
    for i, m in enumerate(selected_modes):
        ep = m["trajectory"][-1]
        print(f"  Panel {i+1}: Mode {m['mode_idx']} = {m['label']} "
              f"(score={m['score']:.3f}, endpoint=({ep[0]:+.1f}, {ep[1]:+.1f}))")

    # -----------------------------------------------------------------
    # Determine BEV range from data extent
    # -----------------------------------------------------------------
    agent_data = best_result["agent_data"]
    map_data = best_result["map_data"]

    # Compute extent needed to show all trajectories + some context
    all_pts = []
    for m in selected_modes:
        all_pts.append(m["trajectory"])
    ego_history = agent_data["agent_polylines"][0, :, 0:2]
    all_pts.append(ego_history)
    target_future = best_result["batch"]["target_future"][0, 0].cpu().numpy()
    target_valid = best_result["batch"]["target_future_valid"][0, 0].cpu().numpy().astype(bool)
    if target_valid.any():
        all_pts.append(target_future[target_valid])

    all_pts = np.concatenate(all_pts, axis=0)
    max_extent = max(np.abs(all_pts).max(), 15.0)  # at least 15m
    bev_range = min(max_extent + 10.0, 50.0)       # pad 10m, cap at 50m
    bev_range = max(bev_range, 25.0)                # at least 25m
    print(f"\nAuto BEV range: {bev_range:.0f}m")

    # -----------------------------------------------------------------
    # Global heatmap normalization
    # -----------------------------------------------------------------
    raw_heatmaps = []
    for m in selected_modes:
        hm = build_attention_heatmap(
            agent_data, map_data, m["agent_attn"], m["map_attn"],
            bev_range=bev_range, resolution=0.5)
        raw_heatmaps.append(hm)

    all_vals = np.concatenate([h.ravel() for h in raw_heatmaps])
    pos_vals = all_vals[all_vals > 0]
    global_vmax = np.percentile(pos_vals, 95) if len(pos_vals) > 0 else 1.0
    print(f"Global heatmap vmax (p95): {global_vmax:.4f}")

    # -----------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------
    print(f"\nRendering {n_panels}-panel figure...")

    fig_w = 4.5 * n_panels + 0.5
    fig_h = 5.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        1, n_panels + 1,
        width_ratios=[1] * n_panels + [0.035],
        wspace=0.06,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cbar_ax = fig.add_subplot(gs[0, n_panels])

    panel_labels = ["(a)", "(b)", "(c)"]
    last_im = None

    all_trajs = [m["trajectory"] for m in selected_modes]

    for i, mode in enumerate(selected_modes):
        other_trajs = [t for j, t in enumerate(all_trajs) if j != i]

        im = render_mode_panel(
            ax=axes[i],
            agent_data=agent_data,
            map_data=map_data,
            agent_attn=mode["agent_attn"],
            map_attn=mode["map_attn"],
            mode_traj=mode["trajectory"],
            mode_label=mode["label"],
            mode_color=MODE_COLORS[i],
            mode_score=mode["score"],
            ego_history=ego_history,
            target_future=target_future,
            target_valid=target_valid,
            other_trajs=other_trajs,
            bev_range=bev_range,
            resolution=0.5,
            global_vmax=global_vmax,
            panel_label=panel_labels[i] if i < len(panel_labels) else None,
            show_y_label=(i == 0),
            mode_rank=i,
        )
        last_im = im

    # Shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Attention weight", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Shared legend
    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_HISTORY, marker="o", markersize=3,
                    linewidth=1.5, label="History"),
        plt.Line2D([0], [0], color=COLOR_GT, linewidth=1.5, linestyle="--",
                    label="Ground truth"),
    ]
    for i, mode in enumerate(selected_modes):
        legend_handles.append(
            plt.Line2D([0], [0], color=MODE_COLORS[i], linewidth=2.5,
                        label=f"Mode {i+1}: {mode['label']}"))
    legend_handles.append(
        plt.Line2D([0], [0], color="#999999", linewidth=0.8, alpha=0.4,
                    label="Other modes"))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        bbox_to_anchor=(0.47, -0.01),
        handlelength=2.0,
    )

    # Save
    save_path = OUTPUT_DIR / "fig_mode_attention_comparison.pdf"
    fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("DONE -- Multi-modal attention comparison figure generated")
    print(f"  Output: {save_path}")
    print(f"  Panels: {n_panels}")
    for i, mode in enumerate(selected_modes):
        ep = mode["trajectory"][-1]
        print(f"    Panel {i+1}: Mode {mode['mode_idx']} = {mode['label']} "
              f"(score={mode['score']:.3f}, endpoint=({ep[0]:+.1f}, {ep[1]:+.1f}))")
    print(f"  Time: {elapsed:.1f}s")
    size_kb = save_path.stat().st_size / 1024
    print(f"  File size: {size_kb:.0f} KB")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Attention statistics for paper text
    # -----------------------------------------------------------------
    print("\n--- Attention statistics per mode (for paper text) ---")
    agent_mask = agent_data["agent_mask"]
    map_mask_np = map_data["map_mask"].astype(bool)

    for i, mode in enumerate(selected_modes):
        va = mode["agent_attn"][agent_mask]
        vm = mode["map_attn"][map_mask_np]
        total = va.sum() + vm.sum()
        a_frac = va.sum() / max(total, 1e-8) * 100
        m_frac = vm.sum() / max(total, 1e-8) * 100

        a_order = np.argsort(mode["agent_attn"])[::-1]
        m_order = np.argsort(mode["map_attn"])[::-1]

        print(f"\n  Mode {mode['mode_idx']} ({mode['label']}):")
        print(f"    Agent attn fraction: {a_frac:.1f}%, Map: {m_frac:.1f}%")
        print(f"    Top-3 agents: {a_order[:3]} ({mode['agent_attn'][a_order[:3]]})")
        print(f"    Top-3 lanes:  {m_order[:3]} ({mode['map_attn'][m_order[:3]]})")

        for j, other in enumerate(selected_modes):
            if j == i:
                continue
            p = mode["map_attn"][map_mask_np]
            q = other["map_attn"][map_mask_np]
            p_n = p / max(p.sum(), 1e-8)
            q_n = q / max(q.sum(), 1e-8)
            mm = 0.5 * (p_n + q_n)
            eps = 1e-10
            kl1 = np.sum(p_n * np.log((p_n + eps) / (mm + eps)))
            kl2 = np.sum(q_n * np.log((q_n + eps) / (mm + eps)))
            jsd = 0.5 * (kl1 + kl2)
            print(f"    JSD(map vs Mode {other['mode_idx']} "
                  f"{other['label']}): {jsd:.4f}")


if __name__ == "__main__":
    main()
