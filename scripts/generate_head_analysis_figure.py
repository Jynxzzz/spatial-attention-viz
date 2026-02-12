"""Generate attention head disentanglement figure for the trajectory prediction paper.

Analyzes per-head attention patterns across all 4 encoder layers to determine
whether the "map attention reversal" in Layer 3 is driven by all 8 heads
uniformly or by a few specialized "map heads".

Produces a 2-panel figure (fig_head_disentanglement.pdf):
  Panel (a): Head-wise agent vs map attention ratio across all 4 layers (32 bars)
  Panel (b): Spatial BEV heatmaps for the 3 most agent-focused and 3 most
             map-focused heads in Layer 3 (2x3 grid)

Usage:
    cd /home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper
    conda run -n scenario-dreamer python scripts/generate_head_analysis_figure.py

Or with yolov8 env (has all deps):
    conda run -n yolov8 python scripts/generate_head_analysis_figure.py
"""

import os
import sys
import random
import time

import numpy as np
import torch
import yaml

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter

from training.lightning_module import MTRLiteModule
from data.polyline_dataset import PolylineDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/last.ckpt"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "mtr_lite.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")

DEVICE = "cpu"

# Number of scenes to average over for robust statistics
NUM_SCENES_TO_AVERAGE = 20

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "text.usetex": False,
    "axes.grid": False,
}

# Colors
COLOR_AGENT = "#4285F4"   # Google blue
COLOR_MAP = "#34A853"     # Google green
COLOR_LANE_BG = "#BDBDBD"
COLOR_EGO = "#0D47A1"
COLOR_HIGH_ATTN = "#C62828"

# Layer colors for annotations
LAYER_COLORS = ["#E8EAF6", "#C5CAE9", "#9FA8DA", "#7986CB"]  # indigo gradient


def load_model(config_path, checkpoint_path, device="cpu"):
    """Load trained MTR-Lite from a Lightning checkpoint."""
    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading checkpoint: {checkpoint_path}")
    module = MTRLiteModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    module.eval()
    module.to(device)
    model = module.model
    print(f"  Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, cfg


def run_inference(model, sample, device="cpu"):
    """Run a single sample with attention capture."""
    batch = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0).to(device)
        else:
            batch[key] = val
    with torch.no_grad():
        output = model(batch, capture_attention=True)
    return output, batch


def compute_head_stats_single(output, batch):
    """Compute per-head agent/map attention stats for one scene.

    Returns:
        head_stats: dict with keys:
            agent_pct[layer][head]: fraction of ego's attention on agent tokens
            map_pct[layer][head]: fraction of ego's attention on map tokens
            entropy[layer][head]: Shannon entropy of ego's attention distribution
            per_head_attn[layer][head]: (96,) raw attention vector from ego token
    """
    attn_maps = output["attention_maps"]
    scene_attns = attn_maps.scene_attentions  # list of (B, nhead, N, N)
    num_layers = len(scene_attns)

    agent_mask = batch["agent_mask"][0].cpu().numpy()  # (32,)
    map_mask = batch["map_mask"][0].cpu().numpy()       # (64,)

    A = 32  # num agent tokens
    M = 64  # num map tokens

    # Valid token mask for normalization
    valid_agent = agent_mask.astype(bool)
    valid_map = map_mask.astype(bool)

    result = {
        "agent_pct": [],
        "map_pct": [],
        "entropy": [],
        "per_head_attn": [],
    }

    for layer_i in range(num_layers):
        attn = scene_attns[layer_i][0]  # (nhead, N, N) where N = A + M = 96

        # Ego token is index 0 (always the first agent)
        # ego_attn[h, :] = attention from ego token to all tokens, for head h
        ego_attn = attn[:, 0, :].cpu().numpy()  # (nhead, 96)
        nhead = ego_attn.shape[0]

        layer_agent_pct = []
        layer_map_pct = []
        layer_entropy = []
        layer_per_head = []

        for h in range(nhead):
            attn_h = ego_attn[h]  # (96,)

            # Agent attention: sum of weights on valid agent tokens
            agent_sum = attn_h[:A][valid_agent].sum()
            # Map attention: sum of weights on valid map tokens
            map_sum = attn_h[A:][valid_map].sum()

            total = agent_sum + map_sum
            if total > 1e-8:
                agent_frac = agent_sum / total
                map_frac = map_sum / total
            else:
                agent_frac = 0.5
                map_frac = 0.5

            # Shannon entropy (in bits)
            eps = 1e-10
            valid_weights = np.concatenate([attn_h[:A][valid_agent], attn_h[A:][valid_map]])
            valid_weights = valid_weights + eps
            valid_weights = valid_weights / valid_weights.sum()
            entropy = -np.sum(valid_weights * np.log2(valid_weights))

            layer_agent_pct.append(float(agent_frac))
            layer_map_pct.append(float(map_frac))
            layer_entropy.append(float(entropy))
            layer_per_head.append(attn_h.copy())

        result["agent_pct"].append(layer_agent_pct)
        result["map_pct"].append(layer_map_pct)
        result["entropy"].append(layer_entropy)
        result["per_head_attn"].append(layer_per_head)

    return result


def aggregate_head_stats(all_stats):
    """Average head_stats across multiple scenes."""
    num_scenes = len(all_stats)
    num_layers = len(all_stats[0]["agent_pct"])
    nhead = len(all_stats[0]["agent_pct"][0])

    avg = {
        "agent_pct": [[0.0] * nhead for _ in range(num_layers)],
        "map_pct": [[0.0] * nhead for _ in range(num_layers)],
        "entropy": [[0.0] * nhead for _ in range(num_layers)],
        "agent_pct_std": [[0.0] * nhead for _ in range(num_layers)],
        "map_pct_std": [[0.0] * nhead for _ in range(num_layers)],
    }

    for l in range(num_layers):
        for h in range(nhead):
            agent_vals = [s["agent_pct"][l][h] for s in all_stats]
            map_vals = [s["map_pct"][l][h] for s in all_stats]
            ent_vals = [s["entropy"][l][h] for s in all_stats]

            avg["agent_pct"][l][h] = np.mean(agent_vals)
            avg["map_pct"][l][h] = np.mean(map_vals)
            avg["entropy"][l][h] = np.mean(ent_vals)
            avg["agent_pct_std"][l][h] = np.std(agent_vals)
            avg["map_pct_std"][l][h] = np.std(map_vals)

    return avg


def generate_panel_a(ax, avg_stats):
    """Panel (a): Head-wise agent/map attention ratio across all 4 layers.

    32 stacked bars (4 layers x 8 heads), grouped by layer.
    Each bar is stacked: bottom = agent%, top = map%.
    """
    num_layers = len(avg_stats["agent_pct"])
    nhead = len(avg_stats["agent_pct"][0])
    total_bars = num_layers * nhead

    x_positions = np.arange(total_bars)
    agent_vals = []
    map_vals = []
    agent_stds = []
    map_stds = []
    bar_labels = []

    for l in range(num_layers):
        for h in range(nhead):
            a_pct = avg_stats["agent_pct"][l][h] * 100
            m_pct = avg_stats["map_pct"][l][h] * 100
            agent_vals.append(a_pct)
            map_vals.append(m_pct)
            agent_stds.append(avg_stats["agent_pct_std"][l][h] * 100)
            map_stds.append(avg_stats["map_pct_std"][l][h] * 100)
            bar_labels.append(f"H{h}")

    agent_vals = np.array(agent_vals)
    map_vals = np.array(map_vals)
    agent_stds = np.array(agent_stds)

    # Stacked bar chart
    bar_width = 0.75
    bars_agent = ax.bar(x_positions, agent_vals, bar_width,
                        label="Agent tokens", color=COLOR_AGENT, alpha=0.85,
                        edgecolor="white", linewidth=0.3)
    bars_map = ax.bar(x_positions, map_vals, bar_width, bottom=agent_vals,
                      label="Map tokens", color=COLOR_MAP, alpha=0.85,
                      edgecolor="white", linewidth=0.3)

    # Error bars on agent portion
    ax.errorbar(x_positions, agent_vals, yerr=agent_stds,
                fmt="none", ecolor="0.3", elinewidth=0.6, capsize=1.5, capthick=0.5)

    # 50% reference line
    ax.axhline(y=50, color="0.5", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.text(total_bars - 0.5, 51, "50%", fontsize=7, color="0.5", ha="right", va="bottom")

    # Layer grouping: background shading and labels
    for l in range(num_layers):
        start = l * nhead - 0.5
        end = (l + 1) * nhead - 0.5
        ax.axvspan(start, end, alpha=0.08, color=LAYER_COLORS[l], zorder=0)
        # Layer label at top
        mid_x = l * nhead + (nhead - 1) / 2
        ax.text(mid_x, 103, f"Layer {l}", fontsize=9, fontweight="bold",
                ha="center", va="bottom", color="0.2")

    # Vertical separators between layers
    for l in range(1, num_layers):
        ax.axvline(x=l * nhead - 0.5, color="0.5", linewidth=0.8, linestyle="-", alpha=0.4)

    # X axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, fontsize=7, rotation=0)
    ax.set_xlim(-0.7, total_bars - 0.3)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Attention share (%)")
    ax.set_title("(a)  Head-wise agent vs. map attention across encoder layers",
                 fontweight="bold", pad=12, loc="left")

    # Legend
    legend = ax.legend(loc="upper left", framealpha=0.9, edgecolor="none",
                       ncol=2, handlelength=1.2, handletextpad=0.4)

    # Annotate key findings
    # Find the most map-focused head in Layer 3
    l3_map = [avg_stats["map_pct"][3][h] for h in range(nhead)]
    max_map_head = np.argmax(l3_map)
    max_map_val = l3_map[max_map_head] * 100
    bar_idx = 3 * nhead + max_map_head

    # Find the most agent-focused head in Layer 3
    l3_agent = [avg_stats["agent_pct"][3][h] for h in range(nhead)]
    max_agent_head = np.argmax(l3_agent)
    max_agent_val = l3_agent[max_agent_head] * 100

    # Print the summary stat
    l3_avg_map = np.mean(l3_map) * 100
    l3_avg_agent = np.mean([avg_stats["agent_pct"][3][h] for h in range(nhead)]) * 100
    print(f"\n  Layer 3 summary: avg agent={l3_avg_agent:.1f}%, avg map={l3_avg_map:.1f}%")
    print(f"  Most map-focused: H{max_map_head} ({max_map_val:.1f}% map)")
    print(f"  Most agent-focused: H{max_agent_head} ({max_agent_val:.1f}% agent)")

    return ax


def generate_panel_b(axes_grid, single_scene_stats, sample, layer_idx=3):
    """Panel (b): BEV spatial attention heatmaps for top-3 agent-focused
    and top-3 map-focused heads in the specified encoder layer.

    axes_grid: 2x3 array of axes
    single_scene_stats: stats from a single representative scene
    sample: dataset sample (for spatial data)
    """
    nhead = len(single_scene_stats["agent_pct"][layer_idx])

    # Rank heads by map attention fraction
    head_map_pct = single_scene_stats["map_pct"][layer_idx]
    head_order = np.argsort(head_map_pct)  # ascending: most agent-focused first

    # Top-3 agent-focused (lowest map_pct) and top-3 map-focused (highest map_pct)
    agent_heads = head_order[:3]
    map_heads = head_order[-3:][::-1]  # descending by map%

    # Spatial data
    lane_cl_bev = sample["lane_centerlines_bev"].numpy()  # (M, 20, 2)
    map_mask = sample["map_mask"].numpy()                  # (M,)
    agent_poly = sample["agent_polylines"].numpy()         # (A, 11, 29)
    agent_mask_np = sample["agent_mask"].numpy()           # (A,)

    A = 32
    bev_range = 50.0
    resolution = 0.5
    grid_size = int(2 * bev_range / resolution)

    def bev_to_grid(xy):
        gx = int((xy[0] + bev_range) / resolution)
        gy = int((xy[1] + bev_range) / resolution)
        return np.clip(gx, 0, grid_size - 1), np.clip(gy, 0, grid_size - 1)

    agent_positions = agent_poly[:, -1, 0:2]  # (A, 2) last timestep positions

    def render_head_heatmap(ax, head_idx, title_str):
        """Render BEV spatial attention for a single head."""
        attn_vec = single_scene_stats["per_head_attn"][layer_idx][head_idx]  # (96,)

        # Agent attention weights
        agent_attn = attn_vec[:A]
        # Map attention weights
        map_attn = attn_vec[A:]

        # Build heatmap
        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Agent Gaussian splats
        sigma_px = 4.0 / resolution
        for i in range(A):
            if not agent_mask_np[i] or agent_attn[i] < 1e-6:
                continue
            pos = agent_positions[i]
            if abs(pos[0]) > bev_range or abs(pos[1]) > bev_range:
                continue
            cx, cy = bev_to_grid(pos)
            span = int(3 * sigma_px)
            for dy in range(-span, span + 1):
                for dx in range(-span, span + 1):
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < grid_size and 0 <= gy < grid_size:
                        dist2 = dx**2 + dy**2
                        heatmap[gy, gx] += agent_attn[i] * np.exp(-dist2 / (2 * sigma_px**2))

        # Lane painting
        lane_width_px = int(2.0 / resolution)
        for i in range(len(lane_cl_bev)):
            if not map_mask[i] or map_attn[i] < 1e-6:
                continue
            pts = lane_cl_bev[i]
            for p in range(pts.shape[0] - 1):
                p1, p2 = pts[p], pts[p + 1]
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

        # Smooth
        heatmap = gaussian_filter(heatmap, sigma=2.5)
        if heatmap.max() > 0:
            p95 = np.percentile(heatmap[heatmap > 0], 95) if (heatmap > 0).any() else 1.0
            heatmap = np.clip(heatmap / max(p95, 1e-8), 0, 1)

        # Draw background lanes
        for i in range(len(lane_cl_bev)):
            if map_mask[i]:
                pts = lane_cl_bev[i]
                ax.plot(pts[:, 0], pts[:, 1], "-", color=COLOR_LANE_BG,
                        linewidth=0.4, alpha=0.4, zorder=1)

        # Heatmap overlay
        extent = [-bev_range, bev_range, -bev_range, bev_range]
        ax.imshow(heatmap, extent=extent, origin="lower", cmap="magma",
                  alpha=0.65, vmin=0, vmax=1, zorder=2)

        # Ego marker
        ax.plot(0, 0, "^", color=COLOR_EGO, markersize=5, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)

        # Agent markers (small dots for context)
        for i in range(A):
            if agent_mask_np[i] and i != 0:
                pos = agent_positions[i]
                if abs(pos[0]) < bev_range and abs(pos[1]) < bev_range:
                    ax.plot(pos[0], pos[1], "s", color="#666666", markersize=2,
                            alpha=0.5, zorder=4)

        ax.set_xlim(-bev_range, bev_range)
        ax.set_ylim(-bev_range, bev_range)
        ax.set_aspect("equal")
        ax.set_title(title_str, fontsize=8, pad=3)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))

    # Top row: most agent-focused heads
    for col, h in enumerate(agent_heads):
        a_pct = single_scene_stats["agent_pct"][layer_idx][h] * 100
        m_pct = single_scene_stats["map_pct"][layer_idx][h] * 100
        title = f"H{h} (agent {a_pct:.0f}%)"
        render_head_heatmap(axes_grid[0, col], h, title)

    # Bottom row: most map-focused heads
    for col, h in enumerate(map_heads):
        a_pct = single_scene_stats["agent_pct"][layer_idx][h] * 100
        m_pct = single_scene_stats["map_pct"][layer_idx][h] * 100
        title = f"H{h} (map {m_pct:.0f}%)"
        render_head_heatmap(axes_grid[1, col], h, title)

    # Row labels
    axes_grid[0, 0].set_ylabel("Agent-focused\nheads", fontsize=8, fontweight="bold")
    axes_grid[1, 0].set_ylabel("Map-focused\nheads", fontsize=8, fontweight="bold")


def main():
    t0 = time.time()

    plt.rcParams.update(PAPER_RC)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load model ----
    model, cfg = load_model(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)

    # ---- Create dataset ----
    print("\nCreating validation dataset...")
    dataset = PolylineDataset(
        scene_list_path=cfg["data"]["scene_list"],
        split="val",
        val_ratio=cfg["data"]["val_ratio"],
        data_fraction=cfg["data"]["data_fraction"],
        history_len=cfg["data"]["history_len"],
        future_len=cfg["data"]["future_len"],
        max_agents=cfg["data"]["max_agents"],
        max_map_polylines=cfg["data"]["max_map_polylines"],
        map_points_per_lane=cfg["data"]["map_points_per_lane"],
        max_targets=cfg["model"]["max_targets"],
        neighbor_distance=cfg["data"]["neighbor_distance"],
        anchor_frames=cfg["data"]["anchor_frames"],
        augment=False,
        seed=cfg.get("seed", 42),
    )
    print(f"  Validation set: {len(dataset)} scenes")

    # ---- Run inference on multiple scenes and collect per-head stats ----
    print(f"\nCollecting per-head attention stats from {NUM_SCENES_TO_AVERAGE} scenes...")
    all_stats = []
    representative_sample = None
    representative_stats = None
    representative_output = None
    best_scene_score = -1

    # Try scenes in order, skip failures
    scene_idx = 0
    while len(all_stats) < NUM_SCENES_TO_AVERAGE and scene_idx < len(dataset):
        sample = dataset[scene_idx]
        scene_idx += 1
        if sample is None:
            continue

        try:
            output, batch = run_inference(model, sample, device=DEVICE)
        except Exception as e:
            print(f"  Scene {scene_idx - 1}: inference failed ({e})")
            continue

        if output.get("attention_maps") is None:
            continue

        stats = compute_head_stats_single(output, batch)
        all_stats.append(stats)

        # Pick a representative scene for Panel (b) BEV heatmaps
        # Prefer scenes with many lanes + agents for visual clarity
        n_lanes = sample["map_mask"].sum().item()
        n_agents = sample["agent_mask"].sum().item()
        score = n_lanes * 2 + n_agents
        if score > best_scene_score:
            best_scene_score = score
            representative_sample = sample
            representative_stats = stats
            representative_output = output

        if len(all_stats) % 5 == 0:
            print(f"  Processed {len(all_stats)}/{NUM_SCENES_TO_AVERAGE} scenes")

    print(f"  Collected stats from {len(all_stats)} scenes")

    if len(all_stats) < 3:
        print("ERROR: Not enough valid scenes for analysis")
        return

    # ---- Aggregate statistics ----
    avg_stats = aggregate_head_stats(all_stats)

    # ---- Print detailed statistics table ----
    num_layers = len(avg_stats["agent_pct"])
    nhead = len(avg_stats["agent_pct"][0])

    print("\n" + "=" * 80)
    print("PER-HEAD ATTENTION STATISTICS (averaged over {} scenes)".format(len(all_stats)))
    print("=" * 80)
    print(f"{'Layer':<8} {'Head':<6} {'Agent%':>8} {'Map%':>8} {'Entropy':>10} {'Std(A%)':>8}")
    print("-" * 50)

    for l in range(num_layers):
        for h in range(nhead):
            a_pct = avg_stats["agent_pct"][l][h] * 100
            m_pct = avg_stats["map_pct"][l][h] * 100
            ent = avg_stats["entropy"][l][h]
            a_std = avg_stats["agent_pct_std"][l][h] * 100
            print(f"  L{l:<5} H{h:<4} {a_pct:>7.1f}% {m_pct:>7.1f}% {ent:>9.2f}b {a_std:>7.1f}%")
        # Layer summary
        l_agent_avg = np.mean(avg_stats["agent_pct"][l]) * 100
        l_map_avg = np.mean(avg_stats["map_pct"][l]) * 100
        l_ent_avg = np.mean(avg_stats["entropy"][l])
        print(f"  L{l} avg:       {l_agent_avg:>7.1f}% {l_map_avg:>7.1f}% {l_ent_avg:>9.2f}b")
        print("-" * 50)

    # Check for head specialization in Layer 3
    l3_agent = np.array(avg_stats["agent_pct"][3])
    l3_map = np.array(avg_stats["map_pct"][3])
    l3_range = l3_map.max() - l3_map.min()
    l3_std = np.std(l3_map)
    print(f"\nLayer 3 head specialization:")
    print(f"  Map% range: {l3_map.min()*100:.1f}% - {l3_map.max()*100:.1f}% (spread = {l3_range*100:.1f}pp)")
    print(f"  Map% std: {l3_std*100:.1f}pp")
    if l3_range > 0.2:
        print(f"  FINDING: Strong head specialization detected!")
        print(f"  Some heads specialize in map ({l3_map.max()*100:.0f}%), others in agents ({l3_agent.max()*100:.0f}%)")
    else:
        print(f"  FINDING: Heads shift relatively uniformly toward map attention.")

    # ---- Generate figure ----
    print("\nGenerating figure...")

    # Layout: Panel (a) on top (full width), Panel (b) on bottom (2x3 grid)
    fig = plt.figure(figsize=(12.0, 9.5))

    # Top: Panel (a) - bar chart
    gs_top = fig.add_gridspec(
        nrows=1, ncols=1,
        left=0.06, right=0.98, top=0.97, bottom=0.56,
    )
    ax_a = fig.add_subplot(gs_top[0, 0])
    generate_panel_a(ax_a, avg_stats)

    # Bottom: Panel (b) - 2x3 BEV grid
    gs_bot = fig.add_gridspec(
        nrows=2, ncols=3,
        left=0.06, right=0.92, top=0.48, bottom=0.02,
        wspace=0.18, hspace=0.28,
    )
    axes_b = np.array([[fig.add_subplot(gs_bot[r, c]) for c in range(3)] for r in range(2)])

    # Panel (b) title
    fig.text(0.06, 0.50, "(b)  Head specialization in Layer 3: spatial attention patterns",
             fontsize=11, fontweight="bold", va="bottom")

    generate_panel_b(axes_b, representative_stats, representative_sample, layer_idx=3)

    # Add colorbar for the BEV heatmaps
    cbar_ax = fig.add_axes([0.94, 0.08, 0.015, 0.35])
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention intensity", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ---- Save ----
    save_path = os.path.join(OUTPUT_DIR, "fig_head_disentanglement.pdf")
    fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {save_path}")

    # Also save PNG for quick preview
    save_path_png = os.path.join(OUTPUT_DIR, "fig_head_disentanglement.png")
    fig2 = plt.figure(figsize=(12.0, 9.5))
    gs_top2 = fig2.add_gridspec(nrows=1, ncols=1, left=0.06, right=0.98, top=0.97, bottom=0.56)
    ax_a2 = fig2.add_subplot(gs_top2[0, 0])
    generate_panel_a(ax_a2, avg_stats)
    gs_bot2 = fig2.add_gridspec(nrows=2, ncols=3, left=0.06, right=0.92, top=0.48, bottom=0.02, wspace=0.18, hspace=0.28)
    axes_b2 = np.array([[fig2.add_subplot(gs_bot2[r, c]) for c in range(3)] for r in range(2)])
    fig2.text(0.06, 0.50, "(b)  Head specialization in Layer 3: spatial attention patterns",
              fontsize=11, fontweight="bold", va="bottom")
    generate_panel_b(axes_b2, representative_stats, representative_sample, layer_idx=3)
    cbar_ax2 = fig2.add_axes([0.94, 0.08, 0.015, 0.35])
    cbar2 = fig2.colorbar(sm, cax=cbar_ax2)
    cbar2.set_label("Attention intensity", fontsize=8)
    cbar2.ax.tick_params(labelsize=7)
    fig2.savefig(save_path_png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {save_path_png}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE  ({elapsed:.1f}s)")
    print(f"  Figure: {save_path}")
    print(f"  Preview: {save_path_png}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
