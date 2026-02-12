"""Generate publication-quality figures for the paper:
  1. Lane Token Activation Map (fig_lane_activation.pdf)
  2. Time-Attention Refinement Diagram (fig_time_attention.pdf)

Loads the best MTR-Lite checkpoint on CPU, picks a good validation scene,
extracts attention maps, and produces both figures.

Usage:
    cd /home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper
    conda run -n yolov8 python scripts/generate_lane_time_figures.py
"""

import os
import sys
import pickle

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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import rcParams

from training.lightning_module import MTRLiteModule
from data.polyline_dataset import PolylineDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "/mnt/hdd12t/outputs/mtr_lite/checkpoints/"
    "mtr_lite-epoch=44-val/minADE@6=2.670.ckpt"
)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "mtr_lite.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")

DEVICE = "cpu"

# Scene indices to try (want an intersection with clear lane structure)
CANDIDATE_SCENE_INDICES = [5, 15, 25, 35, 45, 55, 0, 10, 20, 30]


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.5,
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(config_path, checkpoint_path, device="cpu"):
    """Load trained MTR-Lite from a Lightning checkpoint."""
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load Lightning module
    module = MTRLiteModule.load_from_checkpoint(
        checkpoint_path,
        model_cfg=cfg["model"],
        training_cfg=cfg["training"],
        loss_cfg=cfg["loss"],
        map_location=device,
    )
    module.eval()
    module.to(device)
    model = module.model
    print(f"  Model loaded successfully on {device}")
    return model, cfg


# ---------------------------------------------------------------------------
# Scene selection: pick one with rich lane structure
# ---------------------------------------------------------------------------
def score_scene(sample):
    """Heuristic score: prefer scenes with many valid lanes and agents."""
    if sample is None:
        return -1
    n_lanes = sample["map_mask"].sum().item()
    n_agents = sample["agent_mask"].sum().item()
    # Favour intersection-like scenes (many lanes, many agents)
    return n_lanes * 2 + n_agents


def pick_best_scene(dataset, candidate_indices):
    """Try candidate indices and return the best one by heuristic score."""
    best_idx = None
    best_score = -1
    best_sample = None
    for idx in candidate_indices:
        if idx >= len(dataset):
            continue
        sample = dataset[idx]
        s = score_scene(sample)
        print(f"  Scene {idx}: score={s:.0f} "
              f"(lanes={sample['map_mask'].sum().item() if sample else 0}, "
              f"agents={sample['agent_mask'].sum().item() if sample else 0})")
        if s > best_score:
            best_score = s
            best_idx = idx
            best_sample = sample
    print(f"  => Selected scene index {best_idx} (score={best_score:.0f})")
    return best_idx, best_sample


# ---------------------------------------------------------------------------
# Forward pass with attention capture
# ---------------------------------------------------------------------------
def run_inference(model, sample, device="cpu"):
    """Run a single sample through the model with attention capture."""
    # Build batch of size 1
    batch = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0).to(device)
        else:
            batch[key] = val

    with torch.no_grad():
        output = model(batch, capture_attention=True)

    return output, batch


# ---------------------------------------------------------------------------
# Figure 1: Lane Token Activation Map
# ---------------------------------------------------------------------------
def generate_lane_activation_figure(
    output, batch, sample, save_path, bev_range=55.0, top_k_sidebar=10,
):
    """Generate BEV lane activation map with sidebar bar chart."""
    # ---- Extract attention data ----
    attn_maps = output["attention_maps"]
    # decoder_map_attentions is [target][layer] = (B, nhead, K, M)
    # We want target=0 (ego), cumulative across all layers, winning mode
    # Get NMS-selected mode indices for target 0
    nms_indices = output["nms_indices"]  # (B, T, M_out)
    winning_intention_idx = nms_indices[0, 0, 0].item()  # best mode for target 0

    # Accumulate map attention across all decoder layers for target 0
    target_decoder_map_attns = attn_maps.decoder_map_attentions[0]  # list of (B, nhead, K, M) per layer
    num_layers = len(target_decoder_map_attns)
    M = batch["map_mask"].shape[1]

    cumulative_map_attn = np.zeros(M, dtype=np.float32)
    for layer_i in range(num_layers):
        layer_attn = target_decoder_map_attns[layer_i]  # (B, nhead, K, M)
        # Pick batch=0, winning intention query, average over heads
        attn_mode = layer_attn[0, :, winning_intention_idx, :]  # (nhead, M)
        attn_avg = attn_mode.cpu().numpy().mean(axis=0)  # (M,)
        cumulative_map_attn += attn_avg

    # ---- Lane centerlines and masks ----
    lane_cl_bev = sample["lane_centerlines_bev"].numpy()  # (M, 20, 2)
    map_mask = sample["map_mask"].numpy()  # (M,)

    # ---- Trajectory data ----
    # History: agent 0 (ego), positions are first 2 dims of polyline features
    agent_poly = sample["agent_polylines"].numpy()  # (A, 11, 29)
    ego_history = agent_poly[0, :, :2]  # (11, 2)
    ego_history_valid = sample["agent_valid"].numpy()[0]  # (11,)
    ego_history = ego_history[ego_history_valid]

    # GT future for target 0
    gt_future = sample["target_future"].numpy()[0]  # (80, 2)
    gt_future_valid = sample["target_future_valid"].numpy()[0]  # (80,)
    gt_future_plot = gt_future[gt_future_valid]

    # Predicted trajectories (all NMS-selected modes for target 0)
    pred_trajs = output["trajectories"][0, 0].cpu().numpy()  # (M_out, 80, 2)
    pred_scores = output["scores"][0, 0].cpu().numpy()  # (M_out,)

    # Lane labels
    lane_ids = sample.get("lane_ids", [])
    lane_labels = [f"Lane {lid}" if not isinstance(lid, int) else f"Lane {lid}"
                   for lid in lane_ids]
    # Pad to M entries
    while len(lane_labels) < M:
        lane_labels.append(f"Lane {len(lane_labels)}")

    # ---- Build figure ----
    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.12)
    ax_bev = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # Normalize attention for coloring
    valid_attn = cumulative_map_attn[map_mask.astype(bool)]
    if len(valid_attn) > 0 and valid_attn.max() > 0:
        norm = mcolors.Normalize(vmin=0, vmax=valid_attn.max())
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)

    cmap_name = "RdYlBu_r"
    colormap = plt.get_cmap(cmap_name)

    # Draw lanes
    for i in range(M):
        if not map_mask[i]:
            continue
        pts = lane_cl_bev[i]
        attn = cumulative_map_attn[i]
        color = colormap(norm(attn))
        width = float(1.0 + 4.0 * norm(attn))

        ax_bev.plot(pts[:, 0], pts[:, 1], color=color, linewidth=width,
                    alpha=0.85, zorder=3, solid_capstyle="round")

        # Direction arrow at midpoint
        mid = len(pts) // 2
        if mid < len(pts) - 1:
            dx = pts[mid + 1, 0] - pts[mid, 0]
            dy = pts[mid + 1, 1] - pts[mid, 1]
            arrow_len = np.sqrt(dx**2 + dy**2)
            if arrow_len > 0.3:
                ax_bev.annotate(
                    "", xy=(pts[mid, 0] + dx * 0.4, pts[mid, 1] + dy * 0.4),
                    xytext=(pts[mid, 0], pts[mid, 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    zorder=4,
                )

    # Draw history
    if len(ego_history) > 0:
        ax_bev.plot(ego_history[:, 0], ego_history[:, 1], "-o",
                    color="#1565C0", linewidth=2.0, markersize=2.5,
                    label="History", zorder=6)

    # Draw GT future
    if len(gt_future_plot) > 0:
        ax_bev.plot(gt_future_plot[:, 0], gt_future_plot[:, 1], "--",
                    color="#2E7D32", linewidth=2.0,
                    label="Ground Truth", zorder=6)

    # Draw predicted trajectories (top 3 by score)
    score_order = np.argsort(pred_scores)[::-1]
    pred_colors = ["#E53935", "#F57C00", "#FDD835"]
    for rank, k in enumerate(score_order[:3]):
        alpha = 0.9 if rank == 0 else 0.5
        lw = 2.0 if rank == 0 else 1.2
        label = "Prediction (best)" if rank == 0 else (
            "Prediction (alt.)" if rank == 1 else None)
        ax_bev.plot(pred_trajs[k, :, 0], pred_trajs[k, :, 1], "-",
                    color=pred_colors[rank], linewidth=lw, alpha=alpha,
                    label=label, zorder=7)

    # Ego marker at origin
    ax_bev.plot(0, 0, "^", color="#0D47A1", markersize=10, zorder=8,
                label="Ego Vehicle")

    ax_bev.set_xlim(-bev_range, bev_range)
    ax_bev.set_ylim(-bev_range, bev_range)
    ax_bev.set_xlabel("Lateral (m)")
    ax_bev.set_ylabel("Longitudinal (m)")
    ax_bev.set_aspect("equal")
    ax_bev.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax_bev.set_title("Lane-Token Activation Map", fontweight="bold")

    # Colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_bev, label="Cumulative Attention",
                        shrink=0.6, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=8)

    # ---- Sidebar bar chart: top-K lanes ----
    valid_indices = np.where(map_mask.astype(bool))[0]
    valid_attns = cumulative_map_attn[valid_indices]
    order = np.argsort(valid_attns)[::-1][:top_k_sidebar]

    bar_indices = valid_indices[order]
    bar_values = valid_attns[order]
    bar_labels_list = [lane_labels[i] for i in bar_indices]
    bar_colors = [colormap(norm(v)) for v in bar_values]

    y_pos = np.arange(len(bar_values))
    ax_bar.barh(y_pos, bar_values, color=bar_colors, alpha=0.9,
                edgecolor="0.3", linewidth=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels_list, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Cumulative Attention")
    ax_bar.set_title("Top-10 Lanes", fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.savefig(save_path, format="pdf", dpi=300)
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Time-Attention Refinement Diagram
# ---------------------------------------------------------------------------
# Token type colors
COLOR_AGENT = "#1565C0"
COLOR_MAP = "#43A047"
COLOR_EGO = "#E53935"


def generate_time_attention_figure(
    output, batch, sample, save_path, top_k=10,
):
    """Generate 4-panel decoder-layer attention refinement diagram."""
    attn_maps = output["attention_maps"]
    nms_indices = output["nms_indices"]  # (B, T, M_out)
    winning_intention_idx = nms_indices[0, 0, 0].item()

    # decoder attentions for target 0: list of (B, nhead, K, A) and (B, nhead, K, M)
    target_agent_attns = attn_maps.decoder_agent_attentions[0]  # list per layer
    target_map_attns = attn_maps.decoder_map_attentions[0]

    num_layers = len(target_agent_attns)

    # Masks
    agent_mask = sample["agent_mask"].numpy()  # (A,)
    map_mask = sample["map_mask"].numpy()       # (M,)

    # Build labels
    agent_ids = sample.get("agent_ids", [])
    lane_ids = sample.get("lane_ids", [])
    A = agent_mask.shape[0]
    M = map_mask.shape[0]

    agent_labels = []
    for i in range(A):
        if i < len(agent_ids):
            if i == 0:
                agent_labels.append("Ego")
            else:
                agent_labels.append(f"Veh_{i}")
        else:
            agent_labels.append(f"A_{i}")

    map_labels = []
    for i in range(M):
        if i < len(lane_ids):
            map_labels.append(f"Lane_{i}")
        else:
            map_labels.append(f"M_{i}")

    # ---- Collect per-layer combined ranked tokens ----
    # For each layer, merge agent and map attention into a single ranked list
    layer_data = []
    global_max_attn = 0.0

    for layer_i in range(num_layers):
        # Agent attention: (B, nhead, K, A)
        a_attn = target_agent_attns[layer_i][0, :, winning_intention_idx, :]  # (nhead, A)
        a_weights = a_attn.cpu().numpy().mean(axis=0)  # (A,)
        a_weights = a_weights * agent_mask.astype(np.float32)

        # Map attention: (B, nhead, K, M)
        m_attn = target_map_attns[layer_i][0, :, winning_intention_idx, :]  # (nhead, M)
        m_weights = m_attn.cpu().numpy().mean(axis=0)  # (M,)
        m_weights = m_weights * map_mask.astype(np.float32)

        # Combine into a single list: (label, value, type)
        entries = []
        for i in range(A):
            if agent_mask[i]:
                entries.append((agent_labels[i], a_weights[i], "agent"))
        for i in range(M):
            if map_mask[i]:
                entries.append((map_labels[i], m_weights[i], "map"))

        # Sort descending by attention
        entries.sort(key=lambda x: -x[1])
        entries = entries[:top_k]

        # Track global max for consistent scale
        if entries:
            local_max = entries[0][1]
            if local_max > global_max_attn:
                global_max_attn = local_max

        layer_data.append(entries)

    # ---- Build figure: 4 panels horizontal ----
    fig, axes = plt.subplots(1, num_layers, figsize=(4.2 * num_layers, 5.5),
                             sharey=False)
    if num_layers == 1:
        axes = [axes]

    x_limit = global_max_attn * 1.25 if global_max_attn > 0 else 0.1

    for layer_i, (entries, ax) in enumerate(zip(layer_data, axes)):
        labels = [e[0] for e in entries]
        values = [e[1] for e in entries]
        types = [e[2] for e in entries]

        colors = []
        for lbl, _, t in entries:
            if lbl == "Ego":
                colors.append(COLOR_EGO)
            elif t == "agent":
                colors.append(COLOR_AGENT)
            else:
                colors.append(COLOR_MAP)

        y_pos = np.arange(len(values))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85,
                       edgecolor="0.3", linewidth=0.4, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, x_limit)
        ax.set_xlabel("Attention Weight", fontsize=9)
        ax.set_title(f"Decoder Layer {layer_i + 1}", fontweight="bold", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value annotations on bars
        for bar, val in zip(bars, values):
            if val > x_limit * 0.02:
                ax.text(bar.get_width() + x_limit * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=7, color="0.3")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_EGO, edgecolor="0.3", label="Ego Vehicle"),
        Patch(facecolor=COLOR_AGENT, edgecolor="0.3", label="Other Agents"),
        Patch(facecolor=COLOR_MAP, edgecolor="0.3", label="Lane Tokens"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Decoder Attention Refinement Across Layers",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_publication_style()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load model
    model, cfg = load_model(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)

    # Create dataset (validation split)
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

    # Pick the best scene
    print("\nEvaluating candidate scenes...")
    scene_idx, sample = pick_best_scene(dataset, CANDIDATE_SCENE_INDICES)

    if sample is None:
        print("ERROR: Could not find a valid scene. Trying sequential search...")
        for idx in range(min(100, len(dataset))):
            sample = dataset[idx]
            if sample is not None and score_scene(sample) > 30:
                scene_idx = idx
                print(f"  Found scene {idx} with score {score_scene(sample)}")
                break
        if sample is None:
            print("FATAL: No valid scene found.")
            return

    # Run inference
    print(f"\nRunning inference on scene {scene_idx} (CPU, may take a moment)...")
    output, batch = run_inference(model, sample, device=DEVICE)

    # Print some diagnostics
    pred_trajs = output["trajectories"]  # (B, T, M_out, 80, 2)
    print(f"  Predictions shape: {pred_trajs.shape}")
    print(f"  Scores: {output['scores'][0, 0].cpu().numpy()}")
    n_decoder_layers = len(output['attention_maps'].decoder_agent_attentions[0])
    print(f"  Decoder layers with attention: {n_decoder_layers}")

    # Generate Figure 1: Lane Token Activation Map
    print("\nGenerating Figure 1: Lane Token Activation Map...")
    lane_fig_path = os.path.join(OUTPUT_DIR, "fig_lane_activation.pdf")
    generate_lane_activation_figure(output, batch, sample, lane_fig_path)

    # Generate Figure 2: Time-Attention Refinement Diagram
    print("\nGenerating Figure 2: Time-Attention Refinement Diagram...")
    time_fig_path = os.path.join(OUTPUT_DIR, "fig_time_attention.pdf")
    generate_time_attention_figure(output, batch, sample, time_fig_path)

    print("\n" + "=" * 60)
    print("Done! Generated figures:")
    print(f"  1. {lane_fig_path}")
    print(f"  2. {time_fig_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
