"""Temporal attention animation: watch the full 9-second scenario unfold.

For each timestep, shows:
  - All agents moving in bird's-eye view (fixed world camera)
  - Lane network (static)
  - Attention heatmap: which agents and lanes the model attends to
  - Ego's predicted future trajectory
  - Ground truth future
  - Timestamp and attention statistics

Model re-runs at key anchor points (every 0.5s) to refresh attention.
Between anchors, positions update but attention holds from last anchor.
"""

import sys
import os
import math
import pickle
import io

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention


# ── helpers ──────────────────────────────────────────────────────────────────

def get_agent_world_positions(scene, frame_idx):
    """Get all agent positions in world coordinates at a given frame."""
    positions = {}  # obj_idx -> (x, y)
    for i, obj in enumerate(scene["objects"]):
        if frame_idx < len(obj["valid"]) and obj["valid"][frame_idx]:
            positions[i] = (
                float(obj["position"][frame_idx]["x"]),
                float(obj["position"][frame_idx]["y"]),
            )
    return positions


def get_lane_world_coords(scene):
    """Get all lane centerlines in world coordinates."""
    lanes = {}
    for lid, pts in scene["lane_graph"]["lanes"].items():
        if pts is not None and len(pts) >= 2:
            lanes[lid] = pts[:, :2].astype(np.float64)
    return lanes


def get_ego_trajectory(scene, start_frame=0, end_frame=91):
    """Get ego vehicle's full trajectory in world coords."""
    av_idx = scene["av_idx"]
    ego_obj = scene["objects"][av_idx]
    traj = []
    for f in range(start_frame, min(end_frame, len(ego_obj["position"]))):
        if ego_obj["valid"][f]:
            traj.append((
                f,
                float(ego_obj["position"][f]["x"]),
                float(ego_obj["position"][f]["y"]),
            ))
    return traj


def compute_view_bounds(scene, padding=15.0):
    """Compute fixed BEV view bounds that cover the full scenario."""
    all_x, all_y = [], []
    av_idx = scene["av_idx"]
    ego_obj = scene["objects"][av_idx]
    for f in range(len(ego_obj["position"])):
        if ego_obj["valid"][f]:
            all_x.append(float(ego_obj["position"][f]["x"]))
            all_y.append(float(ego_obj["position"][f]["y"]))

    # Also include nearby agents at mid-frame
    mid = len(ego_obj["position"]) // 2
    ego_mid = np.array([float(ego_obj["position"][mid]["x"]),
                        float(ego_obj["position"][mid]["y"])])
    for obj in scene["objects"]:
        if mid < len(obj["valid"]) and obj["valid"][mid]:
            x = float(obj["position"][mid]["x"])
            y = float(obj["position"][mid]["y"])
            if abs(x - ego_mid[0]) < 60 and abs(y - ego_mid[1]) < 60:
                all_x.append(x)
                all_y.append(y)

    cx = (min(all_x) + max(all_x)) / 2
    cy = (min(all_y) + max(all_y)) / 2
    half = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2 + padding
    half = max(half, 40)  # at least 40m
    return cx - half, cx + half, cy - half, cy + half


AGENT_TYPE_COLORS = {
    "vehicle": "#1976D2",
    "pedestrian": "#FF8F00",
    "cyclist": "#2E7D32",
    "other": "#7B1FA2",
    "unknown": "#757575",
}
AGENT_TYPE_MARKERS = {
    "vehicle": "s",
    "pedestrian": "o",
    "cyclist": "D",
    "other": "^",
    "unknown": "x",
}


def render_temporal_frame(
    scene,
    frame_idx,
    lanes_world,
    view_bounds,
    # attention data (from last model run, may be None)
    attention_agent_weights=None,  # dict: obj_idx -> attention_weight
    attention_lane_weights=None,   # dict: lane_id -> attention_weight
    # prediction data
    pred_trajs_world=None,  # (K, T, 2) predicted trajectories in world coords
    pred_scores=None,       # (K,)
    gt_future_world=None,   # (T, 2) ground truth in world coords
    gt_future_valid=None,   # (T,)
    # ego trail
    ego_trail=None,  # list of (x, y) past ego positions
    # labels
    timestamp_s=None,
    anchor_label=None,
    agent_attn_pct=None,
    map_attn_pct=None,
):
    """Render one frame of the temporal animation in fixed world coordinates."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    xmin, xmax, ymin, ymax = view_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("#2B2B2B")

    # ── 1. Draw lanes ────────────────────────────────────────────────────
    for lid, pts in lanes_world.items():
        attn_w = 0.0
        if attention_lane_weights and lid in attention_lane_weights:
            attn_w = attention_lane_weights[lid]

        if attn_w > 0.001:
            # Color by attention: low=cool blue, high=warm red
            color = plt.cm.hot(min(attn_w * 15, 1.0))  # scale up for visibility
            lw = 1.5 + attn_w * 30
            alpha = 0.6 + min(attn_w * 5, 0.4)
        else:
            color = "#555555"
            lw = 0.8
            alpha = 0.4

        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=alpha, zorder=2)

    # ── 2. Draw agent attention halos (behind agents) ────────────────────
    if attention_agent_weights:
        max_attn = max(attention_agent_weights.values()) if attention_agent_weights else 1.0
        for obj_idx, attn_w in attention_agent_weights.items():
            if attn_w < 0.001:
                continue
            obj = scene["objects"][obj_idx]
            if frame_idx >= len(obj["valid"]) or not obj["valid"][frame_idx]:
                continue
            x = float(obj["position"][frame_idx]["x"])
            y = float(obj["position"][frame_idx]["y"])
            # Glow circle
            radius = 2.0 + attn_w / (max_attn + 1e-8) * 8.0
            circle = plt.Circle(
                (x, y), radius, color=plt.cm.hot(min(attn_w * 12, 1.0)),
                alpha=0.35, zorder=3,
            )
            ax.add_patch(circle)

    # ── 3. Draw all agents at current positions ──────────────────────────
    av_idx = scene["av_idx"]
    for i, obj in enumerate(scene["objects"]):
        if frame_idx >= len(obj["valid"]) or not obj["valid"][frame_idx]:
            continue
        x = float(obj["position"][frame_idx]["x"])
        y = float(obj["position"][frame_idx]["y"])
        agent_type = obj.get("type", "unknown").lower()
        color = AGENT_TYPE_COLORS.get(agent_type, "#757575")
        marker = AGENT_TYPE_MARKERS.get(agent_type, "x")
        size = 80 if i == av_idx else 50

        # Ego vehicle is special
        if i == av_idx:
            ax.plot(x, y, marker="^", color="#FF1744", markersize=14,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=10)
        else:
            ax.plot(x, y, marker=marker, color=color, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    # ── 4. Ego trail ─────────────────────────────────────────────────────
    if ego_trail and len(ego_trail) >= 2:
        trail = np.array(ego_trail)
        n = len(trail)
        for j in range(1, n):
            alpha = 0.15 + 0.6 * (j / n)
            ax.plot(trail[j-1:j+1, 0], trail[j-1:j+1, 1],
                    color="#FF1744", linewidth=2, alpha=alpha, zorder=8)

    # ── 5. Ground truth future ───────────────────────────────────────────
    if gt_future_world is not None and gt_future_valid is not None:
        valid_pts = gt_future_world[gt_future_valid.astype(bool)]
        if len(valid_pts) > 1:
            ax.plot(valid_pts[:, 0], valid_pts[:, 1],
                    color="#00E676", linewidth=2.5, linestyle="--",
                    alpha=0.8, zorder=7, label="Ground Truth")

    # ── 6. Predicted trajectories ────────────────────────────────────────
    if pred_trajs_world is not None and pred_scores is not None:
        top_k = min(3, len(pred_scores))
        sorted_idx = pred_scores.argsort()[-top_k:][::-1]
        pred_colors = ["#FF5252", "#FF9800", "#FFEB3B"]
        for rank, midx in enumerate(sorted_idx):
            traj = pred_trajs_world[midx]
            ax.plot(traj[:, 0], traj[:, 1],
                    color=pred_colors[rank], linewidth=2.0 - rank * 0.4,
                    alpha=0.8 - rank * 0.15, zorder=6)
            # endpoint dot
            ax.plot(traj[-1, 0], traj[-1, 1], "o",
                    color=pred_colors[rank], markersize=5, zorder=6)

    # ── 7. Title & info ──────────────────────────────────────────────────
    title_parts = []
    if timestamp_s is not None:
        title_parts.append(f"t = {timestamp_s:.1f}s / 9.0s")
    if anchor_label:
        title_parts.append(anchor_label)
    ax.set_title("  |  ".join(title_parts), fontsize=13, color="white",
                 fontweight="bold", pad=10)

    # Attention stats text
    if agent_attn_pct is not None and map_attn_pct is not None:
        stats_text = f"Agent Attn: {agent_attn_pct:.0f}%  |  Map Attn: {map_attn_pct:.0f}%"
        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11, color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8))

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="^", color="#FF1744", linestyle="None",
                   markersize=10, label="Ego"),
        plt.Line2D([0], [0], marker="s", color="#1976D2", linestyle="None",
                   markersize=8, label="Vehicle"),
        plt.Line2D([0], [0], marker="o", color="#FF8F00", linestyle="None",
                   markersize=8, label="Pedestrian"),
        plt.Line2D([0], [0], marker="D", color="#2E7D32", linestyle="None",
                   markersize=8, label="Cyclist"),
        plt.Line2D([0], [0], color="#00E676", linewidth=2, linestyle="--",
                   label="GT Future"),
        plt.Line2D([0], [0], color="#FF5252", linewidth=2, label="Prediction"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              facecolor="#333333", edgecolor="#555555", labelcolor="white",
              framealpha=0.9)

    ax.set_xlabel("X (m)", color="white", fontsize=10)
    ax.set_ylabel("Y (m)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    fig.patch.set_facecolor("#1A1A1A")

    # Convert to PIL
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def bev_to_world(pts_bev, ego_pos, ego_heading_deg):
    """Transform BEV coordinates back to world coordinates."""
    heading_rad = math.radians(ego_heading_deg)
    adjusted = heading_rad - np.pi / 2
    c, s = np.cos(adjusted), np.sin(adjusted)  # note: forward rotation (inverse of _world_to_bev)
    R_inv = np.array([[c, -s], [s, c]], dtype=np.float64)
    ego_xy = np.array(ego_pos, dtype=np.float64)
    return (pts_bev.astype(np.float64) @ R_inv.T + ego_xy).astype(np.float64)


# ── main ─────────────────────────────────────────────────────────────────────

def create_temporal_animation(
    model,
    scene_path,
    output_path,
    anchor_step=5,        # run model every N frames (N/10 seconds)
    display_step=2,       # render every N frames
    device="cpu",
):
    """Create temporal animation for a full scene."""
    with open(scene_path, "rb") as f:
        scene = pickle.load(f)

    n_frames = len(scene["objects"][0]["position"])
    av_idx = scene["av_idx"]

    print(f"  Scene: {n_frames} frames, {len(scene['objects'])} agents, av_idx={av_idx}")

    # Pre-compute static data
    lanes_world = get_lane_world_coords(scene)
    view_bounds = compute_view_bounds(scene, padding=15.0)
    ego_full_traj = get_ego_trajectory(scene)

    # Determine anchor frames (where we run the model)
    # Need at least 11 history frames, so earliest anchor = 10
    anchor_frames = list(range(10, n_frames - 1, anchor_step))
    print(f"  Anchor frames: {anchor_frames} ({len(anchor_frames)} model runs)")

    # Run model at each anchor and store results
    anchor_results = {}
    for a_idx, anchor in enumerate(anchor_frames):
        print(f"  [{a_idx+1}/{len(anchor_frames)}] Model forward at t={anchor} ({anchor/10:.1f}s)...")

        result = extract_scene_attention(
            model=model,
            scene_path=scene_path,
            anchor_frame=anchor,
            history_len=11,
            future_len=min(80, n_frames - 1 - anchor),  # might be shorter at end
            device=device,
        )

        attn_maps = result["attention_maps"]
        agent_data = result["agent_data"]
        map_data = result["map_data"]

        # Extract scene encoder attention (last layer, ego row)
        if attn_maps is not None and attn_maps.scene_attentions:
            last_attn = attn_maps.scene_attentions[-1][0]  # (H, N, N)
            ego_attn = last_attn.mean(dim=0)[0].cpu().numpy()  # (N,)
            agent_attn_raw = ego_attn[:32]
            map_attn_raw = ego_attn[32:96]
        else:
            agent_attn_raw = np.zeros(32)
            map_attn_raw = np.zeros(64)

        # Map attention weights to original entity IDs
        agent_weights = {}
        for slot_idx in range(32):
            if agent_data["agent_mask"][slot_idx]:
                obj_idx = agent_data["agent_ids"][slot_idx]
                agent_weights[obj_idx] = float(agent_attn_raw[slot_idx])

        lane_weights = {}
        for slot_idx in range(64):
            if map_data["map_mask"][slot_idx] and slot_idx < len(map_data["lane_ids"]):
                lid = map_data["lane_ids"][slot_idx]
                lane_weights[lid] = float(map_attn_raw[slot_idx])

        # Prediction trajectories -> world coordinates
        preds = result["predictions"]
        pred_trajs_bev = preds["trajectories"][0, 0].cpu().numpy()  # (K, T, 2)
        pred_scores = preds["scores"][0, 0].cpu().numpy()           # (K,)

        ego_pos = result["ego_pos"]
        ego_heading = result["ego_heading"]
        pred_trajs_world = np.zeros_like(pred_trajs_bev)
        for k in range(len(pred_trajs_bev)):
            pred_trajs_world[k] = bev_to_world(pred_trajs_bev[k], ego_pos, ego_heading)

        # GT future -> world coords
        gt_future_bev = result["batch"]["target_future"][0, 0].cpu().numpy()  # (T, 2)
        gt_future_valid = result["batch"]["target_future_valid"][0, 0].cpu().numpy()
        gt_future_world = bev_to_world(gt_future_bev, ego_pos, ego_heading)

        # Attention percentages
        agent_sum = agent_attn_raw.sum()
        map_sum = map_attn_raw.sum()
        total = agent_sum + map_sum + 1e-8
        agent_pct = agent_sum / total * 100
        map_pct = map_sum / total * 100

        anchor_results[anchor] = {
            "agent_weights": agent_weights,
            "lane_weights": lane_weights,
            "pred_trajs_world": pred_trajs_world,
            "pred_scores": pred_scores,
            "gt_future_world": gt_future_world,
            "gt_future_valid": gt_future_valid,
            "agent_pct": agent_pct,
            "map_pct": map_pct,
        }

    # ── Render all display frames ────────────────────────────────────────
    display_frames_idx = list(range(0, n_frames, display_step))
    print(f"  Rendering {len(display_frames_idx)} display frames...")

    frames = []
    ego_trail = []

    for f_idx in display_frames_idx:
        # Update ego trail
        ego_obj = scene["objects"][av_idx]
        if f_idx < len(ego_obj["valid"]) and ego_obj["valid"][f_idx]:
            ego_trail.append((
                float(ego_obj["position"][f_idx]["x"]),
                float(ego_obj["position"][f_idx]["y"]),
            ))

        # Find nearest anchor (most recent one at or before this frame)
        best_anchor = None
        for a in anchor_frames:
            if a <= f_idx:
                best_anchor = a
            else:
                break

        # Get attention and prediction data
        if best_anchor is not None and best_anchor in anchor_results:
            ar = anchor_results[best_anchor]
            attn_agent = ar["agent_weights"]
            attn_lane = ar["lane_weights"]
            pred_trajs = ar["pred_trajs_world"]
            pred_sc = ar["pred_scores"]
            gt_fut = ar["gt_future_world"]
            gt_val = ar["gt_future_valid"]
            a_pct = ar["agent_pct"]
            m_pct = ar["map_pct"]
            anchor_label = f"Attention @ t={best_anchor/10:.1f}s"
        else:
            attn_agent = None
            attn_lane = None
            pred_trajs = None
            pred_sc = None
            gt_fut = None
            gt_val = None
            a_pct = None
            m_pct = None
            anchor_label = "Pre-history (no model)"

        frame_img = render_temporal_frame(
            scene=scene,
            frame_idx=f_idx,
            lanes_world=lanes_world,
            view_bounds=view_bounds,
            attention_agent_weights=attn_agent,
            attention_lane_weights=attn_lane,
            pred_trajs_world=pred_trajs,
            pred_scores=pred_sc,
            gt_future_world=gt_fut,
            gt_future_valid=gt_val,
            ego_trail=ego_trail.copy(),
            timestamp_s=f_idx / 10.0,
            anchor_label=anchor_label,
            agent_attn_pct=a_pct,
            map_attn_pct=m_pct,
        )
        frames.append(frame_img)

    # ── Save GIF ─────────────────────────────────────────────────────────
    if frames:
        # Add pause on last frame
        durations = [300] * len(frames)  # 300ms per frame
        durations[-1] = 2000  # 2s pause on last frame

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )
        print(f"  GIF saved: {output_path} ({len(frames)} frames, "
              f"{sum(durations)/1000:.1f}s total)")

    return frames


def main():
    import yaml

    checkpoint = "/mnt/hdd12t/outputs/mtr_lite/checkpoints/mtr_lite-epoch=39-val/minADE@6=2.687.ckpt"
    config_path = Path(__file__).resolve().parent.parent / "configs" / "mtr_lite.yaml"
    output_dir = "/tmp/attention_demo/temporal"
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading model on CPU...")
    module = MTRLiteModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    model = module.model

    # Get val scenes
    scene_list_path = cfg["data"]["scene_list"]
    with open(scene_list_path) as f:
        all_scenes = [l.strip() for l in f if l.strip()]
    import random
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * 0.15))
    val_indices = indices[:n_val]
    val_scenes = [all_scenes[i] for i in val_indices if os.path.exists(all_scenes[i])]

    # Find diverse scenes
    scene_picks = {}

    print(f"Scanning {min(100, len(val_scenes))} val scenes for diversity...")
    for s_idx in range(min(100, len(val_scenes))):
        sp = val_scenes[s_idx]
        try:
            with open(sp, "rb") as f:
                sc = pickle.load(f)
            n_obj = len(sc["objects"])
            av_idx = sc["av_idx"]
            n_frames = len(sc["objects"][0]["position"])
            types = set()
            for obj in sc["objects"]:
                if obj.get("valid", [False])[min(10, n_frames - 1)]:
                    types.add(obj.get("type", "unknown").lower())

            # Compute ego displacement (how much ego moves)
            ego_obj = sc["objects"][av_idx]
            if ego_obj["valid"][0] and ego_obj["valid"][-1]:
                p0 = np.array([float(ego_obj["position"][0]["x"]),
                               float(ego_obj["position"][0]["y"])])
                p1 = np.array([float(ego_obj["position"][-1]["x"]),
                               float(ego_obj["position"][-1]["y"])])
                ego_displacement = np.linalg.norm(p1 - p0)
            else:
                ego_displacement = 0

            # Categorize
            has_ped = "pedestrian" in types
            has_cyc = "cyclist" in types
            is_dense = sum(1 for o in sc["objects"]
                          if o.get("valid", [False])[min(10, n_frames - 1)]) > 20
            ego_moves = ego_displacement > 15  # at least 15m over 9s

            if has_ped and "with_pedestrian" not in scene_picks:
                scene_picks["with_pedestrian"] = (sp, f"Pedestrian Interaction ({n_obj} agents)")
            elif has_cyc and "with_cyclist" not in scene_picks:
                scene_picks["with_cyclist"] = (sp, f"Cyclist Interaction ({n_obj} agents)")
            elif is_dense and ego_moves and "dense_moving" not in scene_picks:
                scene_picks["dense_moving"] = (sp, f"Dense Traffic in Motion ({n_obj} agents)")
            elif ego_moves and not is_dense and "general_driving" not in scene_picks:
                scene_picks["general_driving"] = (sp, f"General Driving ({n_obj} agents)")

            if len(scene_picks) >= 3:
                break

        except Exception:
            continue

    if not scene_picks:
        # Fallback: just use the first scene
        scene_picks["default"] = (val_scenes[0], "Default Scene")

    print(f"\nSelected {len(scene_picks)} scenes:")
    for cat, (sp, desc) in scene_picks.items():
        print(f"  {cat}: {desc}")

    # Generate animations
    print(f"\nGenerating temporal animations...")
    for cat, (sp, desc) in scene_picks.items():
        print(f"\n{'='*60}")
        print(f"[{cat}] {desc}")
        print(f"{'='*60}")
        gif_path = f"{output_dir}/temporal_{cat}.gif"
        create_temporal_animation(
            model=model,
            scene_path=sp,
            output_path=gif_path,
            anchor_step=5,   # model every 0.5s
            display_step=2,  # render every 0.2s
            device="cpu",
        )

    print(f"\nAll animations saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
