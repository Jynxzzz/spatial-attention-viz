"""BEV Attention Overlay Visualization.

Renders Scenario Dreamer style BEV scenes with attention heatmap overlays.
This is the core visualization that shows WHAT the model attends to and WHERE
in physical space.

Multi-layer rendering order (bottom to top):
1. Background lanes (gray)
2. Waterflow lanes (colored by topology)
3. Traffic lights (colored squares)
4. Neighbor vehicles (gray rectangles)
5. Attention heatmap (semi-transparent, alpha=0.6)
6. Ego vehicle (blue rectangle)
7. Trajectories (history + GT + pred)
8. Annotations (top-3 attended objects highlighted)
"""

import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Import Scenario Dreamer rendering functions
sys.path.insert(0, '/home/xingnan/projects/scenario-dreamer')
try:
    from scripts.visualize_predictions import (
        draw_waterflow_lanes,
        draw_traffic_lights,
        draw_vehicle,
        draw_neighbors,
    )
    SCENARIO_DREAMER_AVAILABLE = True
except ImportError:
    SCENARIO_DREAMER_AVAILABLE = False
    print("Warning: Scenario Dreamer visualization functions not available. "
          "Using fallback rendering.")

# Import our spatial utilities
from visualization.spatial_utils import normalize_heatmap


# Color palette (matching Scenario Dreamer conventions)
COLOR_EGO = '#0D47A1'
COLOR_HISTORY = '#1E88E5'
COLOR_GT_FUTURE = '#2E7D32'
COLOR_PRED = '#E53935'
COLOR_NEIGHBOR = '#78909C'
COLOR_ATTENTION_TOP = '#00FFFF'  # Cyan for highlighting top attended objects


def _fallback_draw_lanes(ax, scene, ego_pos, ego_heading, radius=60):
    """Fallback lane rendering when Scenario Dreamer is not available."""
    from data.map_features import _world_to_bev

    lane_graph = scene["lane_graph"]

    for lane_id, pts in lane_graph["lanes"].items():
        if pts is None or len(pts) < 2:
            continue

        pts_bev = _world_to_bev(pts[:, :2].astype(np.float64), ego_pos, ego_heading)

        # Only draw lanes within radius
        if np.abs(pts_bev).max() > radius:
            continue

        ax.plot(pts_bev[:, 0], pts_bev[:, 1], color='#B0BEC5',
                linewidth=1.0, alpha=0.5)


def _fallback_draw_vehicles(ax, scene, ego_pos, ego_heading, frame_idx, radius=60):
    """Fallback vehicle rendering."""
    from data.map_features import _world_to_bev

    av_idx = scene["av_idx"]
    ego_world = np.array([
        float(scene["objects"][av_idx]["position"][frame_idx]["x"]),
        float(scene["objects"][av_idx]["position"][frame_idx]["y"]),
    ])

    for i, obj in enumerate(scene["objects"]):
        if i == av_idx or not obj["valid"][frame_idx]:
            continue

        pos = obj["position"][frame_idx]
        world_pt = np.array([float(pos["x"]), float(pos["y"])])

        if np.linalg.norm(world_pt - ego_world) > radius:
            continue

        bev_pt = _world_to_bev(world_pt.reshape(1, 2), ego_pos, ego_heading)[0]

        ax.plot(bev_pt[0], bev_pt[1], 's', color=COLOR_NEIGHBOR,
                markersize=4, alpha=0.6)


def _draw_ego_vehicle(ax, color=COLOR_EGO, alpha=0.9):
    """Draw ego vehicle at origin (0, 0) facing forward."""
    length, width = 4.5, 2.0

    rect = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.2",
        facecolor=color, edgecolor='black',
        linewidth=1.0, alpha=alpha, zorder=12
    )

    ax.add_patch(rect)


def _draw_trajectories(ax, batch, target_idx=0, pred_traj=None):
    """Draw history, GT future, and predicted trajectories."""
    # Extract history from agent polylines
    agent_polylines = batch['agent_polylines']  # (A, 11, 29)
    target_agent_indices = batch['target_agent_indices']  # (T,)

    if target_idx >= len(target_agent_indices):
        return

    agent_slot = target_agent_indices[target_idx].item()
    if agent_slot < 0:
        return

    # History trajectory (pos is first 2 dims)
    history = agent_polylines[agent_slot, :, 0:2].cpu().numpy()  # (11, 2)

    # Draw history
    ax.plot(history[:, 0], history[:, 1], '-o', color=COLOR_HISTORY,
            linewidth=2.0, markersize=3, alpha=0.8, label='History', zorder=11)

    # Draw GT future
    target_future = batch['target_future']  # (T, 80, 2)
    target_future_valid = batch['target_future_valid']  # (T, 80)

    gt_future = target_future[target_idx].cpu().numpy()  # (80, 2)
    gt_valid = target_future_valid[target_idx].cpu().numpy()  # (80,)

    if gt_valid.any():
        gt_future_masked = gt_future[gt_valid]
        ax.plot(gt_future_masked[:, 0], gt_future_masked[:, 1], '-',
                color=COLOR_GT_FUTURE, linewidth=2.0, alpha=0.8,
                label='GT Future', zorder=11)

    # Draw prediction if provided
    if pred_traj is not None:
        pred_traj = pred_traj.cpu().numpy() if hasattr(pred_traj, 'cpu') else pred_traj
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], '--',
                color=COLOR_PRED, linewidth=2.0, alpha=0.8,
                label='Prediction', zorder=11)


def _highlight_top_attended(ax, top_objects, spatial_bookkeeper):
    """Draw cyan circles and labels for top attended objects."""
    for i, (token_idx, attn_weight, obj_type, info) in enumerate(top_objects):
        if obj_type == 'agent':
            pos = info['pos']
            ax.scatter(pos[0], pos[1], s=200, facecolors='none',
                      edgecolors=COLOR_ATTENTION_TOP, linewidths=2.5, zorder=15)

            label_text = f"#{i+1}: Agent {info['agent_idx']}\nAttn: {attn_weight:.3f}"
            ax.text(pos[0] + 3, pos[1] + 3, label_text,
                   color=COLOR_ATTENTION_TOP, fontsize=9, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                   zorder=16)

        elif obj_type == 'map':
            # Highlight lane centerline
            centerline = info['centerline']
            if len(centerline) > 0:
                # Draw highlight along centerline
                ax.plot(centerline[:, 0], centerline[:, 1],
                       color=COLOR_ATTENTION_TOP, linewidth=3.0, alpha=0.8, zorder=15)

                # Label at midpoint
                mid_idx = len(centerline) // 2
                mid_pos = centerline[mid_idx]
                label_text = f"#{i+1}: Lane\nAttn: {attn_weight:.3f}"
                ax.text(mid_pos[0] + 2, mid_pos[1] + 2, label_text,
                       color=COLOR_ATTENTION_TOP, fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                       zorder=16)


def render_attention_overlay_bev(
    scene,
    batch,
    attention_maps,
    spatial_bookkeeper,
    query_agent_idx=0,
    layer_idx=-1,
    ego_pos=None,
    ego_heading=None,
    anchor_frame=10,
    radius=60,
    save_path=None,
    pred_traj=None,
    show_top_k=3,
    title=None,
):
    """Render BEV scene with attention heatmap overlay.

    Args:
        scene: Scenario dict loaded from pkl (for lane graph, traffic lights)
        batch: Dataset batch dict from PolylineDataset
        attention_maps: AttentionMaps object from model forward pass
        spatial_bookkeeper: SpatialTokenBookkeeper with BEV coordinates
        query_agent_idx: Which agent's attention to visualize (default 0 = ego)
        layer_idx: Which scene encoder layer to visualize (default -1 = last)
        ego_pos: (2,) tuple of ego world position at anchor frame
        ego_heading: float, ego heading in degrees at anchor frame
        anchor_frame: Frame index for scene rendering (default 10)
        radius: BEV radius in meters (default 60)
        save_path: Path to save figure (if None, displays instead)
        pred_traj: Optional (80, 2) predicted trajectory to overlay
        show_top_k: Number of top attended objects to highlight (default 3)
        title: Optional custom title

    Returns:
        fig, ax: matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=(14, 14))

    # Layer 0-1: Background lanes and environment
    if SCENARIO_DREAMER_AVAILABLE and ego_pos is not None and ego_heading is not None:
        draw_waterflow_lanes(ax, scene, ego_pos, ego_heading, radius)
        draw_traffic_lights(ax, scene, ego_pos, ego_heading, anchor_frame)
        draw_neighbors(ax, scene, ego_pos, ego_heading, anchor_frame, max_dist=radius)
    else:
        # Fallback rendering
        if ego_pos is not None and ego_heading is not None:
            _fallback_draw_lanes(ax, scene, ego_pos, ego_heading, radius)
            _fallback_draw_vehicles(ax, scene, ego_pos, ego_heading, anchor_frame, radius)

    # Layer 2: Attention heatmap overlay
    # Extract attention from specified layer
    scene_attn = attention_maps.scene_attentions[layer_idx][0]  # (nhead, N, N)

    # Get attention FROM query_agent TO all other tokens
    if query_agent_idx in spatial_bookkeeper.agent_token_ranges:
        query_token_idx = spatial_bookkeeper.agent_token_ranges[query_agent_idx][0]

        # Average across heads
        attn_weights = scene_attn.mean(dim=0)[query_token_idx].cpu().numpy()  # (N,)

        # Convert to spatial heatmap
        heatmap, extent = spatial_bookkeeper.get_spatial_map(
            attn_weights, resolution=0.5, radius=radius
        )

        # Normalize heatmap
        heatmap_norm = normalize_heatmap(heatmap, percentile=95)

        # Overlay heatmap with transparency
        if heatmap_norm.max() > 0:
            im = ax.imshow(
                heatmap_norm, extent=extent, origin='lower',
                cmap='magma', alpha=0.6, interpolation='bilinear',
                vmin=0, vmax=1.0, zorder=5
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight (normalized)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        # Get top attended objects
        top_objects = spatial_bookkeeper.get_top_attended_objects(
            attn_weights, top_k=show_top_k
        )

    # Layer 3: Ego vehicle (on top of heatmap)
    _draw_ego_vehicle(ax, color=COLOR_EGO, alpha=0.9)

    # Layer 4: Trajectories
    _draw_trajectories(ax, batch, target_idx=0, pred_traj=pred_traj)

    # Layer 5: Annotations - highlight top attended objects
    if query_agent_idx in spatial_bookkeeper.agent_token_ranges and 'top_objects' in locals():
        _highlight_top_attended(ax, top_objects, spatial_bookkeeper)

    # Formatting
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_xlabel('X (meters, BEV)', fontsize=14)
    ax.set_ylabel('Y (meters, BEV)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Title
    if title is None:
        title = f'BEV Attention Overlay (Layer {layer_idx}, Agent {query_agent_idx})'
    ax.set_title(title, fontsize=16, weight='bold', pad=20)

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention overlay to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def render_multi_layer_attention(
    scene,
    batch,
    attention_maps,
    spatial_bookkeeper,
    query_agent_idx=0,
    layer_indices=None,
    ego_pos=None,
    ego_heading=None,
    anchor_frame=10,
    radius=60,
    save_path=None,
):
    """Render attention overlays for multiple layers in a grid.

    Args:
        scene, batch, spatial_bookkeeper: same as render_attention_overlay_bev
        attention_maps: AttentionMaps object
        query_agent_idx: Agent to visualize
        layer_indices: List of layer indices to visualize (default: all layers)
        Other args: same as render_attention_overlay_bev

    Returns:
        fig with subplots
    """
    if layer_indices is None:
        # Use all scene encoder layers
        layer_indices = list(range(len(attention_maps.scene_attentions)))

    n_layers = len(layer_indices)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))

    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer_idx in enumerate(layer_indices):
        ax = axes[i]

        # Simplified rendering for subplots
        plt.sca(ax)

        # Draw lanes (simplified)
        if ego_pos is not None and ego_heading is not None:
            _fallback_draw_lanes(ax, scene, ego_pos, ego_heading, radius)

        # Extract and render attention
        scene_attn = attention_maps.scene_attentions[layer_idx][0]

        if query_agent_idx in spatial_bookkeeper.agent_token_ranges:
            query_token_idx = spatial_bookkeeper.agent_token_ranges[query_agent_idx][0]
            attn_weights = scene_attn.mean(dim=0)[query_token_idx].cpu().numpy()

            heatmap, extent = spatial_bookkeeper.get_spatial_map(
                attn_weights, resolution=0.5, radius=radius
            )

            heatmap_norm = normalize_heatmap(heatmap, percentile=95)

            if heatmap_norm.max() > 0:
                ax.imshow(
                    heatmap_norm, extent=extent, origin='lower',
                    cmap='magma', alpha=0.6, interpolation='bilinear',
                    vmin=0, vmax=1.0, zorder=5
                )

        # Draw ego vehicle
        _draw_ego_vehicle(ax, color=COLOR_EGO, alpha=0.9)

        # Formatting
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_aspect('equal')
        ax.set_title(f'Layer {layer_idx}', fontsize=12, weight='bold')

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-layer attention to: {save_path}")
        plt.close(fig)

    return fig
