#!/usr/bin/env python3
"""Simplified test for spatial attention visualization.

Creates a synthetic BEV scene with attention overlay to demonstrate the full pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from visualization.spatial_utils import gaussian_splat_2d, paint_polyline_2d


def create_synthetic_scene():
    """Create a synthetic intersection scene."""
    # Intersection at origin
    # 4 vehicles: ego at (0, -20), oncoming at (0, 15), left at (-15, 0), right at (20, 5)

    agents = [
        {'pos': np.array([0.0, -20.0]), 'heading': np.pi/2, 'color': 'blue', 'label': 'Ego'},  # Ego
        {'pos': np.array([0.0, 15.0]), 'heading': -np.pi/2, 'color': 'red', 'label': 'Oncoming'},  # Oncoming
        {'pos': np.array([-15.0, 0.0]), 'heading': 0.0, 'color': 'gray', 'label': 'Left'},  # Left
        {'pos': np.array([20.0, 5.0]), 'heading': np.pi, 'color': 'gray', 'label': 'Right'},  # Right
    ]

    # Lanes: straight N-S, E-W
    lanes = [
        {'centerline': np.array([[-5.0, -30.0], [-5.0, 30.0]]), 'label': 'Lane 1 (N-S left)'},
        {'centerline': np.array([[5.0, 30.0], [5.0, -30.0]]), 'label': 'Lane 2 (S-N right)'},
        {'centerline': np.array([[-30.0, -5.0], [30.0, -5.0]]), 'label': 'Lane 3 (W-E)'},
        {'centerline': np.array([[30.0, 5.0], [-30.0, 5.0]]), 'label': 'Lane 4 (E-W)'},
    ]

    # Attention weights (synthetic but realistic)
    # Ego vehicle is turning left, so attends to:
    # - Oncoming vehicle (conflict check): 0.85
    # - Left vehicle (gap check): 0.45
    # - Right vehicle (low priority): 0.15
    # - Ego itself: 0.30
    agent_attention = np.array([0.30, 0.85, 0.45, 0.15])

    # Lane attention:
    # - Current lane (N-S left): 0.60
    # - Target lane (E-W): 0.75
    # - Other lanes: 0.20, 0.10
    lane_attention = np.array([0.60, 0.20, 0.75, 0.10])

    return agents, lanes, agent_attention, lane_attention


def draw_vehicle(ax, pos, heading, color='gray', alpha=0.8):
    """Draw oriented vehicle rectangle."""
    length, width = 4.5, 2.0

    # Create rectangle in local frame
    rect = patches.Rectangle((-length/2, -width/2), length, width,
                             linewidth=1.5, edgecolor='black', facecolor=color, alpha=alpha)

    # Rotate and translate
    import matplotlib.transforms as transforms
    t = transforms.Affine2D().rotate(heading).translate(pos[0], pos[1]) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


def main():
    print("Generating synthetic BEV attention overlay...")

    # Create scene
    agents, lanes, agent_attn, lane_attn = create_synthetic_scene()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    radius = 35

    # ========== Panel 1: BEV Scene (no attention) ==========
    ax = axes[0]
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)
    ax.set_title('BEV Scene (Intersection)', fontsize=13, weight='bold')

    # Draw lanes
    for lane in lanes:
        cl = lane['centerline']
        ax.plot(cl[:, 0], cl[:, 1], 'gray', linewidth=2, alpha=0.6, linestyle='--')

    # Draw vehicles
    for i, agent in enumerate(agents):
        color = agent['color']
        draw_vehicle(ax, agent['pos'], agent['heading'], color=color, alpha=0.9)

        # Label
        ax.text(agent['pos'][0], agent['pos'][1] + 3, agent['label'],
               ha='center', fontsize=9, weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========== Panel 2: Attention Heatmap ==========
    ax = axes[1]
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)
    ax.set_title('Attention Heatmap (Gaussian Splat + Lane Paint)', fontsize=13, weight='bold')

    # Create heatmap
    resolution = 0.5  # 0.5m per pixel
    H = W = int(2 * radius / resolution)
    heatmap = np.zeros((H, W))

    # Splat agent attention
    for i, agent in enumerate(agents):
        gaussian_splat_2d(heatmap, agent['pos'], agent_attn[i],
                         sigma=3.0, resolution=resolution, radius=radius)

    # Paint lane attention
    for i, lane in enumerate(lanes):
        paint_polyline_2d(heatmap, lane['centerline'], lane_attn[i],
                         width=2.5, resolution=resolution, radius=radius)

    # Normalize for visualization
    if heatmap.max() > 0:
        heatmap_norm = heatmap / np.percentile(heatmap[heatmap > 0], 95)
        heatmap_norm = np.clip(heatmap_norm, 0, 1)
    else:
        heatmap_norm = heatmap

    # Display heatmap
    im = ax.imshow(heatmap_norm, extent=(-radius, radius, -radius, radius),
                   origin='lower', cmap='magma', alpha=1.0, interpolation='bilinear')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight (normalized)', fontsize=10)

    # ========== Panel 3: Overlay (Scene + Attention) ==========
    ax = axes[2]
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)
    ax.set_title('BEV Scene with Attention Overlay', fontsize=13, weight='bold')

    # Draw lanes (background)
    for lane in lanes:
        cl = lane['centerline']
        ax.plot(cl[:, 0], cl[:, 1], 'gray', linewidth=2, alpha=0.5, linestyle='--')

    # Overlay heatmap (semi-transparent)
    ax.imshow(heatmap_norm, extent=(-radius, radius, -radius, radius),
             origin='lower', cmap='magma', alpha=0.6, interpolation='bilinear')

    # Draw vehicles (on top of heatmap)
    for i, agent in enumerate(agents):
        color = agent['color']
        draw_vehicle(ax, agent['pos'], agent['heading'], color=color, alpha=0.9)

    # Annotations for top-2 attention
    top_indices = np.argsort(agent_attn)[-2:][::-1]
    for idx in top_indices:
        agent = agents[idx]
        ax.scatter(agent['pos'][0], agent['pos'][1], s=200,
                  facecolors='none', edgecolors='cyan', linewidths=3, zorder=20)
        ax.text(agent['pos'][0] + 4, agent['pos'][1] + 4,
               f"Attn: {agent_attn[idx]:.2f}",
               color='cyan', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Colorbar
    cbar2 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Attention Weight', fontsize=10)

    plt.tight_layout()

    # Save
    output_path = '/tmp/bev_attention_overlay_demo.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    print("\nKey observations:")
    print("  - Oncoming vehicle: HIGH attention (0.85) → bright yellow/red hotspot")
    print("  - Target lane (E-W): HIGH attention (0.75) → bright lane")
    print("  - Left vehicle: MEDIUM attention (0.45) → visible glow")
    print("  - Right vehicle: LOW attention (0.15) → faint")
    print("\nThis demonstrates:")
    print("  ✓ Spatial mapping: tokens → physical BEV coordinates")
    print("  ✓ Gaussian splatting: agent attention → 2D heatmap")
    print("  ✓ Polyline painting: lane attention → along centerlines")
    print("  ✓ Multi-layer rendering: lanes + heatmap + vehicles")
    print("  ✓ Intuitive interpretation: model 'sees' conflicts and targets")


if __name__ == "__main__":
    main()
