"""Unit tests for SpatialTokenBookkeeper and spatial utilities.

Tests:
1. Coordinate transformation (BEV meters <-> grid pixels)
2. Gaussian splatting (visual validation)
3. Polyline painting (visual validation)
4. Full pipeline from batch to spatial heatmap
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.spatial_bookkeeper import SpatialTokenBookkeeper
from visualization.spatial_utils import (
    coords_to_grid,
    grid_to_coords,
    gaussian_splat_2d,
    paint_polyline_2d,
    normalize_heatmap,
)


def test_coords_to_grid():
    """Test coordinate transformation."""
    print("Testing coords_to_grid...")

    radius = 60
    resolution = 0.5

    # Test center point (0, 0) -> should map to grid center
    center_bev = np.array([0.0, 0.0])
    center_grid = coords_to_grid(center_bev, resolution, radius)

    expected = int(radius / resolution)  # Grid center
    assert center_grid[0] == expected and center_grid[1] == expected, \
        f"Center failed: {center_grid} != ({expected}, {expected})"

    # Test corner points
    corner_bev = np.array([radius - resolution, radius - resolution])
    corner_grid = coords_to_grid(corner_bev, resolution, radius)

    H = W = int(2 * radius / resolution)
    assert corner_grid[0] == H - 1 and corner_grid[1] == W - 1, \
        f"Corner failed: {corner_grid}"

    # Test inverse transformation
    recovered_bev = grid_to_coords(center_grid, resolution, radius)
    assert np.allclose(recovered_bev, center_bev, atol=resolution), \
        f"Inverse failed: {recovered_bev} != {center_bev}"

    print("  ✓ Coordinate transformation tests passed")


def test_gaussian_splat():
    """Test Gaussian splatting with visual output."""
    print("Testing Gaussian splatting...")

    radius = 30
    resolution = 0.5
    H = W = int(2 * radius / resolution)

    heatmap = np.zeros((H, W), dtype=np.float32)

    # Splat at a few positions
    positions = [
        (0.0, 0.0),      # center
        (10.0, 10.0),    # front-right
        (-15.0, 5.0),    # back-left
        (5.0, -8.0),     # front-left
    ]
    weights = [1.0, 0.8, 0.6, 0.4]

    for pos, weight in zip(positions, weights):
        gaussian_splat_2d(heatmap, np.array(pos), weight,
                         sigma=3.0, resolution=resolution, radius=radius)

    # Check that heatmap is non-zero
    assert heatmap.max() > 0, "Heatmap is empty"

    # Visual check
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, extent=(-radius, radius, -radius, radius),
                   origin='lower', cmap='magma', interpolation='bilinear')
    plt.colorbar(im, ax=ax)

    # Mark input positions
    for pos, weight in zip(positions, weights):
        ax.plot(pos[0], pos[1], 'c+', markersize=12, markeredgewidth=2,
               label=f'Weight={weight}')

    ax.set_title('Gaussian Splat Test')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = '/tmp/attention_viz_demo/test_gaussian_splat.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gaussian splat test passed. Visual saved to: {save_path}")


def test_polyline_painting():
    """Test polyline painting with visual output."""
    print("Testing polyline painting...")

    radius = 30
    resolution = 0.5
    H = W = int(2 * radius / resolution)

    heatmap = np.zeros((H, W), dtype=np.float32)

    # Define a curved lane
    t = np.linspace(0, 1, 20)
    x = 20 * t - 10
    y = 5 * np.sin(3 * np.pi * t)
    polyline = np.stack([x, y], axis=-1).astype(np.float32)

    # Paint polyline
    paint_polyline_2d(heatmap, polyline, weight=1.0,
                     width=2.0, resolution=resolution, radius=radius)

    # Check that heatmap is non-zero
    assert heatmap.max() > 0, "Polyline heatmap is empty"

    # Visual check
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, extent=(-radius, radius, -radius, radius),
                   origin='lower', cmap='magma', interpolation='bilinear')
    plt.colorbar(im, ax=ax)

    # Overlay original polyline
    ax.plot(polyline[:, 0], polyline[:, 1], 'c-', linewidth=2, label='Original Polyline')

    ax.set_title('Polyline Paint Test')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = '/tmp/attention_viz_demo/test_polyline_paint.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Polyline paint test passed. Visual saved to: {save_path}")


def test_combined_heatmap():
    """Test combined agent + map heatmap."""
    print("Testing combined heatmap...")

    radius = 40
    resolution = 0.5
    H = W = int(2 * radius / resolution)

    heatmap = np.zeros((H, W), dtype=np.float32)

    # Agent positions (Gaussians)
    agent_positions = [
        (0.0, 0.0),    # ego
        (15.0, -3.0),  # front vehicle
        (-8.0, 5.0),   # back vehicle
    ]
    agent_weights = [0.5, 0.9, 0.4]

    for pos, weight in zip(agent_positions, agent_weights):
        gaussian_splat_2d(heatmap, np.array(pos), weight,
                         sigma=3.0, resolution=resolution, radius=radius)

    # Map polylines
    # Straight lane
    lane1 = np.array([[i, 0.0] for i in np.linspace(-20, 30, 30)], dtype=np.float32)
    paint_polyline_2d(heatmap, lane1, weight=0.6,
                     width=2.0, resolution=resolution, radius=radius)

    # Curved lane
    t = np.linspace(0, 1, 25)
    x = 25 * t - 10
    y = 8 * np.sin(2 * np.pi * t) + 5
    lane2 = np.stack([x, y], axis=-1).astype(np.float32)
    paint_polyline_2d(heatmap, lane2, weight=0.7,
                     width=2.0, resolution=resolution, radius=radius)

    # Normalize
    heatmap_norm = normalize_heatmap(heatmap, percentile=95)

    # Visual check
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Raw heatmap
    im1 = ax1.imshow(heatmap, extent=(-radius, radius, -radius, radius),
                     origin='lower', cmap='magma', interpolation='bilinear')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Raw Heatmap')

    # Normalized heatmap
    im2 = ax2.imshow(heatmap_norm, extent=(-radius, radius, -radius, radius),
                     origin='lower', cmap='magma', interpolation='bilinear')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Normalized Heatmap')

    # Mark agents and lanes on both
    for ax in [ax1, ax2]:
        for pos in agent_positions:
            ax.plot(pos[0], pos[1], 'c*', markersize=15, markeredgewidth=1,
                   markeredgecolor='white')
        ax.plot(lane1[:, 0], lane1[:, 1], 'c--', linewidth=1, alpha=0.5)
        ax.plot(lane2[:, 0], lane2[:, 1], 'c--', linewidth=1, alpha=0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

    save_path = '/tmp/attention_viz_demo/test_combined_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Combined heatmap test passed. Visual saved to: {save_path}")


def test_spatial_bookkeeper():
    """Test SpatialTokenBookkeeper with mock data."""
    print("Testing SpatialTokenBookkeeper...")

    bookkeeper = SpatialTokenBookkeeper()

    # Register agents
    bookkeeper.register_agent(
        token_idx=0, agent_idx=0,
        pos_bev=np.array([0.0, 0.0]),
        heading=0.0,
        velocity=np.array([5.0, 0.0])
    )

    bookkeeper.register_agent(
        token_idx=1, agent_idx=1,
        pos_bev=np.array([20.0, -5.0]),
        heading=np.pi / 4,
        velocity=np.array([8.0, 2.0])
    )

    # Register map lanes
    lane1 = np.array([[i, 0.0] for i in np.linspace(-30, 30, 30)], dtype=np.float32)
    bookkeeper.register_map(
        token_idx=32, map_idx=0,
        lane_id='lane_0001',
        centerline_bev=lane1
    )

    t = np.linspace(0, 1, 25)
    x = 30 * t - 15
    y = 10 * np.sin(2 * np.pi * t)
    lane2 = np.stack([x, y], axis=-1).astype(np.float32)
    bookkeeper.register_map(
        token_idx=33, map_idx=1,
        lane_id='lane_0002',
        centerline_bev=lane2
    )

    # Create mock attention weights
    n_tokens = 96
    attention_weights = np.zeros(n_tokens, dtype=np.float32)
    attention_weights[0] = 0.3   # ego agent
    attention_weights[1] = 0.8   # other agent (high attention)
    attention_weights[32] = 0.5  # lane 1
    attention_weights[33] = 0.7  # lane 2 (high attention)

    # Generate spatial heatmap
    heatmap, extent = bookkeeper.get_spatial_map(attention_weights, resolution=0.5, radius=50)

    assert heatmap.max() > 0, "Bookkeeper heatmap is empty"

    # Get top attended objects
    top_objects = bookkeeper.get_top_attended_objects(attention_weights, top_k=3)

    assert len(top_objects) == 3, f"Expected 3 top objects, got {len(top_objects)}"

    # Check that the highest weight is 0.8 (with floating point tolerance)
    max_weight = max([obj[1] for obj in top_objects])
    assert np.isclose(max_weight, 0.8, atol=1e-6), f"Top object should have weight 0.8, got {max_weight}"

    print(f"  Top-3 attended objects:")
    for i, (token_idx, weight, obj_type, info) in enumerate(top_objects):
        if obj_type == 'agent':
            print(f"    #{i+1}: Agent {info['agent_idx']} (token {token_idx}), weight={weight:.3f}")
        else:
            print(f"    #{i+1}: Lane {info['lane_id']} (token {token_idx}), weight={weight:.3f}")

    # Visual check
    fig, ax = plt.subplots(figsize=(10, 10))

    heatmap_norm = normalize_heatmap(heatmap, percentile=95)
    im = ax.imshow(heatmap_norm, extent=extent, origin='lower',
                   cmap='magma', alpha=0.8, interpolation='bilinear')
    plt.colorbar(im, ax=ax)

    # Draw agents
    for token_idx, info in bookkeeper.agent_tokens.items():
        pos = info['pos']
        ax.plot(pos[0], pos[1], 'c*', markersize=20, markeredgewidth=2,
               markeredgecolor='white', label=f"Agent {info['agent_idx']}")

    # Draw lanes
    for token_idx, info in bookkeeper.map_tokens.items():
        cl = info['centerline']
        ax.plot(cl[:, 0], cl[:, 1], 'c--', linewidth=2, alpha=0.7,
               label=f"Lane {info['lane_id']}")

    ax.set_title('SpatialTokenBookkeeper Test')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    save_path = '/tmp/attention_viz_demo/test_spatial_bookkeeper.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ SpatialTokenBookkeeper test passed. Visual saved to: {save_path}")


def run_all_tests():
    """Run all unit tests."""
    print("="*60)
    print("Running SpatialTokenBookkeeper and spatial_utils tests")
    print("="*60)

    test_coords_to_grid()
    test_gaussian_splat()
    test_polyline_painting()
    test_combined_heatmap()
    test_spatial_bookkeeper()

    print("="*60)
    print("All tests passed! ✓")
    print("Visual outputs saved to: /tmp/attention_viz_demo/")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
