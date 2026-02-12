"""Test visualization pipeline with dummy attention data.

This script verifies that all visualization modules work correctly:
1. Space-Attention BEV Heatmap
2. Time-Attention Refinement Diagram
3. Lane-Token Activation Map
4. Composite Figure
5. Animation

Usage:
    python scripts/test_visualizations.py --output-dir /tmp/viz_test
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.space_attention_bev import render_space_attention_bev
from visualization.time_attention_diagram import render_time_attention_diagram
from visualization.lane_token_activation import render_lane_activation_map
from visualization.animation import create_layer_refinement_gif


def generate_dummy_data():
    """Generate synthetic test data for visualization."""
    np.random.seed(42)

    # Scene dimensions
    num_agents = 32
    num_map = 64
    map_points_per_lane = 20
    future_len = 80
    history_len = 11
    num_modes = 6
    num_intentions = 64
    nhead = 8
    num_dec_layers = 4

    # Agent data
    agent_positions_bev = np.random.randn(num_agents, 2) * 20  # scatter around origin
    agent_mask = np.random.rand(num_agents) > 0.3
    agent_attention = np.random.rand(num_agents)
    agent_attention = agent_attention * agent_mask.astype(float)
    agent_attention /= (agent_attention.sum() + 1e-8)

    # Map data
    lane_centerlines_bev = np.zeros((num_map, map_points_per_lane, 2))
    for i in range(num_map):
        # Generate curved lanes
        t = np.linspace(0, 1, map_points_per_lane)
        angle = np.random.rand() * 2 * np.pi
        radius = np.random.rand() * 30 + 20
        offset_x = np.random.randn() * 10
        offset_y = np.random.randn() * 10

        lane_centerlines_bev[i, :, 0] = offset_x + radius * t * np.cos(angle)
        lane_centerlines_bev[i, :, 1] = offset_y + radius * t * np.sin(angle)

    map_mask = np.random.rand(num_map) > 0.3
    map_attention = np.random.rand(num_map)
    map_attention = map_attention * map_mask.astype(float)
    map_attention /= (map_attention.sum() + 1e-8)

    # Target trajectory
    target_history_bev = np.zeros((history_len, 2))
    target_history_bev[:, 0] = np.linspace(-5, 0, history_len)
    target_history_bev[:, 1] = 0

    target_future_bev = np.zeros((future_len, 2))
    target_future_bev[:, 0] = np.linspace(0, 40, future_len)
    target_future_bev[:, 1] = 5 * np.sin(np.linspace(0, 2 * np.pi, future_len))

    # Predictions
    pred_trajectories_bev = np.zeros((num_modes, future_len, 2))
    for k in range(num_modes):
        pred_trajectories_bev[k, :, 0] = np.linspace(0, 40, future_len)
        pred_trajectories_bev[k, :, 1] = (
            5 * np.sin(np.linspace(0, 2 * np.pi, future_len) + k * 0.2)
            + np.random.randn() * 2
        )

    # Decoder attentions (simulating refinement: increasing focus)
    decoder_agent_attns = []
    decoder_map_attns = []

    for layer_i in range(num_dec_layers):
        # Agent attention: becomes more focused with depth
        focus_factor = 1.0 + layer_i * 0.5
        agent_attn_layer = np.random.gamma(focus_factor, 1.0, (nhead, num_intentions, num_agents))
        agent_attn_layer = agent_attn_layer / agent_attn_layer.sum(axis=-1, keepdims=True)
        decoder_agent_attns.append(agent_attn_layer)

        # Map attention: also becomes more focused
        map_attn_layer = np.random.gamma(focus_factor, 1.0, (nhead, num_intentions, num_map))
        map_attn_layer = map_attn_layer / map_attn_layer.sum(axis=-1, keepdims=True)
        decoder_map_attns.append(map_attn_layer)

    # Agent/map labels
    agent_labels = [f"Agent {i}" for i in range(num_agents)]
    map_labels = [f"Lane {i}" for i in range(num_map)]

    # Cumulative lane attention
    cumulative_lane_attn = sum(
        attn[:, 0, :].mean(0) for attn in decoder_map_attns
    )
    cumulative_lane_attn = cumulative_lane_attn * map_mask.astype(float)

    return {
        "agent_positions_bev": agent_positions_bev,
        "agent_attention": agent_attention,
        "agent_mask": agent_mask,
        "lane_centerlines_bev": lane_centerlines_bev,
        "map_attention": map_attention,
        "map_mask": map_mask,
        "target_history_bev": target_history_bev,
        "target_future_bev": target_future_bev,
        "pred_trajectories_bev": pred_trajectories_bev,
        "decoder_agent_attns": decoder_agent_attns,
        "decoder_map_attns": decoder_map_attns,
        "agent_labels": agent_labels,
        "map_labels": map_labels,
        "cumulative_lane_attn": cumulative_lane_attn,
    }


def test_space_attention_bev(data, output_dir):
    """Test Space-Attention BEV rendering."""
    print("Testing Space-Attention BEV...")
    fig = render_space_attention_bev(
        agent_positions_bev=data["agent_positions_bev"],
        agent_attention=data["agent_attention"],
        agent_mask=data["agent_mask"],
        lane_centerlines_bev=data["lane_centerlines_bev"],
        map_attention=data["map_attention"],
        map_mask=data["map_mask"],
        target_history_bev=data["target_history_bev"],
        target_future_bev=data["target_future_bev"],
        pred_trajectories_bev=data["pred_trajectories_bev"],
        all_lane_points=data["lane_centerlines_bev"],
        bev_range=50.0,
        title="Test: Space-Attention BEV Heatmap",
    )
    save_path = os.path.join(output_dir, "test_space_attention_bev.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def test_time_attention_diagram(data, output_dir):
    """Test Time-Attention Refinement Diagram."""
    print("Testing Time-Attention Diagram...")
    fig = render_time_attention_diagram(
        decoder_agent_attns=data["decoder_agent_attns"],
        decoder_map_attns=data["decoder_map_attns"],
        mode_idx=0,
        agent_labels=data["agent_labels"],
        map_labels=data["map_labels"],
        agent_mask=data["agent_mask"],
        map_mask=data["map_mask"],
        top_k=10,
        title="Test: Time-Attention Refinement",
    )
    save_path = os.path.join(output_dir, "test_time_attention_diagram.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def test_lane_activation_map(data, output_dir):
    """Test Lane-Token Activation Map."""
    print("Testing Lane-Token Activation Map...")
    fig = render_lane_activation_map(
        lane_centerlines_bev=data["lane_centerlines_bev"],
        lane_attention=data["cumulative_lane_attn"],
        lane_mask=data["map_mask"],
        lane_labels=data["map_labels"],
        target_history_bev=data["target_history_bev"],
        target_future_bev=data["target_future_bev"],
        pred_trajectories_bev=data["pred_trajectories_bev"],
        bev_range=50.0,
        title="Test: Lane-Token Activation Map",
    )
    save_path = os.path.join(output_dir, "test_lane_activation_map.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def test_animation(data, output_dir):
    """Test animation GIF generation."""
    print("Testing Animation GIF...")
    save_path = os.path.join(output_dir, "test_layer_refinement.gif")
    create_layer_refinement_gif(
        decoder_agent_attns=data["decoder_agent_attns"],
        decoder_map_attns=data["decoder_map_attns"],
        mode_idx=0,
        agent_positions_bev=data["agent_positions_bev"],
        agent_mask=data["agent_mask"],
        lane_centerlines_bev=data["lane_centerlines_bev"],
        map_mask=data["map_mask"],
        target_history_bev=data["target_history_bev"],
        target_future_bev=data["target_future_bev"],
        pred_trajectories_bev=data["pred_trajectories_bev"],
        save_path=save_path,
        bev_range=50.0,
        fps=2,
    )
    print(f"  Saved: {save_path}")


def test_composite_figure(data, output_dir):
    """Test composite 3-column figure (simplified version)."""
    print("Testing Composite Figure...")

    fig = plt.figure(figsize=(20, 7))

    # Column 1: Space-Attention BEV
    ax1 = fig.add_subplot(1, 3, 1)
    render_space_attention_bev(
        agent_positions_bev=data["agent_positions_bev"],
        agent_attention=data["agent_attention"],
        agent_mask=data["agent_mask"],
        lane_centerlines_bev=data["lane_centerlines_bev"],
        map_attention=data["map_attention"],
        map_mask=data["map_mask"],
        target_history_bev=data["target_history_bev"],
        target_future_bev=data["target_future_bev"],
        pred_trajectories_bev=data["pred_trajectories_bev"],
        all_lane_points=data["lane_centerlines_bev"],
        bev_range=50.0,
        title="Space-Attention BEV",
        ax=ax1,
    )

    # Column 2: Time-Attention Diagram (simplified)
    ax2 = fig.add_subplot(1, 3, 2)
    layer_idx = 3  # Last layer
    agent_attn = data["decoder_agent_attns"][layer_idx][:, 0, :].mean(0)
    agent_attn = agent_attn * data["agent_mask"].astype(float)
    top_k = min(10, int(data["agent_mask"].sum()))
    order = np.argsort(agent_attn)[::-1][:top_k]
    ax2.barh(range(top_k), agent_attn[order], color="#1565C0", alpha=0.8)
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels([f"A{i}" for i in order], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Attention Weight")
    ax2.set_title("Time-Attention (Layer 4)")

    # Column 3: Lane Activation (simplified BEV)
    ax3 = fig.add_subplot(1, 3, 3)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    valid_attn = data["cumulative_lane_attn"][data["map_mask"].astype(bool)]
    vmax = valid_attn.max() if len(valid_attn) > 0 and valid_attn.max() > 0 else 1
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    colormap = cm.get_cmap("RdYlBu_r")

    for i in range(len(data["lane_centerlines_bev"])):
        if not data["map_mask"][i]:
            continue
        pts = data["lane_centerlines_bev"][i]
        attn = data["cumulative_lane_attn"][i]
        color = colormap(norm(attn))
        width = 1.0 + 4.0 * norm(attn)
        ax3.plot(pts[:, 0], pts[:, 1], color=color, linewidth=width, alpha=0.9)

    ax3.plot(0, 0, "^", color="#0D47A1", markersize=10)
    ax3.set_xlim(-50, 50)
    ax3.set_ylim(-50, 50)
    ax3.set_aspect("equal")
    ax3.set_title("Lane-Token Activation")
    ax3.set_xlabel("Lateral (m)")
    ax3.set_ylabel("Forward (m)")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "test_composite_figure.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test visualization pipeline")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/viz_test",
        help="Output directory for test figures",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    # Generate dummy data
    print("Generating dummy test data...")
    data = generate_dummy_data()
    print("  Done.\n")

    # Run all tests
    test_space_attention_bev(data, args.output_dir)
    test_time_attention_diagram(data, args.output_dir)
    test_lane_activation_map(data, args.output_dir)
    test_composite_figure(data, args.output_dir)
    test_animation(data, args.output_dir)

    print(f"\nAll tests completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
