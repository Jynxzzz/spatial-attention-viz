"""Generate example attention visualizations from a trained model.

This script demonstrates the complete workflow:
1. Load trained MTR-Lite model
2. Extract attention from a test scene
3. Generate all visualization types
4. Save publication-quality figures

Usage:
    python scripts/generate_example_figures.py \
        --checkpoint /path/to/model.ckpt \
        --scene-path /path/to/scene.pkl \
        --output-dir ./figures/example \
        --device cuda
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visualization.attention_extractor import extract_scene_attention
from visualization.space_attention_bev import render_space_attention_bev
from visualization.time_attention_diagram import render_time_attention_diagram
from visualization.lane_token_activation import render_lane_activation_map
from visualization.animation import create_layer_refinement_gif
from visualization.utils import compute_attention_statistics


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained MTR-Lite model from checkpoint.

    Args:
        checkpoint_path: path to .ckpt file
        device: device to load model on

    Returns:
        model: loaded MTR-Lite model in eval mode
    """
    from training.lightning_module import TrajPredModule

    print(f"Loading model from {checkpoint_path}...")
    module = TrajPredModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    module.eval()
    model = module.model
    model.to(device)
    print(f"  Model loaded successfully on {device}")
    return model


def generate_all_visualizations(
    model,
    scene_path: str,
    output_dir: str,
    device: str = "cuda",
    target_idx: int = 0,
    bev_range: float = 60.0,
):
    """Generate all visualization types for a single scene.

    Args:
        model: trained MTR-Lite model
        scene_path: path to Waymo pkl scene
        output_dir: directory to save figures
        device: device for inference
        target_idx: which target agent to visualize (default 0 = ego)
        bev_range: BEV spatial extent in meters

    Returns:
        dict with paths to all generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]

    print(f"\nProcessing scene: {scene_name}")
    print("=" * 60)

    # Step 1: Extract attention from scene
    print("\n1. Extracting attention from model...")
    result = extract_scene_attention(
        model=model,
        scene_path=scene_path,
        anchor_frame=10,
        device=device,
    )

    attn_maps = result["attention_maps"]
    bookkeeper = result["bookkeeper"]
    agent_data = result["agent_data"]
    map_data = result["map_data"]
    predictions = result["predictions"]

    if attn_maps is None:
        print("  ERROR: No attention maps captured!")
        return {}

    print(f"  ✓ Captured {len(attn_maps.scene_attentions)} scene encoder layers")
    print(f"  ✓ Captured {len(attn_maps.decoder_agent_attentions)} decoder layers")

    # Extract data for visualization
    agent_positions = agent_data["agent_polylines"][:, -1, :2]  # (A, 2)
    lane_centerlines = map_data["lane_centerlines_bev"]  # (M, P, 2)

    # Get target agent's trajectory
    target_agent_slot = bookkeeper.target_agent_indices[target_idx] if target_idx < len(bookkeeper.target_agent_indices) else 0
    target_history = agent_data["agent_polylines"][target_agent_slot, :, :2]  # (11, 2)

    # Get GT and predictions for target
    target_future = predictions["target_future"][0, target_idx].cpu().numpy()  # (80, 2)
    pred_trajectories = predictions["pred_trajectories"][0, target_idx].cpu().numpy()  # (K, 80, 2)

    # Get NMS-selected mode
    if attn_maps.nms_indices is not None and attn_maps.nms_indices.dim() >= 3:
        winning_mode_idx = attn_maps.nms_indices[0, target_idx, 0].item()
    else:
        winning_mode_idx = 0

    # --- Visualization 1: Space-Attention BEV ---
    print("\n2. Generating Space-Attention BEV heatmap...")

    # Use last scene encoder layer, target agent's row
    scene_attn = attn_maps.scene_attentions[-1][0]  # (nhead, A+M, A+M)
    target_attn_row = scene_attn[:, target_agent_slot, :].mean(0).cpu().numpy()  # (A+M,)

    agent_attn = target_attn_row[:bookkeeper.num_agents]
    map_attn = target_attn_row[bookkeeper.num_agents:]

    fig1 = render_space_attention_bev(
        agent_positions_bev=agent_positions,
        agent_attention=agent_attn,
        agent_mask=agent_data["agent_mask"],
        lane_centerlines_bev=lane_centerlines,
        map_attention=map_attn,
        map_mask=map_data["map_mask"],
        target_history_bev=target_history,
        target_future_bev=target_future,
        pred_trajectories_bev=pred_trajectories,
        all_lane_points=lane_centerlines,
        bev_range=bev_range,
        title=f"Space-Attention BEV: {scene_name}",
        figsize=(12, 12),
    )

    path1 = os.path.join(output_dir, f"{scene_name}_space_attention_bev.png")
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  ✓ Saved: {path1}")

    # --- Visualization 2: Time-Attention Diagram ---
    print("\n3. Generating Time-Attention Refinement Diagram...")

    # Extract decoder attentions for the winning mode
    # Note: decoder attentions are per-target, organized as list[layer] per target
    decoder_agent_attns = []
    decoder_map_attns = []

    for layer_i in range(len(attn_maps.decoder_agent_attentions)):
        # Get attention for this target and layer
        if isinstance(attn_maps.decoder_agent_attentions[0], list):
            # Organized per-target: [target][layer]
            agent_attn_layer = attn_maps.decoder_agent_attentions[target_idx][layer_i][0].cpu().numpy()
            map_attn_layer = attn_maps.decoder_map_attentions[target_idx][layer_i][0].cpu().numpy()
        else:
            # Organized per-layer: [layer] (all targets batched)
            agent_attn_layer = attn_maps.decoder_agent_attentions[layer_i][0].cpu().numpy()
            map_attn_layer = attn_maps.decoder_map_attentions[layer_i][0].cpu().numpy()

        decoder_agent_attns.append(agent_attn_layer)  # (nhead, K, A)
        decoder_map_attns.append(map_attn_layer)      # (nhead, K, M)

    # Create labels
    agent_labels = [f"Agent {i}" if i != target_agent_slot else "Agent 0 (Ego)" for i in range(bookkeeper.num_agents)]
    map_labels = [f"Lane {i}" for i in range(bookkeeper.num_map)]

    fig2 = render_time_attention_diagram(
        decoder_agent_attns=decoder_agent_attns,
        decoder_map_attns=decoder_map_attns,
        mode_idx=winning_mode_idx,
        agent_labels=agent_labels,
        map_labels=map_labels,
        agent_mask=agent_data["agent_mask"],
        map_mask=map_data["map_mask"],
        top_k=10,
        title=f"Time-Attention Refinement: {scene_name}",
        figsize=(20, 6),
    )

    path2 = os.path.join(output_dir, f"{scene_name}_time_attention_diagram.png")
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved: {path2}")

    # --- Visualization 3: Lane Activation Map ---
    print("\n4. Generating Lane-Token Activation Map...")

    # Cumulative map attention across all decoder layers
    cumulative_lane_attn = np.zeros(bookkeeper.num_map, dtype=np.float32)
    for layer_i in range(len(decoder_map_attns)):
        # Average over heads, extract winning mode
        mode_attn = decoder_map_attns[layer_i][:, winning_mode_idx, :].mean(0)  # (M,)
        cumulative_lane_attn += mode_attn

    fig3 = render_lane_activation_map(
        lane_centerlines_bev=lane_centerlines,
        lane_attention=cumulative_lane_attn,
        lane_mask=map_data["map_mask"],
        lane_labels=map_labels,
        target_history_bev=target_history,
        target_future_bev=target_future,
        pred_trajectories_bev=pred_trajectories,
        bev_range=bev_range,
        top_k_sidebar=15,
        title=f"Lane-Token Activation: {scene_name}",
        figsize=(14, 8),
    )

    path3 = os.path.join(output_dir, f"{scene_name}_lane_activation_map.png")
    fig3.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  ✓ Saved: {path3}")

    # --- Visualization 4: Animation ---
    print("\n5. Generating Attention Refinement Animation...")

    path4 = os.path.join(output_dir, f"{scene_name}_attention_evolution.gif")
    create_layer_refinement_gif(
        decoder_agent_attns=decoder_agent_attns,
        decoder_map_attns=decoder_map_attns,
        mode_idx=winning_mode_idx,
        agent_positions_bev=agent_positions,
        agent_mask=agent_data["agent_mask"],
        lane_centerlines_bev=lane_centerlines,
        map_mask=map_data["map_mask"],
        target_history_bev=target_history,
        target_future_bev=target_future,
        pred_trajectories_bev=pred_trajectories,
        save_path=path4,
        bev_range=bev_range,
        fps=2,
    )
    print(f"  ✓ Saved: {path4}")

    # --- Attention Statistics ---
    print("\n6. Computing Attention Statistics...")

    scene_stats = compute_attention_statistics(target_attn_row)
    print(f"  Scene Encoder Attention (Last Layer, Target Row):")
    print(f"    Entropy: {scene_stats['entropy']:.2f} bits")
    print(f"    Gini: {scene_stats['gini']:.3f}")
    print(f"    Top-1 ratio: {scene_stats['top1_ratio']:.3f}")
    print(f"    Top-5 ratio: {scene_stats['top5_ratio']:.3f}")

    # Per-layer decoder attention statistics
    print(f"\n  Decoder Attention Evolution:")
    for layer_i, (agent_attn_l, map_attn_l) in enumerate(zip(decoder_agent_attns, decoder_map_attns)):
        agent_mode_attn = agent_attn_l[:, winning_mode_idx, :].mean(0)
        map_mode_attn = map_attn_l[:, winning_mode_idx, :].mean(0)

        agent_stats = compute_attention_statistics(agent_mode_attn)
        map_stats = compute_attention_statistics(map_mode_attn)

        print(f"    Layer {layer_i+1}:")
        print(f"      Agent: entropy={agent_stats['entropy']:.2f}, gini={agent_stats['gini']:.3f}")
        print(f"      Map:   entropy={map_stats['entropy']:.2f}, gini={map_stats['gini']:.3f}")

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")

    return {
        "space_attention_bev": path1,
        "time_attention_diagram": path2,
        "lane_activation_map": path3,
        "animation": path4,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate attention visualizations from trained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--scene-path",
        type=str,
        required=True,
        help="Path to Waymo scene (.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures/example",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--target-idx",
        type=int,
        default=0,
        help="Target agent index to visualize (default 0 = ego)",
    )
    parser.add_argument(
        "--bev-range",
        type=float,
        default=60.0,
        help="BEV spatial range in meters",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return 1

    if not os.path.exists(args.scene_path):
        print(f"ERROR: Scene file not found: {args.scene_path}")
        return 1

    # Load model
    model = load_model(args.checkpoint, device=args.device)

    # Generate visualizations
    paths = generate_all_visualizations(
        model=model,
        scene_path=args.scene_path,
        output_dir=args.output_dir,
        device=args.device,
        target_idx=args.target_idx,
        bev_range=args.bev_range,
    )

    print(f"\nAll figures saved to: {args.output_dir}")
    print("\nGenerated files:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
