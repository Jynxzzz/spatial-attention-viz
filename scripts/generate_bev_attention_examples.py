"""Generate BEV attention overlay examples for paper figures.

This script:
1. Loads a trained MTR-Lite checkpoint
2. Selects diverse scenarios (intersection, highway, lane change, etc.)
3. Generates BEV attention overlays for each scenario
4. Saves publication-quality figures to demo_outputs/bev_attention/

Usage:
    python scripts/generate_bev_attention_examples.py \
        --checkpoint checkpoints/best_model.pt \
        --scene_list data/scene_list.txt \
        --output_dir demo_outputs/bev_attention \
        --num_examples 10
"""

import argparse
import os
import pickle
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.polyline_dataset import PolylineDataset
from data.spatial_bookkeeper import SpatialTokenBookkeeper
from models.mtr_lite import MTRLite
from visualization.bev_attention_overlay import (
    render_attention_overlay_bev,
    render_multi_layer_attention,
)


def load_model(checkpoint_path, device='cuda'):
    """Load trained MTR-Lite model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    config = checkpoint.get('config', {})

    # Build model
    model = MTRLite(
        agent_dim=config.get('agent_dim', 29),
        map_dim=config.get('map_dim', 9),
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.1),
        pred_len=config.get('pred_len', 80),
        num_modes=config.get('num_modes', 6),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Model loaded successfully (epoch {checkpoint.get('epoch', '?')})")

    return model, config


def select_diverse_scenarios(dataset, num_examples=10):
    """Select diverse scenarios for visualization.

    Heuristic selection based on scene characteristics:
    - Different lane topology structures
    - Presence of traffic lights
    - Number of surrounding vehicles
    - Lane change vs straight driving

    Returns:
        List of sample indices
    """
    print(f"Selecting {num_examples} diverse scenarios from {len(dataset)} scenes...")

    selected_indices = []
    scenarios_found = set()

    # Try to find one of each type
    desired_scenarios = [
        'intersection',
        'highway',
        'lane_change',
        'stop_sign',
        'traffic_light',
        'crowded',
        'sparse',
        'curved_road',
    ]

    for idx in range(len(dataset)):
        if len(selected_indices) >= num_examples:
            break

        sample = dataset[idx]
        if sample is None:
            continue

        # Load scene to check characteristics
        scene_path = sample['scene_path']
        try:
            with open(scene_path, 'rb') as f:
                scene = pickle.load(f)
        except:
            continue

        # Classify scenario (simple heuristics)
        scenario_type = None

        # Check for traffic lights
        if len(scene.get('traffic_lights', [])) > 0:
            if 'traffic_light' not in scenarios_found:
                scenario_type = 'traffic_light'

        # Check number of nearby agents
        n_agents = sample['agent_mask'].sum().item()
        if n_agents > 15 and 'crowded' not in scenarios_found:
            scenario_type = 'crowded'
        elif n_agents < 5 and 'sparse' not in scenarios_found:
            scenario_type = 'sparse'

        # Check lane topology complexity
        n_lanes = sample['map_mask'].sum().item()
        if n_lanes > 40 and 'intersection' not in scenarios_found:
            scenario_type = 'intersection'
        elif n_lanes < 20 and 'highway' not in scenarios_found:
            scenario_type = 'highway'

        if scenario_type is not None:
            selected_indices.append(idx)
            scenarios_found.add(scenario_type)
            print(f"  Selected scene {idx}: {scenario_type}")

        # Fill remaining slots with random selection
        if len(selected_indices) < num_examples and idx % (len(dataset) // num_examples) == 0:
            if idx not in selected_indices:
                selected_indices.append(idx)
                print(f"  Selected scene {idx}: general")

    # If still not enough, add random scenes
    while len(selected_indices) < num_examples:
        idx = np.random.randint(len(dataset))
        if idx not in selected_indices:
            sample = dataset[idx]
            if sample is not None:
                selected_indices.append(idx)
                print(f"  Selected scene {idx}: random")

    return selected_indices[:num_examples]


def generate_single_example(
    model,
    sample,
    scene,
    output_dir,
    scene_idx,
    device='cuda',
    query_agent_idx=0,
    layer_idx=-1,
):
    """Generate BEV attention overlay for a single scene.

    Args:
        model: MTR-Lite model
        sample: Dataset sample dict
        scene: Loaded scene dict
        output_dir: Output directory
        scene_idx: Scene index for naming
        device: torch device
        query_agent_idx: Which agent's attention to visualize
        layer_idx: Which layer to visualize
    """
    # Move sample to device
    batch = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0).to(device)  # Add batch dim
        else:
            batch[key] = val

    # Forward pass with attention extraction
    with torch.no_grad():
        outputs = model(batch, return_attention=True)

    # Build spatial bookkeeper
    spatial_bookkeeper = SpatialTokenBookkeeper.from_batch(
        batch,
        agent_token_start=0,
        map_token_start=32,  # Assuming max 32 agent tokens
    )

    # Extract ego position and heading
    ego_pos = batch.get('ego_pos')
    ego_heading = batch.get('ego_heading')
    anchor_frame = batch.get('anchor_frame', 10)

    if isinstance(anchor_frame, torch.Tensor):
        anchor_frame = anchor_frame.item()

    # Get prediction for visualization
    pred_traj = outputs['predictions'][0, 0, :, :].cpu()  # (80, 2) - mode 0, target 0

    # Render single-layer overlay
    save_path = os.path.join(output_dir, f'scene_{scene_idx:03d}_layer{layer_idx}.png')
    render_attention_overlay_bev(
        scene=scene,
        batch=batch,
        attention_maps=outputs['attention_maps'],
        spatial_bookkeeper=spatial_bookkeeper,
        query_agent_idx=query_agent_idx,
        layer_idx=layer_idx,
        ego_pos=ego_pos,
        ego_heading=ego_heading,
        anchor_frame=anchor_frame,
        radius=60,
        save_path=save_path,
        pred_traj=pred_traj,
        show_top_k=3,
        title=f'BEV Attention Overlay - Scene {scene_idx}',
    )

    # Render multi-layer comparison
    save_path_multi = os.path.join(output_dir, f'scene_{scene_idx:03d}_all_layers.png')
    render_multi_layer_attention(
        scene=scene,
        batch=batch,
        attention_maps=outputs['attention_maps'],
        spatial_bookkeeper=spatial_bookkeeper,
        query_agent_idx=query_agent_idx,
        layer_indices=[0, 2, 4, -1],  # First, middle, and last layers
        ego_pos=ego_pos,
        ego_heading=ego_heading,
        anchor_frame=anchor_frame,
        radius=60,
        save_path=save_path_multi,
    )

    print(f"  Generated examples for scene {scene_idx}")


def main():
    parser = argparse.ArgumentParser(description='Generate BEV attention overlay examples')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--scene_list', type=str, required=True,
                       help='Path to scene list file')
    parser.add_argument('--output_dir', type=str, default='demo_outputs/bev_attention',
                       help='Output directory for visualizations')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of examples to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--query_agent', type=int, default=0,
                       help='Agent index to visualize attention for (0=ego)')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer index to visualize (-1=last layer)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, config = load_model(args.checkpoint, device=args.device)

    # Create dataset
    dataset = PolylineDataset(
        scene_list_path=args.scene_list,
        split='val',
        val_ratio=0.15,
        data_fraction=1.0,
        augment=False,
    )

    print(f"Dataset loaded: {len(dataset)} scenes")

    # Select diverse scenarios
    selected_indices = select_diverse_scenarios(dataset, num_examples=args.num_examples)

    # Generate visualizations
    print(f"\nGenerating BEV attention overlays...")

    for i, scene_idx in enumerate(selected_indices):
        print(f"Processing scene {i+1}/{len(selected_indices)} (index {scene_idx})...")

        sample = dataset[scene_idx]
        if sample is None:
            print(f"  Skipping scene {scene_idx}: failed to load")
            continue

        # Load scene
        scene_path = sample['scene_path']
        try:
            with open(scene_path, 'rb') as f:
                scene = pickle.load(f)
        except Exception as e:
            print(f"  Skipping scene {scene_idx}: {e}")
            continue

        # Generate visualizations
        try:
            generate_single_example(
                model=model,
                sample=sample,
                scene=scene,
                output_dir=args.output_dir,
                scene_idx=scene_idx,
                device=args.device,
                query_agent_idx=args.query_agent,
                layer_idx=args.layer,
            )
        except Exception as e:
            print(f"  Error generating scene {scene_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(selected_indices)} examples")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
