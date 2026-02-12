"""Demo BEV attention overlay with synthetic attention weights.

This script demonstrates the BEV attention visualization system WITHOUT requiring
a trained model. It uses synthetic attention weights to show how the visualization
works.

Useful for:
- Testing the visualization pipeline
- Creating example figures for documentation
- Debugging spatial mapping issues

Usage:
    python scripts/demo_bev_attention_synthetic.py \
        --scene_list data/scene_list.txt \
        --output_dir demo_outputs/synthetic_bev_attention \
        --num_examples 3
"""

import argparse
import os
import pickle
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.polyline_dataset import PolylineDataset
from data.spatial_bookkeeper import SpatialTokenBookkeeper
from visualization.bev_attention_overlay import render_attention_overlay_bev


class SyntheticAttentionMaps:
    """Synthetic attention maps for demonstration."""

    def __init__(self, n_layers=6, n_heads=8, n_tokens=96):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_tokens = n_tokens
        self.scene_attentions = []

        # Generate synthetic attention for each layer
        for layer_idx in range(n_layers):
            # (1, nhead, N, N) - batch_size=1
            attn = self._generate_layer_attention(layer_idx)
            self.scene_attentions.append(attn)

    def _generate_layer_attention(self, layer_idx):
        """Generate synthetic but realistic-looking attention patterns."""
        n_agent_tokens = 32
        n_map_tokens = 64

        # (nhead, N, N)
        attn = torch.zeros(self.n_heads, self.n_tokens, self.n_tokens)

        # Strategy: Earlier layers attend locally, later layers attend globally
        local_factor = 1.0 - (layer_idx / self.n_layers)
        global_factor = layer_idx / self.n_layers

        for h in range(self.n_heads):
            # Each head has different attention pattern
            for query_idx in range(self.n_tokens):
                # Determine if query is agent or map
                is_query_agent = query_idx < n_agent_tokens

                if is_query_agent:
                    # Agent queries: attend to nearby agents and lanes
                    # Self-attention (always present)
                    attn[h, query_idx, query_idx] = 0.3

                    # Attend to other agents (distance-based)
                    for key_idx in range(n_agent_tokens):
                        if key_idx != query_idx:
                            # Simulate distance-based attention
                            dist = abs(key_idx - query_idx)
                            weight = 0.2 * np.exp(-dist / 5.0) * local_factor
                            attn[h, query_idx, key_idx] = weight

                    # Attend to lanes (especially ego lane and successors)
                    for key_idx in range(n_agent_tokens, n_agent_tokens + n_map_tokens):
                        # First few lanes get more attention (ego lane + successors)
                        lane_priority = max(0, 1.0 - (key_idx - n_agent_tokens) / 20.0)
                        weight = 0.15 * lane_priority * global_factor
                        attn[h, query_idx, key_idx] = weight

                else:
                    # Map queries: attend to agents and connected lanes
                    # Self-attention
                    attn[h, query_idx, query_idx] = 0.2

                    # Attend to agents (all agents pay some attention to lanes)
                    for key_idx in range(n_agent_tokens):
                        weight = 0.1 * global_factor
                        attn[h, query_idx, key_idx] = weight

                    # Attend to other lanes (topology-based, simulated)
                    for key_idx in range(n_agent_tokens, n_agent_tokens + n_map_tokens):
                        if key_idx != query_idx:
                            # Lanes close in index → likely topologically connected
                            dist = abs(key_idx - query_idx)
                            weight = 0.15 * np.exp(-dist / 10.0)
                            attn[h, query_idx, key_idx] = weight

                # Add some random noise for realism
                attn[h, query_idx, :] += torch.randn(self.n_tokens) * 0.01

            # Softmax normalization per query
            attn[h, :, :] = torch.softmax(attn[h, :, :], dim=-1)

        return attn.unsqueeze(0)  # Add batch dimension


def create_synthetic_prediction(batch):
    """Create a simple synthetic prediction trajectory."""
    # Extract ego current position (last timestep of history)
    agent_polylines = batch['agent_polylines']  # (1, A, 11, 29)
    ego_pos = agent_polylines[0, 0, -1, 0:2].cpu().numpy()  # (2,)
    ego_vel = agent_polylines[0, 0, -1, 4:6].cpu().numpy()  # (2,)

    # Generate simple constant-velocity prediction
    pred_len = 80
    dt = 0.1  # 0.1s per step

    pred_traj = np.zeros((pred_len, 2), dtype=np.float32)
    for t in range(pred_len):
        pred_traj[t] = ego_pos + ego_vel * (t + 1) * dt

    return torch.from_numpy(pred_traj)


def generate_demo_example(dataset, scene_idx, output_dir):
    """Generate one demo example with synthetic attention."""
    print(f"Generating demo for scene {scene_idx}...")

    sample = dataset[scene_idx]
    if sample is None:
        print(f"  Skipping scene {scene_idx}: failed to load")
        return

    # Load scene
    scene_path = sample['scene_path']
    try:
        with open(scene_path, 'rb') as f:
            scene = pickle.load(f)
    except Exception as e:
        print(f"  Skipping scene {scene_idx}: {e}")
        return

    # Create batch (add batch dimension)
    batch = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0)
        else:
            batch[key] = val

    # Generate synthetic attention
    attention_maps = SyntheticAttentionMaps(n_layers=6, n_heads=8, n_tokens=96)

    # Build spatial bookkeeper
    bookkeeper = SpatialTokenBookkeeper.from_batch(
        batch,
        agent_token_start=0,
        map_token_start=32,
    )

    # Generate synthetic prediction
    pred_traj = create_synthetic_prediction(batch)

    # Extract ego info
    ego_pos = batch.get('ego_pos')
    ego_heading = batch.get('ego_heading')
    anchor_frame = batch.get('anchor_frame', 10)

    if isinstance(anchor_frame, torch.Tensor):
        anchor_frame = anchor_frame.item()

    # Render attention overlay
    save_path = os.path.join(output_dir, f'demo_scene_{scene_idx:03d}_synthetic.png')

    try:
        render_attention_overlay_bev(
            scene=scene,
            batch=batch,
            attention_maps=attention_maps,
            spatial_bookkeeper=bookkeeper,
            query_agent_idx=0,
            layer_idx=-1,
            ego_pos=ego_pos,
            ego_heading=ego_heading,
            anchor_frame=anchor_frame,
            radius=60,
            save_path=save_path,
            pred_traj=pred_traj,
            show_top_k=3,
            title=f'BEV Attention Demo (Synthetic) - Scene {scene_idx}',
        )
        print(f"  ✓ Saved to: {save_path}")
    except Exception as e:
        print(f"  Error rendering scene {scene_idx}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Demo BEV attention overlay with synthetic attention'
    )
    parser.add_argument('--scene_list', type=str, required=True,
                       help='Path to scene list file')
    parser.add_argument('--output_dir', type=str,
                       default='demo_outputs/synthetic_bev_attention',
                       help='Output directory')
    parser.add_argument('--num_examples', type=int, default=3,
                       help='Number of examples to generate')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset
    print("Loading dataset...")
    dataset = PolylineDataset(
        scene_list_path=args.scene_list,
        split='val',
        val_ratio=0.15,
        data_fraction=1.0,
        augment=False,
    )

    print(f"Dataset loaded: {len(dataset)} scenes")

    # Generate demos for first N valid scenes
    count = 0
    for scene_idx in range(len(dataset)):
        if count >= args.num_examples:
            break

        generate_demo_example(dataset, scene_idx, args.output_dir)
        count += 1

    print(f"\n{'='*60}")
    print(f"Demo complete! Generated {count} examples")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")
    print("\nNote: These use SYNTHETIC attention weights for demonstration.")
    print("For real attention, use generate_bev_attention_examples.py with a trained model.")


if __name__ == '__main__':
    main()
