#!/usr/bin/env python3
"""Quick visualization demo - Generate attention maps from current model (even during training)"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("Quick Attention Visualization Demo")
    print("="*70)

    # Load one scene
    print("\n[1/6] Loading a test scene...")
    from data.polyline_dataset import PolylineDataset
    from data.collate import mtr_collate_fn

    dataset = PolylineDataset(
        scene_list_path="/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt",
        split="val",
        data_fraction=0.01,  # Just 1% for quick test
    )
    print(f"  ✓ Loaded dataset with {len(dataset)} validation scenes")

    # Get one interesting scene (pick one with many agents and lanes)
    sample = dataset[0]
    print(f"  ✓ Selected scene: {sample['scene_path']}")
    print(f"    - Agents: {sample['agent_mask'].sum().item()}")
    print(f"    - Map lanes: {sample['map_mask'].sum().item()}")
    print(f"    - Targets: {sample['target_mask'].sum().item()}")

    # Create batch
    from torch.utils.data import DataLoader
    loader = DataLoader([sample], batch_size=1, collate_fn=mtr_collate_fn)
    batch = next(iter(loader))

    # Load model (randomly initialized or current training weights)
    print("\n[2/6] Loading model...")
    from model.mtr_lite import MTRLite

    intent_points = torch.from_numpy(
        np.load("/mnt/hdd12t/models/mtr_lite/intent_points_64.npy")
    ).float()

    model = MTRLite(
        agent_feat_dim=29,
        map_feat_dim=9,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_intentions=64,
        num_modes_output=6,
        future_len=80,
        intention_points=intent_points,
    )

    # Try to load training checkpoint if exists
    checkpoint_dir = Path("/mnt/hdd12t/outputs/mtr_lite/checkpoints")
    if checkpoint_dir.exists():
        ckpts = list(checkpoint_dir.glob("*.ckpt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
            print(f"  ✓ Loading checkpoint: {latest_ckpt.name}")
            ckpt = torch.load(latest_ckpt, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            print("  ⚠ No checkpoint found, using random weights")
    else:
        print("  ⚠ No checkpoint directory, using random weights")

    model.eval()
    print(f"  ✓ Model ready: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # Forward pass with attention capture
    print("\n[3/6] Running model with attention capture...")
    with torch.no_grad():
        output = model(batch, capture_attention=True)

    attn = output['attention_maps']
    print(f"  ✓ Captured attention:")
    print(f"    - Scene encoder: {len(attn.scene_attentions)} layers")
    print(f"    - Decoder: {len(attn.decoder_agent_attentions)} targets")

    # Prepare visualization data
    print("\n[4/6] Preparing visualization data...")

    # Get scene encoder attention (last layer, first sample)
    scene_attn = attn.scene_attentions[-1][0]  # (nhead, N, N)
    print(f"  ✓ Scene attention shape: {scene_attn.shape}")

    # Get decoder attention (first target, first layer)
    if len(attn.decoder_agent_attentions) > 0 and len(attn.decoder_agent_attentions[0]) > 0:
        dec_agent_attn = attn.decoder_agent_attentions[0][0][0]  # (nhead, K, A)
        dec_map_attn = attn.decoder_map_attentions[0][0][0]      # (nhead, K, M)
        print(f"  ✓ Decoder agent attention: {dec_agent_attn.shape}")
        print(f"  ✓ Decoder map attention: {dec_map_attn.shape}")

    # Generate visualizations
    print("\n[5/6] Generating visualizations...")
    output_dir = Path("/tmp/attention_viz_demo")
    output_dir.mkdir(exist_ok=True)

    # Viz 1: Scene encoder attention matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    attn_avg = scene_attn.mean(dim=0).cpu().numpy()  # Average across heads
    im = ax.imshow(attn_avg, cmap='hot', interpolation='nearest')
    ax.set_title('Scene Encoder Self-Attention (Layer 4, Head Average)', fontsize=14)
    ax.set_xlabel('Key Tokens (32 agents + 64 lanes = 96)', fontsize=12)
    ax.set_ylabel('Query Tokens (32 agents + 64 lanes = 96)', fontsize=12)

    # Add grid lines at agent/map boundary
    ax.axhline(y=32, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Agent/Map boundary')
    ax.axvline(x=32, color='cyan', linestyle='--', linewidth=2, alpha=0.7)

    plt.colorbar(im, ax=ax, label='Attention Weight')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / "scene_attention_matrix.png", dpi=150)
    print(f"  ✓ Saved: scene_attention_matrix.png")

    # Viz 2: Decoder attention to agents (per-head)
    if len(attn.decoder_agent_attentions) > 0:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for head in range(8):
            ax = axes[head // 4, head % 4]
            attn_head = dec_agent_attn[head].cpu().numpy()  # (K=64, A=32)
            im = ax.imshow(attn_head, cmap='viridis', aspect='auto', interpolation='nearest')
            ax.set_title(f'Head {head}', fontsize=12)
            ax.set_xlabel('Agent Tokens (32)', fontsize=10)
            ax.set_ylabel('Intention Queries (64)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Attention')

        plt.suptitle('Decoder Agent Attention (Layer 0, All Heads)', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / "decoder_agent_attention_heads.png", dpi=150)
        print(f"  ✓ Saved: decoder_agent_attention_heads.png")

    # Viz 3: Decoder attention to map (average across heads)
    if len(attn.decoder_map_attentions) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        attn_map_avg = dec_map_attn.mean(dim=0).cpu().numpy()  # (K=64, M=64)
        im = ax.imshow(attn_map_avg, cmap='plasma', aspect='auto', interpolation='nearest')
        ax.set_title('Decoder Map Attention (Layer 0, Head Average)', fontsize=14)
        ax.set_xlabel('Map Lane Tokens (64)', fontsize=12)
        ax.set_ylabel('Intention Queries (64)', fontsize=12)
        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()
        plt.savefig(output_dir / "decoder_map_attention.png", dpi=150)
        print(f"  ✓ Saved: decoder_map_attention.png")

    # Viz 4: Attention statistics
    print("\n[6/6] Computing attention statistics...")

    # Entropy (measure of focus)
    def entropy(attn_weights):
        """Compute entropy of attention distribution"""
        # attn_weights: (nhead, N, N) or similar
        probs = attn_weights + 1e-10  # avoid log(0)
        return -(probs * torch.log2(probs)).sum(dim=-1).mean().item()

    scene_entropy = entropy(scene_attn)
    print(f"  Scene attention entropy: {scene_entropy:.2f} bits")
    print(f"    (Lower = more focused, Higher = more uniform)")

    # Top-K attention targets
    print(f"\n  Top-5 most attended tokens in scene encoder:")
    attn_sum = attn_avg.sum(axis=0)  # Sum across queries
    top5_idx = np.argsort(attn_sum)[-5:][::-1]
    for i, idx in enumerate(top5_idx):
        token_type = "Agent" if idx < 32 else f"Lane (map index {idx-32})"
        print(f"    {i+1}. Token {idx:2d} ({token_type}): attention = {attn_sum[idx]:.2f}")

    print("\n" + "="*70)
    print("✓ Visualization Demo Complete!")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in output_dir.glob("*.png"):
        print(f"  • {f.name}")

    print("\n" + "="*70)
    print("What these visualizations show:")
    print("="*70)
    print("""
1. scene_attention_matrix.png
   → Shows which tokens (agents + lanes) attend to each other
   → Diagonal = self-attention
   → Off-diagonal = cross-attention between tokens
   → Cyan lines separate agent tokens (0-31) from map tokens (32-95)

2. decoder_agent_attention_heads.png
   → Shows 8 attention heads separately
   → Each head may specialize (e.g., head 0 focuses on front vehicles)
   → Multi-head allows model to attend to different aspects simultaneously

3. decoder_map_attention.png
   → Shows which lanes (64 lanes) each intention query (64 queries) attends to
   → Bright spots = model thinks this lane is relevant for this trajectory mode

⚠️ NOTE: Current visualizations use UNTRAINED or PARTIALLY TRAINED weights!
   Attention patterns will be much more meaningful after 60 epochs of training.

   But the PIPELINE WORKS! Once training completes, same code will generate
   interpretable attention maps showing real decision-making patterns.
""")

    print("\n" + "="*70)
    print("Next steps:")
    print("="*70)
    print("""
1. Wait for training to complete (~48 hours remaining)
2. Re-run this script with trained checkpoint
3. Generate BEV spatial visualizations (need scene geometry)
4. Select interesting scenarios for paper figures
5. Write paper with these visualizations as核心贡献!
""")

if __name__ == "__main__":
    main()
