"""Batch evaluation on validation set.

Loads a trained checkpoint, runs inference on the validation set,
and computes minADE@6, minFDE@6, MR@6, and horizon ADE metrics.
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from data.collate import mtr_collate_fn
from data.polyline_dataset import PolylineDataset
from training.lightning_module import MTRLiteModule
from training.metrics import MetricAggregator


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    config_path: str,
    data_fraction: float = None,
    batch_size: int = None,
    device: str = "cuda",
) -> dict:
    """Run full evaluation on validation set.

    Args:
        checkpoint_path: path to .ckpt file
        config_path: path to config YAML
        data_fraction: override data fraction (for quick eval)
        batch_size: override batch size
        device: cuda or cpu

    Returns:
        dict of metric name -> value
    """
    cfg = yaml.safe_load(open(config_path))
    data_cfg = cfg["data"]

    # Load model
    module = MTRLiteModule.load_from_checkpoint(checkpoint_path)
    module = module.to(device)
    module.eval()

    # Build val dataset
    val_dataset = PolylineDataset(
        scene_list_path=data_cfg["scene_list"],
        split="val",
        val_ratio=data_cfg.get("val_ratio", 0.15),
        data_fraction=data_fraction or data_cfg.get("data_fraction", 1.0),
        history_len=data_cfg["history_len"],
        future_len=data_cfg["future_len"],
        max_agents=data_cfg["max_agents"],
        max_map_polylines=data_cfg["max_map_polylines"],
        map_points_per_lane=data_cfg["map_points_per_lane"],
        neighbor_distance=data_cfg.get("neighbor_distance", 50.0),
        anchor_frames=data_cfg.get("anchor_frames", [10]),
        augment=False,
        seed=cfg.get("seed", 42),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size or cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=mtr_collate_fn,
        pin_memory=True,
    )

    aggregator = MetricAggregator()

    print(f"Evaluating {len(val_dataset)} validation samples...")
    for batch_idx, batch in enumerate(val_loader):
        if batch is None:
            continue

        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        output = module(batch, capture_attention=False)

        aggregator.update(
            pred_trajectories=output["trajectories"].detach(),
            pred_scores=output["scores"].detach(),
            target_future=batch["target_future"],
            target_future_valid=batch["target_future_valid"],
            target_mask=batch["target_mask"],
        )

        if (batch_idx + 1) % 100 == 0:
            interim = aggregator.compute()
            print(f"  Batch {batch_idx+1}: minADE@6={interim['minADE@6']:.3f}")

    metrics = aggregator.compute()
    print("\n=== Evaluation Results ===")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-fraction", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate(
        args.checkpoint, args.config,
        data_fraction=args.data_fraction,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
