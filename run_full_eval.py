"""Full validation set evaluation of MTR-Lite model.

Evaluates on the FULL Waymo validation set (all val scenes, not just 20%).
Reports overall + per-agent-type metrics and saves to JSON.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.collate import mtr_collate_fn
from data.polyline_dataset import PolylineDataset
from training.lightning_module import MTRLiteModule
from training.metrics import MetricAggregator, AGENT_TYPE_NAMES


@torch.no_grad()
def run_full_evaluation(
    checkpoint_path: str,
    config_path: str,
    batch_size: int = 4,
    num_workers: int = 8,
    device: str = "cuda:0",
    output_json: str = "eval_results.json",
):
    """Run full evaluation on the complete validation set.

    Uses data_fraction=1.0 to evaluate ALL val scenes.
    """
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    print("=" * 70)
    print("MTR-Lite Full Validation Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print()

    # Load model
    print("Loading model checkpoint...")
    t0 = time.time()
    module = MTRLiteModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    module = module.to(device)
    module.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Count parameters
    n_params = sum(p.numel() for p in module.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Build FULL validation dataset (data_fraction=1.0)
    print("\nBuilding validation dataset (data_fraction=1.0, full val set)...")
    val_dataset = PolylineDataset(
        scene_list_path=data_cfg["scene_list"],
        split="val",
        val_ratio=data_cfg.get("val_ratio", 0.15),
        data_fraction=1.0,  # FULL validation set
        history_len=data_cfg["history_len"],
        future_len=data_cfg["future_len"],
        max_agents=data_cfg["max_agents"],
        max_map_polylines=data_cfg["max_map_polylines"],
        map_points_per_lane=data_cfg["map_points_per_lane"],
        neighbor_distance=data_cfg.get("neighbor_distance", 50.0),
        anchor_frames=data_cfg.get("anchor_frames", [10]),
        max_targets=data_cfg["model"].get("max_targets", 8) if "model" in data_cfg else cfg.get("model", {}).get("max_targets", 8),
        augment=False,
        seed=cfg.get("seed", 42),
    )

    n_val = len(val_dataset)
    print(f"Validation set size: {n_val} scenes")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mtr_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    total_batches = len(val_loader)
    print(f"Total batches: {total_batches}")

    # Run evaluation
    print("\n--- Starting evaluation ---")
    aggregator = MetricAggregator()

    skipped_batches = 0
    total_samples = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(val_loader):
        if batch is None:
            skipped_batches += 1
            continue

        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # Forward pass
        output = module(batch, capture_attention=False)

        # Update metrics (with per-type tracking)
        aggregator.update(
            pred_trajectories=output["trajectories"].detach(),
            pred_scores=output["scores"].detach(),
            target_future=batch["target_future"],
            target_future_valid=batch["target_future_valid"],
            target_mask=batch["target_mask"],
            target_agent_types=batch.get("target_agent_types"),
        )

        batch_samples = batch["target_mask"].sum().item()
        total_samples += batch_samples

        # Progress reporting
        if (batch_idx + 1) % 200 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - t_start
            pct = 100 * (batch_idx + 1) / total_batches
            speed = (batch_idx + 1) / elapsed
            eta = (total_batches - batch_idx - 1) / speed if speed > 0 else 0
            interim = aggregator.compute()
            print(
                f"  [{batch_idx+1:5d}/{total_batches}] ({pct:5.1f}%) "
                f"minADE@6={interim['minADE@6']:.3f} "
                f"minFDE@6={interim['minFDE@6']:.3f} "
                f"MR@6={interim['MR@6']:.3f} "
                f"[{speed:.1f} batch/s, ETA {eta:.0f}s]"
            )

    elapsed_total = time.time() - t_start

    # Compute final metrics
    metrics = aggregator.compute()

    # Print results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"  Total val scenes:     {n_val}")
    print(f"  Skipped batches:      {skipped_batches}")
    print(f"  Total target samples: {total_samples}")
    print(f"  Evaluation time:      {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print()
    print(f"  minADE@6:  {metrics['minADE@6']:.4f} m")
    print(f"  minFDE@6:  {metrics['minFDE@6']:.4f} m")
    print(f"  MR@6:      {metrics['MR@6']:.4f}")
    if "ade_3s" in metrics:
        print(f"  ADE@3s:    {metrics['ade_3s']:.4f} m")
    if "ade_5s" in metrics:
        print(f"  ADE@5s:    {metrics['ade_5s']:.4f} m")
    if "ade_8s" in metrics:
        print(f"  ADE@8s:    {metrics['ade_8s']:.4f} m")

    # Per-type results
    print("\n" + "=" * 70)
    print("PER-TYPE BREAKDOWN")
    print("=" * 70)
    print(f"  {'Type':<15} {'minADE@6':>10} {'minFDE@6':>10} {'MR@6':>8} {'Count':>8}")
    print(f"  {'-' * 51}")

    per_type_results = {}
    for type_idx, type_name in enumerate(AGENT_TYPE_NAMES[:3]):
        ade_key = f"{type_name}/minADE@6"
        if ade_key in metrics:
            count = int(metrics[f"{type_name}/count"])
            ade = metrics[f"{type_name}/minADE@6"]
            fde = metrics[f"{type_name}/minFDE@6"]
            mr = metrics[f"{type_name}/MR@6"]
            print(f"  {type_name:<15} {ade:>10.4f} {fde:>10.4f} {mr:>8.4f} {count:>8}")
            per_type_results[type_name] = {
                "minADE@6": float(ade),
                "minFDE@6": float(fde),
                "MR@6": float(mr),
                "count": count,
            }
        else:
            print(f"  {type_name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'0':>8}")

    print(f"  {'-' * 51}")
    print(f"  {'ALL':<15} {metrics['minADE@6']:>10.4f} {metrics['minFDE@6']:>10.4f} {metrics['MR@6']:>8.4f} {total_samples:>8}")

    # Prepare JSON output
    results = {
        "model": "MTR-Lite",
        "checkpoint": checkpoint_path,
        "config": config_path,
        "evaluation_date": datetime.now().isoformat(),
        "device": device,
        "batch_size": batch_size,
        "dataset": {
            "scene_list": data_cfg["scene_list"],
            "total_scenes": n_val,
            "data_fraction": 1.0,
            "val_ratio": data_cfg.get("val_ratio", 0.15),
            "skipped_batches": skipped_batches,
            "total_target_samples": total_samples,
        },
        "model_info": {
            "parameters": n_params,
            "parameters_M": round(n_params / 1e6, 1),
        },
        "overall_metrics": {
            "minADE@6": float(metrics["minADE@6"]),
            "minFDE@6": float(metrics["minFDE@6"]),
            "MR@6": float(metrics["MR@6"]),
        },
        "horizon_metrics": {},
        "per_type_metrics": per_type_results,
        "evaluation_time_seconds": round(elapsed_total, 1),
    }

    # Add horizon metrics
    for key in ["ade_3s", "ade_5s", "ade_8s"]:
        if key in metrics:
            results["horizon_metrics"][key] = float(metrics[key])

    # Save JSON
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        output_json,
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Full validation evaluation of MTR-Lite")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/hdd12t/outputs/mtr_lite/checkpoints/last.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mtr_lite.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Make config path absolute if relative
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    run_full_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=config_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
