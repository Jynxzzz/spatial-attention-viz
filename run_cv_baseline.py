"""Constant Velocity (CV) baseline evaluation.

Evaluates a simple constant-velocity extrapolation baseline on the same
Waymo validation set used by MTR-Lite.

CV model: position(t) = position(t_last) + velocity * (t - t_last)
          velocity = (position(t_last) - position(t_{last-1})) / dt

Since this is a deterministic single-mode baseline, minADE@K = ADE for any K.
"""

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
from training.metrics import MetricAggregator, AGENT_TYPE_NAMES


def constant_velocity_predict(
    agent_polylines: torch.Tensor,
    agent_valid: torch.Tensor,
    target_agent_indices: torch.Tensor,
    target_mask: torch.Tensor,
    future_len: int = 80,
    dt: float = 0.1,
) -> torch.Tensor:
    """Generate constant-velocity predictions for target agents.

    Uses the last two valid history positions to compute velocity,
    then extrapolates linearly for future_len timesteps.

    The agent_polylines feature layout is:
        pos(2) + prev_pos(2) + vel(2) + accel(2) + heading_sincos(2) +
        bbox(2) + type_onehot(5) + temporal_embed(11) + is_ego(1) = 29

    We use the velocity field (indices 4:6) at the last history timestep
    (the anchor frame, index 10) for extrapolation.

    Args:
        agent_polylines: (B, A, H, 29) agent history features
        agent_valid: (B, A, H) validity mask
        target_agent_indices: (B, T) which agent slots are targets
        target_mask: (B, T) which target slots are active
        future_len: number of future timesteps to predict
        dt: time between frames (0.1s for 10Hz)

    Returns:
        pred_trajectories: (B, T, 1, future_len, 2) single-mode predictions
    """
    B, A, H, F = agent_polylines.shape
    T = target_agent_indices.shape[1]
    device = agent_polylines.device

    pred = torch.zeros(B, T, 1, future_len, 2, device=device)

    for b in range(B):
        for t in range(T):
            if not target_mask[b, t]:
                continue

            a_idx = target_agent_indices[b, t].item()
            if a_idx < 0 or a_idx >= A:
                continue

            # Get the agent's history polyline
            polyline = agent_polylines[b, a_idx]  # (H, 29)
            valid = agent_valid[b, a_idx]          # (H,)

            # Find the last valid timestep (should be H-1 = 10, the anchor)
            valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
            if len(valid_indices) == 0:
                continue

            last_idx = valid_indices[-1].item()

            # Position at last valid timestep: features[0:2]
            last_pos = polyline[last_idx, 0:2]  # (2,)

            # Velocity at last valid timestep: features[4:6]
            # This is already in BEV coordinates
            vel = polyline[last_idx, 4:6]  # (2,)

            # Extrapolate: pos(t) = last_pos + vel * (t+1) * dt
            # t goes from 0..future_len-1, so time offset is (t+1)*dt
            time_offsets = torch.arange(1, future_len + 1, device=device, dtype=torch.float32) * dt
            pred[b, t, 0, :, 0] = last_pos[0] + vel[0] * time_offsets
            pred[b, t, 0, :, 1] = last_pos[1] + vel[1] * time_offsets

    return pred


def run_cv_evaluation(
    config_path: str,
    batch_size: int = 8,
    num_workers: int = 8,
    output_json: str = "cv_baseline_results.json",
):
    """Run CV baseline evaluation on the full validation set."""

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    print("=" * 70)
    print("Constant Velocity Baseline Evaluation")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print()

    # Build validation dataset (same split as MTR-Lite)
    print("Building validation dataset (data_fraction=1.0, full val set)...")
    val_dataset = PolylineDataset(
        scene_list_path=data_cfg["scene_list"],
        split="val",
        val_ratio=data_cfg.get("val_ratio", 0.15),
        data_fraction=1.0,
        history_len=data_cfg["history_len"],
        future_len=data_cfg["future_len"],
        max_agents=data_cfg["max_agents"],
        max_map_polylines=data_cfg["max_map_polylines"],
        map_points_per_lane=data_cfg["map_points_per_lane"],
        neighbor_distance=data_cfg.get("neighbor_distance", 50.0),
        anchor_frames=data_cfg.get("anchor_frames", [10]),
        max_targets=cfg.get("model", {}).get("max_targets", 8),
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
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    total_batches = len(val_loader)
    print(f"Total batches: {total_batches}")

    # Run evaluation
    print("\n--- Starting CV baseline evaluation ---")
    aggregator = MetricAggregator()

    skipped_batches = 0
    total_samples = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(val_loader):
        if batch is None:
            skipped_batches += 1
            continue

        # CV prediction (CPU is fine, no model needed)
        pred_trajectories = constant_velocity_predict(
            agent_polylines=batch["agent_polylines"],
            agent_valid=batch["agent_valid"],
            target_agent_indices=batch["target_agent_indices"],
            target_mask=batch["target_mask"],
            future_len=data_cfg["future_len"],
            dt=0.1,
        )

        # For minADE@6 compatibility, expand single mode to 6 modes
        # (all 6 modes are identical for deterministic baseline)
        pred_trajectories_6 = pred_trajectories.expand(-1, -1, 6, -1, -1)

        # Compute scores (uniform since all modes are identical)
        B, T = batch["target_mask"].shape
        pred_scores = torch.ones(B, T, 6) / 6.0

        # Update metrics
        aggregator.update(
            pred_trajectories=pred_trajectories_6,
            pred_scores=pred_scores,
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
                f"ADE={interim['minADE@6']:.3f} "
                f"FDE={interim['minFDE@6']:.3f} "
                f"MR={interim['MR@6']:.3f} "
                f"[{speed:.1f} batch/s, ETA {eta:.0f}s]"
            )

    elapsed_total = time.time() - t_start

    # Compute final metrics
    metrics = aggregator.compute()

    # Print results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS (Constant Velocity Baseline)")
    print("=" * 70)
    print(f"  Total val scenes:     {n_val}")
    print(f"  Skipped batches:      {skipped_batches}")
    print(f"  Total target samples: {total_samples}")
    print(f"  Evaluation time:      {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print()
    print(f"  minADE@6 (=ADE@1):  {metrics['minADE@6']:.4f} m")
    print(f"  minFDE@6 (=FDE@1):  {metrics['minFDE@6']:.4f} m")
    print(f"  MR@6:               {metrics['MR@6']:.4f}")
    if "ade_3s" in metrics:
        print(f"  ADE@3s:             {metrics['ade_3s']:.4f} m")
    if "ade_5s" in metrics:
        print(f"  ADE@5s:             {metrics['ade_5s']:.4f} m")
    if "ade_8s" in metrics:
        print(f"  ADE@8s:             {metrics['ade_8s']:.4f} m")

    # Per-type results
    print("\n" + "=" * 70)
    print("PER-TYPE BREAKDOWN")
    print("=" * 70)
    print(f"  {'Type':<15} {'ADE':>10} {'FDE':>10} {'MR':>8} {'Count':>8}")
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
        "model": "Constant Velocity Baseline",
        "description": "Linear extrapolation using last observed velocity: pos(t) = pos(t_last) + vel * (t - t_last)",
        "evaluation_date": datetime.now().isoformat(),
        "dataset": {
            "scene_list": data_cfg["scene_list"],
            "total_scenes": n_val,
            "data_fraction": 1.0,
            "val_ratio": data_cfg.get("val_ratio", 0.15),
            "seed": cfg.get("seed", 42),
            "skipped_batches": skipped_batches,
            "total_target_samples": total_samples,
        },
        "prediction_setup": {
            "history_len": data_cfg["history_len"],
            "future_len": data_cfg["future_len"],
            "dt": 0.1,
            "num_modes": 1,
            "note": "Deterministic baseline: minADE@1 = minADE@6 since only one mode exists",
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Constant Velocity baseline evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mtr_lite.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--output", type=str, default="cv_baseline_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Make config path absolute if relative
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    run_cv_evaluation(
        config_path=config_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_json=args.output,
    )
