"""Per-type evaluation: vehicle / pedestrian / cyclist breakdown.

Runs on CPU to avoid interfering with ongoing GPU training.
Uses the best checkpoint and a subset of validation scenes.
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lightning_module import MTRLiteModule
from data.polyline_dataset import PolylineDataset
from training.metrics import MetricAggregator, AGENT_TYPE_NAMES


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/mtr_lite.yaml")
    parser.add_argument("--n-scenes", type=int, default=500,
                        help="Number of val scenes to evaluate (0=all)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Load model
    module = MTRLiteModule.load_from_checkpoint(
        args.checkpoint,
        map_location=args.device,
    )
    module.eval()
    module.to(args.device)
    model = module.model

    # Load validation dataset
    intent_points_path = "/mnt/hdd12t/models/mtr_lite/intent_points_64.npy"
    intent_points = np.load(intent_points_path)

    dataset = PolylineDataset(
        scene_list_path=cfg["data"]["scene_list"],
        split="val",
        history_len=cfg["data"]["history_len"],
        future_len=cfg["data"]["future_len"],
        anchor_frames=cfg["data"]["anchor_frames"],
        max_agents=cfg["data"]["max_agents"],
        max_map_polylines=cfg["data"]["max_map_polylines"],
        map_points_per_lane=cfg["data"]["map_points_per_lane"],
        neighbor_distance=cfg["data"]["neighbor_distance"],
        data_fraction=cfg["data"]["data_fraction"],
        val_ratio=cfg["data"]["val_ratio"],
        augment=False,
    )

    n_total = len(dataset)
    n_eval = min(args.n_scenes, n_total) if args.n_scenes > 0 else n_total
    print(f"Validation set: {n_total} scenes, evaluating {n_eval}")

    # Evaluate
    aggregator = MetricAggregator()
    skipped = 0

    with torch.no_grad():
        for i in range(n_eval):
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n_eval}]")

            sample = dataset[i]
            if sample is None:
                skipped += 1
                continue

            # Create batch of size 1
            batch = {}
            for key, val in sample.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.unsqueeze(0).to(args.device)
                else:
                    batch[key] = [val]

            try:
                output = model(batch, capture_attention=False)

                aggregator.update(
                    pred_trajectories=output["trajectories"],
                    pred_scores=output["scores"],
                    target_future=batch["target_future"],
                    target_future_valid=batch["target_future_valid"],
                    target_mask=batch["target_mask"],
                    target_agent_types=batch.get("target_agent_types"),
                )
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  Warning: scene {i} failed: {e}")
                continue

    results = aggregator.compute()

    print(f"\nEvaluated {n_eval - skipped} scenes (skipped {skipped})")
    print("=" * 60)
    print(f"{'OVERALL':^60}")
    print("=" * 60)
    print(f"  minADE@6:  {results['minADE@6']:.3f} m")
    print(f"  minFDE@6:  {results['minFDE@6']:.3f} m")
    print(f"  MR@6:      {results['MR@6']:.3f}")
    if "ade_3s" in results:
        print(f"  ADE@3s:    {results['ade_3s']:.3f} m")
    if "ade_5s" in results:
        print(f"  ADE@5s:    {results['ade_5s']:.3f} m")
    if "ade_8s" in results:
        print(f"  ADE@8s:    {results['ade_8s']:.3f} m")

    print("\n" + "=" * 60)
    print(f"{'PER-TYPE BREAKDOWN':^60}")
    print("=" * 60)

    for type_idx, type_name in enumerate(AGENT_TYPE_NAMES[:3]):  # vehicle, ped, cyclist
        ade_key = f"{type_name}/minADE@6"
        if ade_key in results:
            count = int(results[f"{type_name}/count"])
            print(f"\n  {type_name.upper()} (n={count}):")
            print(f"    minADE@6:  {results[f'{type_name}/minADE@6']:.3f} m")
            print(f"    minFDE@6:  {results[f'{type_name}/minFDE@6']:.3f} m")
            print(f"    MR@6:      {results[f'{type_name}/MR@6']:.3f}")
        else:
            print(f"\n  {type_name.upper()}: no samples")

    # Summary comparison table
    print("\n" + "=" * 60)
    print(f"{'COMPARISON TABLE':^60}")
    print("=" * 60)
    print(f"  {'Type':<15} {'minADE@6':>10} {'minFDE@6':>10} {'MR@6':>8} {'Count':>8}")
    print(f"  {'-'*51}")
    for type_name in AGENT_TYPE_NAMES[:3]:
        ade_key = f"{type_name}/minADE@6"
        if ade_key in results:
            count = int(results[f"{type_name}/count"])
            print(f"  {type_name:<15} {results[f'{type_name}/minADE@6']:>10.3f} "
                  f"{results[f'{type_name}/minFDE@6']:>10.3f} "
                  f"{results[f'{type_name}/MR@6']:>8.3f} {count:>8}")
    print(f"  {'-'*51}")
    print(f"  {'ALL':<15} {results['minADE@6']:>10.3f} "
          f"{results['minFDE@6']:>10.3f} "
          f"{results['MR@6']:>8.3f} {n_eval - skipped:>8}")


if __name__ == "__main__":
    main()
