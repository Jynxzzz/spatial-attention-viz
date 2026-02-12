"""Run all ablation experiments.

Usage:
    python scripts/run_ablation_suite.py \
        --config configs/mtr_lite.yaml \
        --checkpoints-dir /mnt/hdd12t/outputs/mtr_lite/ablations/ \
        --output-dir /mnt/hdd12t/outputs/mtr_lite/ablation_results/
"""

import argparse

from evaluation.ablation import run_ablation_suite


def main():
    parser = argparse.ArgumentParser(description="Run ablation study suite")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--ablations", type=str, nargs="+", default=None,
        help="Specific ablation names to run (default: all)",
    )
    args = parser.parse_args()

    run_ablation_suite(
        base_config_path=args.config,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        ablation_names=args.ablations,
        device=args.device,
    )


if __name__ == "__main__":
    main()
