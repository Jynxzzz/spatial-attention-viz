"""Generate k-means intention points from training data.

Usage:
    python scripts/generate_intent_points.py \
        --scene-list /home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt \
        --output /mnt/hdd12t/outputs/mtr_lite/intent_points_64.npy \
        --k 64 --max-scenes 10000
"""

import argparse

from data.intent_points import generate_and_save


def main():
    parser = argparse.ArgumentParser(description="Generate intention anchor points via k-means")
    parser.add_argument(
        "--scene-list", type=str, required=True,
        help="Path to scene list txt file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output .npy file path for intention points",
    )
    parser.add_argument("--k", type=int, default=64, help="Number of clusters")
    parser.add_argument("--max-scenes", type=int, default=10000, help="Max scenes to scan")
    parser.add_argument("--future-len", type=int, default=80, help="Future horizon")
    args = parser.parse_args()

    generate_and_save(
        scene_list_path=args.scene_list,
        output_path=args.output,
        k=args.k,
        future_len=args.future_len,
        max_scenes=args.max_scenes,
    )


if __name__ == "__main__":
    main()
