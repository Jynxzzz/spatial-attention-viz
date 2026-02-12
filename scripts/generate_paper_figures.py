"""Generate all paper figures from a trained model.

Usage:
    python scripts/generate_paper_figures.py \
        --checkpoint /mnt/hdd12t/outputs/mtr_lite/checkpoints/last.ckpt \
        --config configs/mtr_lite.yaml \
        --output-dir paper/figures/
"""

import argparse
import os

import torch
import yaml

from evaluation.qualitative import select_interesting_scenes
from training.lightning_module import MTRLiteModule
from visualization.attention_extractor import extract_scene_attention
from visualization.composite_figure import generate_composite_figure
from visualization.space_attention_bev import render_space_attention_bev
from visualization.time_attention_diagram import render_time_attention_diagram
from visualization.lane_token_activation import render_lane_activation_map


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="paper/figures/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-scenes", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))

    # Load model
    print("Loading model...")
    module = MTRLiteModule.load_from_checkpoint(args.checkpoint)
    model = module.model.to(args.device)
    model.eval()

    # Select interesting scenes
    print("Selecting interesting scenes...")
    categories = select_interesting_scenes(
        cfg["data"]["scene_list"],
        n_scenes=args.n_scenes * 4,
        max_scan=1000,
    )

    # Generate figures for each category
    for cat_name, scenes in categories.items():
        if not scenes:
            continue

        print(f"\n--- Generating figures for: {cat_name} ---")
        extraction_results = []

        for i, scene_props in enumerate(scenes[:args.n_scenes]):
            scene_path = scene_props["scene_path"]
            print(f"  Processing scene {i+1}: {os.path.basename(scene_path)}")

            try:
                result = extract_scene_attention(
                    model, scene_path,
                    anchor_frame=10,
                    history_len=cfg["data"]["history_len"],
                    future_len=cfg["data"]["future_len"],
                    max_agents=cfg["model"]["max_agents"],
                    max_map_polylines=cfg["model"]["max_map_polylines"],
                    map_points_per_lane=cfg["model"]["map_points_per_lane"],
                    device=args.device,
                )
                extraction_results.append(result)
            except Exception as e:
                print(f"    Failed: {e}")
                continue

        if extraction_results:
            # Composite figure
            save_path = os.path.join(args.output_dir, f"composite_{cat_name}.pdf")
            generate_composite_figure(
                extraction_results,
                scenario_names=[f"{cat_name.title()} {i+1}" for i in range(len(extraction_results))],
                save_path=save_path,
                dpi=300,
            )

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
