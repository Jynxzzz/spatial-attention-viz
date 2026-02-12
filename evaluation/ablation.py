"""Ablation study runner.

Runs predefined ablation experiments by modifying configs:
1. Encoder depth: 2/4/6 layers
2. No-map: remove map polylines
3. No-agent: only use ego agent
4. Intention count: 32 vs 64 vs 128
"""

import copy
import os

import yaml

from evaluation.evaluate import evaluate


ABLATION_CONFIGS = {
    "enc_2_layers": {"model.num_encoder_layers": 2},
    "enc_6_layers": {"model.num_encoder_layers": 6},
    "no_map": {"model.max_map_polylines": 0},
    "no_neighbor_agents": {"model.max_agents": 1},
    "intent_32": {"model.num_intentions": 32},
    "intent_128": {"model.num_intentions": 128},
    "dec_2_layers": {"model.num_decoder_layers": 2},
    "d_model_192": {"model.d_model": 192, "model.dim_feedforward": 768},
}


def modify_config(base_cfg: dict, modifications: dict) -> dict:
    """Apply dotted-key modifications to config dict."""
    cfg = copy.deepcopy(base_cfg)
    for key, value in modifications.items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = value
    return cfg


def run_ablation_suite(
    base_config_path: str,
    checkpoints_dir: str,
    output_dir: str,
    ablation_names: list = None,
    device: str = "cuda",
):
    """Run ablation experiments and collect results.

    This function evaluates pre-trained ablation checkpoints.
    Each ablation model should be trained separately with the modified config.

    Args:
        base_config_path: path to base config YAML
        checkpoints_dir: directory containing ablation checkpoints
        output_dir: where to save results
        ablation_names: specific ablations to run (None = all)
        device: cuda or cpu
    """
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)

    ablations = ablation_names or list(ABLATION_CONFIGS.keys())
    results = {}

    for name in ablations:
        if name not in ABLATION_CONFIGS:
            print(f"Unknown ablation: {name}")
            continue

        ckpt_path = os.path.join(checkpoints_dir, f"{name}", "last.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {name}: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"{'='*60}")

        # Create modified config
        mods = ABLATION_CONFIGS[name]
        cfg = modify_config(base_cfg, mods)

        # Save temp config
        temp_cfg_path = os.path.join(output_dir, f"{name}_config.yaml")
        with open(temp_cfg_path, "w") as f:
            yaml.dump(cfg, f)

        try:
            metrics = evaluate(ckpt_path, temp_cfg_path, device=device)
            results[name] = metrics
        except Exception as e:
            print(f"Ablation {name} failed: {e}")
            results[name] = {"error": str(e)}

    # Save results
    results_path = os.path.join(output_dir, "ablation_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n=== Ablation Results Summary ===")
    print(f"{'Name':<25} {'minADE@6':>10} {'minFDE@6':>10} {'MR@6':>10}")
    print("-" * 60)
    for name, metrics in results.items():
        if "error" in metrics:
            print(f"{name:<25} {'ERROR':>10}")
        else:
            print(f"{name:<25} {metrics.get('minADE@6', 0):>10.3f} "
                  f"{metrics.get('minFDE@6', 0):>10.3f} {metrics.get('MR@6', 0):>10.3f}")

    return results
