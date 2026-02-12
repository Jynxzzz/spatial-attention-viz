# Spatial Attention Visualization for Trajectory Prediction

**Discovering Safety Blind Spots in Transformer-Based Autonomous Driving Through Counterfactual Analysis**

[![Paper](https://img.shields.io/badge/Paper-MDPI%20Sustainability-blue)](https://obsicat.com/attention-visualization.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

<p align="center">
  <img src="assets/fig_spatial_attention_composite.png" width="100%" alt="Spatial attention patterns across diverse driving scenarios">
  <br>
  <em>Bird's-eye-view attention visualization across diverse driving scenarios, revealing how the Transformer allocates attention to agents and map elements.</em>
</p>

---

## Key Findings

We present a **spatial attention visualization framework** that maps abstract Transformer attention weights onto bird's-eye-view (BEV) traffic scenes. Built on **MTR-Lite** (8.48M parameters) trained on the Waymo Open Motion Dataset (~17,800 scenes), our analysis reveals:

| Finding | Key Statistic | Figure |
|---------|--------------|--------|
| **Non-Monotonic Layer Specialization** | Entropy: 5.64 &rarr; 5.36 &rarr; 5.92 bits | [Fig 5](assets/fig_entropy_evolution.png) |
| **Tunnel Vision Failure Mode** | 40% higher self-attention in failed predictions | [Fig 6](assets/fig_failure_diagnosis.png) |
| **Cyclist Safety Blind Spot** | 73% less attention, 88.1% miss rate | [Fig 7](assets/fig_cyclist_failure.png) |
| **Far-Range Attention is Essential** | +4.7% minADE from distance masking | [Fig 13](assets/fig_distance_ablation.png) |

<p align="center">
  <img src="assets/fig_entropy_evolution.png" width="48%" alt="Entropy evolution across layers">
  <img src="assets/fig_cyclist_failure.png" width="48%" alt="Cyclist attention deficit">
</p>

---

## Architecture

```
MTR-Lite (8.48M parameters)
  Encoder: 4 layers, 8 heads, d=256, ff=1024
    Input: 32 agent tokens + 64 map tokens
    Global self-attention with attention capture

  Decoder: 4 layers, 8 heads, d=256
    64 intention queries -> K=6 output modes (NMS)
    Agent cross-attention + Map cross-attention

  Visualization Framework:
    Attention-Capture Layers -> Spatial Token Bookkeeping -> BEV Heatmaps
```

## Project Structure

```
spatial-attention-viz/
  model/            # MTR-Lite Transformer (encoder, decoder, attention hooks)
  data/             # Waymo dataset loader, token bookkeeping, polyline features
  training/         # PyTorch Lightning training pipeline
  evaluation/       # Metrics (minADE, minFDE, Miss Rate), ablation studies
  visualization/    # BEV heatmaps, attention animations, composite figures
  experiments/      # Counterfactual scene editing framework
  scripts/          # Analysis & figure generation scripts
  configs/          # Training configurations (YAML)
  paper/            # LaTeX source (MDPI Sustainability format)
  tests/            # Unit & integration tests
```

## Installation

```bash
git clone https://github.com/Jynxzzz/spatial-attention-viz.git
cd spatial-attention-viz
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, PyTorch Lightning, matplotlib, numpy, scipy, networkx, Pillow, PyYAML

**Data:** Download the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) (requires license agreement).

## Usage

### Training

```bash
# Full training (~50h on RTX 4090)
python training/train.py --config configs/mtr_lite.yaml

# Quick smoke test (~5 min)
python training/train.py --config configs/mtr_lite_smoke_test.yaml
```

### Evaluation

```bash
# Full validation set evaluation
python run_full_eval.py --checkpoint path/to/checkpoint.ckpt

# Constant-velocity baseline comparison
python run_cv_baseline.py
```

### Visualization

```bash
# BEV attention heatmaps
python scripts/demo_attention_viz.py

# Animated attention evolution (GIF)
python scripts/demo_attention_gif.py

# Temporal attention dynamics
python scripts/demo_temporal_animation.py

# Paper figures
python scripts/generate_paper_figures.py
```

### Analysis

```bash
# Entropy evolution across layers
python scripts/analysis_entropy.py

# Failure mode attention diagnosis
python scripts/analysis_failure_attention.py

# Scene-type attention adaptation
python scripts/analysis_scene_type_attention.py

# Distance masking ablation
python scripts/experiment_distance_mask.py
```

## Results

| Model | minADE@6 | minFDE@6 | Miss Rate |
|-------|----------|----------|-----------|
| Constant Velocity | 5.076 m | 14.141 m | - |
| **MTR-Lite (Ours)** | **2.314 m** | **6.401 m** | 54.0% (vehicles) |

Cyclists show 88.1% miss rate vs 54.0% for vehicles, receiving 73% less self-attention (0.026 vs 0.045).

## Citation

```bibtex
@article{zhou2026spatial,
  title={Spatial Attention Visualization for Interpretable Trajectory Prediction
         in Autonomous Driving: Discovering Safety Blind Spots Through
         Counterfactual Analysis},
  author={Zhou, Xingnan and Alecsandru, Ciprian},
  journal={Sustainability},
  year={2026},
  publisher={MDPI}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) for training data
- [Motion Transformer (MTR)](https://github.com/sshaoshuai/MTR) for architectural inspiration
- Concordia University for computational resources
