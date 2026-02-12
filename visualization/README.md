# Attention Visualization Pipeline

This directory contains all visualization modules for interpreting transformer attention in trajectory prediction models.

## Overview

The visualization pipeline provides three core visualization types for understanding how the MTR-Lite model attends to agents and map elements:

1. **Space-Attention BEV Heatmap**: Projects attention weights onto bird's-eye-view coordinates
2. **Time-Attention Refinement Diagram**: Shows attention evolution across decoder layers
3. **Lane-Token Activation Map**: Visualizes which lanes guide the prediction

## Module Structure

```
visualization/
├── attention_extractor.py    # Forward pass with attention capture
├── space_attention_bev.py     # BEV heatmap rendering
├── time_attention_diagram.py  # Decoder attention evolution
├── lane_token_activation.py   # Lane activation visualization
├── composite_figure.py        # Multi-panel paper figures
├── animation.py               # Animated attention GIFs
├── utils.py                   # Utility functions
├── colormap.py               # Custom colormaps and colors
└── README.md                 # This file
```

## Quick Start

### 1. Extract Attention from a Scene

```python
from visualization.attention_extractor import extract_scene_attention

# Run model and extract attention
result = extract_scene_attention(
    model=trained_model,
    scene_path="/path/to/scene.pkl",
    anchor_frame=10,
    device="cuda",
)

# result contains:
# - attention_maps: AttentionMaps object with all captured attention
# - bookkeeper: TokenBookkeeper for mapping tokens to entities
# - predictions: model output
# - scene_data: raw scene dict
# - agent_data, map_data: extracted features
```

### 2. Generate Space-Attention BEV Heatmap

```python
from visualization.space_attention_bev import render_space_attention_bev

fig = render_space_attention_bev(
    agent_positions_bev=agent_positions,      # (A, 2)
    agent_attention=agent_attn_weights,       # (A,)
    agent_mask=agent_mask,                    # (A,)
    lane_centerlines_bev=lane_centerlines,    # (M, P, 2)
    map_attention=map_attn_weights,           # (M,)
    map_mask=map_mask,                        # (M,)
    target_history_bev=history_traj,          # (11, 2) optional
    target_future_bev=gt_future,              # (80, 2) optional
    pred_trajectories_bev=predictions,        # (K, 80, 2) optional
    bev_range=60.0,
    cmap="magma",
)
fig.savefig("space_attention.png", dpi=300)
```

### 3. Generate Time-Attention Diagram

```python
from visualization.time_attention_diagram import render_time_attention_diagram

fig = render_time_attention_diagram(
    decoder_agent_attns=[layer1_attn, layer2_attn, layer3_attn, layer4_attn],
    decoder_map_attns=[layer1_map, layer2_map, layer3_map, layer4_map],
    mode_idx=0,  # Winning mode index
    agent_labels=["Agent 0 (Ego)", "Agent 1", ...],
    map_labels=["Lane 0", "Lane 1", ...],
    agent_mask=agent_mask,
    map_mask=map_mask,
    top_k=10,
)
fig.savefig("time_attention.png", dpi=300)
```

### 4. Generate Lane Activation Map

```python
from visualization.lane_token_activation import render_lane_activation_map

# Cumulative attention across all decoder layers
cumulative_attn = sum(layer_map_attn for layer_map_attn in decoder_map_attns)

fig = render_lane_activation_map(
    lane_centerlines_bev=lane_centerlines,
    lane_attention=cumulative_attn,
    lane_mask=map_mask,
    target_history_bev=history_traj,
    target_future_bev=gt_future,
    pred_trajectories_bev=predictions,
    bev_range=60.0,
    cmap="RdYlBu_r",
)
fig.savefig("lane_activation.png", dpi=300)
```

### 5. Generate Composite Paper Figure

```python
from visualization.composite_figure import generate_composite_figure

fig = generate_composite_figure(
    extraction_results=[result1, result2, result3],  # Multiple scenes
    scenario_names=["Intersection", "Lane Change", "Highway"],
    figsize=(20, 7),
    save_path="paper_figure.png",
    dpi=300,
)
```

### 6. Create Animated GIF

```python
from visualization.animation import create_layer_refinement_gif

create_layer_refinement_gif(
    decoder_agent_attns=decoder_agent_attns,
    decoder_map_attns=decoder_map_attns,
    mode_idx=0,
    agent_positions_bev=agent_positions,
    agent_mask=agent_mask,
    lane_centerlines_bev=lane_centerlines,
    map_mask=map_mask,
    target_history_bev=history_traj,
    target_future_bev=gt_future,
    pred_trajectories_bev=predictions,
    save_path="attention_evolution.gif",
    fps=2,
)
```

## Utility Functions

### Spatial Attention Heatmap

```python
from visualization.utils import get_spatial_attention_map

heatmap = get_spatial_attention_map(
    attn_weights=combined_attention,  # (A+M,)
    agent_positions=agent_pos,
    agent_mask=agent_mask,
    map_centerlines=lane_centerlines,
    map_mask=map_mask,
    bev_range=60.0,
    resolution=0.5,  # meters per pixel
)
# heatmap: (H, W) normalized attention grid
```

### Entity-Level Attention Aggregation

```python
from visualization.utils import aggregate_attention_to_entity

agg = aggregate_attention_to_entity(
    attn_weights=scene_attention,  # (nhead, A+M) or (A+M,)
    num_agents=32,
    num_map=64,
    agent_mask=agent_mask,
    map_mask=map_mask,
    agg_method="mean",
)
# Returns:
# {
#   "agent_attention": (A,),
#   "map_attention": (M,),
#   "agent_total": scalar,
#   "map_total": scalar,
# }
```

### Attention Statistics

```python
from visualization.utils import compute_attention_statistics

stats = compute_attention_statistics(attn_weights)
# Returns:
# {
#   "mean": float,
#   "std": float,
#   "min": float,
#   "max": float,
#   "entropy": float (bits),
#   "gini": float [0, 1],
#   "top1_ratio": float,
#   "top5_ratio": float,
# }
```

## Colormaps

### Traffic Engineering Colors

```python
from visualization.colormap import (
    COLOR_LANE_EGO,
    COLOR_HISTORY,
    COLOR_GT_FUTURE,
    COLOR_PRED_TRAJECTORY,
    get_color_palette,
)

palette = get_color_palette("traffic")
# Returns dict with standard colors
```

### Attention Colormaps

```python
from visualization.colormap import get_attention_colormap

cmap = get_attention_colormap("magma")
# Recommended colormaps:
# - "magma": purple-red-yellow (attention intensity)
# - "viridis": purple-green-yellow (perceptually uniform)
# - "YlOrRd": yellow-orange-red (lane activation)
# - "RdYlBu_r": red-yellow-blue (diverging)
```

### Color Utilities

```python
from visualization.colormap import (
    attention_to_color,
    attention_to_alpha,
    attention_to_linewidth,
)

color = attention_to_color(0.7, vmin=0, vmax=1, cmap_name="magma")
alpha = attention_to_alpha(0.7, alpha_range=(0.2, 1.0))
linewidth = attention_to_linewidth(0.7, width_range=(1.0, 5.0))
```

## Coordinate Systems

### BEV Coordinate System

The visualization pipeline uses a bird's-eye-view (BEV) coordinate system centered on the ego vehicle:

- **X+**: Forward (aligned with ego heading)
- **Y+**: Left
- **Origin**: Ego position at anchor frame
- **Units**: Meters

### Coordinate Transformations

```python
from visualization.utils import world_to_bev, bev_to_world

# World -> BEV
points_bev = world_to_bev(
    points_world,  # (N, 2) or (N, M, 2)
    ego_pos=(x, y),
    ego_heading=heading_radians,
)

# BEV -> World
points_world = bev_to_world(
    points_bev,
    ego_pos=(x, y),
    ego_heading=heading_radians,
)
```

## Testing

Run the test script to verify all visualizations work with dummy data:

```bash
python scripts/test_visualizations.py --output-dir /tmp/viz_test
```

This generates:
- `test_space_attention_bev.png`
- `test_time_attention_diagram.png`
- `test_lane_activation_map.png`
- `test_composite_figure.png`
- `test_layer_refinement.gif`

## Attention Data Format

### Scene Encoder Attention

```python
scene_attentions[layer]: (B, nhead, A+M, A+M)
```

- **B**: Batch size
- **nhead**: Number of attention heads (8)
- **A**: Number of agent tokens (32)
- **M**: Number of map tokens (64)

The attention matrix is structured as:
```
       | Agent 0 ... Agent A-1 | Map 0 ... Map M-1 |
-------|----------------------|-------------------|
Agent 0 | Agent-to-Agent      | Agent-to-Map      |
...     |                      |                   |
Agent A-1|                     |                   |
-------|----------------------|-------------------|
Map 0   | Map-to-Agent        | Map-to-Map        |
...     |                      |                   |
Map M-1 |                      |                   |
```

### Decoder Attention

```python
decoder_agent_attentions[layer]: (B, nhead, K, A)
decoder_map_attentions[layer]: (B, nhead, K, M)
```

- **K**: Number of intention queries (64)

Each intention query attends to agent tokens and map tokens separately via cross-attention.

## Design Principles

1. **Perceptually Uniform Colormaps**: Use viridis/magma for accurate visual perception
2. **Traffic Engineering Standards**: Follow established color conventions (blue=history, green=GT, red=prediction)
3. **Publication Quality**: All figures designed for 300 DPI paper output
4. **Reusable Components**: Modular functions that can be composed
5. **Efficient Rendering**: Vectorized operations, sparse heatmap computation

## Performance Tips

- Use `resolution=0.5` for heatmaps (0.5m per pixel) for good balance of quality and speed
- For animations, use `fps=2` to keep file size reasonable
- Aggregate attention heads with `mean` for most cases; use `max` to highlight peak attention
- Pre-compute cumulative attention once if generating multiple visualizations

## Citation

If you use this visualization pipeline in your research, please cite:

```bibtex
@article{mtr_lite_attention_viz,
  title={Thinking on the Map: Interpretable Attention Visualization for Trajectory Prediction},
  author={...},
  journal={...},
  year={2026}
}
```

## License

Same license as the parent project.
