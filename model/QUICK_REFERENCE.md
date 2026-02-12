# MTR-Lite Model Quick Reference

## Basic Usage

```python
from model.mtr_lite import MTRLite

# Initialize model
model = MTRLite(
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    num_intentions=64,
    num_modes_output=6,
    future_len=80,
    agent_feat_dim=29,
    map_feat_dim=9,
    max_agents=32,
    max_map_polylines=64,
    max_targets=8,
)

# Prepare batch (from data pipeline)
batch = {
    "agent_polylines": torch.randn(B, 32, 11, 29),  # (B, agents, timesteps, features)
    "agent_valid": torch.ones(B, 32, 11, dtype=torch.bool),
    "agent_mask": torch.ones(B, 32, dtype=torch.bool),
    "map_polylines": torch.randn(B, 64, 20, 9),  # (B, lanes, points, features)
    "map_valid": torch.ones(B, 64, 20, dtype=torch.bool),
    "map_mask": torch.ones(B, 64, dtype=torch.bool),
    "target_agent_indices": torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1]]),  # (B, max_targets)
    "target_mask": torch.tensor([[True, True, True, False, False, False, False, False]]),
}

# Forward pass (training)
output = model(batch, capture_attention=False)
trajectories = output["trajectories"]  # (B, T, 6, 80, 2)
scores = output["scores"]              # (B, T, 6)

# Forward pass (with attention capture for visualization)
output = model(batch, capture_attention=True)
attention_maps = output["attention_maps"]
```

## Input Specification

### Agent Features (29-dim per timestep)
- Position: x, y (2)
- Previous position: x_prev, y_prev (2)
- Velocity: vx, vy (2)
- Acceleration: ax, ay (2)
- Heading: sin(θ), cos(θ) (2)
- Bounding box: length, width (2)
- Object type: one-hot 5 classes (5)
- Temporal embedding: 11-dim one-hot (11)
- Is ego flag: 0/1 (1)

### Map Features (9-dim per point)
- Position: x, y (2)
- Direction: dx, dy (2)
- Lane flags: is_intersection, speed_limit, lane_type (3)
- Previous point: x_prev, y_prev (2)

## Output Specification

### Main Predictions
```python
{
    "trajectories": (B, T_targets, 6, 80, 2),  # NMS-selected 6 modes
    "scores": (B, T_targets, 6),                # Confidence scores
    "nms_indices": (B, T_targets, 6),           # Which intentions were selected
    "target_mask": (B, T_targets),              # Valid targets
}
```

### Deep Supervision (for training)
```python
{
    "layer_trajectories": [
        (B, T, 64, 80, 2),  # Layer 0: all 64 intentions
        (B, T, 64, 80, 2),  # Layer 1
        (B, T, 64, 80, 2),  # Layer 2
        (B, T, 64, 80, 2),  # Layer 3 (final)
    ],
    "layer_scores": [
        (B, T, 64),  # Layer 0
        (B, T, 64),  # Layer 1
        (B, T, 64),  # Layer 2
        (B, T, 64),  # Layer 3 (final)
    ],
}
```

### Attention Maps (if capture_attention=True)
```python
attention_maps = AttentionMaps(
    scene_attentions=[
        (B, 8, 96, 96),  # Layer 0: self-attention over 32 agents + 64 map tokens
        (B, 8, 96, 96),  # Layer 1
        (B, 8, 96, 96),  # Layer 2
        (B, 8, 96, 96),  # Layer 3
    ],
    decoder_agent_attentions=[  # List of T targets
        [  # Target 0
            (B, 8, 64, 32),  # Layer 0: 64 intentions attend to 32 agents
            (B, 8, 64, 32),  # Layer 1
            (B, 8, 64, 32),  # Layer 2
            (B, 8, 64, 32),  # Layer 3
        ],
        ...  # More targets
    ],
    decoder_map_attentions=[  # List of T targets
        [  # Target 0
            (B, 8, 64, 64),  # Layer 0: 64 intentions attend to 64 map tokens
            (B, 8, 64, 64),  # Layer 1
            (B, 8, 64, 64),  # Layer 2
            (B, 8, 64, 64),  # Layer 3
        ],
        ...  # More targets
    ],
    nms_indices=(B, T, 6),  # Map final modes back to intentions
    num_agents=32,
    num_map=64,
)
```

## Attention Extraction (for Visualization)

```python
from model.attention_hooks import AttentionMaps

# Forward pass with capture
output = model(batch, capture_attention=True)
attn_maps = output["attention_maps"]

# Scene encoder: agent-to-agent attention
agent_to_agent = attn_maps.get_scene_agent_to_agent(layer=0, batch_idx=0)  # (8, 32, 32)
agent_to_agent_avg = attn_maps.aggregate_heads(agent_to_agent, method="mean")  # (32, 32)

# Scene encoder: agent-to-map attention
agent_to_map = attn_maps.get_scene_agent_to_map(layer=3, batch_idx=0)  # (8, 32, 64)

# Decoder: winning mode attention (for best prediction)
target_idx = 0  # First target
winning_mode_idx = 0  # Best mode (highest score)
winning_attn = attn_maps.get_winning_mode_attention(
    layer=3,  # Final decoder layer
    winning_nms_idx=winning_mode_idx,
    batch_idx=0,
)
agent_attn = winning_attn["agent_attn"]  # (8, 32)
map_attn = winning_attn["map_attn"]      # (8, 64)

# Compute attention entropy (measure of focus)
entropy = attn_maps.compute_entropy(map_attn)  # (8,) per head
mean_entropy = entropy.mean().item()  # Average across heads
```

## Training Setup

```python
import torch
from model.mtr_lite import MTRLite

device = torch.device("cuda")
model = MTRLite().to(device)

# Load intention points (from k-means, generated by data pipeline)
intent_points = torch.load("data/intention_points_K64.pt")  # (64, 2)
model.motion_decoder.load_intention_points(intent_points)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Training loop
model.train()
for batch in dataloader:
    # Move to device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Forward (no attention capture during training)
    output = model(batch, capture_attention=False)

    # Compute loss (implement in training/losses.py)
    loss = compute_loss(output, batch)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

## Inference / Visualization

```python
model.eval()
with torch.no_grad():
    output = model(batch, capture_attention=True)

    # Get predictions for first target
    trajs = output["trajectories"][0, 0]  # (6, 80, 2)
    scores = output["scores"][0, 0]        # (6,)

    # Get attention maps
    attn_maps = output["attention_maps"]

    # Extract final layer decoder attention
    # For first target, best mode (mode 0)
    decoder_map_attn = attn_maps.decoder_map_attentions[0][3]  # Target 0, layer 3
    # Shape: (1, 8, 64, 64) - batch, heads, intentions, map_tokens

    # Get winning mode attention
    winning_nms_idx = torch.argmax(scores).item()
    intention_idx = output["nms_indices"][0, 0, winning_nms_idx].item()

    winning_map_attn = decoder_map_attn[0, :, intention_idx, :]  # (8, 64)
    winning_map_attn_avg = winning_map_attn.mean(dim=0)  # (64,) average over heads
```

## Common Patterns

### Getting attention for a specific agent
```python
target_agent_idx = 5  # Agent index in the scene
layer = 3  # Final encoder layer

# How does this agent attend to others?
scene_attn = attn_maps.scene_attentions[layer][batch_idx]  # (8, 96, 96)
agent_attn = scene_attn[:, target_agent_idx, :]  # (8, 96) - this agent's attention to all tokens

# Average over heads
agent_attn_avg = agent_attn.mean(dim=0)  # (96,)

# Split into agent-to-agent and agent-to-map
agent_to_agent = agent_attn_avg[:32]  # (32,)
agent_to_map = agent_attn_avg[32:]    # (64,)
```

### Getting attention for a specific lane
```python
lane_idx = 10  # Lane index in the scene
layer = 3  # Final decoder layer
target = 0  # First target

# How much does the winning mode attend to this lane?
decoder_map_attn = attn_maps.decoder_map_attentions[target][layer][batch_idx]  # (8, 64, 64)
winning_mode_idx = torch.argmax(scores[batch_idx, target]).item()
intention_idx = output["nms_indices"][batch_idx, target, winning_mode_idx].item()

lane_attn = decoder_map_attn[:, intention_idx, lane_idx]  # (8,) per head
lane_attn_avg = lane_attn.mean().item()  # Scalar
```

## Tips

1. **Attention capture overhead**: Only use `capture_attention=True` during visualization/evaluation, not training
2. **Batch processing**: Attention tensors can be large (~100MB for B=4), process one scene at a time for visualization
3. **Head aggregation**: Use `mean` for smooth heatmaps, `max` to see strongest attention
4. **Entropy analysis**: Lower entropy = more focused attention (better for interpretability)
5. **NMS mapping**: Use `nms_indices` to map final 6 modes back to original 64 intention queries

## Files

- **Model**: `/home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper/model/mtr_lite.py`
- **Tests**: `/home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper/tests/test_model.py`
- **Summary**: `/home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper/MODEL_ARCHITECTURE_SUMMARY.md`
