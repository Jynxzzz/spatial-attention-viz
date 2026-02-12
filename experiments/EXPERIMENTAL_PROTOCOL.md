# Experimental Protocol

**Document**: 2 of 7 | Experimental Design
**Paper**: "Thinking on the Map: Visualizing Spatio-Temporal Attention in Transformer-Based Trajectory Prediction"
**Target**: MDPI Sustainability
**Author**: Agent G (Experimental Design Architect)
**Date**: 2026-02-10

---

## 1. Overview

This document specifies all experiments required for the paper, including quantitative evaluation, visualization generation, attention analysis, ablation studies, and qualitative case studies. Each experiment has defined inputs, outputs, expected results, and success criteria.

---

## 2. Experiment E1: Main Performance Evaluation

### 2.1 Objective

Establish that MTR-Lite achieves competitive trajectory prediction performance, validating that its attention patterns are meaningful (not from an untrained or poorly performing model).

### 2.2 Setup

| Parameter | Value |
|-----------|-------|
| Model | MTR-Lite (8.48M params) |
| Dataset | Waymo Open Motion, 20% subset (~17,850 scenes) |
| Split | 85% train (~15,173) / 15% val (~2,678) |
| Prediction horizon | 8 seconds (80 timesteps @ 10Hz) |
| Number of modes | K=6 (after NMS from 64 intentions) |
| Evaluation set | Full validation split |

### 2.3 Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| minADE@6 | min_k mean_t \|\|pred_k(t) - gt(t)\|\| | Min average displacement across 6 modes |
| minFDE@6 | min_k \|\|pred_k(T) - gt(T)\|\| | Min final displacement across 6 modes |
| MR@6 | fraction where minFDE > 2.0m | Miss rate at 2m threshold |
| ADE@3s | minADE up to t=30 | Short horizon performance |
| ADE@5s | minADE up to t=50 | Medium horizon performance |
| ADE@8s | minADE up to t=80 | Full horizon performance |

### 2.4 Baselines for Comparison

| Model | Source | Description |
|-------|--------|-------------|
| Constant Velocity (CV) | Implemented in-house | Linear extrapolation of last observed velocity, K=1 mode |
| Social LSTM | Literature reference | RNN with social pooling (Alahi et al., 2016) |
| TransformerLaneCond | Literature reference | Transformer encoder with lane conditioning (project's prior work) |
| MTR (Full) | Literature reference | Full MTR model from Shi et al., 2022, reported on full Waymo |

**Note on fair comparison**: Since we use 20% of Waymo, we cannot directly compare to published numbers on 100% Waymo. We will:
1. Report our results on our 20% subset
2. Also report published results from literature (on full Waymo) with a clear notation
3. Emphasize that our contribution is visualization, not SOTA performance

### 2.5 Expected Results

| Model | minADE@6 (m) | minFDE@6 (m) | MR@6 |
|-------|-------------|-------------|------|
| Constant Velocity | 2.5-3.5 | 5.0-8.0 | 0.60-0.75 |
| Social LSTM (literature) | 1.5-2.0 | 3.0-4.5 | 0.40-0.55 |
| TransformerLaneCond (literature) | 0.8-1.2 | 2.0-3.0 | 0.25-0.35 |
| **MTR-Lite (ours)** | **0.6-1.0** | **1.5-3.0** | **0.20-0.30** |

### 2.6 Execution

```bash
# Step 1: Evaluate MTR-Lite on validation set
python evaluation/evaluate.py \
  --checkpoint /mnt/hdd12t/outputs/mtr_lite/checkpoints/best.ckpt \
  --config configs/mtr_lite.yaml \
  --device cuda

# Step 2: Evaluate Constant Velocity baseline
python evaluation/evaluate.py \
  --checkpoint CONSTANT_VELOCITY \
  --config configs/mtr_lite.yaml \
  --device cpu
```

### 2.7 Success Criteria

- MTR-Lite minADE@6 < 1.5 meters (competitive, not necessarily SOTA)
- MTR-Lite outperforms CV baseline by at least 50%
- All metrics computed on at least 2000 validation samples
- Training converged (loss plateau visible in TensorBoard)

---

## 3. Experiment E2: Attention Visualization Generation

### 3.1 Objective

Generate the three core visualization types for all 50 selected scenes, creating the visual evidence base for the paper.

### 3.2 Visualization Types

#### V1: Space-Attention BEV Heatmap

**What it shows**: Where in physical space the model allocates attention for a given query agent.

**Technical details**:
- Extract scene encoder attention from the last layer (layer 3)
- For the ego agent token (query), aggregate attention to all other tokens via mean across 8 heads
- Map token attention values to BEV coordinates using SpatialTokenBookkeeper
- Agent attention: Gaussian splatted at agent positions (sigma=3.0m)
- Map attention: Painted along lane centerlines (width=2.0m)
- Overlay on BEV scene rendering with alpha=0.6
- Use 'magma' colormap, clipped at 95th percentile

**Output**: `space_attention_{scene_id}_layer{L}.png` at 300 DPI

#### V2: Time-Attention Diagram (Decoder Layer Evolution)

**What it shows**: How the decoder's attention distribution changes across 4 decoder layers, demonstrating iterative refinement.

**Technical details**:
- For the winning mode (highest score after NMS), extract cross-attention from each of the 4 decoder layers
- Separate agent attention (K, A) and map attention (K, M) for the winning intention query
- Display as 4-panel strip chart (one per layer)
- Each panel: bar chart of attention to top-10 tokens (labeled as "Vehicle_3", "Lane_12", etc.)
- Mean aggregation across heads
- Same vertical scale across all 4 panels for comparability

**Output**: `time_attention_{scene_id}.png` at 300 DPI

#### V3: Lane-Token Activation Map

**What it shows**: Which lane tokens the model's winning mode attends to most, overlaid on the BEV map.

**Technical details**:
- Extract decoder map cross-attention from the last layer (layer 3) for the winning mode
- Aggregate across heads (mean)
- Color each lane polyline by its attention weight
- Line width proportional to attention (min 0.5pt, max 5pt)
- Use 'RdYlGn' colormap (green = high attention = selected lane)
- Add sidebar bar chart ranking top-10 lanes

**Output**: `lane_activation_{scene_id}.png` at 300 DPI

#### V4: Composite Figure (for paper)

**What it shows**: All three visualizations for one scene in a single figure.

**Technical details**:
- 3-panel horizontal layout: [Space BEV | Time Evolution | Lane Activation]
- Shared scene ID label
- Individual panel titles
- Uniform figure size: 18 x 6 inches

**Output**: `composite_{scene_id}.png` at 300 DPI

### 3.3 Execution

```bash
# Generate all visualizations for selected scenes
python scripts/generate_paper_figures.py \
  --checkpoint /mnt/hdd12t/outputs/mtr_lite/checkpoints/best.ckpt \
  --scene-list experiments/selected_scenes.json \
  --output paper/figures/ \
  --dpi 300
```

### 3.4 Expected Output Count

| Visualization | Per Scene | Total (50 scenes) |
|---------------|-----------|-------------------|
| Space-Attention BEV (4 layers) | 4 | 200 |
| Time-Attention Diagram | 1 | 50 |
| Lane-Token Activation | 1 | 50 |
| Composite Figure | 1 | 50 |
| **Total images** | **7** | **350** |

### 3.5 Success Criteria

- All 50 scenes produce valid visualizations (no rendering errors)
- Attention heatmaps are not uniform (visual inspection)
- At least 30/50 scenes show clearly interpretable attention patterns
- Figures are publication quality (300 DPI, clean labels, proper color bars)

---

## 4. Experiment E3: Quantitative Attention Analysis

### 4.1 Objective

Provide statistical evidence that attention is meaningful and correlates with prediction quality.

### 4.2 Analysis E3.1: Attention Entropy Across Layers

**Hypothesis**: Attention entropy decreases from early to late encoder layers, indicating progressive focusing.

**Method**:
1. For each of the 50 selected scenes, compute mean attention entropy at each of the 4 scene encoder layers
2. Also compute for each of the 4 decoder layers (using the winning mode's cross-attention)
3. Aggregate: mean and std of entropy per layer across all 50 scenes
4. Plot: line plot with error bars, x=layer index, y=entropy (bits)

**Expected result**: Monotonically decreasing entropy from layer 0 to layer 3 in both encoder and decoder.

**Statistical test**: Paired t-test between layer 0 and layer 3 entropy (expect p < 0.01).

### 4.3 Analysis E3.2: Attention-to-GT-Lane Correlation

**Hypothesis**: Higher attention to ground-truth-relevant lanes correlates with lower prediction error.

**Method**:
1. For each scene, identify the "GT lane" -- the lane polyline closest to the ground truth future trajectory (using mean point-to-polyline distance)
2. Extract the model's attention weight to this GT lane token from the decoder (last layer, winning mode)
3. Also compute the scene's minADE@6
4. Scatter plot: x = attention to GT lane, y = minADE
5. Compute Pearson correlation coefficient and p-value

**Expected result**: Negative correlation (r < -0.3, p < 0.05). When the model attends more to the correct lane, prediction error is lower.

**Computation on**: Full validation set (not just 50 selected scenes), sample up to 500 scenes for statistical power.

### 4.4 Analysis E3.3: Agent-Type Attention Distribution

**Hypothesis**: The model allocates different amounts of attention to vehicles, pedestrians, and cyclists, with more attention to interacting agents.

**Method**:
1. For each scene, classify agent tokens by type (vehicle, pedestrian, cyclist)
2. Compute total attention allocated to each agent type
3. Also compute "relevance-weighted" attention: attention to agents that are within 20m of ego's predicted path vs. agents far away
4. Box plot or violin plot: attention by agent type

**Expected result**: Vehicles receive the most total attention; pedestrians/cyclists receive disproportionately high attention relative to their count when they are near the ego path.

### 4.5 Analysis E3.4: Attention Consistency

**Hypothesis**: Similar scenarios produce similar attention patterns.

**Method**:
1. Group the 50 selected scenes by category (intersection, highway, VRU, complex, failure)
2. For each category, compute the pairwise cosine similarity of attention vectors (last encoder layer, ego token row)
3. Compare within-category similarity vs. across-category similarity

**Expected result**: Within-category similarity > across-category similarity (by at least 0.1).

**Statistical test**: Permutation test or Mann-Whitney U test.

### 4.6 Analysis E3.5: Attention Sparsity (Gini Coefficient)

**Hypothesis**: Attention becomes sparser (more focused) in later layers.

**Method**:
1. For each scene and each layer, compute the Gini coefficient of the attention weight vector
2. Gini = 0 means perfectly uniform; Gini = 1 means all attention on one token
3. Plot: Gini vs. layer index (expect increasing trend)

```python
def gini_coefficient(weights):
    """Compute Gini coefficient of attention weights."""
    sorted_w = np.sort(weights)
    n = len(sorted_w)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_w) / (n * np.sum(sorted_w))) - (n + 1) / n
```

### 4.7 Execution

```bash
python evaluation/attention_analysis.py \
  --checkpoint /mnt/hdd12t/outputs/mtr_lite/checkpoints/best.ckpt \
  --scene-list experiments/selected_scenes.json \
  --output experiments/attention_analysis_results/ \
  --n-val-scenes 500
```

### 4.8 Success Criteria

- Entropy decreasing trend: p < 0.01
- Attention-ADE correlation: r < -0.2, p < 0.05
- Within-category similarity > across-category similarity
- Gini coefficient increasing with layer depth
- At least 4 out of 5 analyses support the interpretability hypothesis

---

## 5. Ablation Studies

### 5.1 Overview

All ablations require separate training runs with modified configs. Each ablation trains for the same 60 epochs as the base model. Due to compute constraints, we may reduce to 30 epochs for ablations if needed (note this in the paper).

### 5.2 Ablation A1: Scene Encoder Depth

**Hypothesis**: More encoder layers lead to better attention refinement and lower prediction error.

| Variant | Encoder Layers | Decoder Layers | Approx Params |
|---------|---------------|----------------|---------------|
| enc_2_layers | 2 | 4 | ~7.4M |
| **base (MTR-Lite)** | **4** | **4** | **~8.5M** |
| enc_6_layers | 6 | 4 | ~9.6M |

**Config modifications**:
```yaml
# enc_2_layers
model.num_encoder_layers: 2

# enc_6_layers
model.num_encoder_layers: 6
```

**Metrics**: minADE@6, minFDE@6, MR@6, plus attention entropy at each available layer.

**Expected result**: Performance improves from 2 to 4 layers, marginal improvement from 4 to 6. Entropy decreases more steeply with more layers.

### 5.3 Ablation A2: No Map Tokens

**Hypothesis**: Removing map context severely degrades lane-following predictions and makes attention patterns for lane selection impossible.

| Variant | Map Polylines | Description |
|---------|--------------|-------------|
| **base** | **64** | Full map context |
| no_map | 0 | Agent tokens only, no lane information |

**Config modifications**:
```yaml
model.max_map_polylines: 0
```

**Expected result**: Significant performance drop (minADE increase by 30-50%). Predictions lose lane structure, become less goal-directed.

**Attention insight**: Without map tokens, the model can only rely on agent-agent attention. This should degrade performance particularly in intersection and lane-change scenarios.

### 5.4 Ablation A3: No Neighbor Agents

**Hypothesis**: Removing neighbor agent context degrades interaction-heavy scenarios but may not affect simple lane-following.

| Variant | Max Agents | Description |
|---------|-----------|-------------|
| **base** | **32** | Ego + 31 neighbors |
| no_neighbor | 1 | Ego only, no neighbor context |

**Config modifications**:
```yaml
model.max_agents: 1
```

**Expected result**: Moderate performance drop overall. Large performance drop in interaction scenarios (Category D). Small drop in highway lane-keeping (Category B1).

### 5.5 Ablation A4: Number of Intention Queries

**Hypothesis**: More intention queries provide better mode coverage but with diminishing returns.

| Variant | Intentions | Modes Output | Description |
|---------|-----------|-------------|-------------|
| intent_32 | 32 | 6 | Fewer anchor points |
| **base** | **64** | **6** | Default |
| intent_128 | 128 | 6 | More anchor points |

**Config modifications**:
```yaml
# intent_32
model.num_intentions: 32

# intent_128
model.num_intentions: 128
```

**Expected result**: 32 intentions may miss some modes (higher miss rate). 128 intentions provide minimal improvement over 64 (diminishing returns). Decoder attention entropy is higher with more intentions.

### 5.6 Ablation A5: Decoder Depth

**Hypothesis**: Fewer decoder layers reduce the model's ability to iteratively refine predictions through attention.

| Variant | Decoder Layers | Description |
|---------|---------------|-------------|
| dec_2_layers | 2 | Fewer refinement steps |
| **base** | **4** | Default |

**Config modifications**:
```yaml
model.num_decoder_layers: 2
```

**Expected result**: 2-layer decoder performs noticeably worse, and attention evolution shows less refinement (entropy does not decrease as much).

### 5.7 Ablation A6: Model Dimension

**Hypothesis**: Smaller model dimension reduces capacity but maintains basic structure.

| Variant | d_model | FFN dim | Approx Params |
|---------|---------|---------|---------------|
| d_model_192 | 192 | 768 | ~5.0M |
| **base** | **256** | **1024** | **~8.5M** |

**Config modifications**:
```yaml
model.d_model: 192
model.dim_feedforward: 768
```

**Expected result**: Modest performance decrease. Attention patterns may be less specialized (lower head diversity).

### 5.8 Ablation Training Schedule

Due to compute constraints (single RTX 4090), ablations will be trained sequentially after the main model:

| Ablation | Priority | Training Time | GPU Hours |
|----------|----------|---------------|-----------|
| A2: no_map | High | 40 hours | 40 |
| A3: no_neighbor | High | 40 hours | 40 |
| A1: enc_2_layers | Medium | 35 hours | 35 |
| A5: dec_2_layers | Medium | 35 hours | 35 |
| A4: intent_32 | Low | 45 hours | 45 |
| A4: intent_128 | Low | 55 hours | 55 |
| A6: d_model_192 | Low | 30 hours | 30 |

**Minimum viable set**: A1 (enc depth), A2 (no map), A3 (no agents) -- 3 ablations, ~115 GPU hours.

**Pragmatic approach**: If time is limited, train ablations for 30 epochs instead of 60 (note in paper that ablations used reduced training, which may slightly understate their performance).

### 5.9 Ablation Execution

```bash
# Train each ablation
python scripts/run_ablation_suite.py \
  --base-config configs/mtr_lite.yaml \
  --ablations enc_2_layers no_map no_neighbor_agents intent_32 \
  --output-dir /mnt/hdd12t/outputs/mtr_lite_ablations/ \
  --max-epochs 30

# Evaluate all ablation checkpoints
python evaluation/ablation.py \
  --base-config configs/mtr_lite.yaml \
  --checkpoints-dir /mnt/hdd12t/outputs/mtr_lite_ablations/ \
  --output-dir experiments/ablation_results/
```

### 5.10 Expected Ablation Results Table

| Variant | minADE@6 | minFDE@6 | MR@6 | Delta ADE |
|---------|----------|----------|------|-----------|
| MTR-Lite (base) | 0.80 | 2.10 | 0.25 | -- |
| enc_2_layers | 0.95 | 2.50 | 0.30 | +0.15 |
| enc_6_layers | 0.78 | 2.05 | 0.24 | -0.02 |
| no_map | 1.20 | 3.50 | 0.45 | +0.40 |
| no_neighbor | 0.95 | 2.60 | 0.32 | +0.15 |
| intent_32 | 0.85 | 2.25 | 0.28 | +0.05 |
| intent_128 | 0.79 | 2.08 | 0.24 | -0.01 |
| dec_2_layers | 0.90 | 2.40 | 0.29 | +0.10 |
| d_model_192 | 0.88 | 2.30 | 0.27 | +0.08 |

(Values are educated estimates; actual results will replace these.)

---

## 6. Qualitative Case Studies

### 6.1 Case Study Q1: Intersection Left Turn (Success)

**Scene requirements**: Unprotected left turn, oncoming traffic, traffic light, ego vehicle successfully turns.

**Analysis protocol**:
1. Show BEV scene context with all agents and lanes
2. Space-Attention BEV: highlight attention to oncoming vehicle (safety check) and target lane (goal)
3. Time-Attention Evolution: show how attention shifts across 4 decoder layers
   - Layer 1: broad attention to intersection area
   - Layer 2: focus on oncoming traffic and traffic light
   - Layer 3: attention shifts to target lane and gap
   - Layer 4: precise focus on turning path
4. Lane Activation: show the selected left-turn lane highlighted
5. Compare predicted trajectory with GT (overlay)

**Narrative**: "The model first surveys the intersection broadly, then checks for conflicts with oncoming traffic, and finally commits to the turning path -- mirroring human driving behavior."

### 6.2 Case Study Q2: Failure Case Analysis

**Scene requirements**: High minFDE (> 5m), attention visibly misallocated.

**Analysis protocol**:
1. Show BEV scene with predicted trajectory vs. GT trajectory
2. Space-Attention BEV: identify where attention was focused vs. where it should have been
3. Root cause analysis:
   - Does the model attend to the wrong lane? (lane confusion)
   - Does the model ignore a key agent? (interaction miss)
   - Does the model fail to detect a signal change? (temporal miss)
4. Propose counterfactual: "If attention were allocated to [correct element], prediction would improve"

**Narrative**: "This failure case reveals that the model allocated 78% of attention to the straight-through lane while the ground truth shows a left turn. The model's attention to the left-turn signal was only 5%, suggesting the signal encoding could be improved."

### 6.3 Case Study Q3: Trained vs. Untrained Attention

**Scene requirements**: Any clear intersection scene.

**Analysis protocol**:
1. Run the same scene through two models:
   - Model A: randomly initialized (no training), `model.eval()` mode
   - Model B: fully trained (60 epochs), `model.eval()` mode
2. Generate Space-Attention BEV for both with IDENTICAL color scale
3. Side-by-side comparison figure

**Expected result**:
- Untrained: near-uniform attention (entropy ~ log2(96) = 6.58 bits)
- Trained: focused attention (entropy ~ 2-4 bits)

**Narrative**: "Training transforms attention from random noise to structured, interpretable patterns. The trained model's attention map reveals its learned strategy for navigating intersections."

### 6.4 Case Study Q4: Highway Lane Change (Success)

**Scene requirements**: Active lane change on highway, multiple adjacent vehicles.

**Analysis protocol**:
1. Space-Attention: show attention to gap vehicles in target lane
2. Lane Activation: show both current and target lane highlighted
3. Decoder evolution: show attention shifting from current lane to target lane

**Narrative**: "During a lane change, the model progressively shifts attention from the current lane to the target lane while monitoring the gap vehicle in the adjacent lane."

### 6.5 Case Study Q5: Pedestrian Interaction (Safety Critical)

**Scene requirements**: Pedestrian crossing or near the ego path, model correctly yields.

**Analysis protocol**:
1. Space-Attention: show high attention to pedestrian despite being a small object
2. Compare attention to pedestrian vs. nearby vehicles
3. Show prediction: model predicts deceleration / yielding behavior

**Narrative**: "Despite pedestrians being represented by a single token, the model allocates disproportionately high attention (3x average) to the crossing pedestrian, demonstrating learned safety awareness."

---

## 7. Additional Experiments (If Time Permits)

### 7.1 E4: Per-Category Performance Breakdown

Break down minADE@6 by scenario category (intersection vs highway vs VRU vs complex).

**Hypothesis**: Model performs best on highway (simple), worst on complex interactions.

### 7.2 E5: Attention Head Specialization Analysis

For each of the 8 attention heads, visualize what it specializes in:
- Head 1: nearby agents?
- Head 2: current lane?
- Head 3: traffic signals?
- etc.

**Method**: For each head, find the token type that receives highest attention on average.

### 7.3 E6: Temporal Horizon Analysis

Compare attention patterns when predicting at different horizons (3s vs 5s vs 8s).

**Hypothesis**: Longer horizons require broader attention (higher entropy).

---

## 8. Compute Budget

| Experiment | GPU Hours | Priority |
|------------|-----------|----------|
| E1: Main evaluation | 2 | Critical |
| E2: Visualization (50 scenes) | 3 | Critical |
| E3: Attention analysis (500 scenes) | 5 | Critical |
| A1-A3: Core ablations (3 runs) | 115 | High |
| A4-A6: Extended ablations (4 runs) | 165 | Medium |
| Q1-Q5: Case studies | 2 | Critical |
| E4-E6: Additional experiments | 10 | Low |
| **Total** | **~300** | |

**Minimum viable paper**: E1 + E2 + E3 + Q1-Q3 = ~12 GPU hours (post-training).

**Full paper**: E1-E3 + A1-A3 + Q1-Q5 = ~127 GPU hours.

---

## 9. Reproducibility Checklist

For the paper's reproducibility section:

- [ ] Random seed = 42 for all experiments
- [ ] Exact training config (mtr_lite.yaml) provided
- [ ] Dataset split method documented (hash-based, reproducible)
- [ ] Waymo dataset version specified (v1.x, 20% subset)
- [ ] Hardware specified (RTX 4090, CUDA 12.1)
- [ ] PyTorch version specified
- [ ] Training convergence curve (TensorBoard screenshot or extracted data)
- [ ] Checkpoint selection criterion (best val/minADE@6)
- [ ] NMS threshold documented (2.0 meters for miss rate)
- [ ] All hyperparameters listed in a table
