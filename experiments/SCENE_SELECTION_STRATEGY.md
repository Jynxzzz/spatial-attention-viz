# Scene Selection Strategy

**Document**: 1 of 7 | Experimental Design
**Paper**: "Thinking on the Map: Visualizing Spatio-Temporal Attention in Transformer-Based Trajectory Prediction"
**Target**: MDPI Sustainability
**Author**: Agent G (Experimental Design Architect)
**Date**: 2026-02-10

---

## 1. Overview

This document defines a systematic strategy for selecting 50 representative scenes from the Waymo Open Motion Dataset (20% subset, ~17,850 scenes) for attention visualization and qualitative analysis. The selection is designed to maximize diversity, interpretability, and storytelling value for the paper.

**Goal**: Select 50 scenes (10 per category) that collectively demonstrate:
- The model attends to semantically meaningful traffic elements
- Attention patterns differ meaningfully across scenario types
- Both success and failure cases are represented
- Scenarios connect to sustainable transportation themes

---

## 2. Scene Categories (5 categories, 10 scenes each)

### Category A: Intersection Scenarios (10 scenes)

**Definition**: Scenes where the ego vehicle is within 30 meters of an intersection with traffic signals and/or lane branching.

**Subcategories** (aim for 3-4 each):
- **A1: Protected left turn** (2-3 scenes) -- green arrow, clear path
- **A2: Unprotected left turn** (2-3 scenes) -- must yield to oncoming traffic
- **A3: Right turn** (2 scenes) -- checking pedestrians and cyclists
- **A4: Straight through intersection** (2-3 scenes) -- responding to traffic light state

**Selection criteria**:
- `has_traffic_light == True`
- `lane_branches >= 2` (indicates intersection topology)
- `n_agents >= 5` (at least a few other vehicles present for interaction)
- `lateral_motion >= 2.0` for turns, `lateral_motion < 1.0` for straight-through

**Attention interpretability target**: Model should attend to traffic light tokens, oncoming vehicles (for left turns), target lane, and crosswalk areas.

**Sustainability angle**: Intersection management is the primary cause of urban congestion; understanding how models reason at intersections supports better signal timing and reduced idle emissions.

### Category B: Highway Scenarios (10 scenes)

**Definition**: Scenes where the ego vehicle is traveling at speed > 15 m/s on multi-lane roads.

**Subcategories**:
- **B1: Lane keeping** (3 scenes) -- stable straight driving
- **B2: Lane change** (4 scenes) -- active lateral motion
- **B3: Highway merge** (3 scenes) -- entering highway with conflict

**Selection criteria**:
- `ego_speed >= 15.0` m/s (highway speed)
- `n_lanes >= 4` (multi-lane road)
- For B2: `lateral_motion >= 3.0` meters
- For B1: `lateral_motion < 1.0` meters

**Attention interpretability target**: Lane change scenes should show attention shifting from current lane to target lane; merge scenarios should show attention to gap vehicles.

**Sustainability angle**: Smooth highway driving (no phantom jams) directly reduces fuel consumption and emissions. Predictive lane changes reduce congestion waves.

### Category C: Vulnerable Road User (VRU) Interaction (10 scenes)

**Definition**: Scenes with pedestrians and/or cyclists near the ego vehicle's path.

**Subcategories**:
- **C1: Pedestrian crosswalk** (4 scenes) -- pedestrian crossing or waiting
- **C2: Cyclist adjacent** (3 scenes) -- cyclist in or near ego lane
- **C3: Mixed VRU** (3 scenes) -- both pedestrians and cyclists present

**Selection criteria**:
- At least 1 pedestrian or cyclist agent within 30 meters of ego
- Agent type filtering from the object type field in scene data
- Prefer scenes where VRU trajectory crosses or comes close to the ego path

**Attention interpretability target**: Model should show elevated attention to VRU tokens, especially when they are on a collision course. This is the most safety-critical visualization.

**Sustainability angle**: Pedestrian and cyclist safety is central to SDG 11 (Sustainable Cities). Explainable attention to VRUs builds trust in AV safety for active transportation modes.

### Category D: Complex Multi-Agent Interaction (10 scenes)

**Definition**: Dense traffic scenarios with many interacting agents.

**Subcategories**:
- **D1: Dense urban** (4 scenes) -- 15+ agents in vicinity
- **D2: Roundabout/complex junction** (3 scenes) -- unusual road topology
- **D3: Parking lot / low-speed maneuver** (3 scenes) -- complex but slow

**Selection criteria**:
- `n_agents >= 15` for D1
- `lane_branches >= 5` for D2
- `ego_speed < 5.0` for D3
- Scene complexity score = n_agents * n_lanes * (1 + has_traffic_light)

**Attention interpretability target**: In dense scenes, attention should be selective (not uniform), demonstrating the model can prioritize relevant agents. Expect higher attention entropy in dense scenes vs. highway.

**Sustainability angle**: Efficient navigation in congested urban environments reduces emissions from stop-and-go driving. Understanding multi-agent attention enables cooperative driving strategies.

### Category E: Failure Cases (10 scenes)

**Definition**: Scenes where the trained model produces high prediction error.

**Selection method**: Run the trained model on the full validation set, rank by minFDE@6, and select the top-10 worst predictions.

**Subcategories** (determined post-hoc):
- **E1: High FDE, clear reason** (5 scenes) -- attention analysis reveals root cause
- **E2: High FDE, unclear reason** (3 scenes) -- model fundamentally confused
- **E3: Near-miss correct** (2 scenes) -- model barely got it right, interesting attention

**Selection criteria**:
- `minFDE@6 >= 4.0` meters (significant prediction error)
- Manually verify that the scene is not a data quality issue
- Prefer diverse failure modes (not all the same type)

**Attention interpretability target**: Show where attention is misallocated. For example, the model attends to the wrong lane, ignores a relevant agent, or fails to detect a traffic signal change.

**Sustainability angle**: Identifying and explaining failure modes is essential for safe AV deployment. Only interpretable systems can be systematically debugged, which accelerates the path to sustainable, trustworthy autonomous transportation.

---

## 3. Quantitative Selection Metrics

For each candidate scene, compute the following scores to guide selection:

### 3.1 Attention Entropy Score

```python
def compute_scene_entropy_score(attention_maps, layer=-1, batch_idx=0):
    """Compute mean attention entropy for a scene.

    Low entropy = focused attention (interesting for interpretability).
    High entropy = diffuse attention (interesting for complexity analysis).

    We want BOTH extremes in our selected scenes.
    """
    scene_attn = attention_maps.scene_attentions[layer][batch_idx]  # (nhead, N, N)
    attn_mean = scene_attn.mean(dim=0)  # (N, N) -- average over heads

    # Entropy per query token
    eps = 1e-8
    attn_clamped = attn_mean.clamp(min=eps)
    entropy_per_token = -(attn_clamped * attn_clamped.log2()).sum(dim=-1)  # (N,)

    return {
        "mean_entropy": entropy_per_token.mean().item(),
        "min_entropy": entropy_per_token.min().item(),
        "max_entropy": entropy_per_token.max().item(),
        "entropy_std": entropy_per_token.std().item(),
    }
```

**Selection rule**: For each category, include at least 2 scenes with low mean entropy (< 2.0 bits) and 2 scenes with high mean entropy (> 4.0 bits).

### 3.2 Prediction Quality Score

```python
def compute_prediction_quality(output, batch):
    """Compute prediction metrics for scene selection."""
    from training.metrics import compute_min_ade_fde

    pred = output["trajectories"][0, 0]  # (6, 80, 2) -- first target
    gt = batch["target_future"][0, 0]     # (80, 2)
    valid = batch["target_future_valid"][0, 0]  # (80,)

    metrics = compute_min_ade_fde(pred, gt, valid)
    return {
        "minADE": metrics["min_ade"],
        "minFDE": metrics["min_fde"],
        "is_miss": metrics["miss"],
    }
```

**Selection rule**: For categories A-D, primarily select success cases (minFDE < 2.0m) with 2 marginal cases (minFDE 1.5-3.0m) per category. Category E is explicitly failure cases (minFDE > 4.0m).

### 3.3 Scene Complexity Score

```python
def compute_complexity_score(scene_properties):
    """Composite complexity score for diversity."""
    return (
        scene_properties["n_agents"] * 0.3 +
        scene_properties["n_lanes"] * 0.2 +
        scene_properties["lane_branches"] * 0.2 +
        (10.0 if scene_properties["has_traffic_light"] else 0.0) * 0.15 +
        scene_properties["ego_speed"] * 0.15
    )
```

**Selection rule**: Across all 50 scenes, ensure the complexity score has good spread (std > 5.0).

### 3.4 Attention Diversity Score (Cross-Head)

```python
def compute_head_diversity(attention_maps, layer=-1, batch_idx=0):
    """How different are the 8 attention heads?

    High diversity = heads specialize (interesting).
    Low diversity = heads are redundant.
    """
    scene_attn = attention_maps.scene_attentions[layer][batch_idx]  # (8, N, N)
    nhead = scene_attn.shape[0]

    # Flatten each head's attention to a vector
    head_vectors = scene_attn.view(nhead, -1)  # (8, N*N)

    # Pairwise cosine similarity
    norms = head_vectors.norm(dim=1, keepdim=True)
    head_vectors_normed = head_vectors / (norms + 1e-8)
    sim_matrix = head_vectors_normed @ head_vectors_normed.T  # (8, 8)

    # Mean off-diagonal similarity
    mask = ~torch.eye(nhead, dtype=torch.bool)
    mean_sim = sim_matrix[mask].mean().item()

    return {
        "mean_head_similarity": mean_sim,
        "head_diversity": 1.0 - mean_sim,
    }
```

**Selection rule**: Prefer scenes with `head_diversity > 0.3` (heads show different specializations).

---

## 4. Selection Pipeline

### Step 1: Initial Scan (Automated)

Run `evaluation/qualitative.py:select_interesting_scenes()` on the full validation set (scan up to 3000 scenes). This produces candidate pools for intersection, lane_change, highway, and crowded categories.

```bash
# After training completes
python -c "
from evaluation.qualitative import select_interesting_scenes
categories = select_interesting_scenes(
    scene_list_path='/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt',
    n_scenes=50,
    max_scan=3000,
    seed=42,
)
# Save candidate pools
import json
with open('experiments/candidate_scenes.json', 'w') as f:
    json.dump(categories, f, indent=2)
"
```

### Step 2: Model-Based Filtering (Requires Trained Checkpoint)

For each candidate scene, run the model with `capture_attention=True` and compute:
- Prediction quality (minADE, minFDE)
- Attention entropy
- Head diversity

```bash
python scripts/score_candidate_scenes.py \
  --checkpoint /mnt/hdd12t/outputs/mtr_lite/checkpoints/best.ckpt \
  --candidates experiments/candidate_scenes.json \
  --output experiments/scored_scenes.json
```

### Step 3: Failure Case Detection

```bash
python -c "
from evaluation.qualitative import select_failure_cases
from training.lightning_module import MTRLiteModule

model = MTRLiteModule.load_from_checkpoint(
    '/mnt/hdd12t/outputs/mtr_lite/checkpoints/best.ckpt'
).to('cuda')

failures = select_failure_cases(
    model,
    scene_list_path='/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt',
    config_path='configs/mtr_lite.yaml',
    n_cases=20,
)
# Save top-20 failure cases (select 10 manually)
import json
with open('experiments/failure_cases.json', 'w') as f:
    json.dump(failures, f, indent=2)
"
```

### Step 4: Manual Curation

From the scored candidates, manually select the final 50 scenes using these criteria:

| Priority | Criterion | Weight |
|----------|-----------|--------|
| 1 | Category balance (10 per category) | Required |
| 2 | Subcategory coverage (at least 1 per subcategory) | Required |
| 3 | Attention clarity (visual inspection of attention maps) | High |
| 4 | Storytelling value (can explain model behavior in text) | High |
| 5 | Prediction quality (mostly successes, some failures) | Medium |
| 6 | Diversity of scene complexity | Medium |
| 7 | Visual appeal (for paper figures) | Medium |

### Step 5: Output

Save the final selection as:

```
experiments/selected_scenes.json
```

Format:
```json
{
  "intersection": [
    {
      "scene_path": "/path/to/scene.pkl",
      "subcategory": "unprotected_left_turn",
      "minADE": 0.52,
      "minFDE": 1.23,
      "mean_entropy": 3.45,
      "head_diversity": 0.41,
      "n_agents": 12,
      "notes": "Clear attention to oncoming vehicle during left turn"
    }
  ],
  "highway": [...],
  "vru_interaction": [...],
  "complex_interaction": [...],
  "failure_cases": [...]
}
```

---

## 5. Special Scenes for Key Figures

Beyond the 50 general scenes, identify specific "star scenes" for the paper's main figures:

### Figure 1 Teaser Scene

**Requirements**:
- Intersection left turn (most visually compelling)
- At least 8 agents visible
- Traffic light present
- Model prediction is accurate (minFDE < 1.5m)
- Attention shows clear focus on oncoming traffic + target lane
- All 3 visualization types look good

### Figure 7 Failure Analysis Scene

**Requirements**:
- High prediction error (minFDE > 5.0m)
- Attention is visibly misallocated
- Root cause is explainable (e.g., model attends to wrong lane, ignores pedestrian)
- Scene is not ambiguous (clear GT behavior)

### Trained vs. Untrained Comparison Scene

**Requirements**:
- Any category, preferably intersection
- Run same scene through: (a) randomly initialized model, (b) trained model
- Verify that random model shows uniform/noisy attention
- Trained model shows focused, interpretable attention
- Use the same color scale for both

---

## 6. Scene Selection Quality Assurance

Before finalizing the 50 scenes, verify:

- [ ] All 5 categories have exactly 10 scenes each
- [ ] No duplicate scene_paths across categories
- [ ] All scene .pkl files exist and are loadable
- [ ] All scenes produce valid attention maps (no NaN, shapes correct)
- [ ] At least 35/50 scenes are success cases (minFDE < 2.0m)
- [ ] At least 10/50 scenes are failure cases (minFDE > 4.0m)
- [ ] Category A has both left-turn and straight-through subcategories
- [ ] Category B has both lane-keeping and lane-change subcategories
- [ ] Category C has at least one pedestrian and one cyclist scene
- [ ] Category D has at least one scene with 20+ agents
- [ ] Attention entropy spans at least 1.5 to 5.5 bits across all scenes
- [ ] Head diversity spans at least 0.15 to 0.65 across all scenes

---

## 7. Timing and Dependencies

| Step | Dependency | Estimated Time |
|------|------------|----------------|
| Step 1: Initial scan | Scene list file | ~30 minutes |
| Step 2: Model scoring | Trained checkpoint | ~2 hours (3000 scenes) |
| Step 3: Failure detection | Trained checkpoint | ~1 hour |
| Step 4: Manual curation | Steps 1-3 complete | ~2 hours |
| Step 5: Generate all visualizations | Selected scenes | ~3 hours |

**Total**: ~8 hours after training completes.
