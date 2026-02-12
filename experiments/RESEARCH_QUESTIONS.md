# Research Questions & Hypotheses

**Project**: Thinking on the Map - Attention Visualization for Trajectory Prediction
**Purpose**: Define deep research questions that transform visualization into scientific insights
**Date**: 2026-02-10

---

## RQ1: Safety-Critical Attention Allocation

### 1.1 Vulnerable Road Users (VRUs)

**Hypothesis**: Current trajectory prediction models systematically under-attend to pedestrians and cyclists compared to vehicles, creating safety blind spots.

**Motivation**:
- Pedestrians/cyclists are smaller → fewer visual features → potentially lower attention
- Training data bias: vehicles are more common than VRUs in Waymo dataset
- Safety criticality: VRUs are fragile, require more cautious predictions

**Experiments**:
1. **Attention Weight Statistics by Agent Type**:
   ```python
   For each scene with VRUs:
       - Measure mean attention to vehicles vs pedestrians vs cyclists
       - Compute attention per unit distance (normalize by proximity)
       - Statistical test: Do vehicles get disproportionately more attention?
   ```

2. **Safety-Critical Scenario Analysis**:
   - Pedestrian crossing scenarios: Does model attend before pedestrian enters path?
   - Cyclist merging scenarios: Does model track cyclist trajectory?
   - Compare with human driver eye-tracking data (if available)

3. **Attention Deficit → Prediction Error Correlation**:
   ```
   Hypothesis: Scenes where VRU has low attention → higher prediction error
   Measure: Pearson correlation between attention_to_VRU and minADE
   Expected: Negative correlation (more attention → lower error)
   ```

**Expected Findings**:
- ✗ Pedestrians receive ~40% less attention than vehicles at equal distance
- ✗ Model "notices" pedestrian too late (within 2m instead of 5m)
- ✓ After adding attention regularization: VRU attention increases, safety improves

**Impact**:
- **Safety**: Identify systematic blind spots in prediction models
- **Regulatory**: Evidence for stricter VRU-focused testing standards
- **Sustainability**: Safe AVs enable walkable cities (SDG 11: Sustainable Cities)

---

### 1.2 Traffic Light Compliance

**Hypothesis**: Models may predict physically feasible but rule-violating trajectories due to insufficient attention to traffic signals.

**Experiments**:
1. **Red Light Attention Analysis**:
   ```
   For scenes with red lights in field of view:
       - Measure attention to traffic light tokens
       - Compare: Stopped vehicles vs vehicles violating red light
       - Check: Does attention to red light correlate with stopping behavior?
   ```

2. **Counterfactual Test** (using Scenario Dreamer):
   ```
   Scenario: Ego approaches intersection with green light
   Manipulation: Change light to red (via Scenario Dreamer API)
   Question: Does attention shift to red light? Does prediction change?
   ```

3. **Failure Case Study**:
   - Find cases where model predicts running red light
   - Visualize attention: Is red light ignored?
   - Root cause: Encoding issue? Attention mechanism?

**Expected Findings**:
- ✗ Traffic lights receive only 2-5% of total attention (too low!)
- ✗ In 15% of red light scenarios, model predicts running the light
- ✓ After explicit traffic light embedding: attention increases to 10-15%

**Impact**:
- **Safety**: Prevent illegal and dangerous predictions
- **Trust**: Demonstrate rule-awareness to regulators
- **Sustainability**: Rule compliance → fewer accidents → less congestion

---

### 1.3 Occluded and Emerging Agents

**Hypothesis**: Models trained on visible agents may fail to reason about occluded or about-to-emerge agents (e.g., car hidden behind truck).

**Experiments** (requires Scenario Dreamer's occlusion reasoning):
1. **Occlusion Scenarios**:
   ```
   Setup: Large truck blocks view of vehicle behind it
   Ground Truth: Human driver slows down (defensive driving)
   Test: Does model attend to occluded region? Or only visible agents?
   ```

2. **Emerging Agent Prediction**:
   ```
   Setup: Vehicle about to enter scene from side street
   Question: Does model allocate attention to empty space where agent will appear?
   Measure: Attention entropy in regions with high emergence probability
   ```

**Expected Findings**:
- ✗ Model does not reason about occlusion (no attention to hidden regions)
- ✗ Predictions become overconfident when visibility is reduced
- ✓ Adding occlusion-aware tokens: improves defensive driving behavior

**Impact**:
- **Safety**: Defensive driving requires reasoning about unseen agents
- **Research**: Novel problem for trajectory prediction community

---

## RQ2: Lane and Map Attention Patterns

### 2.1 Lane Selection Interpretability

**Hypothesis**: Model's lane attention distribution reveals its understanding of lane topology and goal-directed planning.

**Experiments**:
1. **Lane Hierarchy Analysis**:
   ```
   For lane-change scenarios:
       - Current lane: Expected high attention (ego position)
       - Target lane: Expected high attention (goal)
       - Irrelevant lanes: Expected low attention

   Measure: Top-3 attended lanes, correlation with GT future path
   ```

2. **Failure Mode: Wrong Lane Selection**:
   ```
   Find: Scenes where model predicts wrong lane
   Visualize: Which lane did model attend to?
   Insight: Did model confuse similar lanes? Ignore traffic rules?
   ```

3. **Lane Graph Topology Understanding**:
   ```
   Test: Does model attend to successor lanes?
   Method: Measure attention propagation along lane graph edges
   Expected: Attention follows waterflow graph structure
   ```

**Expected Findings**:
- ✓ In 85% of successful predictions, top-2 attended lanes match GT path
- ✗ In failures, model attends to geometrically similar but rule-violating lanes
- ✓ Attention flows along lane graph → model understands topology

**Impact**:
- **Interpretability**: Lane attention = model's "mental map"
- **Debugging**: Visualize lane confusion cases
- **Urban Planning**: Identify confusing lane configurations

---

### 2.2 Intersection Complexity and Attention Entropy

**Hypothesis**: Model requires broader attention distribution (higher entropy) in complex intersections vs simple highway driving.

**Experiments**:
1. **Entropy by Scenario Type**:
   ```
   Compute attention entropy for:
       - Highway lane keeping: Expected low entropy (focused)
       - 4-way intersection: Expected high entropy (survey multiple conflicts)
       - Pedestrian crossing: Expected medium entropy

   Statistical test: ANOVA on entropy across scenario types
   ```

2. **Entropy vs Prediction Confidence**:
   ```
   Hypothesis: Higher entropy → lower confidence (more uncertainty)
   Measure: Correlation between entropy and prediction variance
   ```

3. **Attention Coverage**:
   ```
   Metric: What % of agents/lanes in scene receive >5% of total attention?
   Question: Does model attend to all relevant elements or miss some?
   ```

**Expected Findings**:
- ✓ Highway entropy: 2.5 bits, Intersection entropy: 4.2 bits
- ✓ Higher entropy correlates with multi-modal predictions (branching paths)
- ✗ 20% of relevant agents receive <1% attention (attention bottleneck)

**Impact**:
- **Model Design**: Justify multi-scale attention mechanisms
- **Computational Efficiency**: Low entropy → can prune low-attention tokens

---

## RQ3: Attention Evolution Across Layers

### 3.1 Progressive Refinement Hypothesis

**Hypothesis**: Attention becomes increasingly focused and task-relevant from early to late Transformer layers.

**Experiments**:
1. **Layer-wise Entropy Decrease**:
   ```
   For each scene, compute attention entropy at layers 0, 1, 2, 3
   Expected: Monotonic decrease (broad → focused)
   Statistical test: Paired t-test between consecutive layers
   ```

2. **Attention Specialization**:
   ```
   Early layers: Attend broadly to local neighborhood
   Late layers: Attend selectively to goal-relevant elements

   Measure: Attention radius (mean distance to attended tokens)
   Expected: Radius decreases in late layers
   ```

3. **Task-Relevance Score**:
   ```
   Define: Attention to GT future path elements
   Measure: Does this score increase in late layers?
   ```

**Expected Findings**:
- ✓ Entropy decreases: 5.2 → 4.1 → 3.3 → 2.8 bits (Layer 0 → 3)
- ✓ Layer 0: Local neighborhood attention
- ✓ Layer 3: Goal-directed attention (attends to target lane, conflicts)

**Impact**:
- **Architecture Design**: Justify deep Transformers (4+ layers needed)
- **Interpretability**: Early layers = perception, Late layers = reasoning

---

### 3.2 Multi-Head Attention Specialization

**Hypothesis**: Different attention heads learn complementary spatial or semantic roles.

**Experiments**:
1. **Head Clustering**:
   ```
   For each head, compute attention pattern across all validation scenes
   Cluster heads by similarity: Do distinct groups emerge?
   ```

2. **Semantic Role Labeling**:
   ```
   Manually label heads by analyzing their attention patterns:
       - "Front vehicle head": Attends to vehicles ahead
       - "Lane head": Attends to current/target lanes
       - "Conflict head": Attends to potential collisions
   ```

3. **Ablation: Remove One Head**:
   ```
   Train 8 models, each with one head ablated
   Question: Which head removal causes largest performance drop?
   Insight: Which attention patterns are most critical?
   ```

**Expected Findings**:
- ✓ Head specialization emerges: 3 distinct clusters
  - Cluster 1: Local agents (front/side vehicles)
  - Cluster 2: Lane geometry
  - Cluster 3: Traffic signals and long-range agents
- ✗ Removing "conflict head" → +35% miss rate (safety-critical!)

**Impact**:
- **Model Efficiency**: Can prune redundant heads
- **Interpretability**: Each head has a "job"

---

## RQ4: Scenario Dreamer's Unique Contributions

### 4.1 Counterfactual Attention Analysis

**Hypothesis**: By systematically manipulating scenes, we can isolate causal effects on attention.

**Experiments** (leveraging Scenario Dreamer's scene editing):

1. **Agent Removal Study**:
   ```
   Original scene: Ego vehicle approaching intersection, oncoming vehicle present
   Attention: 0.85 to oncoming vehicle, 0.75 to target lane

   Edited scene: Remove oncoming vehicle (via Scenario Dreamer API)
   Prediction: Does attention redistribute? Where does it go?

   Expected: Attention shifts to target lane (0.85 → 0.95), ego proceeds faster
   ```

2. **Traffic Light Flip Study**:
   ```
   Original: Green light → Ego goes straight → Attends to straight lane (0.80)
   Edited: Red light → Ego should stop → Attention shifts to traffic light?

   Measure: ΔAttention to traffic light before/after flip
   ```

3. **Lane Blocking Study**:
   ```
   Original: Target lane clear → Ego changes lane
   Edited: Block target lane with parked vehicle
   Question: Does attention shift to alternative lanes?
   ```

**Expected Findings**:
- ✓ Counterfactual editing reveals causal attention dependencies
- ✓ Model attention is reactive to scene changes (not just learned bias)
- ✗ In 10% of cases, model's attention does not adapt (failure mode)

**Impact**:
- **Scientific**: Causal inference in deep learning (rare in trajectory prediction!)
- **Scenario Dreamer Showcase**: Demonstrates unique value of controllable simulator
- **Debugging**: Identify when model fails to adapt to scene changes

---

### 4.2 Safety-Critical Scenario Stress Testing

**Hypothesis**: Scenario Dreamer enables systematic generation of edge cases that reveal attention failures.

**Experiments**:

1. **Pedestrian Jaywalking**:
   ```
   Generate: 100 scenes with pedestrian suddenly entering road
   Test: Does model attend to pedestrian before collision?
   Measure: Attention latency (time from pedestrian appearance to high attention)
   ```

2. **Sudden Brake Events**:
   ```
   Generate: Lead vehicle sudden brake (deceleration >8 m/s²)
   Test: Does model's attention spike to front vehicle?
   Measure: Attention response time
   ```

3. **Adverse Weather** (if Scenario Dreamer supports):
   ```
   Compare: Clear weather vs fog/rain
   Question: Does attention spread more broadly in low visibility?
   Expected: Entropy increases in adverse weather (more uncertainty)
   ```

**Expected Findings**:
- ✗ Pedestrian jaywalking: Model reacts 0.3s too late (dangerous!)
- ✓ Sudden brake: Model attention spikes correctly
- ✗ Adverse weather: Model does not adapt attention strategy

**Impact**:
- **Safety Validation**: Systematic edge case testing
- **Regulatory**: Evidence for safety certification
- **Model Improvement**: Identify failure modes → targeted fixes

---

## RQ5: Attention-to-Sustainability Link

### 5.1 Computational Efficiency

**Hypothesis**: Sparse attention (low entropy) enables computational savings without performance loss.

**Experiments**:
1. **Attention Pruning**:
   ```
   Method: Zero out attention to tokens with weight < threshold (e.g., 0.05)
   Measure: FLOPs saved, latency reduction, performance change
   ```

2. **Entropy-Adaptive Computation**:
   ```
   Idea: Simple scenes (low entropy) → use fewer layers/heads
   Test: Can we predict scene difficulty from early-layer entropy?
   ```

**Expected Findings**:
- ✓ Pruning 50% of low-attention tokens → 30% FLOPs saved, <1% performance drop
- ✓ Entropy predicts scene difficulty (correlation r=0.65)

**Impact**:
- **Sustainability**: Lower computation → less energy → lower carbon footprint
- **Deployment**: Enables real-time prediction on edge devices
- **MDPI Sustainability Angle**: Direct contribution to energy-efficient AI

---

### 5.2 Human Trust and AV Adoption

**Hypothesis**: Interpretable attention visualization increases user trust in AV systems, accelerating adoption → shared mobility → reduced emissions.

**Experiments** (Human-subject study, optional):
1. **Trust Survey**:
   ```
   Show users: Black-box prediction vs Attention-overlaid prediction
   Question: "How much do you trust this system?"
   Hypothesis: Attention visualization increases trust scores
   ```

2. **Acceptance of Edge Cases**:
   ```
   Show: Failure case with attention misallocation visible
   Question: "Do you understand why the system failed?"
   Hypothesis: Explainability makes failures more acceptable
   ```

**Expected Findings**:
- ✓ Attention visualization increases trust by 25% (p < 0.01)
- ✓ Users can correctly identify attention failures 80% of the time

**Impact**:
- **Adoption**: Trust → acceptance → deployment → sustainability benefits
- **Regulation**: Explainability is increasingly required (EU AI Act 2024)
- **MDPI Sustainability**: Human-AI trust enables sustainable transportation transformation

---

## Summary: From Visualization to Insight

| Research Question | Key Hypothesis | Expected Finding | Impact |
|-------------------|---------------|------------------|--------|
| RQ1.1: VRU Attention | VRUs under-attended | ✗ 40% attention deficit | **Safety blind spot identified** |
| RQ1.2: Traffic Lights | Signals under-attended | ✗ Only 2-5% attention | **Rule compliance issue** |
| RQ1.3: Occlusion | No hidden agent reasoning | ✗ No defensive driving | **Novel research direction** |
| RQ2.1: Lane Selection | Attention reveals "mental map" | ✓ 85% lane correlation | **Interpretability unlock** |
| RQ2.2: Intersection Complexity | Higher entropy in complex scenes | ✓ 2.5 → 4.2 bits | **Validates architecture** |
| RQ3.1: Layer Refinement | Progressive focusing | ✓ Entropy decreases | **Justifies deep models** |
| RQ3.2: Head Specialization | Heads learn distinct roles | ✓ 3 functional clusters | **Efficient design** |
| RQ4.1: Counterfactuals | Causal attention analysis | ✓ Reactive attention | **Scientific rigor** |
| RQ4.2: Edge Cases | Systematic stress testing | ✗ Jaywalking failures | **Safety certification** |
| RQ5.1: Efficiency | Sparse attention saves compute | ✓ 30% FLOPs reduction | **Sustainability** |
| RQ5.2: Trust | Explainability increases adoption | ✓ +25% trust score | **Societal impact** |

---

## Execution Plan

**Phase 1** (Post-training, Week 1):
- RQ1.1, RQ1.2, RQ2.1: Basic attention statistics and failure analysis
- Generate: ~200 visualizations across all scenario types

**Phase 2** (Week 2):
- RQ3.1, RQ3.2: Layer-wise and head-wise analysis
- RQ4.1: Counterfactual experiments (requires Scenario Dreamer integration)

**Phase 3** (Week 3):
- RQ4.2: Edge case generation and testing
- RQ5.1: Efficiency experiments (pruning, adaptive computation)

**Phase 4** (Optional, Week 4):
- RQ5.2: Human-subject study (if time permits)

**Paper Sections**:
- **Results**: RQ1.1, RQ2.1, RQ3.1 (core findings)
- **Discussion**: RQ1.2, RQ1.3, RQ4.1 (deeper insights)
- **Ablation**: RQ3.2 (head specialization)
- **Sustainability**: RQ5.1, RQ5.2 (MDPI requirement)

---

**This is NOT just visualization — it's a comprehensive audit of model attention behavior, revealing blind spots, validating architectures, and enabling safer, more efficient, and more trustworthy autonomous driving.**
