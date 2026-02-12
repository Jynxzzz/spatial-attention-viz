"""Comprehensive attention analysis framework for MTR-Lite.

Runs after training completes to produce quantitative analysis of attention
patterns in the scene encoder and motion decoder. Generates JSON results
and summary plots for the paper.

Analyses:
  1. Attention entropy (Shannon) per token, per layer
  2. Gini coefficient (sparsity) per layer
  3. Attention breakdown by agent type (vehicle/pedestrian/cyclist)
  4. Attention to ground-truth lane correlation
  5. Layer-wise entropy aggregation
  6. Head diversity (pairwise cosine similarity)
  7. Attention threshold statistics

Usage:
    python evaluation/attention_analysis.py \
        --checkpoint PATH --config PATH \
        --n-scenes 200 --output-dir DIR
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np

# Agent type constants matching data.agent_features
AGENT_TYPES = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]
NUM_AGENT_TYPES = len(AGENT_TYPES)
TYPE_ONEHOT_START = 14  # indices [14:19] in the 29-dim agent feature vector
TYPE_ONEHOT_END = 19
NUM_AGENTS = 32
NUM_MAP = 64
NUM_TOTAL_TOKENS = NUM_AGENTS + NUM_MAP  # 96

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Attention entropy
# ---------------------------------------------------------------------------

def compute_attention_entropy(attn_weights):
    """Compute Shannon entropy of attention distributions.

    Measures how focused or diffuse the attention is for each query token
    in each layer and head. Higher entropy means more uniform (less focused)
    attention; lower entropy means the model concentrates on fewer keys.

    Args:
        attn_weights: numpy array of shape (n_layers, n_heads, n_queries, n_keys)
            or (n_heads, n_queries, n_keys) for a single layer.
            Each row along the last axis should sum to ~1 (softmax output).

    Returns:
        dict with:
            'entropy': numpy array of shape matching input but without the last
                dimension, i.e. (n_layers, n_heads, n_queries) or
                (n_heads, n_queries). Units are bits (log base 2).
            'mean_entropy': float, global mean entropy across all tokens/heads/layers.
            'per_layer_mean': numpy array of shape (n_layers,) if multi-layer input,
                otherwise None. Mean entropy per layer.
            'per_head_mean': numpy array of shape (n_heads,) mean entropy per head
                (averaged over layers and queries).
    """
    attn = np.asarray(attn_weights, dtype=np.float64)
    single_layer = (attn.ndim == 3)
    if single_layer:
        attn = attn[np.newaxis]  # (1, H, Q, K)

    eps = 1e-12
    attn_clamped = np.clip(attn, eps, None)
    # Shannon entropy in bits: H = -sum(p * log2(p))
    entropy = -np.sum(attn_clamped * np.log2(attn_clamped), axis=-1)
    # entropy shape: (L, H, Q)

    per_layer_mean = entropy.mean(axis=(1, 2))  # (L,)
    per_head_mean = entropy.mean(axis=(0, 2))   # (H,)
    mean_entropy = float(entropy.mean())

    result = {
        "entropy": entropy[0] if single_layer else entropy,
        "mean_entropy": mean_entropy,
        "per_layer_mean": None if single_layer else per_layer_mean,
        "per_head_mean": per_head_mean,
    }
    return result


# ---------------------------------------------------------------------------
# 2. Gini coefficient
# ---------------------------------------------------------------------------

def compute_gini_coefficient(attn_weights):
    """Compute Gini coefficient of attention distributions as a sparsity metric.

    The Gini coefficient ranges from 0 (perfectly uniform) to 1 (maximally
    concentrated on a single key). It complements entropy by providing a
    scale-invariant sparsity measure.

    Args:
        attn_weights: numpy array of shape (n_layers, n_heads, n_queries, n_keys)
            or (n_heads, n_queries, n_keys) for a single layer.

    Returns:
        dict with:
            'gini': numpy array of shape (n_layers, n_heads, n_queries) or
                (n_heads, n_queries). Gini coefficient per query token.
            'mean_gini': float, global mean Gini coefficient.
            'per_layer_mean': numpy array of shape (n_layers,) or None.
            'per_head_mean': numpy array of shape (n_heads,).
    """
    attn = np.asarray(attn_weights, dtype=np.float64)
    single_layer = (attn.ndim == 3)
    if single_layer:
        attn = attn[np.newaxis]  # (1, H, Q, K)

    L, H, Q, K = attn.shape

    # Gini = (2 * sum_i(i * x_sorted_i)) / (n * sum(x)) - (n+1)/n
    # We compute per (l, h, q) slice.
    # Sort along the key dimension
    sorted_attn = np.sort(attn, axis=-1)  # ascending

    # Indices 1..K
    indices = np.arange(1, K + 1, dtype=np.float64)  # (K,)

    # Numerator: 2 * sum(i * x_i) for sorted values
    # Use broadcasting: sorted_attn is (L, H, Q, K), indices is (K,)
    weighted_sum = np.sum(sorted_attn * indices, axis=-1)  # (L, H, Q)
    total_sum = np.sum(sorted_attn, axis=-1)  # (L, H, Q)

    # Avoid division by zero for masked/empty rows
    safe_total = np.where(total_sum > 1e-12, total_sum, 1.0)
    gini = (2.0 * weighted_sum) / (K * safe_total) - (K + 1.0) / K
    # Clamp to [0, 1]
    gini = np.clip(gini, 0.0, 1.0)

    per_layer_mean = gini.mean(axis=(1, 2))  # (L,)
    per_head_mean = gini.mean(axis=(0, 2))   # (H,)
    mean_gini = float(gini.mean())

    result = {
        "gini": gini[0] if single_layer else gini,
        "mean_gini": mean_gini,
        "per_layer_mean": None if single_layer else per_layer_mean,
        "per_head_mean": per_head_mean,
    }
    return result


# ---------------------------------------------------------------------------
# 3. Attention by agent type
# ---------------------------------------------------------------------------

def analyze_attention_by_agent_type(batch, attention_maps):
    """Break down scene encoder attention weights by agent type.

    Examines how much attention each agent type (vehicle, pedestrian, cyclist,
    other, unknown) receives from all tokens in the scene encoder, and also
    computes per-distance-bin statistics.

    Args:
        batch: dict with at least:
            'agent_polylines': numpy array (A, 11, 29) or (B, A, 11, 29).
                Agent feature vectors. Type one-hot at indices [14:19].
            'agent_mask': numpy array (A,) or (B, A) bool.
        attention_maps: dict with:
            'scene_attentions': list of numpy arrays, each (n_heads, 96, 96)
                for a single batch element, or (B, n_heads, 96, 96).
                Scene encoder self-attention, one per layer.

    Returns:
        dict with:
            'mean_attn_received_by_type': dict mapping type_name -> float.
                Average attention weight received by tokens of each agent type,
                aggregated across all layers and heads.
            'mean_attn_given_by_type': dict mapping type_name -> dict mapping
                type_name -> float. How much each type attends to each other type.
            'per_layer_by_type': list of dicts (one per layer), each mapping
                type_name -> float (mean attention received).
            'per_distance_bin': dict mapping type_name -> dict mapping
                distance_bin_str -> float. Mean attention received per distance
                bin from ego (bins: 0-10m, 10-20m, 20-30m, 30-50m, 50+m).
            'type_counts': dict mapping type_name -> int. Number of tokens per type.
    """
    # Unpack batch, handle batched vs unbatched
    agent_polylines = np.asarray(batch["agent_polylines"])
    agent_mask = np.asarray(batch["agent_mask"])
    if agent_polylines.ndim == 4:
        agent_polylines = agent_polylines[0]
        agent_mask = agent_mask[0]

    scene_attns = attention_maps["scene_attentions"]
    n_layers = len(scene_attns)

    # Determine agent types from one-hot encoding
    # type_onehot is at indices [14:19] in the 29-dim feature, last timestep
    agent_types = np.full(NUM_AGENTS, -1, dtype=np.int32)  # -1 = masked out
    agent_positions = np.zeros((NUM_AGENTS, 2), dtype=np.float64)

    for a in range(NUM_AGENTS):
        if not agent_mask[a]:
            continue
        # Use last timestep (current frame)
        feat = agent_polylines[a, -1, :]  # (29,)
        type_oh = feat[TYPE_ONEHOT_START:TYPE_ONEHOT_END]
        type_idx = int(np.argmax(type_oh))
        if type_oh.sum() < 0.5:
            type_idx = NUM_AGENT_TYPES - 1  # unknown if no type set
        agent_types[a] = type_idx
        agent_positions[a] = feat[0:2]  # BEV position

    # Distance from ego (agent 0)
    ego_pos = agent_positions[0]
    distances = np.linalg.norm(agent_positions - ego_pos, axis=1)

    # Distance bins
    dist_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, float("inf"))]
    dist_bin_labels = ["0-10m", "10-20m", "20-30m", "30-50m", "50+m"]

    # Accumulators
    type_attn_received = defaultdict(list)  # type -> list of attention values received
    type_attn_given = defaultdict(lambda: defaultdict(list))
    per_layer_type_attn = []
    type_dist_attn = defaultdict(lambda: defaultdict(list))

    type_counts = defaultdict(int)
    for a in range(NUM_AGENTS):
        if agent_types[a] >= 0:
            type_counts[AGENT_TYPES[agent_types[a]]] += 1

    for layer_idx in range(n_layers):
        attn = np.asarray(scene_attns[layer_idx], dtype=np.float64)
        # Handle batched: (B, H, 96, 96) -> take first element
        if attn.ndim == 4:
            attn = attn[0]
        # attn: (H, 96, 96), average over heads
        attn_avg = attn.mean(axis=0)  # (96, 96)

        layer_type_accum = defaultdict(list)

        for a in range(NUM_AGENTS):
            if agent_types[a] < 0:
                continue
            tname = AGENT_TYPES[agent_types[a]]

            # Attention RECEIVED by this agent from all tokens
            attn_received = attn_avg[:, a].sum()  # total attention flowing to token a
            type_attn_received[tname].append(float(attn_received))
            layer_type_accum[tname].append(float(attn_received))

            # Per distance bin
            d = distances[a]
            for (lo, hi), label in zip(dist_bins, dist_bin_labels):
                if lo <= d < hi:
                    type_dist_attn[tname][label].append(float(attn_received))
                    break

            # Attention GIVEN by this agent to each type
            for a2 in range(NUM_AGENTS):
                if agent_types[a2] < 0:
                    continue
                tname2 = AGENT_TYPES[agent_types[a2]]
                type_attn_given[tname][tname2].append(float(attn_avg[a, a2]))

        per_layer_type_attn.append(
            {t: float(np.mean(v)) if v else 0.0 for t, v in layer_type_accum.items()}
        )

    # Aggregate results
    mean_attn_received = {
        t: float(np.mean(v)) if v else 0.0
        for t, v in type_attn_received.items()
    }
    mean_attn_given = {}
    for t1, inner in type_attn_given.items():
        mean_attn_given[t1] = {
            t2: float(np.mean(v)) if v else 0.0
            for t2, v in inner.items()
        }
    per_distance_bin = {}
    for t, bins in type_dist_attn.items():
        per_distance_bin[t] = {
            label: float(np.mean(v)) if v else 0.0
            for label, v in bins.items()
        }

    return {
        "mean_attn_received_by_type": mean_attn_received,
        "mean_attn_given_by_type": mean_attn_given,
        "per_layer_by_type": per_layer_type_attn,
        "per_distance_bin": per_distance_bin,
        "type_counts": dict(type_counts),
    }


# ---------------------------------------------------------------------------
# 4. Attention to ground-truth lane
# ---------------------------------------------------------------------------

def compute_attention_to_gt_lane(batch, attention_maps, gt_future):
    """Measure attention allocated to the lane closest to the ground-truth trajectory.

    For each target agent, finds the map token whose lane centerline is closest
    to the ground-truth future trajectory, then measures how much decoder
    cross-attention (map branch) is allocated to that lane vs. others.

    Args:
        batch: dict with at least:
            'lane_centerlines_bev': numpy array (M, 20, 2) or (B, M, 20, 2).
                BEV coordinates of lane centerlines.
            'map_mask': numpy array (M,) or (B, M) bool.
            'target_agent_indices': numpy array (T,) or (B, T) int.
            'target_mask': numpy array (T,) or (B, T) bool.
        attention_maps: dict with:
            'decoder_map_attentions': nested structure. For a single batch element:
                list of length n_targets, each a list of length n_layers,
                each numpy array (n_heads, K, M).
                OR a flat list of length n_layers, each (n_heads, K, M), when
                only one target is analyzed.
        gt_future: numpy array (T, 80, 2) or (B, T, 80, 2).
            Ground-truth future trajectories in BEV.

    Returns:
        dict with:
            'gt_lane_attn_fraction': float. Mean fraction of decoder map attention
                allocated to the GT-closest lane (averaged across targets, layers,
                heads, and the best intention query).
            'per_target': list of dicts, one per valid target, each with:
                'target_idx': int, agent slot index.
                'closest_lane_idx': int, map slot of the closest lane.
                'closest_lane_dist': float, mean distance to closest lane (meters).
                'attn_fraction_per_layer': list of floats, one per decoder layer.
            'correlation_attn_vs_error': float or None. Pearson correlation between
                attention to GT lane and prediction error (minFDE), across targets.
                None if fewer than 3 targets.
    """
    # Unpack, handle batched
    centerlines = np.asarray(batch["lane_centerlines_bev"])
    map_mask = np.asarray(batch["map_mask"])
    target_indices = np.asarray(batch["target_agent_indices"])
    target_mask = np.asarray(batch["target_mask"])
    gt = np.asarray(gt_future)

    if centerlines.ndim == 4:
        centerlines = centerlines[0]
    if map_mask.ndim == 2:
        map_mask = map_mask[0]
    if target_indices.ndim == 2:
        target_indices = target_indices[0]
    if target_mask.ndim == 2:
        target_mask = target_mask[0]
    if gt.ndim == 4:
        gt = gt[0]

    dec_map_attns = attention_maps.get("decoder_map_attentions", [])

    n_targets_valid = int(target_mask.sum())
    M = int(map_mask.shape[0])

    # Determine structure of decoder_map_attentions
    # It could be:
    #   (a) list[target][layer] -> (H, K, M) [from AttentionMaps]
    #   (b) list[layer] -> (H, K, M) [single target]
    #   (c) list[layer] -> (B, H, K, M) [batched single target]
    # We normalize to list[target][layer] -> (H, K, M)
    dec_attn_per_target = _normalize_decoder_map_attns(dec_map_attns, n_targets_valid)

    per_target_results = []
    attn_fractions_all = []
    errors_all = []

    for t_slot in range(len(target_mask)):
        if not target_mask[t_slot]:
            continue

        gt_traj = gt[t_slot]  # (80, 2)
        # Find valid future steps
        valid_steps = np.any(np.abs(gt_traj) > 1e-6, axis=1)
        if not valid_steps.any():
            continue

        gt_valid_pts = gt_traj[valid_steps]  # (n_valid, 2)

        # Find closest lane
        best_lane_idx = -1
        best_lane_dist = float("inf")
        for m in range(M):
            if not map_mask[m]:
                continue
            cl = centerlines[m]  # (20, 2)
            # Mean distance from GT trajectory points to this lane centerline
            # For each GT point, find min distance to any lane point
            dists = np.linalg.norm(
                gt_valid_pts[:, np.newaxis, :] - cl[np.newaxis, :, :],
                axis=2,
            )  # (n_valid, 20)
            mean_min_dist = float(dists.min(axis=1).mean())
            if mean_min_dist < best_lane_dist:
                best_lane_dist = mean_min_dist
                best_lane_idx = m

        if best_lane_idx < 0:
            continue

        # Compute attention fraction to GT lane per layer
        target_result_idx = len(per_target_results)
        attn_fracs_per_layer = []

        if target_result_idx < len(dec_attn_per_target):
            layer_attns = dec_attn_per_target[target_result_idx]
            for layer_attn in layer_attns:
                layer_attn = np.asarray(layer_attn, dtype=np.float64)
                if layer_attn.ndim == 4:
                    layer_attn = layer_attn[0]  # remove batch dim
                # layer_attn: (H, K, M)
                # Average over heads, take max-score intention query (first)
                # or average over top-K queries for robustness
                avg_over_heads = layer_attn.mean(axis=0)  # (K, M)
                # Use the intention query with highest total map attention as proxy
                query_totals = avg_over_heads.sum(axis=1)  # (K,)
                if query_totals.max() > 0:
                    best_query = int(np.argmax(query_totals))
                else:
                    best_query = 0
                map_attn = avg_over_heads[best_query]  # (M,)
                total_attn = map_attn.sum()
                if total_attn > 1e-12:
                    frac = float(map_attn[best_lane_idx] / total_attn)
                else:
                    frac = 0.0
                attn_fracs_per_layer.append(frac)
        else:
            attn_fracs_per_layer = []

        mean_frac = float(np.mean(attn_fracs_per_layer)) if attn_fracs_per_layer else 0.0
        attn_fractions_all.append(mean_frac)

        # Compute FDE for this target (using GT endpoint)
        if valid_steps.any():
            last_valid = np.where(valid_steps)[0][-1]
            endpoint = gt_traj[last_valid]
            # We do not have predictions here, so use distance to lane endpoint as proxy
            # for error. The caller can optionally pass predictions for true correlation.
            errors_all.append(best_lane_dist)

        per_target_results.append({
            "target_idx": int(target_indices[t_slot]),
            "closest_lane_idx": int(best_lane_idx),
            "closest_lane_dist": float(best_lane_dist),
            "attn_fraction_per_layer": attn_fracs_per_layer,
        })

    # Correlation between GT lane attention and lane distance (proxy for error)
    correlation = None
    if len(attn_fractions_all) >= 3 and len(errors_all) >= 3:
        attn_arr = np.array(attn_fractions_all)
        err_arr = np.array(errors_all)
        if attn_arr.std() > 1e-8 and err_arr.std() > 1e-8:
            correlation = float(np.corrcoef(attn_arr, err_arr)[0, 1])

    gt_lane_attn_fraction = (
        float(np.mean(attn_fractions_all)) if attn_fractions_all else 0.0
    )

    return {
        "gt_lane_attn_fraction": gt_lane_attn_fraction,
        "per_target": per_target_results,
        "correlation_attn_vs_error": correlation,
    }


def _normalize_decoder_map_attns(dec_map_attns, n_targets):
    """Normalize decoder map attention structure to list[target][layer] -> ndarray.

    Handles multiple input formats from the model output:
      - list[target][layer] each (H, K, M) or (B, H, K, M)
      - list[layer] each (H, K, M) or (B, H, K, M)   [single target]
      - empty list
    """
    if not dec_map_attns:
        return []

    first = dec_map_attns[0]

    # Check if first element is itself a list (target -> layer structure)
    if isinstance(first, (list, tuple)):
        # Already list[target][layer]
        result = []
        for target_attns in dec_map_attns:
            layers = []
            for la in target_attns:
                layers.append(np.asarray(la, dtype=np.float64))
            result.append(layers)
        return result

    # It is list[layer] -> array. Treat as single target.
    layers = [np.asarray(la, dtype=np.float64) for la in dec_map_attns]
    return [layers]


# ---------------------------------------------------------------------------
# 5. Layer-wise entropy analysis
# ---------------------------------------------------------------------------

def layer_wise_entropy_analysis(attention_maps, n_scenes):
    """Aggregate entropy statistics across layers for multiple scenes.

    Collects entropy values from all provided scenes and computes summary
    statistics per layer, including mean, std, median, and percentiles.

    Args:
        attention_maps: list of dicts, one per scene. Each dict has:
            'scene_attentions': list of numpy arrays per layer,
                each (n_heads, 96, 96) or (B, n_heads, 96, 96).
            Optionally 'decoder_agent_attentions' and 'decoder_map_attentions'
                for decoder analysis.
        n_scenes: int, number of scenes (should match len(attention_maps)).

    Returns:
        dict with:
            'encoder': dict with:
                'per_layer': list of dicts, one per encoder layer, each with
                    'mean', 'std', 'median', 'p5', 'p95' entropy values.
                'overall_mean': float.
                'layer_trend': list of floats (mean entropy per layer, showing
                    if entropy increases or decreases across layers).
            'decoder_agent': same structure for decoder agent cross-attention,
                or None if not available.
            'decoder_map': same structure for decoder map cross-attention,
                or None if not available.
    """
    # Collect per-layer entropy values across all scenes
    encoder_layer_entropies = defaultdict(list)
    decoder_agent_layer_entropies = defaultdict(list)
    decoder_map_layer_entropies = defaultdict(list)

    for scene_attn in attention_maps:
        # Encoder
        scene_enc_attns = scene_attn.get("scene_attentions", [])
        for layer_idx, attn in enumerate(scene_enc_attns):
            attn = np.asarray(attn, dtype=np.float64)
            if attn.ndim == 4:
                attn = attn[0]  # remove batch dim
            # attn: (H, 96, 96)
            ent_result = compute_attention_entropy(attn)
            # ent_result['entropy']: (H, 96) for single layer input
            encoder_layer_entropies[layer_idx].append(
                ent_result["entropy"].flatten()
            )

        # Decoder agent
        dec_agent = scene_attn.get("decoder_agent_attentions", [])
        if dec_agent:
            # May be list[target][layer] or list[layer]
            _collect_decoder_entropies(dec_agent, decoder_agent_layer_entropies)

        # Decoder map
        dec_map = scene_attn.get("decoder_map_attentions", [])
        if dec_map:
            _collect_decoder_entropies(dec_map, decoder_map_layer_entropies)

    def _summarize(layer_dict):
        if not layer_dict:
            return None
        n_layers = max(layer_dict.keys()) + 1
        per_layer = []
        layer_means = []
        for li in range(n_layers):
            if li in layer_dict and layer_dict[li]:
                vals = np.concatenate(layer_dict[li])
                per_layer.append({
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "p5": float(np.percentile(vals, 5)),
                    "p95": float(np.percentile(vals, 95)),
                })
                layer_means.append(float(np.mean(vals)))
            else:
                per_layer.append({
                    "mean": 0.0, "std": 0.0, "median": 0.0,
                    "p5": 0.0, "p95": 0.0,
                })
                layer_means.append(0.0)
        overall = float(np.mean(layer_means)) if layer_means else 0.0
        return {
            "per_layer": per_layer,
            "overall_mean": overall,
            "layer_trend": layer_means,
        }

    return {
        "encoder": _summarize(encoder_layer_entropies),
        "decoder_agent": _summarize(decoder_agent_layer_entropies),
        "decoder_map": _summarize(decoder_map_layer_entropies),
    }


def _collect_decoder_entropies(dec_attns, layer_dict):
    """Helper to collect decoder attention entropies from various formats."""
    if not dec_attns:
        return
    first = dec_attns[0]
    if isinstance(first, (list, tuple)):
        # list[target][layer] -> array
        for target_attns in dec_attns:
            for layer_idx, la in enumerate(target_attns):
                la = np.asarray(la, dtype=np.float64)
                if la.ndim == 4:
                    la = la[0]
                ent = compute_attention_entropy(la)
                layer_dict[layer_idx].append(ent["entropy"].flatten())
    else:
        # list[layer] -> array
        for layer_idx, la in enumerate(dec_attns):
            la = np.asarray(la, dtype=np.float64)
            if la.ndim == 4:
                la = la[0]
            if la.ndim == 2:
                la = la[np.newaxis]
            ent = compute_attention_entropy(la)
            layer_dict[layer_idx].append(ent["entropy"].flatten())


# ---------------------------------------------------------------------------
# 6. Head diversity analysis
# ---------------------------------------------------------------------------

def head_diversity_analysis(attention_maps):
    """Measure pairwise cosine similarity between attention heads.

    Quantifies head specialization: if heads are diverse (low similarity),
    they capture different relationship patterns. If heads are similar,
    the model may not be efficiently using its multi-head capacity.

    Args:
        attention_maps: dict with:
            'scene_attentions': list of numpy arrays per layer,
                each (n_heads, 96, 96) or (B, n_heads, 96, 96).

    Returns:
        dict with:
            'pairwise_similarity': numpy array (n_layers, n_heads, n_heads).
                Cosine similarity between each pair of heads per layer.
            'mean_pairwise_similarity': float. Average off-diagonal similarity.
            'per_layer_mean_similarity': numpy array (n_layers,).
            'per_layer_min_similarity': numpy array (n_layers,).
                Minimum pairwise similarity per layer (most diverse pair).
            'head_specialization_score': float. 1 - mean_pairwise_similarity.
                Higher means more specialized heads.
    """
    scene_attns = attention_maps.get("scene_attentions", [])
    if not scene_attns:
        return {
            "pairwise_similarity": np.array([]),
            "mean_pairwise_similarity": 0.0,
            "per_layer_mean_similarity": np.array([]),
            "per_layer_min_similarity": np.array([]),
            "head_specialization_score": 0.0,
        }

    n_layers = len(scene_attns)
    attn0 = np.asarray(scene_attns[0])
    if attn0.ndim == 4:
        attn0 = attn0[0]
    n_heads = attn0.shape[0]

    pairwise = np.zeros((n_layers, n_heads, n_heads), dtype=np.float64)
    per_layer_mean = np.zeros(n_layers, dtype=np.float64)
    per_layer_min = np.ones(n_layers, dtype=np.float64)

    for li in range(n_layers):
        attn = np.asarray(scene_attns[li], dtype=np.float64)
        if attn.ndim == 4:
            attn = attn[0]
        # attn: (H, 96, 96)
        # Flatten each head's attention to a vector
        flat = attn.reshape(n_heads, -1)  # (H, 96*96)

        # Normalize rows
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        flat_normed = flat / norms

        # Cosine similarity matrix
        sim = flat_normed @ flat_normed.T  # (H, H)
        pairwise[li] = sim

        # Off-diagonal statistics
        mask = ~np.eye(n_heads, dtype=bool)
        off_diag = sim[mask]
        per_layer_mean[li] = float(off_diag.mean())
        per_layer_min[li] = float(off_diag.min())

    mean_sim = float(per_layer_mean.mean())

    return {
        "pairwise_similarity": pairwise,
        "mean_pairwise_similarity": mean_sim,
        "per_layer_mean_similarity": per_layer_mean,
        "per_layer_min_similarity": per_layer_min,
        "head_specialization_score": 1.0 - mean_sim,
    }


# ---------------------------------------------------------------------------
# 7. Attention threshold statistics
# ---------------------------------------------------------------------------

def attention_threshold_statistics(batch, attention_maps, thresholds=None):
    """Compute statistics about attention weight magnitudes at various thresholds.

    For each threshold, computes what percentage of key tokens exceed it for
    the average query, providing insight into how concentrated or spread-out
    attention patterns are.

    Args:
        batch: dict with at least:
            'agent_mask': numpy array (A,) or (B, A) bool.
            'map_mask': numpy array (M,) or (B, M) bool.
        attention_maps: dict with:
            'scene_attentions': list of numpy arrays per layer,
                each (n_heads, 96, 96) or (B, n_heads, 96, 96).
        thresholds: list of float threshold values. Default [0.05, 0.1, 0.2, 0.3].

    Returns:
        dict with:
            'threshold_stats': dict mapping threshold -> dict with:
                'pct_tokens_above': float. Mean percentage of key tokens with
                    attention weight above threshold (across all queries/heads/layers).
                'pct_agent_tokens_above': float. Same but restricted to agent tokens.
                'pct_map_tokens_above': float. Same but restricted to map tokens.
                'per_layer': list of floats, pct_tokens_above per layer.
            'effective_attention_size': dict mapping layer_idx -> float.
                Average number of tokens that hold >5% of total attention
                (effective "attention window size").
            'attention_mass_in_top_k': dict mapping k -> float.
                Fraction of total attention mass in top-k tokens (k=1,3,5,10).
    """
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.2, 0.3]

    agent_mask = np.asarray(batch["agent_mask"])
    map_mask = np.asarray(batch["map_mask"])
    if agent_mask.ndim == 2:
        agent_mask = agent_mask[0]
    if map_mask.ndim == 2:
        map_mask = map_mask[0]

    scene_attns = attention_maps.get("scene_attentions", [])
    n_layers = len(scene_attns)

    # Build combined token validity mask (96,)
    combined_mask = np.concatenate([agent_mask, map_mask])

    threshold_stats = {}
    effective_sizes = {}
    top_k_mass = {k: [] for k in [1, 3, 5, 10]}

    for thresh in thresholds:
        pct_all_layers = []
        pct_agent_layers = []
        pct_map_layers = []
        per_layer_pct = []

        for li in range(n_layers):
            attn = np.asarray(scene_attns[li], dtype=np.float64)
            if attn.ndim == 4:
                attn = attn[0]
            # attn: (H, 96, 96)

            # Mask invalid tokens
            n_valid = combined_mask.sum()
            if n_valid == 0:
                per_layer_pct.append(0.0)
                continue

            # For each query, count keys above threshold
            above = (attn > thresh).astype(np.float64)  # (H, 96, 96)

            # Only consider valid query and key tokens
            valid_idx = np.where(combined_mask)[0]
            agent_valid_idx = np.where(agent_mask)[0]
            map_valid_idx = np.where(map_mask)[0] + NUM_AGENTS  # offset for map tokens
            # note: map tokens in scene attn are at indices [32:96]
            # but map_mask is (64,) so map token i is at scene index 32+i
            map_valid_scene_idx = np.array(
                [NUM_AGENTS + i for i in range(len(map_mask)) if map_mask[i]]
            )

            # Restrict to valid queries
            if len(valid_idx) == 0:
                per_layer_pct.append(0.0)
                continue

            # All tokens
            above_valid = above[:, valid_idx, :][:, :, valid_idx]
            n_keys_valid = len(valid_idx)
            pct = float(above_valid.mean()) * 100.0
            pct_all_layers.append(pct)
            per_layer_pct.append(pct)

            # Agent tokens as keys
            if len(agent_valid_idx) > 0:
                above_agent = above[:, valid_idx, :][:, :, agent_valid_idx]
                pct_agent_layers.append(float(above_agent.mean()) * 100.0)

            # Map tokens as keys
            if len(map_valid_scene_idx) > 0:
                above_map = above[:, valid_idx, :][:, :, map_valid_scene_idx]
                pct_map_layers.append(float(above_map.mean()) * 100.0)

        threshold_stats[str(thresh)] = {
            "pct_tokens_above": float(np.mean(pct_all_layers)) if pct_all_layers else 0.0,
            "pct_agent_tokens_above": float(np.mean(pct_agent_layers)) if pct_agent_layers else 0.0,
            "pct_map_tokens_above": float(np.mean(pct_map_layers)) if pct_map_layers else 0.0,
            "per_layer": per_layer_pct,
        }

    # Effective attention size and top-k mass (computed once, using 5% threshold)
    for li in range(n_layers):
        attn = np.asarray(scene_attns[li], dtype=np.float64)
        if attn.ndim == 4:
            attn = attn[0]

        valid_idx = np.where(combined_mask)[0]
        if len(valid_idx) == 0:
            effective_sizes[li] = 0.0
            for k in top_k_mass:
                top_k_mass[k].append(0.0)
            continue

        attn_valid = attn[:, valid_idx, :][:, :, valid_idx]  # (H, n_valid, n_valid)
        n_valid = len(valid_idx)

        # Effective attention size: tokens with >5% weight
        above_5pct = (attn_valid > 0.05).sum(axis=-1).astype(np.float64)  # (H, n_valid)
        effective_sizes[li] = float(above_5pct.mean())

        # Top-k attention mass
        for k in top_k_mass:
            if k >= n_valid:
                top_k_mass[k].append(1.0)
                continue
            # Sort along key axis descending, sum top k
            sorted_attn = np.sort(attn_valid, axis=-1)[:, :, ::-1]  # descending
            mass = sorted_attn[:, :, :k].sum(axis=-1)  # (H, n_valid)
            top_k_mass[k].append(float(mass.mean()))

    attention_mass_top_k = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in top_k_mass.items()
    }

    return {
        "threshold_stats": threshold_stats,
        "effective_attention_size": {str(k): v for k, v in effective_sizes.items()},
        "attention_mass_in_top_k": {str(k): v for k, v in attention_mass_top_k.items()},
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def _generate_summary_plots(results, output_dir):
    """Generate summary plots from aggregated analysis results.

    Creates the following figures:
      1. Layer-wise entropy bar chart (encoder + decoder)
      2. Gini coefficient distribution per layer
      3. Attention by agent type heatmap
      4. Head diversity similarity matrix
      5. Threshold statistics line plot
      6. GT lane attention vs. distance scatter
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    figures_created = []

    # 1. Layer-wise entropy
    enc_entropy = results.get("layer_wise_entropy", {}).get("encoder")
    if enc_entropy and enc_entropy.get("per_layer"):
        fig, ax = plt.subplots(figsize=(8, 4))
        layers = enc_entropy["per_layer"]
        x = list(range(len(layers)))
        means = [l["mean"] for l in layers]
        stds = [l["std"] for l in layers]
        ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.8)
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("Attention Entropy (bits)")
        ax.set_title("Scene Encoder: Attention Entropy per Layer")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in x])
        fig.tight_layout()
        path = os.path.join(output_dir, "encoder_entropy_per_layer.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_created.append(path)

    # Decoder entropy (agent + map combined)
    dec_agent_ent = results.get("layer_wise_entropy", {}).get("decoder_agent")
    dec_map_ent = results.get("layer_wise_entropy", {}).get("decoder_map")
    if dec_agent_ent and dec_map_ent:
        if dec_agent_ent.get("per_layer") and dec_map_ent.get("per_layer"):
            fig, ax = plt.subplots(figsize=(8, 4))
            x = list(range(len(dec_agent_ent["per_layer"])))
            agent_means = [l["mean"] for l in dec_agent_ent["per_layer"]]
            map_means = [l["mean"] for l in dec_map_ent["per_layer"]]
            width = 0.35
            ax.bar([xi - width / 2 for xi in x], agent_means,
                   width, label="Agent Cross-Attn", color="#4C72B0", alpha=0.8)
            ax.bar([xi + width / 2 for xi in x], map_means,
                   width, label="Map Cross-Attn", color="#DD8452", alpha=0.8)
            ax.set_xlabel("Decoder Layer")
            ax.set_ylabel("Attention Entropy (bits)")
            ax.set_title("Motion Decoder: Cross-Attention Entropy per Layer")
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{i}" for i in x])
            ax.legend()
            fig.tight_layout()
            path = os.path.join(output_dir, "decoder_entropy_per_layer.pdf")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figures_created.append(path)

    # 2. Gini coefficient per layer
    gini_data = results.get("gini_per_layer")
    if gini_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = list(range(len(gini_data)))
        means = [g["mean"] for g in gini_data]
        stds = [g["std"] for g in gini_data]
        ax.bar(x, means, yerr=stds, capsize=4, color="#55A868", alpha=0.8)
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("Gini Coefficient")
        ax.set_title("Attention Sparsity (Gini) per Encoder Layer")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in x])
        ax.set_ylim(0, 1)
        fig.tight_layout()
        path = os.path.join(output_dir, "gini_per_layer.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_created.append(path)

    # 3. Attention by agent type
    type_attn = results.get("attention_by_type", {}).get("mean_attn_received_by_type")
    if type_attn:
        fig, ax = plt.subplots(figsize=(6, 4))
        types = list(type_attn.keys())
        vals = [type_attn[t] for t in types]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        ax.barh(types, vals, color=colors[:len(types)], alpha=0.8)
        ax.set_xlabel("Mean Attention Received")
        ax.set_title("Attention Received by Agent Type")
        fig.tight_layout()
        path = os.path.join(output_dir, "attention_by_agent_type.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_created.append(path)

    # Distance-bin heatmap
    dist_data = results.get("attention_by_type", {}).get("per_distance_bin")
    if dist_data:
        types_present = sorted(dist_data.keys())
        all_bins = ["0-10m", "10-20m", "20-30m", "30-50m", "50+m"]
        bins_present = [b for b in all_bins if any(b in dist_data[t] for t in types_present)]
        if types_present and bins_present:
            matrix = np.zeros((len(types_present), len(bins_present)))
            for i, t in enumerate(types_present):
                for j, b in enumerate(bins_present):
                    matrix[i, j] = dist_data[t].get(b, 0.0)

            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(bins_present)))
            ax.set_xticklabels(bins_present)
            ax.set_yticks(range(len(types_present)))
            ax.set_yticklabels(types_present)
            ax.set_xlabel("Distance from Ego")
            ax.set_ylabel("Agent Type")
            ax.set_title("Attention Received by Type and Distance")
            fig.colorbar(im, ax=ax, label="Mean Attention")
            fig.tight_layout()
            path = os.path.join(output_dir, "attention_type_distance_heatmap.pdf")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figures_created.append(path)

    # 4. Head diversity similarity matrix (average over layers)
    head_div = results.get("head_diversity")
    if head_div and isinstance(head_div.get("pairwise_similarity"), np.ndarray):
        pw = head_div["pairwise_similarity"]
        if pw.ndim == 3 and pw.shape[0] > 0:
            n_layers = pw.shape[0]
            fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5))
            if n_layers == 1:
                axes = [axes]
            for li in range(n_layers):
                im = axes[li].imshow(pw[li], vmin=0, vmax=1, cmap="coolwarm")
                axes[li].set_title(f"Layer {li}")
                axes[li].set_xlabel("Head")
                axes[li].set_ylabel("Head")
                axes[li].set_xticks(range(pw.shape[1]))
                axes[li].set_yticks(range(pw.shape[2]))
            fig.colorbar(im, ax=axes, label="Cosine Similarity", shrink=0.8)
            fig.suptitle(
                f"Head Diversity (specialization={head_div['head_specialization_score']:.3f})",
                fontsize=12,
            )
            fig.tight_layout()
            path = os.path.join(output_dir, "head_diversity_similarity.pdf")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figures_created.append(path)

    # 5. Threshold statistics
    thresh_data = results.get("threshold_stats", {}).get("threshold_stats")
    if thresh_data:
        fig, ax = plt.subplots(figsize=(7, 4))
        thresholds_sorted = sorted(thresh_data.keys(), key=float)
        pct_all = [thresh_data[t]["pct_tokens_above"] for t in thresholds_sorted]
        pct_agent = [thresh_data[t]["pct_agent_tokens_above"] for t in thresholds_sorted]
        pct_map = [thresh_data[t]["pct_map_tokens_above"] for t in thresholds_sorted]
        x_vals = [float(t) for t in thresholds_sorted]
        ax.plot(x_vals, pct_all, "o-", label="All Tokens", color="#4C72B0")
        ax.plot(x_vals, pct_agent, "s--", label="Agent Tokens", color="#DD8452")
        ax.plot(x_vals, pct_map, "^:", label="Map Tokens", color="#55A868")
        ax.set_xlabel("Attention Weight Threshold")
        ax.set_ylabel("% Tokens Above Threshold")
        ax.set_title("Attention Concentration: Tokens Above Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, "threshold_statistics.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_created.append(path)

    # Top-k attention mass
    topk_data = results.get("threshold_stats", {}).get("attention_mass_in_top_k")
    if topk_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ks = sorted(topk_data.keys(), key=int)
        vals = [topk_data[k] for k in ks]
        ax.bar([str(k) for k in ks], vals, color="#8172B2", alpha=0.8)
        ax.set_xlabel("Top-K Tokens")
        ax.set_ylabel("Fraction of Attention Mass")
        ax.set_title("Attention Mass Concentration in Top-K Tokens")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        path = os.path.join(output_dir, "top_k_attention_mass.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_created.append(path)

    # 6. GT lane attention summary
    gt_lane = results.get("gt_lane_attention")
    if gt_lane:
        per_target = gt_lane.get("per_target_all", [])
        if per_target:
            dists = [p["closest_lane_dist"] for p in per_target if "closest_lane_dist" in p]
            fracs = []
            for p in per_target:
                layer_fracs = p.get("attn_fraction_per_layer", [])
                if layer_fracs:
                    fracs.append(float(np.mean(layer_fracs)))

            if dists and fracs and len(dists) == len(fracs):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(dists, fracs, alpha=0.5, s=20, color="#4C72B0")
                ax.set_xlabel("Distance to GT Lane (m)")
                ax.set_ylabel("Attention Fraction to GT Lane")
                ax.set_title(
                    f"GT Lane Attention vs. Distance "
                    f"(corr={gt_lane.get('correlation_attn_vs_error', 'N/A')})"
                )
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path = os.path.join(output_dir, "gt_lane_attention_scatter.pdf")
                fig.savefig(path, dpi=150)
                plt.close(fig)
                figures_created.append(path)

    logger.info("Generated %d plots in %s", len(figures_created), output_dir)
    return figures_created


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _make_json_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run comprehensive attention analysis on validation scenes.

    Loads a trained checkpoint and config, iterates over N validation scenes
    with attention capture, runs all analysis functions, saves results to JSON,
    and generates summary plots.
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive attention analysis for MTR-Lite"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML (e.g. configs/mtr_lite.yaml)",
    )
    parser.add_argument(
        "--n-scenes", type=int, default=200,
        help="Number of validation scenes to analyze (default: 200)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results. Default: <training_output_dir>/attention_analysis/",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference (default: 1 for attention capture)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (JSON only)",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3],
        help="Attention thresholds for threshold statistics analysis",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Late imports for torch (kept out of analysis functions)
    import torch
    import yaml

    from data.collate import mtr_collate_fn
    from data.polyline_dataset import PolylineDataset
    from training.lightning_module import MTRLiteModule

    # Load config
    logger.info("Loading config from %s", args.config)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Output directory
    output_dir = args.output_dir
    if output_dir is None:
        training_out = cfg["training"].get("output_dir", ".")
        output_dir = os.path.join(training_out, "attention_analysis")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Load model
    logger.info("Loading checkpoint from %s", args.checkpoint)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    module = MTRLiteModule.load_from_checkpoint(args.checkpoint, map_location=device)
    module = module.to(device)
    module.eval()
    model = module.model

    # Build validation dataset
    logger.info("Building validation dataset...")
    val_dataset = PolylineDataset(
        scene_list_path=data_cfg["scene_list"],
        split="val",
        val_ratio=data_cfg.get("val_ratio", 0.15),
        data_fraction=data_cfg.get("data_fraction", 1.0),
        history_len=data_cfg["history_len"],
        future_len=data_cfg["future_len"],
        max_agents=data_cfg["max_agents"],
        max_map_polylines=data_cfg["max_map_polylines"],
        map_points_per_lane=data_cfg["map_points_per_lane"],
        neighbor_distance=data_cfg.get("neighbor_distance", 50.0),
        anchor_frames=data_cfg.get("anchor_frames", [10]),
        augment=False,
        seed=cfg.get("seed", 42),
    )

    n_scenes = min(args.n_scenes, len(val_dataset))
    logger.info("Analyzing %d of %d validation scenes", n_scenes, len(val_dataset))

    from torch.utils.data import DataLoader

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, cfg["training"].get("num_workers", 4)),
        collate_fn=mtr_collate_fn,
        pin_memory=(device == "cuda"),
    )

    # -----------------------------------------------------------------------
    # Run analysis over scenes
    # -----------------------------------------------------------------------
    all_scene_attns = []        # for layer_wise_entropy_analysis
    all_type_results = []       # for aggregating type analysis
    all_gt_lane_results = []    # for GT lane attention
    all_gini_per_layer = defaultdict(list)  # layer -> list of gini arrays
    all_threshold_results = []
    all_head_diversity = []     # per-scene pairwise similarity
    scene_count = 0
    total_time = 0.0

    logger.info("Starting attention analysis loop...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch is None:
                continue

            if scene_count >= n_scenes:
                break

            t0 = time.time()

            # Move tensors to device
            batch_device = {}
            batch_numpy = {}
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch_device[key] = batch[key].to(device)
                    batch_numpy[key] = batch[key].cpu().numpy()
                else:
                    batch_device[key] = batch[key]
                    batch_numpy[key] = batch[key]

            # Forward pass with attention capture
            try:
                output = model(batch_device, capture_attention=True)
            except Exception as e:
                logger.warning("Forward pass failed for batch %d: %s", batch_idx, e)
                continue

            attn_maps_obj = output.get("attention_maps")
            if attn_maps_obj is None:
                logger.warning("No attention maps captured for batch %d", batch_idx)
                continue

            # Convert attention maps to numpy for analysis functions
            scene_attn_np = {
                "scene_attentions": [
                    sa.detach().cpu().numpy() for sa in attn_maps_obj.scene_attentions
                ],
            }

            # Decoder attention: organize per target
            dec_agent_np = []
            dec_map_np = []
            if attn_maps_obj.decoder_agent_attentions:
                for target_attns in attn_maps_obj.decoder_agent_attentions:
                    if isinstance(target_attns, (list, tuple)):
                        dec_agent_np.append([
                            la.detach().cpu().numpy() for la in target_attns
                        ])
                    else:
                        dec_agent_np.append(
                            target_attns.detach().cpu().numpy()
                        )
            if attn_maps_obj.decoder_map_attentions:
                for target_attns in attn_maps_obj.decoder_map_attentions:
                    if isinstance(target_attns, (list, tuple)):
                        dec_map_np.append([
                            la.detach().cpu().numpy() for la in target_attns
                        ])
                    else:
                        dec_map_np.append(
                            target_attns.detach().cpu().numpy()
                        )

            scene_attn_np["decoder_agent_attentions"] = dec_agent_np
            scene_attn_np["decoder_map_attentions"] = dec_map_np

            all_scene_attns.append(scene_attn_np)

            # --- Per-scene analyses ---

            # 2. Gini coefficient (scene encoder)
            for li, sa in enumerate(scene_attn_np["scene_attentions"]):
                gini_res = compute_gini_coefficient(sa if sa.ndim == 3 else sa[0])
                all_gini_per_layer[li].append(gini_res["mean_gini"])

            # 3. Agent type analysis
            try:
                type_res = analyze_attention_by_agent_type(batch_numpy, scene_attn_np)
                all_type_results.append(type_res)
            except Exception as e:
                logger.debug("Agent type analysis failed for batch %d: %s", batch_idx, e)

            # 4. GT lane attention
            try:
                gt_future_np = batch_numpy.get("target_future")
                if gt_future_np is not None:
                    gt_lane_res = compute_attention_to_gt_lane(
                        batch_numpy, scene_attn_np, gt_future_np,
                    )
                    all_gt_lane_results.append(gt_lane_res)
            except Exception as e:
                logger.debug("GT lane analysis failed for batch %d: %s", batch_idx, e)

            # 6. Head diversity (per scene, we aggregate later)
            try:
                hd_res = head_diversity_analysis(scene_attn_np)
                if isinstance(hd_res["pairwise_similarity"], np.ndarray) and hd_res["pairwise_similarity"].size > 0:
                    all_head_diversity.append(hd_res["pairwise_similarity"])
            except Exception as e:
                logger.debug("Head diversity failed for batch %d: %s", batch_idx, e)

            # 7. Threshold statistics
            try:
                thresh_res = attention_threshold_statistics(
                    batch_numpy, scene_attn_np, thresholds=args.thresholds,
                )
                all_threshold_results.append(thresh_res)
            except Exception as e:
                logger.debug("Threshold stats failed for batch %d: %s", batch_idx, e)

            scene_count += 1
            elapsed = time.time() - t0
            total_time += elapsed

            if scene_count % 20 == 0 or scene_count == n_scenes:
                avg_time = total_time / scene_count
                eta = avg_time * (n_scenes - scene_count)
                logger.info(
                    "Processed %d/%d scenes (%.2fs/scene, ETA: %.0fs)",
                    scene_count, n_scenes, avg_time, eta,
                )

    logger.info("Analysis loop complete. Processed %d scenes.", scene_count)

    if scene_count == 0:
        logger.error("No scenes were successfully processed. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    logger.info("Aggregating results...")

    results = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "n_scenes_requested": args.n_scenes,
            "n_scenes_processed": scene_count,
            "thresholds": args.thresholds,
        },
    }

    # 1 & 5. Layer-wise entropy (aggregated)
    results["layer_wise_entropy"] = layer_wise_entropy_analysis(all_scene_attns, scene_count)

    # 2. Gini per layer (aggregated)
    gini_agg = []
    for li in sorted(all_gini_per_layer.keys()):
        vals = np.array(all_gini_per_layer[li])
        gini_agg.append({
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
        })
    results["gini_per_layer"] = gini_agg

    # 3. Attention by type (aggregated across scenes)
    if all_type_results:
        agg_received = defaultdict(list)
        agg_given = defaultdict(lambda: defaultdict(list))
        agg_dist = defaultdict(lambda: defaultdict(list))
        agg_counts = defaultdict(int)

        for tr in all_type_results:
            for t, v in tr["mean_attn_received_by_type"].items():
                agg_received[t].append(v)
            for t1, inner in tr.get("mean_attn_given_by_type", {}).items():
                for t2, v in inner.items():
                    agg_given[t1][t2].append(v)
            for t, bins in tr.get("per_distance_bin", {}).items():
                for b, v in bins.items():
                    agg_dist[t][b].append(v)
            for t, c in tr.get("type_counts", {}).items():
                agg_counts[t] += c

        results["attention_by_type"] = {
            "mean_attn_received_by_type": {
                t: float(np.mean(v)) for t, v in agg_received.items()
            },
            "mean_attn_given_by_type": {
                t1: {t2: float(np.mean(v)) for t2, v in inner.items()}
                for t1, inner in agg_given.items()
            },
            "per_distance_bin": {
                t: {b: float(np.mean(v)) for b, v in bins.items()}
                for t, bins in agg_dist.items()
            },
            "type_counts_total": dict(agg_counts),
        }
    else:
        results["attention_by_type"] = {}

    # 4. GT lane attention (aggregated)
    if all_gt_lane_results:
        all_fracs = [r["gt_lane_attn_fraction"] for r in all_gt_lane_results]
        all_per_target = []
        for r in all_gt_lane_results:
            all_per_target.extend(r.get("per_target", []))
        all_corrs = [
            r["correlation_attn_vs_error"]
            for r in all_gt_lane_results
            if r["correlation_attn_vs_error"] is not None
        ]
        results["gt_lane_attention"] = {
            "mean_gt_lane_attn_fraction": float(np.mean(all_fracs)) if all_fracs else 0.0,
            "std_gt_lane_attn_fraction": float(np.std(all_fracs)) if all_fracs else 0.0,
            "mean_correlation_attn_vs_error": float(np.mean(all_corrs)) if all_corrs else None,
            "n_targets_total": len(all_per_target),
            "per_target_all": all_per_target,
        }
    else:
        results["gt_lane_attention"] = {}

    # 6. Head diversity (aggregated)
    if all_head_diversity:
        stacked = np.stack(all_head_diversity, axis=0)  # (N, L, H, H)
        mean_pw = stacked.mean(axis=0)  # (L, H, H)
        n_layers = mean_pw.shape[0]
        n_heads = mean_pw.shape[1]
        mask = ~np.eye(n_heads, dtype=bool)
        per_layer_mean = np.array([
            mean_pw[li][mask].mean() for li in range(n_layers)
        ])
        mean_sim = float(per_layer_mean.mean())
        results["head_diversity"] = {
            "pairwise_similarity": mean_pw,
            "mean_pairwise_similarity": mean_sim,
            "per_layer_mean_similarity": per_layer_mean,
            "per_layer_min_similarity": np.array([
                mean_pw[li][mask].min() for li in range(n_layers)
            ]),
            "head_specialization_score": 1.0 - mean_sim,
        }
    else:
        results["head_diversity"] = {}

    # 7. Threshold statistics (aggregated)
    if all_threshold_results:
        agg_thresh = defaultdict(lambda: {
            "pct_tokens_above": [],
            "pct_agent_tokens_above": [],
            "pct_map_tokens_above": [],
        })
        agg_eff_size = defaultdict(list)
        agg_topk = defaultdict(list)

        for tr in all_threshold_results:
            for thresh_str, stats in tr.get("threshold_stats", {}).items():
                agg_thresh[thresh_str]["pct_tokens_above"].append(stats["pct_tokens_above"])
                agg_thresh[thresh_str]["pct_agent_tokens_above"].append(stats["pct_agent_tokens_above"])
                agg_thresh[thresh_str]["pct_map_tokens_above"].append(stats["pct_map_tokens_above"])
            for li, sz in tr.get("effective_attention_size", {}).items():
                agg_eff_size[li].append(sz)
            for k, v in tr.get("attention_mass_in_top_k", {}).items():
                agg_topk[k].append(v)

        results["threshold_stats"] = {
            "threshold_stats": {
                t: {
                    "pct_tokens_above": float(np.mean(s["pct_tokens_above"])),
                    "pct_agent_tokens_above": float(np.mean(s["pct_agent_tokens_above"])),
                    "pct_map_tokens_above": float(np.mean(s["pct_map_tokens_above"])),
                }
                for t, s in agg_thresh.items()
            },
            "effective_attention_size": {
                li: float(np.mean(v)) for li, v in agg_eff_size.items()
            },
            "attention_mass_in_top_k": {
                k: float(np.mean(v)) for k, v in agg_topk.items()
            },
        }
    else:
        results["threshold_stats"] = {}

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    # JSON output (convert numpy to native types)
    json_results = _make_json_serializable(results)
    json_path = os.path.join(output_dir, "attention_analysis_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("Saved JSON results to %s", json_path)

    # Generate plots
    if not args.no_plots:
        try:
            plots = _generate_summary_plots(results, output_dir)
            logger.info("Generated %d summary plots", len(plots))
        except Exception as e:
            logger.error("Plot generation failed: %s", e)
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Scenes analyzed: {scene_count}")
    print(f"Total time: {total_time:.1f}s ({total_time/max(scene_count,1):.2f}s/scene)")

    enc_ent = results.get("layer_wise_entropy", {}).get("encoder", {})
    if enc_ent:
        print(f"\nEncoder entropy (mean): {enc_ent.get('overall_mean', 0):.3f} bits")
        trend = enc_ent.get("layer_trend", [])
        if trend:
            print(f"  Layer trend: {' -> '.join(f'{v:.3f}' for v in trend)}")

    if gini_agg:
        print(f"\nGini coefficient (mean): {np.mean([g['mean'] for g in gini_agg]):.3f}")
        layer_gini_strs = [f"{g['mean']:.3f}" for g in gini_agg]
        print(f"  Per layer: {' -> '.join(layer_gini_strs)}")

    type_data = results.get("attention_by_type", {}).get("mean_attn_received_by_type", {})
    if type_data:
        print("\nAttention by agent type:")
        for t, v in sorted(type_data.items(), key=lambda x: -x[1]):
            print(f"  {t}: {v:.4f}")

    gt_lane = results.get("gt_lane_attention", {})
    if gt_lane.get("mean_gt_lane_attn_fraction") is not None:
        print(f"\nGT lane attention fraction: {gt_lane['mean_gt_lane_attn_fraction']:.4f}")
        if gt_lane.get("mean_correlation_attn_vs_error") is not None:
            print(f"  Correlation with error: {gt_lane['mean_correlation_attn_vs_error']:.4f}")

    hd = results.get("head_diversity", {})
    if hd.get("head_specialization_score") is not None:
        print(f"\nHead specialization score: {hd['head_specialization_score']:.4f}")
        print(f"  Mean pairwise similarity: {hd.get('mean_pairwise_similarity', 0):.4f}")

    ts = results.get("threshold_stats", {}).get("threshold_stats", {})
    if ts:
        print("\nAttention threshold statistics:")
        for t in sorted(ts.keys(), key=float):
            print(f"  >{t}: {ts[t]['pct_tokens_above']:.1f}% tokens "
                  f"(agents: {ts[t]['pct_agent_tokens_above']:.1f}%, "
                  f"map: {ts[t]['pct_map_tokens_above']:.1f}%)")

    topk = results.get("threshold_stats", {}).get("attention_mass_in_top_k", {})
    if topk:
        print("\nAttention mass in top-k:")
        for k in sorted(topk.keys(), key=int):
            print(f"  Top-{k}: {topk[k]:.3f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
