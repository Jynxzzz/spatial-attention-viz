"""Utility functions for attention visualization.

Provides helper functions for:
- Spatial attention heatmap generation
- Entity-level attention aggregation
- Coordinate transformations
- Common visualization operations
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def get_spatial_attention_map(
    attn_weights: np.ndarray,
    agent_positions: np.ndarray,
    agent_mask: np.ndarray,
    map_centerlines: np.ndarray,
    map_mask: np.ndarray,
    bev_range: float = 60.0,
    resolution: float = 0.5,
    sigma_agent: float = 3.0,
    sigma_lane: float = 1.5,
) -> np.ndarray:
    """Generate spatial attention heatmap by projecting attention weights onto BEV grid.

    Args:
        attn_weights: (N,) attention weights where N = A + M (agents + map)
        agent_positions: (A, 2) agent positions in BEV coordinates
        agent_mask: (A,) bool indicating valid agents
        map_centerlines: (M, P, 2) lane centerline points in BEV
        map_mask: (M,) bool indicating valid lanes
        bev_range: spatial extent in meters
        resolution: grid resolution in meters per pixel
        sigma_agent: Gaussian kernel size for agent attention
        sigma_lane: Gaussian kernel size for lane attention

    Returns:
        heatmap: (H, W) normalized attention heatmap
    """
    num_agents = len(agent_positions)
    num_map = len(map_centerlines)

    assert len(attn_weights) == num_agents + num_map, (
        f"Attention weights length {len(attn_weights)} does not match "
        f"num_agents ({num_agents}) + num_map ({num_map})"
    )

    # Split attention into agent and map portions
    agent_attn = attn_weights[:num_agents]
    map_attn = attn_weights[num_agents:]

    # Create grid
    grid_size = int(2 * bev_range / resolution)
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    def bev_to_grid(xy):
        """Convert BEV coordinates to grid indices."""
        gx = int((xy[0] + bev_range) / resolution)
        gy = int((xy[1] + bev_range) / resolution)
        return np.clip(gx, 0, grid_size - 1), np.clip(gy, 0, grid_size - 1)

    # Splat agent attention onto grid
    for i in range(num_agents):
        if not agent_mask[i]:
            continue
        weight = agent_attn[i]
        if weight < 1e-6:
            continue
        gx, gy = bev_to_grid(agent_positions[i])
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            heatmap[gy, gx] += weight

    # Paint lane attention onto grid
    for i in range(num_map):
        if not map_mask[i]:
            continue
        weight = map_attn[i]
        if weight < 1e-6:
            continue
        for p in range(map_centerlines.shape[1]):
            pt = map_centerlines[i, p]
            gx, gy = bev_to_grid(pt)
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                heatmap[gy, gx] += weight

    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=max(sigma_agent, sigma_lane))

    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def aggregate_attention_to_entity(
    attn_weights: np.ndarray,
    num_agents: int,
    num_map: int,
    agent_mask: np.ndarray = None,
    map_mask: np.ndarray = None,
    agg_method: str = "mean",
) -> dict:
    """Aggregate attention weights to entity level (per-agent, per-lane).

    Args:
        attn_weights: (nhead, N) or (N,) attention weights where N = A + M
        num_agents: number of agent tokens (A)
        num_map: number of map tokens (M)
        agent_mask: (A,) bool indicating valid agents (optional)
        map_mask: (M,) bool indicating valid map tokens (optional)
        agg_method: "mean" or "max" for aggregating across heads

    Returns:
        dict with:
            "agent_attention": (A,) per-agent attention
            "map_attention": (M,) per-map attention
            "agent_total": scalar total attention to agents
            "map_total": scalar total attention to map
    """
    # Handle head dimension
    if attn_weights.ndim == 2:
        # (nhead, N) -> aggregate heads first
        if agg_method == "mean":
            attn = attn_weights.mean(axis=0)
        elif agg_method == "max":
            attn = attn_weights.max(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")
    else:
        attn = attn_weights  # Already (N,)

    assert len(attn) == num_agents + num_map

    # Split into agent and map
    agent_attn = attn[:num_agents]
    map_attn = attn[num_agents:]

    # Apply masks
    if agent_mask is not None:
        agent_attn = agent_attn * agent_mask.astype(np.float32)
    if map_mask is not None:
        map_attn = map_attn * map_mask.astype(np.float32)

    return {
        "agent_attention": agent_attn,
        "map_attention": map_attn,
        "agent_total": agent_attn.sum(),
        "map_total": map_attn.sum(),
    }


def compute_attention_entropy(attn_weights: np.ndarray, eps: float = 1e-8) -> float:
    """Compute entropy (in bits) of attention distribution.

    Higher entropy indicates more uniform (less focused) attention.

    Args:
        attn_weights: (N,) attention weights (should sum to ~1)
        eps: small value to avoid log(0)

    Returns:
        entropy in bits
    """
    attn = np.clip(attn_weights, eps, 1.0)
    attn = attn / (attn.sum() + eps)  # Normalize
    return -np.sum(attn * np.log2(attn + eps))


def compute_attention_gini(attn_weights: np.ndarray) -> float:
    """Compute Gini coefficient of attention distribution.

    Gini = 0: perfectly uniform attention
    Gini = 1: all attention on one token

    Args:
        attn_weights: (N,) attention weights

    Returns:
        Gini coefficient [0, 1]
    """
    attn = np.sort(attn_weights)
    n = len(attn)
    if n == 0 or attn.sum() == 0:
        return 0.0

    index = np.arange(1, n + 1)
    return (2 * np.sum(index * attn)) / (n * attn.sum()) - (n + 1) / n


def extract_attention_for_query_token(
    attention_matrix: np.ndarray,
    query_idx: int,
    num_agents: int,
    num_map: int,
    agg_heads: bool = True,
) -> dict:
    """Extract attention weights from a specific query token to all others.

    Args:
        attention_matrix: (nhead, N, N) or (N, N) attention matrix
        query_idx: index of query token
        num_agents: number of agent tokens (A)
        num_map: number of map tokens (M)
        agg_heads: if True, average across heads

    Returns:
        dict with:
            "agent_attention": (A,) attention to agent tokens
            "map_attention": (M,) attention to map tokens
            "self_attention": scalar attention to self
    """
    if attention_matrix.ndim == 3:
        # (nhead, N, N)
        if agg_heads:
            attn = attention_matrix.mean(axis=0)  # (N, N)
        else:
            attn = attention_matrix  # Keep head dimension
    else:
        attn = attention_matrix  # (N, N)

    # Extract row for query token
    if attn.ndim == 2:
        query_attn = attn[query_idx, :]  # (N,)
    else:
        query_attn = attn[:, query_idx, :]  # (nhead, N)

    # Split into agent and map
    if query_attn.ndim == 1:
        agent_attn = query_attn[:num_agents]
        map_attn = query_attn[num_agents : num_agents + num_map]
        self_attn = query_attn[query_idx]
    else:
        agent_attn = query_attn[:, :num_agents]
        map_attn = query_attn[:, num_agents : num_agents + num_map]
        self_attn = query_attn[:, query_idx]

    return {
        "agent_attention": agent_attn,
        "map_attention": map_attn,
        "self_attention": self_attn,
    }


def world_to_bev(
    points_world: np.ndarray,
    ego_pos: tuple,
    ego_heading: float,
) -> np.ndarray:
    """Transform world coordinates to BEV (ego-centric).

    BEV coordinate system:
        X+ : forward (aligned with ego heading)
        Y+ : left
        Origin: ego position

    Args:
        points_world: (N, 2) or (N, M, 2) points in world coordinates
        ego_pos: (x, y) ego position in world frame
        ego_heading: ego heading in radians

    Returns:
        points_bev: same shape as input, transformed to BEV
    """
    original_shape = points_world.shape
    points = points_world.reshape(-1, 2)

    # Translate to ego position
    dx = points[:, 0] - ego_pos[0]
    dy = points[:, 1] - ego_pos[1]

    # Rotate to ego heading
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)

    x_bev = cos_h * dx - sin_h * dy
    y_bev = sin_h * dx + cos_h * dy

    points_bev = np.stack([x_bev, y_bev], axis=-1)
    return points_bev.reshape(original_shape)


def bev_to_world(
    points_bev: np.ndarray,
    ego_pos: tuple,
    ego_heading: float,
) -> np.ndarray:
    """Transform BEV coordinates back to world frame.

    Args:
        points_bev: (N, 2) or (N, M, 2) points in BEV coordinates
        ego_pos: (x, y) ego position in world frame
        ego_heading: ego heading in radians

    Returns:
        points_world: same shape as input, transformed to world frame
    """
    original_shape = points_bev.shape
    points = points_bev.reshape(-1, 2)

    # Rotate back
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)

    dx = cos_h * points[:, 0] - sin_h * points[:, 1]
    dy = sin_h * points[:, 0] + cos_h * points[:, 1]

    # Translate back
    x_world = dx + ego_pos[0]
    y_world = dy + ego_pos[1]

    points_world = np.stack([x_world, y_world], axis=-1)
    return points_world.reshape(original_shape)


def top_k_attention_tokens(
    attn_weights: np.ndarray,
    k: int,
    mask: np.ndarray = None,
    return_indices: bool = True,
) -> tuple:
    """Find top-K tokens by attention weight.

    Args:
        attn_weights: (N,) attention weights
        k: number of top tokens to return
        mask: (N,) bool indicating valid tokens
        return_indices: if True, return indices; otherwise return (indices, weights)

    Returns:
        if return_indices:
            indices: (K,) indices of top-K tokens
        else:
            (indices, weights): (K,) indices and (K,) attention weights
    """
    if mask is not None:
        attn = attn_weights * mask.astype(np.float32)
    else:
        attn = attn_weights

    k = min(k, len(attn))
    indices = np.argsort(attn)[::-1][:k]

    if return_indices:
        return indices
    else:
        return indices, attn[indices]


def normalize_attention(attn_weights: np.ndarray, method: str = "sum") -> np.ndarray:
    """Normalize attention weights.

    Args:
        attn_weights: (N,) or (nhead, N) attention weights
        method: "sum" (normalize to sum=1) or "max" (normalize to max=1)

    Returns:
        normalized attention weights
    """
    if method == "sum":
        s = attn_weights.sum(axis=-1, keepdims=True)
        return attn_weights / (s + 1e-8)
    elif method == "max":
        m = attn_weights.max(axis=-1, keepdims=True)
        return attn_weights / (m + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_attention_statistics(attn_weights: np.ndarray) -> dict:
    """Compute various statistics for attention distribution.

    Args:
        attn_weights: (N,) attention weights

    Returns:
        dict with statistics:
            - mean, std, min, max
            - entropy (bits)
            - gini coefficient
            - top1_ratio: attention on top token
            - top5_ratio: cumulative attention on top 5 tokens
    """
    attn = attn_weights / (attn_weights.sum() + 1e-8)

    stats = {
        "mean": float(attn.mean()),
        "std": float(attn.std()),
        "min": float(attn.min()),
        "max": float(attn.max()),
        "entropy": compute_attention_entropy(attn),
        "gini": compute_attention_gini(attn),
    }

    # Top-K ratios
    sorted_attn = np.sort(attn)[::-1]
    stats["top1_ratio"] = float(sorted_attn[0]) if len(sorted_attn) > 0 else 0.0
    stats["top5_ratio"] = float(sorted_attn[:5].sum()) if len(sorted_attn) >= 5 else float(sorted_attn.sum())

    return stats
