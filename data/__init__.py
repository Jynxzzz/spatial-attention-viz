"""Data pipeline for MTR-Lite trajectory prediction.

This module provides:
- Agent and map polyline feature extraction
- PyTorch Dataset for multi-agent trajectory prediction
- Token bookkeeping for attention visualization
- Collation functions for batching
- Intention point generation via k-means
"""

from .agent_features import (
    AGENT_FEAT_DIM,
    extract_agent_polyline,
    extract_agent_future,
    extract_all_agents,
)
from .map_features import (
    MAP_FEAT_DIM,
    extract_map_polylines,
    resample_polyline,
    find_ego_lane_id,
)
from .token_bookkeeper import TokenBookkeeper
from .polyline_dataset import PolylineDataset
from .collate import mtr_collate_fn
from .intent_points import (
    collect_gt_endpoints,
    kmeans_intent_points,
    generate_and_save,
)

__all__ = [
    # Constants
    "AGENT_FEAT_DIM",
    "MAP_FEAT_DIM",
    # Agent features
    "extract_agent_polyline",
    "extract_agent_future",
    "extract_all_agents",
    # Map features
    "extract_map_polylines",
    "resample_polyline",
    "find_ego_lane_id",
    # Token bookkeeper
    "TokenBookkeeper",
    # Dataset and collation
    "PolylineDataset",
    "mtr_collate_fn",
    # Intention points
    "collect_gt_endpoints",
    "kmeans_intent_points",
    "generate_and_save",
]
