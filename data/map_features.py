"""Map polyline feature extraction for MTR-Lite.

Extracts 9-dimensional per-point features for map lane polylines:
  pos(2) + direction(2) + lane_flags(3) + prev_point(2) = 9

Uses waterflow graph (BFS from ego lane) expanded to 64 lanes.
Falls back to nearest-distance selection when ego lane is not found.
"""

import math

import networkx as nx
import numpy as np

MAP_FEAT_DIM = 9


def resample_polyline(points: np.ndarray, num_points: int = 20) -> np.ndarray:
    """Arc-length uniform resampling of a polyline.

    Args:
        points: (N, 2) array of polyline coordinates
        num_points: desired number of output points

    Returns:
        (num_points, 2) uniformly sampled points along the polyline
    """
    if len(points) < 2:
        return np.zeros((num_points, 2), dtype=np.float32)

    distances = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    cumulative = np.insert(np.cumsum(distances), 0, 0)

    if cumulative[-1] < 1e-6:
        return np.tile(points[0], (num_points, 1)).astype(np.float32)

    target = np.linspace(0, cumulative[-1], num=num_points)
    x_new = np.interp(target, cumulative, points[:, 0])
    y_new = np.interp(target, cumulative, points[:, 1])
    return np.stack((x_new, y_new), axis=-1).astype(np.float32)


def _world_to_bev(points: np.ndarray, ego_pos: tuple, ego_heading_deg: float) -> np.ndarray:
    """Transform world coordinates to ego-centric BEV frame."""
    heading_rad = math.radians(ego_heading_deg)
    adjusted = heading_rad - np.pi / 2
    c, s = np.cos(-adjusted), np.sin(-adjusted)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    dxdy = points.astype(np.float64) - np.array(ego_pos, dtype=np.float64)
    return (dxdy @ R.T).astype(np.float32)


def find_ego_lane_id(sdc_pos: np.ndarray, lane_graph: dict, threshold: float = 5.0):
    """Find the lane closest to the SDC position within threshold."""
    min_dist = float("inf")
    ego_lane_id = None
    for lane_id, lane_pts in lane_graph["lanes"].items():
        if lane_pts is None or len(lane_pts) < 2:
            continue
        dists = np.linalg.norm(lane_pts[:, :2] - sdc_pos, axis=1)
        d = dists.min()
        if d < min_dist and d < threshold:
            min_dist = d
            ego_lane_id = lane_id
    return ego_lane_id


def build_waterflow_graph(lane_graph: dict, ego_lane_id, max_hops: int = 5):
    """Build waterflow graph with extended BFS to reach ~64 lanes.

    Uses 5 hops (instead of 3) to expand the local subgraph for MTR-Lite.

    Returns:
        G: nx.DiGraph with lane nodes and edges
        stages: list of lists, lanes added at each BFS stage
    """
    G = nx.DiGraph()
    stages = []
    visited = set()

    current_level = [ego_lane_id]
    visited.add(ego_lane_id)
    G.add_node(ego_lane_id)
    stages.append([ego_lane_id])

    for _ in range(max_hops):
        next_level = []
        for node in current_level:
            for succ in lane_graph["suc_pairs"].get(node, []):
                if succ not in visited and succ in lane_graph["lanes"]:
                    G.add_edge(node, succ, type="successor")
                    visited.add(succ)
                    next_level.append(succ)
            for side_key in ["left_pairs", "right_pairs"]:
                for side_id in lane_graph[side_key].get(node, []):
                    if side_id not in visited and side_id in lane_graph["lanes"]:
                        G.add_edge(node, side_id, type=side_key.replace("_pairs", ""))
                        visited.add(side_id)
                        next_level.append(side_id)
        if next_level:
            stages.append(next_level)
        current_level = next_level

    return G, stages


def _get_nearest_lanes(sdc_xy: np.ndarray, lane_graph: dict, max_lanes: int) -> list:
    """Fallback: get nearest lanes by distance when ego lane not found."""
    distances = []
    for lane_id, pts in lane_graph["lanes"].items():
        if pts is None or len(pts) < 2:
            continue
        dists = np.linalg.norm(pts[:, :2] - sdc_xy, axis=1)
        distances.append((lane_id, dists.min()))
    distances.sort(key=lambda x: x[1])
    return [lid for lid, _ in distances[:max_lanes]]


def extract_map_polylines(
    scene: dict,
    ego_pos: tuple,
    ego_heading_deg: float,
    max_polylines: int = 64,
    points_per_lane: int = 20,
    max_hops: int = 5,
) -> dict:
    """Extract map lane polylines with per-point features.

    Each lane is resampled to `points_per_lane` points, each with 9-dim features:
      pos(2) + direction(2) + lane_flags(3) + prev_point(2)

    lane_flags: [is_ego_lane, has_traffic_light, has_stop_sign]

    Returns dict with:
        map_polylines: (max_polylines, points_per_lane, 9) float32
        map_valid: (max_polylines, points_per_lane) bool
        map_mask: (max_polylines,) bool
        lane_ids: list of original lane_id values (for bookkeeping)
        lane_centerlines_bev: (max_polylines, points_per_lane, 2) for visualization
    """
    lane_graph = scene["lane_graph"]
    sdc_xy = np.array([ego_pos[0], ego_pos[1]], dtype=np.float64)

    # Find ego lane
    ego_lane_id = find_ego_lane_id(sdc_xy, lane_graph, threshold=5.0)

    # Build waterflow graph with extended hops
    if ego_lane_id is not None:
        G, stages = build_waterflow_graph(lane_graph, ego_lane_id, max_hops=max_hops)
        lane_ids = list(G.nodes)
    else:
        lane_ids = _get_nearest_lanes(sdc_xy, lane_graph, max_polylines)

    # Collect traffic light and stop sign lanes
    tl_lanes = set()
    for tl_frame in scene.get("traffic_lights", []):
        if isinstance(tl_frame, list):
            for tl in tl_frame:
                if isinstance(tl, dict) and "lane" in tl:
                    tl_lanes.add(tl["lane"])
    ss_lanes = set()
    for ss in lane_graph.get("stop_signs", []):
        if isinstance(ss, dict) and "lane" in ss:
            ss_lanes.add(ss["lane"])

    # Truncate to max_polylines
    lane_ids = lane_ids[:max_polylines]

    # Initialize outputs
    map_polylines = np.zeros((max_polylines, points_per_lane, MAP_FEAT_DIM), dtype=np.float32)
    map_valid = np.zeros((max_polylines, points_per_lane), dtype=bool)
    map_mask = np.zeros(max_polylines, dtype=bool)
    lane_centerlines_bev = np.zeros((max_polylines, points_per_lane, 2), dtype=np.float32)

    for i, lid in enumerate(lane_ids):
        centerline = lane_graph["lanes"].get(lid)
        if centerline is None or len(centerline) < 2:
            continue

        pts_world = centerline[:, :2].astype(np.float64)
        pts_bev = _world_to_bev(pts_world, ego_pos, ego_heading_deg)

        # Resample to fixed points
        pts_resampled = resample_polyline(pts_bev, num_points=points_per_lane)
        lane_centerlines_bev[i] = pts_resampled

        # Lane flags
        is_ego_lane = 1.0 if lid == ego_lane_id else 0.0
        has_tl = 1.0 if lid in tl_lanes else 0.0
        has_ss = 1.0 if lid in ss_lanes else 0.0
        flags = np.array([is_ego_lane, has_tl, has_ss], dtype=np.float32)

        for p in range(points_per_lane):
            pos = pts_resampled[p]

            # Direction: vector to next point (or from previous for last)
            if p < points_per_lane - 1:
                direction = pts_resampled[p + 1] - pts_resampled[p]
            else:
                direction = pts_resampled[p] - pts_resampled[p - 1]
            norm = np.linalg.norm(direction) + 1e-6
            direction = direction / norm

            # Previous point position
            if p > 0:
                prev_pt = pts_resampled[p - 1]
            else:
                prev_pt = pts_resampled[p]

            # Feature: pos(2) + direction(2) + flags(3) + prev_point(2) = 9
            map_polylines[i, p] = np.concatenate([pos, direction, flags, prev_pt])
            map_valid[i, p] = True

        map_mask[i] = True

    return {
        "map_polylines": map_polylines,
        "map_valid": map_valid,
        "map_mask": map_mask,
        "lane_ids": lane_ids,
        "lane_centerlines_bev": lane_centerlines_bev,
        "ego_lane_id": ego_lane_id,
    }
