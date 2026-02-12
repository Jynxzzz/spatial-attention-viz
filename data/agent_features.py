"""Agent feature engineering for MTR-Lite.

Extracts 29-dimensional per-timestep features for each agent:
  pos(2) + prev_pos(2) + vel(2) + accel(2) + heading_sincos(2) +
  bbox(2) + type_onehot(5) + temporal_embed(11) + is_ego(1) = 29

All coordinates are in BEV (ego-centric) frame.
"""

import math

import numpy as np

AGENT_FEAT_DIM = 29
AGENT_TYPES = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]
NUM_AGENT_TYPES = len(AGENT_TYPES)


def _type_to_onehot(agent_type: str) -> np.ndarray:
    """Convert agent type string to 5-dim one-hot vector."""
    oh = np.zeros(NUM_AGENT_TYPES, dtype=np.float32)
    t = agent_type.lower() if isinstance(agent_type, str) else "unknown"
    idx = AGENT_TYPES.index(t) if t in AGENT_TYPES else NUM_AGENT_TYPES - 1
    oh[idx] = 1.0
    return oh


def _temporal_embedding(history_len: int, step: int) -> np.ndarray:
    """One-hot temporal embedding for position within history window."""
    emb = np.zeros(history_len, dtype=np.float32)
    emb[step] = 1.0
    return emb


def extract_agent_polyline(
    obj: dict,
    anchor_frame: int,
    history_len: int,
    ego_pos: tuple,
    ego_heading_deg: float,
    is_ego: bool = False,
) -> tuple:
    """Extract a (history_len, 29) feature polyline for one agent.

    Args:
        obj: scene object dict with position, heading, velocity, valid, type, width, length
        anchor_frame: current frame index
        history_len: number of history frames (including anchor)
        ego_pos: (x, y) ego world position at anchor for BEV transform
        ego_heading_deg: ego heading in degrees at anchor
        is_ego: whether this agent is the ego vehicle

    Returns:
        features: (history_len, 29) float32 array
        valid_mask: (history_len,) bool array
    """
    h_start = anchor_frame - (history_len - 1)
    h_end = anchor_frame + 1

    features = np.zeros((history_len, AGENT_FEAT_DIM), dtype=np.float32)
    valid_mask = np.zeros(history_len, dtype=bool)

    # Precompute BEV rotation
    heading_rad = math.radians(ego_heading_deg)
    adjusted = heading_rad - np.pi / 2
    c, s = np.cos(-adjusted), np.sin(-adjusted)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    ego_xy = np.array(ego_pos, dtype=np.float64)

    # Agent metadata (constant across time)
    type_oh = _type_to_onehot(obj.get("type", "unknown"))
    width = float(obj.get("width", 2.0))
    length = float(obj.get("length", 5.0))
    bbox = np.array([width, length], dtype=np.float32)
    is_ego_flag = np.array([1.0 if is_ego else 0.0], dtype=np.float32)

    for t_idx, frame in enumerate(range(h_start, h_end)):
        if frame < 0 or frame >= len(obj["valid"]) or not obj["valid"][frame]:
            continue

        valid_mask[t_idx] = True

        # Position in world -> BEV
        pos_w = np.array(
            [float(obj["position"][frame]["x"]), float(obj["position"][frame]["y"])],
            dtype=np.float64,
        )
        pos_bev = ((pos_w - ego_xy) @ R.T).astype(np.float32)

        # Previous position
        if frame > 0 and obj["valid"][frame - 1]:
            prev_w = np.array(
                [float(obj["position"][frame - 1]["x"]),
                 float(obj["position"][frame - 1]["y"])],
                dtype=np.float64,
            )
            prev_bev = ((prev_w - ego_xy) @ R.T).astype(np.float32)
        else:
            prev_bev = pos_bev.copy()

        # Velocity in BEV
        vel_w = np.array(
            [float(obj["velocity"][frame]["x"]), float(obj["velocity"][frame]["y"])],
            dtype=np.float64,
        )
        vel_bev = (vel_w @ R.T).astype(np.float32)

        # Acceleration (from velocity difference)
        if frame > 0 and obj["valid"][frame - 1]:
            prev_vel_w = np.array(
                [float(obj["velocity"][frame - 1]["x"]),
                 float(obj["velocity"][frame - 1]["y"])],
                dtype=np.float64,
            )
            prev_vel_bev = (prev_vel_w @ R.T).astype(np.float32)
            accel_bev = (vel_bev - prev_vel_bev) * 10.0  # dt=0.1s -> multiply by 10
        else:
            accel_bev = np.zeros(2, dtype=np.float32)

        # Heading in BEV (sin, cos)
        heading_w = obj["heading"][frame]
        if isinstance(heading_w, (list, tuple)):
            heading_w = float(heading_w[0])
        heading_w = float(heading_w)
        heading_bev_rad = math.radians(heading_w) - adjusted
        heading_sc = np.array(
            [np.sin(heading_bev_rad), np.cos(heading_bev_rad)], dtype=np.float32
        )

        # Temporal embedding
        temp_emb = _temporal_embedding(history_len, t_idx)

        # Concatenate: pos(2) + prev_pos(2) + vel(2) + accel(2) + heading_sc(2) +
        #              bbox(2) + type_onehot(5) + temporal(11) + is_ego(1) = 29
        features[t_idx] = np.concatenate([
            pos_bev,        # 2
            prev_bev,       # 2
            vel_bev,        # 2
            accel_bev,      # 2
            heading_sc,     # 2
            bbox,           # 2
            type_oh,        # 5
            temp_emb,       # 11
            is_ego_flag,    # 1
        ])

    return features, valid_mask


def extract_all_agents(
    scene: dict,
    anchor_frame: int,
    history_len: int,
    ego_pos: tuple,
    ego_heading_deg: float,
    max_agents: int = 32,
    neighbor_distance: float = 50.0,
) -> dict:
    """Extract agent polylines for all valid agents near ego.

    Returns dict with:
        agent_polylines: (max_agents, history_len, 29) float32
        agent_valid: (max_agents, history_len) bool
        agent_mask: (max_agents,) bool - which agent slots are occupied
        agent_ids: list of original object indices (for bookkeeping)
        target_mask: (max_agents,) bool - agents eligible as prediction targets
    """
    av_idx = scene["av_idx"]
    objects = scene["objects"]

    agent_polylines = np.zeros((max_agents, history_len, AGENT_FEAT_DIM), dtype=np.float32)
    agent_valid = np.zeros((max_agents, history_len), dtype=bool)
    agent_mask = np.zeros(max_agents, dtype=bool)
    agent_ids = []
    agent_types = np.full(max_agents, -1, dtype=np.int64)  # -1 = empty slot
    target_mask = np.zeros(max_agents, dtype=bool)

    # Ego position in world at anchor
    ego_world = np.array(ego_pos, dtype=np.float64)

    # Collect (distance, obj_idx) pairs and sort by distance
    candidates = []
    for i, obj in enumerate(objects):
        if anchor_frame >= len(obj["valid"]) or not obj["valid"][anchor_frame]:
            continue
        pos = obj["position"][anchor_frame]
        obj_world = np.array([float(pos["x"]), float(pos["y"])], dtype=np.float64)
        dist = np.linalg.norm(obj_world - ego_world)
        if dist <= neighbor_distance or i == av_idx:
            candidates.append((dist, i))

    # Sort: ego first, then by distance
    candidates.sort(key=lambda x: (-1 if x[1] == av_idx else x[0]))

    count = 0
    for dist, obj_idx in candidates:
        if count >= max_agents:
            break

        obj = objects[obj_idx]
        is_ego = (obj_idx == av_idx)

        feats, valid = extract_agent_polyline(
            obj, anchor_frame, history_len, ego_pos, ego_heading_deg, is_ego=is_ego,
        )

        # Skip agents with no valid history frames
        if not valid.any():
            continue

        agent_polylines[count] = feats
        agent_valid[count] = valid
        agent_mask[count] = True
        agent_ids.append(obj_idx)
        # Store agent type index: 0=vehicle, 1=pedestrian, 2=cyclist, 3=other, 4=unknown
        obj_type = obj.get("type", "unknown")
        t = obj_type.lower() if isinstance(obj_type, str) else "unknown"
        agent_types[count] = AGENT_TYPES.index(t) if t in AGENT_TYPES else NUM_AGENT_TYPES - 1

        # Target eligibility: must have enough valid future frames
        future_start = anchor_frame + 1
        future_end = min(anchor_frame + 81, len(obj["valid"]))
        if future_end > future_start:
            future_valid = obj["valid"][future_start:future_end]
            # Require at least 50% valid future
            if sum(future_valid) >= (future_end - future_start) * 0.5:
                target_mask[count] = True

        count += 1

    return {
        "agent_polylines": agent_polylines,
        "agent_valid": agent_valid,
        "agent_mask": agent_mask,
        "agent_ids": agent_ids,
        "agent_types": agent_types,
        "target_mask": target_mask,
    }


def extract_agent_future(
    obj: dict,
    anchor_frame: int,
    future_len: int,
    ego_pos: tuple,
    ego_heading_deg: float,
) -> tuple:
    """Extract future trajectory for one agent in BEV coordinates.

    Returns:
        future_traj: (future_len, 2) float32 positions in BEV
        future_valid: (future_len,) bool validity mask
    """
    heading_rad = math.radians(ego_heading_deg)
    adjusted = heading_rad - np.pi / 2
    c, s = np.cos(-adjusted), np.sin(-adjusted)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    ego_xy = np.array(ego_pos, dtype=np.float64)

    future_traj = np.zeros((future_len, 2), dtype=np.float32)
    future_valid = np.zeros(future_len, dtype=bool)

    for t in range(future_len):
        frame = anchor_frame + 1 + t
        if frame >= len(obj["valid"]) or not obj["valid"][frame]:
            continue
        pos_w = np.array(
            [float(obj["position"][frame]["x"]),
             float(obj["position"][frame]["y"])],
            dtype=np.float64,
        )
        future_traj[t] = ((pos_w - ego_xy) @ R.T).astype(np.float32)
        future_valid[t] = True

    return future_traj, future_valid
