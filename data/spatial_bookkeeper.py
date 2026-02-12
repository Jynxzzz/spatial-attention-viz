"""Spatial Token Bookkeeper with BEV coordinates for visualization.

Extended bookkeeper that stores BEV spatial information (position, heading, velocity)
for each agent token and lane centerlines for each map token. This enables converting
abstract attention weights into 2D spatial heatmaps overlaid on BEV scenes.
"""

import numpy as np


class SpatialTokenBookkeeper:
    """TokenBookkeeper with BEV spatial information for visualization.

    This class maintains the mapping from token indices to their physical
    BEV coordinates, enabling attention visualization on realistic scenes.
    """

    def __init__(self):
        """Initialize empty bookkeeper."""
        self.agent_tokens = {}  # token_idx -> agent spatial info dict
        self.map_tokens = {}    # token_idx -> map spatial info dict
        self.agent_token_ranges = {}  # agent_idx -> (start_idx, end_idx)
        self.map_token_ranges = {}    # map_idx -> (start_idx, end_idx)
        self.n_agent_tokens = 0
        self.n_map_tokens = 0

    def register_agent(self, token_idx, agent_idx, pos_bev, heading, velocity):
        """Register agent token with BEV coordinates.

        Args:
            token_idx: Token index in the full sequence
            agent_idx: Agent slot index (0 = ego, 1-31 = neighbors)
            pos_bev: (2,) array, current position in BEV coordinates (meters)
            heading: float, current heading in radians (BEV frame)
            velocity: (2,) array, current velocity in BEV frame (m/s)
        """
        # Compute oriented bounding box corners for visualization
        # Assuming standard vehicle dimensions
        length, width = 4.5, 2.0

        # Vehicle corners in local frame (center at origin, heading = 0)
        corners_local = np.array([
            [length/2, width/2],
            [length/2, -width/2],
            [-length/2, -width/2],
            [-length/2, width/2],
        ], dtype=np.float32)

        # Rotate to BEV heading
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        corners_bev = (corners_local @ R.T) + pos_bev

        self.agent_tokens[token_idx] = {
            'agent_idx': agent_idx,
            'pos': np.array(pos_bev, dtype=np.float32),
            'heading': float(heading),
            'velocity': np.array(velocity, dtype=np.float32),
            'bbox_corners': corners_bev,
            'length': length,
            'width': width,
        }

        # Track token range for this agent
        if agent_idx not in self.agent_token_ranges:
            self.agent_token_ranges[agent_idx] = (token_idx, token_idx + 1)
        else:
            start, _ = self.agent_token_ranges[agent_idx]
            self.agent_token_ranges[agent_idx] = (start, token_idx + 1)

    def register_map(self, token_idx, map_idx, lane_id, centerline_bev):
        """Register map token with BEV polyline.

        Args:
            token_idx: Token index in the full sequence
            map_idx: Map polyline slot index (0-63)
            lane_id: Original lane ID string for tracking
            centerline_bev: (N, 2) array of centerline points in BEV (meters)
        """
        centerline = np.array(centerline_bev, dtype=np.float32)

        # Compute bounding box for spatial queries
        if len(centerline) > 0:
            bbox_min = centerline.min(axis=0)
            bbox_max = centerline.max(axis=0)
        else:
            bbox_min = bbox_max = np.zeros(2, dtype=np.float32)

        self.map_tokens[token_idx] = {
            'map_idx': map_idx,
            'lane_id': lane_id,
            'centerline': centerline,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
        }

        # Track token range for this map polyline
        if map_idx not in self.map_token_ranges:
            self.map_token_ranges[map_idx] = (token_idx, token_idx + 1)
        else:
            start, _ = self.map_token_ranges[map_idx]
            self.map_token_ranges[map_idx] = (start, token_idx + 1)

    @classmethod
    def from_batch(cls, batch, agent_token_start=0, map_token_start=32):
        """Construct SpatialTokenBookkeeper from dataset batch.

        Args:
            batch: Dataset sample dict from PolylineDataset
            agent_token_start: Token index where agent tokens start (default 0)
            map_token_start: Token index where map tokens start (default 32)

        Returns:
            SpatialTokenBookkeeper instance
        """
        bookkeeper = cls()

        # Extract agent spatial info from polyline features
        agent_polylines = batch['agent_polylines']  # (A, 11, 29) or (B, A, 11, 29)
        agent_mask = batch['agent_mask']  # (A,) or (B, A)

        # Handle both batched and unbatched inputs
        if agent_polylines.dim() == 4:  # Batched
            agent_polylines = agent_polylines[0]
            agent_mask = agent_mask[0]

        # Agent features: pos(2) + prev_pos(2) + vel(2) + accel(2) + heading_sincos(2) + ...
        token_idx = agent_token_start

        for agent_idx in range(len(agent_polylines)):
            if not agent_mask[agent_idx].item():
                continue

            # Use current frame (last timestep = index 10)
            curr_feats = agent_polylines[agent_idx, -1]  # (29,)

            # Extract spatial information
            pos_bev = curr_feats[0:2].cpu().numpy()
            vel_bev = curr_feats[4:6].cpu().numpy()
            heading_sin = curr_feats[8].item()
            heading_cos = curr_feats[9].item()
            heading_rad = np.arctan2(heading_sin, heading_cos)

            bookkeeper.register_agent(
                token_idx=token_idx,
                agent_idx=agent_idx,
                pos_bev=pos_bev,
                heading=heading_rad,
                velocity=vel_bev,
            )

            token_idx += 1

        bookkeeper.n_agent_tokens = token_idx - agent_token_start

        # Extract map spatial info
        map_mask = batch['map_mask']  # (M,) or (B, M)
        lane_centerlines_bev = batch.get('lane_centerlines_bev')  # (M, 20, 2) or (B, M, 20, 2)
        lane_ids = batch.get('lane_ids', [])

        # Handle batched inputs
        if map_mask.dim() == 2:  # Batched
            map_mask = map_mask[0]
        if lane_centerlines_bev is not None and lane_centerlines_bev.dim() == 4:
            lane_centerlines_bev = lane_centerlines_bev[0]

        if lane_centerlines_bev is not None:
            token_idx = map_token_start

            for map_idx in range(len(map_mask)):
                if not map_mask[map_idx].item():
                    continue

                centerline = lane_centerlines_bev[map_idx].cpu().numpy()  # (20, 2)
                lane_id = lane_ids[map_idx] if map_idx < len(lane_ids) else f"lane_{map_idx}"

                bookkeeper.register_map(
                    token_idx=token_idx,
                    map_idx=map_idx,
                    lane_id=lane_id,
                    centerline_bev=centerline,
                )

                token_idx += 1

            bookkeeper.n_map_tokens = token_idx - map_token_start

        return bookkeeper

    def get_spatial_map(self, attention_weights, resolution=0.5, radius=60):
        """Convert attention weights to 2D spatial heatmap.

        This is the core function that transforms abstract attention distributions
        into physical spatial heatmaps by splatting agent attention as Gaussians
        and painting lane attention along polylines.

        Args:
            attention_weights: (N,) numpy array of attention weights
            resolution: meters per pixel (default 0.5m/px)
            radius: BEV radius in meters (default 60m)

        Returns:
            heatmap: (H, W) numpy array of attention intensity
            extent: (xmin, xmax, ymin, ymax) for matplotlib imshow
        """
        from visualization.spatial_utils import gaussian_splat_2d, paint_polyline_2d

        H = W = int(2 * radius / resolution)
        heatmap = np.zeros((H, W), dtype=np.float32)

        # Splat agent attention as Gaussians
        for token_idx, info in self.agent_tokens.items():
            if token_idx >= len(attention_weights):
                continue

            attn_weight = attention_weights[token_idx]
            if attn_weight <= 0:
                continue

            pos = info['pos']

            # Gaussian splat with sigma=3.0 meters (captures vehicle influence)
            gaussian_splat_2d(
                heatmap, pos, attn_weight,
                sigma=3.0, resolution=resolution, radius=radius
            )

        # Paint lane attention along centerlines
        for token_idx, info in self.map_tokens.items():
            if token_idx >= len(attention_weights):
                continue

            attn_weight = attention_weights[token_idx]
            if attn_weight <= 0:
                continue

            centerline = info['centerline']
            if len(centerline) < 2:
                continue

            # Paint along polyline with width=2.0 meters
            paint_polyline_2d(
                heatmap, centerline, attn_weight,
                width=2.0, resolution=resolution, radius=radius
            )

        extent = (-radius, radius, -radius, radius)
        return heatmap, extent

    def get_top_attended_objects(self, attention_weights, top_k=3):
        """Get the top-k most attended objects with their spatial info.

        Args:
            attention_weights: (N,) numpy array
            top_k: number of top objects to return

        Returns:
            list of tuples: (token_idx, attn_weight, object_type, spatial_info)
        """
        results = []

        # Check agent tokens
        for token_idx, info in self.agent_tokens.items():
            if token_idx >= len(attention_weights):
                continue
            results.append((
                token_idx,
                attention_weights[token_idx],
                'agent',
                info
            ))

        # Check map tokens
        for token_idx, info in self.map_tokens.items():
            if token_idx >= len(attention_weights):
                continue
            results.append((
                token_idx,
                attention_weights[token_idx],
                'map',
                info
            ))

        # Sort by attention weight descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def __repr__(self):
        return (f"SpatialTokenBookkeeper("
                f"n_agent_tokens={self.n_agent_tokens}, "
                f"n_map_tokens={self.n_map_tokens})")
