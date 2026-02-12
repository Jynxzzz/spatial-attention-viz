"""Space-Attention BEV Heatmap visualization.

For a target agent, projects scene encoder attention weights onto BEV coordinates:
- Agent tokens -> Gaussian splat at position
- Map tokens -> paint along lane centerline
- Overlay on grayscale lane map with viridis/magma colormap

Shows "where the model looks" for a given target agent.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter

# Color scheme (traffic engineering standard)
COLOR_HISTORY = "#1E88E5"
COLOR_GT_FUTURE = "#2E7D32"
COLOR_PRED = "#E53935"
COLOR_EGO = "#0D47A1"
COLOR_LANE_BG = "#E0E0E0"


def render_space_attention_bev(
    agent_positions_bev: np.ndarray,
    agent_attention: np.ndarray,
    agent_mask: np.ndarray,
    lane_centerlines_bev: np.ndarray,
    map_attention: np.ndarray,
    map_mask: np.ndarray,
    target_history_bev: np.ndarray = None,
    target_future_bev: np.ndarray = None,
    pred_trajectories_bev: np.ndarray = None,
    all_lane_points: np.ndarray = None,
    bev_range: float = 60.0,
    resolution: float = 0.5,
    sigma_agent: float = 3.0,
    sigma_lane: float = 1.5,
    cmap: str = "magma",
    title: str = "Space-Attention BEV Heatmap",
    figsize: tuple = (10, 10),
    ax: plt.Axes = None,
) -> plt.Figure:
    """Render BEV heatmap of attention weights projected onto spatial coordinates.

    Args:
        agent_positions_bev: (A, 2) agent positions in BEV at anchor frame
        agent_attention: (A,) attention weight per agent (head-averaged)
        agent_mask: (A,) bool valid agents
        lane_centerlines_bev: (M, P, 2) lane centerline points in BEV
        map_attention: (M,) attention weight per lane (head-averaged)
        map_mask: (M,) bool valid lanes
        target_history_bev: (11, 2) target agent history (optional)
        target_future_bev: (future_len, 2) GT future (optional)
        pred_trajectories_bev: (K, future_len, 2) predicted trajectories (optional)
        all_lane_points: (M, P, 2) all lanes for background (optional, same as centerlines)
        bev_range: BEV extent in meters
        resolution: grid resolution in meters per pixel
        sigma_agent: Gaussian kernel size for agent splats
        sigma_lane: Gaussian kernel size for lane painting
        cmap: colormap name
        title: figure title
        figsize: figure size
        ax: existing axes (optional)

    Returns:
        matplotlib Figure
    """
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Create attention heatmap grid
    grid_size = int(2 * bev_range / resolution)
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    def bev_to_grid(xy):
        """Convert BEV coordinates to grid indices."""
        gx = int((xy[0] + bev_range) / resolution)
        gy = int((xy[1] + bev_range) / resolution)
        return np.clip(gx, 0, grid_size - 1), np.clip(gy, 0, grid_size - 1)

    # Splat agent attention
    for i in range(len(agent_positions_bev)):
        if not agent_mask[i]:
            continue
        weight = agent_attention[i]
        if weight < 1e-6:
            continue
        gx, gy = bev_to_grid(agent_positions_bev[i])
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            heatmap[gy, gx] += weight

    # Paint lane attention
    for i in range(len(lane_centerlines_bev)):
        if not map_mask[i]:
            continue
        weight = map_attention[i]
        if weight < 1e-6:
            continue
        for p in range(lane_centerlines_bev.shape[1]):
            pt = lane_centerlines_bev[i, p]
            gx, gy = bev_to_grid(pt)
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                heatmap[gy, gx] += weight

    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=max(sigma_agent, sigma_lane))

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Draw background lanes
    if all_lane_points is not None:
        for i in range(len(all_lane_points)):
            if map_mask[i] if i < len(map_mask) else True:
                pts = all_lane_points[i]
                ax.plot(pts[:, 0], pts[:, 1], color=COLOR_LANE_BG, linewidth=0.5, alpha=0.3, zorder=1)

    # Draw heatmap
    extent = [-bev_range, bev_range, -bev_range, bev_range]
    im = ax.imshow(
        heatmap,
        extent=extent,
        origin="lower",
        cmap=cmap,
        alpha=0.7,
        vmin=0,
        vmax=1,
        zorder=2,
    )

    # Draw valid agent positions as dots
    for i in range(len(agent_positions_bev)):
        if agent_mask[i]:
            ax.plot(
                agent_positions_bev[i, 0], agent_positions_bev[i, 1],
                "s", color="white", markersize=4, markeredgecolor="gray",
                markeredgewidth=0.5, alpha=0.8, zorder=5,
            )

    # Draw target history
    if target_history_bev is not None:
        ax.plot(
            target_history_bev[:, 0], target_history_bev[:, 1],
            "-o", color=COLOR_HISTORY, linewidth=2, markersize=3,
            label="History", zorder=6,
        )

    # Draw GT future
    if target_future_bev is not None:
        ax.plot(
            target_future_bev[:, 0], target_future_bev[:, 1],
            "--", color=COLOR_GT_FUTURE, linewidth=2,
            label="Ground Truth", zorder=6,
        )

    # Draw predictions
    if pred_trajectories_bev is not None:
        for k in range(min(6, pred_trajectories_bev.shape[0])):
            alpha = 0.8 if k == 0 else 0.4
            lw = 2 if k == 0 else 1
            ax.plot(
                pred_trajectories_bev[k, :, 0], pred_trajectories_bev[k, :, 1],
                "-", color=COLOR_PRED, linewidth=lw, alpha=alpha,
                label="Prediction" if k == 0 else None, zorder=7,
            )

    # Target agent marker
    ax.plot(0, 0, "^", color=COLOR_EGO, markersize=12, zorder=8)

    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Forward (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)

    if create_fig:
        plt.colorbar(im, ax=ax, label="Attention Weight", shrink=0.7)

    return fig
