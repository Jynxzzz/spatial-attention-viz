"""Lane-Token Activation Map visualization.

BEV with lanes colored by cumulative decoder map-attention:
- Warm colors (red/yellow) = high attention
- Cool colors (blue/purple) = low attention
- Line width proportional to attention
- Sidebar bar chart ranking lanes by attention

Shows "which lanes guide the prediction."
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def render_lane_activation_map(
    lane_centerlines_bev: np.ndarray,
    lane_attention: np.ndarray,
    lane_mask: np.ndarray,
    lane_labels: list = None,
    target_history_bev: np.ndarray = None,
    target_future_bev: np.ndarray = None,
    pred_trajectories_bev: np.ndarray = None,
    bev_range: float = 60.0,
    top_k_sidebar: int = 15,
    cmap: str = "RdYlBu_r",
    title: str = "Lane-Token Activation Map",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Render BEV with lanes colored by attention + sidebar bar chart.

    Args:
        lane_centerlines_bev: (M, P, 2) lane centerline points
        lane_attention: (M,) cumulative attention per lane
        lane_mask: (M,) bool valid lanes
        lane_labels: list of lane description strings
        target_history_bev: (11, 2) target history
        target_future_bev: (future_len, 2) GT future
        pred_trajectories_bev: (K, future_len, 2) predictions
        bev_range: spatial extent
        top_k_sidebar: number of lanes in sidebar
        cmap: colormap name
        title: figure title
        figsize: figure size

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)

    ax_bev = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # Normalize attention for coloring
    valid_attn = lane_attention[lane_mask.astype(bool)]
    if len(valid_attn) > 0 and valid_attn.max() > 0:
        norm = mcolors.Normalize(vmin=0, vmax=valid_attn.max())
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)

    colormap = cm.get_cmap(cmap)

    # Draw lanes with attention-based color and width
    for i in range(len(lane_centerlines_bev)):
        if not lane_mask[i]:
            continue

        pts = lane_centerlines_bev[i]
        attn = lane_attention[i]

        color = colormap(norm(attn))
        width = float(1.0 + 4.0 * norm(attn))  # 1-5 linewidth

        ax_bev.plot(pts[:, 0], pts[:, 1], color=color, linewidth=width, alpha=0.9, zorder=3)

        # Arrow at midpoint showing direction
        mid = len(pts) // 2
        if mid < len(pts) - 1:
            dx = pts[mid + 1, 0] - pts[mid, 0]
            dy = pts[mid + 1, 1] - pts[mid, 1]
            ax_bev.annotate(
                "", xy=(pts[mid, 0] + dx * 0.3, pts[mid, 1] + dy * 0.3),
                xytext=(pts[mid, 0], pts[mid, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=4,
            )

    # Draw trajectories
    if target_history_bev is not None:
        ax_bev.plot(
            target_history_bev[:, 0], target_history_bev[:, 1],
            "-o", color="#1E88E5", linewidth=2, markersize=3,
            label="History", zorder=6,
        )

    if target_future_bev is not None:
        ax_bev.plot(
            target_future_bev[:, 0], target_future_bev[:, 1],
            "--", color="#2E7D32", linewidth=2,
            label="Ground Truth", zorder=6,
        )

    if pred_trajectories_bev is not None:
        for k in range(min(6, pred_trajectories_bev.shape[0])):
            alpha = 0.8 if k == 0 else 0.4
            ax_bev.plot(
                pred_trajectories_bev[k, :, 0], pred_trajectories_bev[k, :, 1],
                "-", color="#E53935", linewidth=2 if k == 0 else 1, alpha=alpha,
                label="Prediction" if k == 0 else None, zorder=7,
            )

    ax_bev.plot(0, 0, "^", color="#0D47A1", markersize=12, zorder=8)
    ax_bev.set_xlim(-bev_range, bev_range)
    ax_bev.set_ylim(-bev_range, bev_range)
    ax_bev.set_xlabel("Lateral (m)")
    ax_bev.set_ylabel("Forward (m)")
    ax_bev.set_aspect("equal")
    ax_bev.legend(loc="upper left", fontsize=8)
    ax_bev.set_title(title)

    # Colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_bev, label="Attention Weight", shrink=0.5, pad=0.02)

    # Sidebar: top-K lanes bar chart
    valid_indices = np.where(lane_mask.astype(bool))[0]
    valid_attns = lane_attention[valid_indices]
    order = np.argsort(valid_attns)[::-1][:top_k_sidebar]

    bar_indices = valid_indices[order]
    bar_values = valid_attns[order]

    if lane_labels:
        bar_labels = [lane_labels[i] if i < len(lane_labels) else f"Lane {i}" for i in bar_indices]
    else:
        bar_labels = [f"Lane {i}" for i in bar_indices]

    bar_colors = [colormap(norm(v)) for v in bar_values]

    y_pos = np.arange(len(bar_values))
    ax_bar.barh(y_pos, bar_values, color=bar_colors, alpha=0.9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Cumulative Attention")
    ax_bar.set_title("Lane Ranking")

    return fig
