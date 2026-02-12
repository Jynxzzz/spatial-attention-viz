"""Custom colormaps and color utilities for attention visualization.

Provides:
- Traffic engineering standard colors
- Attention-specific colormaps
- Colormap utilities
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ─── Traffic Engineering Color Standards ─────────────────────────
COLOR_LANE_DEFAULT = "#B0BEC5"       # gray - non-ego lanes
COLOR_LANE_EGO = "#1565C0"           # blue - ego lane
COLOR_LANE_SUCCESSOR = "#43A047"     # green - successor lanes
COLOR_LANE_ADJACENT = "#FF8F00"      # amber - left/right lanes
COLOR_HISTORY = "#1E88E5"            # blue - history
COLOR_GT_FUTURE = "#2E7D32"          # green - ground truth
COLOR_PRED_TRAJECTORY = "#E53935"    # red - prediction
COLOR_NEIGHBOR = "#78909C"           # blue-gray - neighbor vehicles
COLOR_EGO_VEHICLE = "#0D47A1"        # dark blue - ego vehicle
COLOR_TRAFFIC_RED = "#D32F2F"
COLOR_TRAFFIC_YELLOW = "#FBC02D"
COLOR_TRAFFIC_GREEN = "#388E3C"

# Attention visualization colors
COLOR_AGENT_ATTN = "#1565C0"         # Blue for agent attention
COLOR_MAP_ATTN = "#43A047"           # Green for map attention
COLOR_HIGHLIGHT = "#E53935"          # Red for highlighting (ego, important)


TRAFFIC_STATE_COLORS = {
    0: COLOR_TRAFFIC_RED,
    1: COLOR_TRAFFIC_YELLOW,
    2: COLOR_TRAFFIC_GREEN,
}

TRAFFIC_STATE_NAMES = {
    0: "Red",
    1: "Yellow",
    2: "Green",
}


# ─── Attention Colormaps ──────────────────────────────────────────

def get_attention_colormap(name: str = "magma"):
    """Get colormap for attention visualization.

    Recommended colormaps:
        - "magma": purple-red-yellow, good for attention intensity
        - "viridis": purple-green-yellow, perceptually uniform
        - "YlOrRd": yellow-orange-red, good for lane activation
        - "RdYlBu_r": red-yellow-blue reversed, diverging colormap

    Args:
        name: colormap name

    Returns:
        matplotlib colormap
    """
    try:
        # Use new API if available (matplotlib >= 3.7)
        return plt.colormaps.get_cmap(name)
    except AttributeError:
        # Fallback to deprecated API
        return plt.cm.get_cmap(name)


def create_attention_colormap_discrete(n_levels: int = 5, base_cmap: str = "magma"):
    """Create discrete attention colormap with fixed levels.

    Useful for categorizing attention into levels (e.g., low/medium/high).

    Args:
        n_levels: number of discrete levels
        base_cmap: base colormap name

    Returns:
        matplotlib colormap
    """
    base = get_attention_colormap(base_cmap)
    colors = base(np.linspace(0, 1, n_levels))
    return mcolors.ListedColormap(colors)


def attention_to_color(
    attention_value: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "magma",
):
    """Convert single attention value to RGB color.

    Args:
        attention_value: attention weight
        vmin: minimum attention value
        vmax: maximum attention value
        cmap_name: colormap name

    Returns:
        (r, g, b, a) tuple
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = get_attention_colormap(cmap_name)
    return cmap(norm(attention_value))


def attention_to_alpha(
    attention_value: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
    alpha_range: tuple = (0.2, 1.0),
):
    """Convert attention value to alpha (transparency).

    Args:
        attention_value: attention weight
        vmin: minimum attention value
        vmax: maximum attention value
        alpha_range: (min_alpha, max_alpha) range

    Returns:
        alpha value in [0, 1]
    """
    norm = (attention_value - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)
    return alpha_range[0] + norm * (alpha_range[1] - alpha_range[0])


def attention_to_linewidth(
    attention_value: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
    width_range: tuple = (1.0, 5.0),
):
    """Convert attention value to line width.

    Useful for lane visualization where width indicates attention intensity.

    Args:
        attention_value: attention weight
        vmin: minimum attention value
        vmax: maximum attention value
        width_range: (min_width, max_width) range

    Returns:
        line width
    """
    norm = (attention_value - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)
    return width_range[0] + norm * (width_range[1] - width_range[0])


# ─── Colormap Visualization Helpers ───────────────────────────────

def plot_colormap_reference(save_path: str = None):
    """Generate a reference figure showing all recommended colormaps.

    Args:
        save_path: if provided, saves figure to disk
    """
    cmaps = ["magma", "viridis", "YlOrRd", "RdYlBu_r", "plasma", "inferno"]

    fig, axes = plt.subplots(len(cmaps), 1, figsize=(8, len(cmaps) * 0.5))

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    for ax, cmap_name in zip(axes, cmaps):
        cmap = get_attention_colormap(cmap_name)
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_yticks([])
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])
        ax.set_ylabel(cmap_name, rotation=0, ha="right", va="center")

    fig.suptitle("Attention Visualization Colormaps", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved colormap reference to {save_path}")

    return fig


def create_custom_diverging_attention_cmap():
    """Create custom diverging colormap for comparing attention.

    Blue (low) -> White (medium) -> Red (high)

    Returns:
        matplotlib colormap
    """
    colors = ["#0D47A1", "#42A5F5", "#FFFFFF", "#EF5350", "#B71C1C"]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list("attention_diverging", colors, N=n_bins)
    return cmap


def create_categorical_entity_colors(num_entities: int, base_hue: float = 0.6):
    """Create distinct colors for multiple entities (agents/lanes).

    Args:
        num_entities: number of entities to color
        base_hue: starting hue (0-1)

    Returns:
        list of (r, g, b, a) tuples
    """
    colors = []
    for i in range(num_entities):
        hue = (base_hue + i / num_entities) % 1.0
        saturation = 0.7 + 0.2 * (i % 3) / 3
        value = 0.8 + 0.15 * (i % 2)
        rgb = mcolors.hsv_to_rgb([hue, saturation, value])
        colors.append((*rgb, 1.0))
    return colors


# ─── Perceptually Uniform Colormap ────────────────────────────────

def create_attention_heatmap_colormap():
    """Create perceptually uniform colormap optimized for attention heatmaps.

    Dark blue (no attention) -> Cyan -> Yellow -> Red (high attention)

    Returns:
        matplotlib colormap
    """
    colors = [
        "#0C0C3E",  # Very dark blue (no attention)
        "#1565C0",  # Blue
        "#00ACC1",  # Cyan
        "#FDD835",  # Yellow
        "#FB8C00",  # Orange
        "#D32F2F",  # Red (high attention)
    ]
    return mcolors.LinearSegmentedColormap.from_list("attention_heat", colors, N=256)


# ─── Color Utilities ───────────────────────────────────────────────

def blend_colors(color1: tuple, color2: tuple, weight: float = 0.5):
    """Blend two colors with given weight.

    Args:
        color1: (r, g, b) or (r, g, b, a)
        color2: (r, g, b) or (r, g, b, a)
        weight: blending weight (0=color1, 1=color2)

    Returns:
        blended (r, g, b, a) color
    """
    c1 = np.array(mcolors.to_rgba(color1))
    c2 = np.array(mcolors.to_rgba(color2))
    return tuple(c1 * (1 - weight) + c2 * weight)


def darken_color(color: tuple, factor: float = 0.7):
    """Darken a color by reducing value in HSV space.

    Args:
        color: (r, g, b) or (r, g, b, a)
        factor: darkening factor (0=black, 1=original)

    Returns:
        darkened (r, g, b, a) color
    """
    rgb = mcolors.to_rgb(color)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[2] *= factor  # Reduce value
    rgb_dark = mcolors.hsv_to_rgb(hsv)
    return (*rgb_dark, 1.0)


def lighten_color(color: tuple, factor: float = 1.3):
    """Lighten a color by increasing value in HSV space.

    Args:
        color: (r, g, b) or (r, g, b, a)
        factor: lightening factor (1=original, >1=lighter)

    Returns:
        lightened (r, g, b, a) color
    """
    rgb = mcolors.to_rgb(color)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[2] = min(1.0, hsv[2] * factor)  # Increase value
    rgb_light = mcolors.hsv_to_rgb(hsv)
    return (*rgb_light, 1.0)


def get_color_palette(name: str = "default"):
    """Get predefined color palette.

    Args:
        name: palette name ("default", "traffic", "attention")

    Returns:
        dict of color names to RGB tuples
    """
    if name == "default" or name == "traffic":
        return {
            "lane_default": COLOR_LANE_DEFAULT,
            "lane_ego": COLOR_LANE_EGO,
            "lane_successor": COLOR_LANE_SUCCESSOR,
            "lane_adjacent": COLOR_LANE_ADJACENT,
            "history": COLOR_HISTORY,
            "gt_future": COLOR_GT_FUTURE,
            "prediction": COLOR_PRED_TRAJECTORY,
            "ego_vehicle": COLOR_EGO_VEHICLE,
            "neighbor": COLOR_NEIGHBOR,
            "traffic_red": COLOR_TRAFFIC_RED,
            "traffic_yellow": COLOR_TRAFFIC_YELLOW,
            "traffic_green": COLOR_TRAFFIC_GREEN,
        }
    elif name == "attention":
        return {
            "agent_attn": COLOR_AGENT_ATTN,
            "map_attn": COLOR_MAP_ATTN,
            "highlight": COLOR_HIGHLIGHT,
        }
    else:
        raise ValueError(f"Unknown palette: {name}")
