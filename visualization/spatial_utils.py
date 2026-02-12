"""Spatial utilities for attention visualization on BEV scenes.

Provides functions for:
- Gaussian splatting: render point attention as 2D Gaussian blobs
- Polyline painting: render line attention along lane centerlines
- Coordinate transforms: BEV meters to grid pixels
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def coords_to_grid(coords, resolution=0.5, radius=60):
    """Convert BEV coordinates (meters) to grid indices (pixels).

    Args:
        coords: (2,) or (N, 2) array of (x, y) in meters
        resolution: meters per pixel
        radius: BEV radius in meters

    Returns:
        Grid coordinates (row, col) or (N, 2) array
        Note: row=0 is top, increasing downward (standard image convention)
              But we use origin='lower' in imshow, so row=0 is bottom
    """
    coords = np.atleast_2d(coords)

    # BEV: x is forward, y is left
    # Grid: col is x, row is y
    # Origin is at center of grid

    H = W = int(2 * radius / resolution)

    # Transform: BEV [-radius, radius] -> grid [0, H or W]
    # x -> col
    # y -> row (with origin='lower', y increases upward)

    cols = ((coords[:, 0] + radius) / resolution).astype(np.int32)
    rows = ((coords[:, 1] + radius) / resolution).astype(np.int32)

    # Clip to grid bounds
    cols = np.clip(cols, 0, W - 1)
    rows = np.clip(rows, 0, H - 1)

    result = np.stack([rows, cols], axis=-1)

    if result.shape[0] == 1:
        return result[0]
    return result


def grid_to_coords(grid_coords, resolution=0.5, radius=60):
    """Convert grid indices (pixels) to BEV coordinates (meters).

    Inverse of coords_to_grid.

    Args:
        grid_coords: (2,) or (N, 2) array of (row, col)
        resolution: meters per pixel
        radius: BEV radius in meters

    Returns:
        BEV coordinates (x, y) in meters
    """
    grid_coords = np.atleast_2d(grid_coords)

    rows = grid_coords[:, 0]
    cols = grid_coords[:, 1]

    x = cols * resolution - radius
    y = rows * resolution - radius

    result = np.stack([x, y], axis=-1).astype(np.float32)

    if result.shape[0] == 1:
        return result[0]
    return result


def gaussian_splat_2d(heatmap, center, weight, sigma=3.0, resolution=0.5, radius=60):
    """Splat attention weight as 2D Gaussian at position.

    Modifies heatmap in-place by adding a Gaussian blob centered at `center`
    with peak value `weight`.

    Args:
        heatmap: (H, W) array to modify in-place
        center: (2,) array, (x, y) in BEV meters
        weight: scalar attention weight
        sigma: Gaussian standard deviation in meters
        resolution: meters per pixel
        radius: BEV radius in meters
    """
    if weight <= 0:
        return

    H, W = heatmap.shape
    cy, cx = coords_to_grid(center, resolution, radius)

    # Check if center is within bounds
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
        return

    # Determine patch size (3-sigma rule)
    patch_radius_m = 3 * sigma
    patch_radius_px = int(np.ceil(patch_radius_m / resolution))

    # Compute patch bounds
    y_min = max(0, cy - patch_radius_px)
    y_max = min(H, cy + patch_radius_px + 1)
    x_min = max(0, cx - patch_radius_px)
    x_max = min(W, cx + patch_radius_px + 1)

    if y_max <= y_min or x_max <= x_min:
        return

    # Create meshgrid for patch
    y_coords = np.arange(y_min, y_max)
    x_coords = np.arange(x_min, x_max)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Compute squared distance from center
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    # Gaussian kernel: exp(-dist^2 / (2 * sigma_px^2))
    sigma_px = sigma / resolution
    gaussian = np.exp(-dist_sq / (2 * sigma_px ** 2))

    # Accumulate weighted Gaussian
    heatmap[y_min:y_max, x_min:x_max] += weight * gaussian


def paint_polyline_2d(heatmap, polyline, weight, width=2.0, resolution=0.5, radius=60):
    """Paint attention weight along polyline with given width.

    Uses Bresenham-style line drawing followed by dilation to achieve
    the desired width.

    Args:
        heatmap: (H, W) array to modify in-place
        polyline: (N, 2) array of points in BEV meters
        weight: scalar attention weight
        width: line width in meters
        resolution: meters per pixel
        radius: BEV radius in meters
    """
    if weight <= 0 or len(polyline) < 2:
        return

    H, W = heatmap.shape

    # Convert polyline to grid coordinates
    grid_points = coords_to_grid(polyline, resolution, radius)  # (N, 2)

    # Create a temporary mask for the polyline
    mask = np.zeros((H, W), dtype=np.float32)

    # Draw lines between consecutive points
    for i in range(len(grid_points) - 1):
        r0, c0 = grid_points[i]
        r1, c1 = grid_points[i + 1]

        # Use Bresenham's line algorithm
        _draw_line_bresenham(mask, r0, c0, r1, c1, value=1.0)

    # Dilate the mask to achieve desired width
    # Convert width to pixels
    width_px = width / resolution

    # Use Gaussian filter for smooth edges
    if width_px > 1.0:
        mask = gaussian_filter(mask, sigma=width_px / 2.0, mode='constant', cval=0.0)

    # Normalize mask to [0, 1] and accumulate
    if mask.max() > 0:
        mask = mask / mask.max()
        heatmap[:] += weight * mask


def _draw_line_bresenham(grid, r0, c0, r1, c1, value=1.0):
    """Draw a line on grid using Bresenham's algorithm.

    Args:
        grid: (H, W) array to modify
        r0, c0: start point (row, col)
        r1, c1: end point (row, col)
        value: value to set on the line
    """
    H, W = grid.shape

    # Bresenham's line algorithm
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)

    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    err = dr - dc

    r, c = r0, c0

    while True:
        # Set pixel if within bounds
        if 0 <= r < H and 0 <= c < W:
            grid[r, c] = value

        # Check if we reached the end
        if r == r1 and c == c1:
            break

        # Update position
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def normalize_heatmap(heatmap, percentile=95, epsilon=1e-8):
    """Normalize heatmap for visualization.

    Clips outliers at the specified percentile and scales to [0, 1].

    Args:
        heatmap: (H, W) array
        percentile: clip values above this percentile
        epsilon: small value to avoid division by zero

    Returns:
        Normalized heatmap (H, W) in [0, 1]
    """
    heatmap = heatmap.copy()

    # Clip outliers
    if heatmap.max() > 0:
        threshold = np.percentile(heatmap[heatmap > 0], percentile)
        heatmap = np.clip(heatmap, 0, threshold)

        # Normalize to [0, 1]
        heatmap = heatmap / (threshold + epsilon)

    return heatmap


def compute_bbox_corners(pos, heading, length=4.5, width=2.0):
    """Compute oriented bounding box corners for a vehicle.

    Args:
        pos: (2,) position in BEV
        heading: heading angle in radians
        length: vehicle length in meters
        width: vehicle width in meters

    Returns:
        corners: (4, 2) array of corner points
    """
    # Corners in local frame (centered at origin)
    corners_local = np.array([
        [length / 2, width / 2],
        [length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
    ], dtype=np.float32)

    # Rotate to heading
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    corners_bev = (corners_local @ R.T) + pos

    return corners_bev


def interpolate_polyline(polyline, spacing=1.0):
    """Densify a polyline by interpolating points at regular intervals.

    Useful for ensuring smooth polyline painting.

    Args:
        polyline: (N, 2) array of points
        spacing: target spacing in meters

    Returns:
        Interpolated polyline with uniform spacing
    """
    if len(polyline) < 2:
        return polyline

    # Compute cumulative arc length
    segments = polyline[1:] - polyline[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)

    total_length = cumulative_lengths[-1]
    if total_length < spacing:
        return polyline

    # Generate target arc lengths
    n_points = int(np.ceil(total_length / spacing)) + 1
    target_lengths = np.linspace(0, total_length, n_points)

    # Interpolate x and y separately
    x_interp = np.interp(target_lengths, cumulative_lengths, polyline[:, 0])
    y_interp = np.interp(target_lengths, cumulative_lengths, polyline[:, 1])

    return np.stack([x_interp, y_interp], axis=-1).astype(np.float32)
