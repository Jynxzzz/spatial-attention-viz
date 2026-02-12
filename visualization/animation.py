"""Animated attention visualization GIF generator.

Creates animated GIFs showing:
1. Attention evolution across decoder layers (refinement animation)
2. Per-head attention comparison
3. Mode competition (multiple intentions narrowing down)
"""

import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from visualization.space_attention_bev import render_space_attention_bev


def create_layer_refinement_gif(
    decoder_agent_attns: list,
    decoder_map_attns: list,
    mode_idx: int,
    agent_positions_bev: np.ndarray,
    agent_mask: np.ndarray,
    lane_centerlines_bev: np.ndarray,
    map_mask: np.ndarray,
    target_history_bev: np.ndarray = None,
    target_future_bev: np.ndarray = None,
    pred_trajectories_bev: np.ndarray = None,
    save_path: str = "attention_refinement.gif",
    bev_range: float = 50.0,
    fps: int = 2,
    figsize: tuple = (10, 10),
) -> str:
    """Create animated GIF showing attention refinement across decoder layers.

    Each frame shows the Space-Attention BEV for one decoder layer,
    creating a smooth transition from diffuse to focused attention.

    Args:
        decoder_agent_attns: list of (nhead, K, A) per layer (numpy arrays)
        decoder_map_attns: list of (nhead, K, M) per layer
        mode_idx: which intention query
        agent_positions_bev: (A, 2) agent BEV positions
        agent_mask: (A,) bool
        lane_centerlines_bev: (M, P, 2) lane points
        map_mask: (M,) bool
        target_history_bev: (11, 2) optional
        target_future_bev: (future_len, 2) optional
        pred_trajectories_bev: (K, future_len, 2) optional
        save_path: output GIF path
        bev_range: spatial range
        fps: frames per second
        figsize: figure size

    Returns:
        save_path
    """
    num_layers = len(decoder_agent_attns)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    def update(frame_idx):
        ax.clear()

        # Get attention for this layer
        agent_attn = decoder_agent_attns[frame_idx][:, mode_idx, :].mean(0)
        agent_attn = agent_attn * agent_mask.astype(np.float32)

        map_attn = decoder_map_attns[frame_idx][:, mode_idx, :].mean(0)
        map_attn = map_attn * map_mask.astype(np.float32)

        render_space_attention_bev(
            agent_positions_bev=agent_positions_bev,
            agent_attention=agent_attn,
            agent_mask=agent_mask,
            lane_centerlines_bev=lane_centerlines_bev,
            map_attention=map_attn,
            map_mask=map_mask,
            target_history_bev=target_history_bev,
            target_future_bev=target_future_bev,
            pred_trajectories_bev=pred_trajectories_bev,
            all_lane_points=lane_centerlines_bev,
            bev_range=bev_range,
            title=f"Decoder Layer {frame_idx + 1}/{num_layers}",
            ax=ax,
        )

    anim = animation.FuncAnimation(fig, update, frames=num_layers, interval=1000 // fps)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved animation to {save_path}")

    return save_path


def create_head_comparison_gif(
    scene_attention: np.ndarray,
    target_agent_idx: int,
    agent_positions_bev: np.ndarray,
    agent_mask: np.ndarray,
    lane_centerlines_bev: np.ndarray,
    map_mask: np.ndarray,
    num_agents: int,
    save_path: str = "head_comparison.gif",
    bev_range: float = 50.0,
    fps: int = 1,
    figsize: tuple = (10, 10),
) -> str:
    """Create GIF cycling through attention heads for scene encoder.

    Shows how different heads focus on different spatial regions.

    Args:
        scene_attention: (nhead, A+M, A+M) scene encoder attention
        target_agent_idx: which agent's attention row to visualize
        agent_positions_bev: (A, 2)
        agent_mask: (A,)
        lane_centerlines_bev: (M, P, 2)
        map_mask: (M,)
        num_agents: A
        save_path: output path
        bev_range: spatial range
        fps: frames per second
        figsize: figure size

    Returns:
        save_path
    """
    nhead = scene_attention.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    def update(head_idx):
        ax.clear()
        attn_row = scene_attention[head_idx, target_agent_idx, :]  # (A+M,)
        agent_attn = attn_row[:num_agents]
        map_attn = attn_row[num_agents:]

        agent_attn = agent_attn * agent_mask.astype(np.float32)
        map_attn = map_attn * map_mask.astype(np.float32)

        render_space_attention_bev(
            agent_positions_bev=agent_positions_bev,
            agent_attention=agent_attn,
            agent_mask=agent_mask,
            lane_centerlines_bev=lane_centerlines_bev,
            map_attention=map_attn,
            map_mask=map_mask,
            all_lane_points=lane_centerlines_bev,
            bev_range=bev_range,
            title=f"Head {head_idx + 1}/{nhead}",
            ax=ax,
        )

    anim = animation.FuncAnimation(fig, update, frames=nhead, interval=1000 // fps)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)

    return save_path
