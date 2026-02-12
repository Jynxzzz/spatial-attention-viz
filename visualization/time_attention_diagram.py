"""Time-Attention Refinement Diagram visualization.

4-panel horizontal strip (one per decoder layer). Each panel shows:
- Top: sorted horizontal bars of attention weights (query -> agent tokens)
- Bottom: sorted horizontal bars (query -> map tokens)

Shows how attention focuses across iterative decoder refinement.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Colors
COLOR_AGENT_ATTN = "#1565C0"  # Blue for agent attention
COLOR_MAP_ATTN = "#43A047"    # Green for map attention
COLOR_HIGHLIGHT = "#E53935"   # Red for ego/important tokens


def render_time_attention_diagram(
    decoder_agent_attns: list,
    decoder_map_attns: list,
    mode_idx: int,
    agent_labels: list = None,
    map_labels: list = None,
    agent_mask: np.ndarray = None,
    map_mask: np.ndarray = None,
    head_aggregation: str = "mean",
    top_k: int = 10,
    title: str = "Decoder Attention Refinement",
    figsize: tuple = (20, 6),
) -> plt.Figure:
    """Render 4-panel strip showing attention evolution across decoder layers.

    Args:
        decoder_agent_attns: list of (nhead, K, A) per layer
        decoder_map_attns: list of (nhead, K, M) per layer
        mode_idx: which intention query to visualize
        agent_labels: list of agent description strings
        map_labels: list of lane description strings
        agent_mask: (A,) bool valid agents
        map_mask: (M,) bool valid lanes
        head_aggregation: "mean" or "max" over heads
        top_k: show top K attended tokens per panel
        title: figure title
        figsize: figure size

    Returns:
        matplotlib Figure
    """
    num_layers = len(decoder_agent_attns)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, num_layers, hspace=0.4, wspace=0.3)

    for layer_i in range(num_layers):
        # Agent attention for this layer and mode
        agent_attn = decoder_agent_attns[layer_i]  # (nhead, K, A)
        if hasattr(agent_attn, 'cpu'):
            agent_attn = agent_attn.cpu().numpy()
        agent_attn_mode = agent_attn[:, mode_idx, :]  # (nhead, A)

        # Aggregate heads
        if head_aggregation == "mean":
            agent_weights = agent_attn_mode.mean(axis=0)  # (A,)
        else:
            agent_weights = agent_attn_mode.max(axis=0)

        # Map attention
        map_attn = decoder_map_attns[layer_i]  # (nhead, K, M)
        if hasattr(map_attn, 'cpu'):
            map_attn = map_attn.cpu().numpy()
        map_attn_mode = map_attn[:, mode_idx, :]  # (nhead, M)

        if head_aggregation == "mean":
            map_weights = map_attn_mode.mean(axis=0)  # (M,)
        else:
            map_weights = map_attn_mode.max(axis=0)

        # Apply masks
        if agent_mask is not None:
            agent_weights = agent_weights * agent_mask.astype(np.float32)
        if map_mask is not None:
            map_weights = map_weights * map_mask.astype(np.float32)

        # Top-K agents
        agent_order = np.argsort(agent_weights)[::-1][:top_k]
        agent_vals = agent_weights[agent_order]
        if agent_labels:
            a_labels = [agent_labels[i] if i < len(agent_labels) else f"A{i}" for i in agent_order]
        else:
            a_labels = [f"Agent {i}" for i in agent_order]

        # Top-K map
        map_order = np.argsort(map_weights)[::-1][:top_k]
        map_vals = map_weights[map_order]
        if map_labels:
            m_labels = [map_labels[i] if i < len(map_labels) else f"M{i}" for i in map_order]
        else:
            m_labels = [f"Lane {i}" for i in map_order]

        # Agent attention panel (top row)
        ax_agent = fig.add_subplot(gs[0, layer_i])
        y_pos = np.arange(len(agent_vals))
        bars = ax_agent.barh(y_pos, agent_vals, color=COLOR_AGENT_ATTN, alpha=0.8)

        # Highlight ego (first agent typically)
        for bar_i, idx in enumerate(agent_order):
            if idx == 0:  # Ego is usually slot 0
                bars[bar_i].set_color(COLOR_HIGHLIGHT)

        ax_agent.set_yticks(y_pos)
        ax_agent.set_yticklabels(a_labels, fontsize=7)
        ax_agent.invert_yaxis()
        ax_agent.set_xlabel("Attention", fontsize=8)
        ax_agent.set_title(f"Layer {layer_i+1}\nAgent Attn", fontsize=9)
        ax_agent.set_xlim(0, max(agent_vals.max() * 1.2, 0.01))

        # Map attention panel (bottom row)
        ax_map = fig.add_subplot(gs[1, layer_i])
        y_pos = np.arange(len(map_vals))
        ax_map.barh(y_pos, map_vals, color=COLOR_MAP_ATTN, alpha=0.8)
        ax_map.set_yticks(y_pos)
        ax_map.set_yticklabels(m_labels, fontsize=7)
        ax_map.invert_yaxis()
        ax_map.set_xlabel("Attention", fontsize=8)
        ax_map.set_title(f"Map Attn", fontsize=9)
        ax_map.set_xlim(0, max(map_vals.max() * 1.2, 0.01))

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    return fig
