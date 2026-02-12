"""Composite figure generator for paper publication.

Creates 3-column paper figures combining all visualization types:
  Column 1: Space-Attention BEV Heatmap
  Column 2: Time-Attention Refinement Diagram
  Column 3: Lane-Token Activation Map

Also supports multi-row layouts for comparing different scenarios.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from visualization.space_attention_bev import render_space_attention_bev
from visualization.time_attention_diagram import render_time_attention_diagram
from visualization.lane_token_activation import render_lane_activation_map


def generate_composite_figure(
    extraction_results: list,
    scenario_names: list = None,
    figsize: tuple = (20, 7),
    save_path: str = None,
    dpi: int = 300,
) -> plt.Figure:
    """Generate a multi-panel composite figure for the paper.

    For a single scenario, creates a 1x3 layout:
        [Space-Attention BEV] | [Time-Attention Diagram] | [Lane Activation Map]

    For multiple scenarios, creates Nx3 layout.

    Args:
        extraction_results: list of dicts from extract_scene_attention()
        scenario_names: list of scenario name strings
        figsize: per-row figure size
        save_path: if provided, saves figure to disk
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure
    """
    n_scenarios = len(extraction_results)
    if scenario_names is None:
        scenario_names = [f"Scenario {i+1}" for i in range(n_scenarios)]

    fig_h = figsize[1] * n_scenarios
    fig = plt.figure(figsize=(figsize[0], fig_h))

    # 3 columns: BEV heatmap, time diagram, lane activation
    outer_gs = gridspec.GridSpec(n_scenarios, 3, figure=fig, wspace=0.3, hspace=0.4)

    for row, (result, name) in enumerate(zip(extraction_results, scenario_names)):
        attn_maps = result.get("attention_maps")
        bookkeeper = result["bookkeeper"]
        map_data = result["map_data"]
        agent_data = result["agent_data"]

        if attn_maps is None:
            continue

        # Get target 0's attention (ego typically)
        target_idx = 0

        # --- Column 1: Space-Attention BEV ---
        ax1 = fig.add_subplot(outer_gs[row, 0])

        # Extract agent positions at anchor (from polylines, last timestep pos)
        agent_pos = agent_data["agent_polylines"][:, -1, :2]  # (A, 2)

        # Scene encoder: agent self-attention for target agent (row = target slot 0)
        if attn_maps.scene_attentions:
            # Average across layers and heads, looking at target's row
            scene_attn = attn_maps.scene_attentions[-1][0]  # (nhead, A+M, A+M)
            agent_self_attn = scene_attn[:, 0, :bookkeeper.num_agents].mean(0).cpu().numpy()
            map_attn_from_scene = scene_attn[:, 0, bookkeeper.num_agents:].mean(0).cpu().numpy()
        else:
            agent_self_attn = np.zeros(bookkeeper.num_agents)
            map_attn_from_scene = np.zeros(bookkeeper.num_map)

        render_space_attention_bev(
            agent_positions_bev=agent_pos,
            agent_attention=agent_self_attn,
            agent_mask=agent_data["agent_mask"],
            lane_centerlines_bev=map_data["lane_centerlines_bev"],
            map_attention=map_attn_from_scene,
            map_mask=map_data["map_mask"],
            all_lane_points=map_data["lane_centerlines_bev"],
            title=f"{name}\nSpace-Attention BEV",
            ax=ax1,
            bev_range=50.0,
        )

        # --- Column 2: Time-Attention Diagram ---
        # This needs its own subplot area - we use the gridspec cell
        ax2_area = outer_gs[row, 1]
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, len(attn_maps.decoder_agent_attentions) if attn_maps.decoder_agent_attentions else 1,
            subplot_spec=ax2_area,
            hspace=0.5,
            wspace=0.3,
        )

        if attn_maps.decoder_agent_attentions:
            # Get the winning mode index from NMS
            nms_idx = attn_maps.nms_indices
            if nms_idx is not None and nms_idx.dim() >= 3:
                winning_intention = nms_idx[0, target_idx, 0].item()
            else:
                winning_intention = 0

            num_dec_layers = len(attn_maps.decoder_agent_attentions)
            for li in range(num_dec_layers):
                # Note: decoder attentions are per-target, stored as list[target][layer]
                # Get attention for target_idx
                if isinstance(attn_maps.decoder_agent_attentions[0], list):
                    agent_attn_layer = attn_maps.decoder_agent_attentions[target_idx][li][0].cpu().numpy()
                    map_attn_layer = attn_maps.decoder_map_attentions[target_idx][li][0].cpu().numpy()
                else:
                    agent_attn_layer = attn_maps.decoder_agent_attentions[li][0].cpu().numpy()
                    map_attn_layer = attn_maps.decoder_map_attentions[li][0].cpu().numpy()

                # Agent panel
                ax_a = fig.add_subplot(inner_gs[0, li])
                attn_vals = agent_attn_layer[:, winning_intention, :].mean(0)
                attn_vals = attn_vals * agent_data["agent_mask"].astype(np.float32)
                top_k = min(8, int(agent_data["agent_mask"].sum()))
                order = np.argsort(attn_vals)[::-1][:top_k]
                ax_a.barh(range(top_k), attn_vals[order], color="#1565C0", alpha=0.8)
                ax_a.set_yticks(range(top_k))
                ax_a.set_yticklabels([f"A{i}" for i in order], fontsize=6)
                ax_a.invert_yaxis()
                ax_a.set_title(f"L{li+1} Agent", fontsize=7)

                # Map panel
                ax_m = fig.add_subplot(inner_gs[1, li])
                map_vals = map_attn_layer[:, winning_intention, :].mean(0)
                map_vals = map_vals * map_data["map_mask"].astype(np.float32)
                top_k_m = min(8, int(map_data["map_mask"].sum()))
                order_m = np.argsort(map_vals)[::-1][:top_k_m]
                ax_m.barh(range(top_k_m), map_vals[order_m], color="#43A047", alpha=0.8)
                ax_m.set_yticks(range(top_k_m))
                ax_m.set_yticklabels([f"L{i}" for i in order_m], fontsize=6)
                ax_m.invert_yaxis()
                ax_m.set_title(f"L{li+1} Map", fontsize=7)

        # --- Column 3: Lane Activation Map ---
        ax3 = fig.add_subplot(outer_gs[row, 2])

        # Cumulative map attention across all decoder layers
        cum_lane_attn = np.zeros(bookkeeper.num_map, dtype=np.float32)
        if attn_maps.decoder_map_attentions:
            nms_idx = attn_maps.nms_indices
            winning_intention = nms_idx[0, target_idx, 0].item() if nms_idx is not None and nms_idx.dim() >= 3 else 0

            for li in range(len(attn_maps.decoder_map_attentions)):
                if isinstance(attn_maps.decoder_map_attentions[0], list):
                    map_attn_l = attn_maps.decoder_map_attentions[target_idx][li][0].cpu().numpy()
                else:
                    map_attn_l = attn_maps.decoder_map_attentions[li][0].cpu().numpy()
                cum_lane_attn += map_attn_l[:, winning_intention, :].mean(0)

        render_lane_activation_map(
            lane_centerlines_bev=map_data["lane_centerlines_bev"],
            lane_attention=cum_lane_attn,
            lane_mask=map_data["map_mask"],
            bev_range=50.0,
            title=f"{name}\nLane Activation",
            figsize=None,
        )
        # Note: lane_activation_map creates its own figure; for composite we'd
        # need to refactor to accept axes. For now we plot directly on ax3.
        _render_lane_activation_inplace(
            ax3, map_data["lane_centerlines_bev"], cum_lane_attn,
            map_data["map_mask"], bev_range=50.0, title=f"{name}\nLane Activation",
        )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved composite figure to {save_path}")

    return fig


def _render_lane_activation_inplace(
    ax, lane_centerlines_bev, lane_attention, lane_mask, bev_range=50.0, title="",
):
    """Simplified lane activation rendering on a given axes."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    valid_attn = lane_attention[lane_mask.astype(bool)]
    vmax = valid_attn.max() if len(valid_attn) > 0 and valid_attn.max() > 0 else 1
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    colormap = cm.get_cmap("RdYlBu_r")

    for i in range(len(lane_centerlines_bev)):
        if not lane_mask[i]:
            continue
        pts = lane_centerlines_bev[i]
        attn = lane_attention[i]
        color = colormap(norm(attn))
        width = 1.0 + 4.0 * norm(attn)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=width, alpha=0.9)

    ax.plot(0, 0, "^", color="#0D47A1", markersize=10, zorder=8)
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
