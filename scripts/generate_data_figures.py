#!/usr/bin/env python3
"""
Generate publication-quality figures for the Scenario Dreamer Transformer
Visualizer paper. All data is from completed experiments -- no model
inference needed.

Figures produced:
  1. fig_entropy_evolution.pdf   -- layer-wise attention entropy
  2. fig_scene_type_comparison.pdf -- scene-type attention adaptation
  3. fig_distance_ablation.pdf   -- distance mask ablation
  4. fig_failure_diagnosis.pdf   -- success vs failure attention comparison

Usage:
    python scripts/generate_data_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Global style (publication / Times-like serif)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paper", "figures",
)
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, stem):
    """Save figure as both PDF (vector) and PNG (raster)."""
    pdf_path = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    png_path = os.path.join(FIGURES_DIR, f"{stem}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  -> {pdf_path}")
    print(f"  -> {png_path}")


# ===================================================================
# Figure 1 -- Layer-wise Attention Entropy Evolution
# ===================================================================
def figure_entropy_evolution():
    print("Figure 1: entropy evolution ...")

    layers = np.arange(4)
    entropy = np.array([5.64, 5.50, 5.36, 5.92])
    agent_pct = np.array([49.7, 55.1, 62.4, 36.4])
    map_pct = np.array([50.3, 44.9, 37.6, 63.6])
    max_theoretical = np.log2(96)  # 6.58 bits

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # -- Left panel: entropy bar chart --
    colors = ["#4878CF"] * 4
    colors[3] = "#E24A33"  # highlight L3 reversal
    bars = ax1.bar(layers, entropy, width=0.55, color=colors,
                   edgecolor="black", linewidth=0.5, zorder=3)
    ax1.axhline(max_theoretical, color="gray", linestyle="--", linewidth=0.9,
                label=f"Max entropy (log$_2$96 = {max_theoretical:.2f})")
    ax1.set_xlabel("Encoder Layer")
    ax1.set_ylabel("Entropy (bits)")
    ax1.set_xticks(layers)
    ax1.set_xticklabels([f"L{i}" for i in layers])
    ax1.set_ylim(4.8, 7.0)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax1.set_title("(a) Attention Entropy per Layer", fontsize=11)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.set_axisbelow(True)

    # Annotate bars with values
    for bar, val in zip(bars, entropy):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # -- Right panel: stacked agent/map percentage --
    width = 0.55
    b1 = ax2.bar(layers, agent_pct, width=width, label="Agent tokens",
                 color="#4878CF", edgecolor="black", linewidth=0.5, zorder=3)
    b2 = ax2.bar(layers, map_pct, width=width, bottom=agent_pct,
                 label="Map tokens", color="#6AB572", edgecolor="black",
                 linewidth=0.5, zorder=3)
    ax2.set_xlabel("Encoder Layer")
    ax2.set_ylabel("Attention Share (%)")
    ax2.set_xticks(layers)
    ax2.set_xticklabels([f"L{i}" for i in layers])
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax2.set_title("(b) Agent vs Map Attention Share", fontsize=11)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.set_axisbelow(True)

    # Annotate percentage in each segment
    for i in range(4):
        # Agent segment (lower)
        ax2.text(i, agent_pct[i] / 2, f"{agent_pct[i]:.1f}%",
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color="white")
        # Map segment (upper)
        ax2.text(i, agent_pct[i] + map_pct[i] / 2, f"{map_pct[i]:.1f}%",
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color="white")

    # -- Annotation on panel (b): "Map Context Aggregation" arrow to L3 bar --
    ax2.annotate(
        "Map Context\nAggregation",
        xy=(3, 100),  # arrow tip: top of L3 stacked bar
        xytext=(3, 112),  # text position directly above L3
        fontsize=8, fontweight="bold", ha="center", va="bottom",
        color="#2E7D32",
        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.15",
                        color="#2E7D32", lw=1.3),
    )
    ax2.set_ylim(0, 128)  # expand to make room for annotation

    # -- Annotation on panel (a): "Agent Focusing" bracket over L0-L2 --
    # Draw a bracket spanning L0 to L2 above the bar value annotations
    bracket_y = 6.25  # above the tallest bar annotation in L0-L2
    ax1.annotate("", xy=(0, bracket_y), xytext=(2, bracket_y),
                 arrowprops=dict(arrowstyle="-", color="black", lw=1.0))
    # Bracket end ticks
    tick_len = 0.06
    ax1.plot([0, 0], [bracket_y - tick_len, bracket_y], color="black", lw=1.0,
             clip_on=False)
    ax1.plot([2, 2], [bracket_y - tick_len, bracket_y], color="black", lw=1.0,
             clip_on=False)
    # Label centred above the bracket
    ax1.text(1.0, bracket_y + 0.04, "Agent Focusing", ha="center",
             va="bottom", fontsize=8, fontstyle="italic", color="#333333")

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig_entropy_evolution")


# ===================================================================
# Figure 2 -- Scene-Type Attention Adaptation
# ===================================================================
def figure_scene_type_comparison():
    print("Figure 2: scene-type comparison ...")

    # Data (from 200-scene analysis)
    scene_types = [
        "Dense traffic", "Sparse", "Highway-like",
        "Intersection", "With pedestrians", "With cyclists",
    ]
    N = np.array([90, 14, 52, 75, 105, 37])
    agent_pct = np.array([42.3, 18.4, 31.6, 42.3, 39.4, 39.2])
    map_pct = np.array([57.7, 81.6, 68.4, 57.7, 60.6, 60.8])
    entropy = np.array([6.11, 5.33, 5.71, 6.10, 5.99, 5.90])
    top5_dist = np.array([18.1, 18.3, 21.4, 17.0, 17.4, 17.2])

    # Sort by agent% descending
    order = np.argsort(-agent_pct)
    scene_types = [scene_types[i] for i in order]
    N = N[order]
    agent_pct = agent_pct[order]
    map_pct = map_pct[order]
    entropy = entropy[order]
    top5_dist = top5_dist[order]

    x = np.arange(len(scene_types))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))

    # -- Left panel: grouped bars agent% vs map% --
    bars_a = ax1.bar(x - width / 2, agent_pct, width, label="Agent %",
                     color="#4878CF", edgecolor="black", linewidth=0.5,
                     zorder=3)
    bars_m = ax1.bar(x + width / 2, map_pct, width, label="Map %",
                     color="#6AB572", edgecolor="black", linewidth=0.5,
                     zorder=3)
    ax1.set_ylabel("Attention Share (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scene_types, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax1.set_title("(a) Attention Share by Scene Type", fontsize=11)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.set_axisbelow(True)

    # Annotate bars
    for bar in bars_a:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=7)
    for bar in bars_m:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=7)

    # -- Right panel: entropy (bar) with bubble size ~ N, plus top-5 dist --
    # Use a dual-axis: bars for entropy, line+markers for top-5 dist
    color_ent = "#8C6BB1"
    color_dist = "#E24A33"

    bars_e = ax2.bar(x, entropy, width=0.45, color=color_ent, alpha=0.85,
                     edgecolor="black", linewidth=0.5, zorder=3,
                     label="Entropy (bits)")
    ax2.set_ylabel("Entropy (bits)", color=color_ent)
    ax2.tick_params(axis="y", labelcolor=color_ent)
    ax2.set_ylim(4.8, 6.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scene_types, rotation=30, ha="right", fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.set_axisbelow(True)

    # Annotate N (sample size) on entropy bars
    for i, bar in enumerate(bars_e):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.04,
                 f"N={N[i]}", ha="center", va="bottom", fontsize=7,
                 color=color_ent)

    ax2b = ax2.twinx()
    ax2b.plot(x, top5_dist, "o-", color=color_dist, linewidth=1.5,
              markersize=6, zorder=4, label="Top-5 Dist (m)")
    ax2b.set_ylabel("Top-5 Distance (m)", color=color_dist)
    ax2b.tick_params(axis="y", labelcolor=color_dist)
    ax2b.set_ylim(15.0, 23.0)

    # Combined legend
    lines_labels_1 = ax2.get_legend_handles_labels()
    lines_labels_2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines_labels_1[0] + lines_labels_2[0],
               lines_labels_1[1] + lines_labels_2[1],
               loc="upper right", framealpha=0.9, fontsize=8)

    ax2.set_title("(b) Entropy & Top-5 Distance", fontsize=11)

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig_scene_type_comparison")


# ===================================================================
# Figure 3 -- Distance Mask Ablation
# ===================================================================
def figure_distance_ablation():
    print("Figure 3: distance ablation ...")

    alphas = [0.00, 0.05, 0.10, 0.20]
    minADE = np.array([2.872, 3.007, 3.032, 3.026])
    delta_pct = np.array([0.0, 4.7, 5.6, 5.4])
    labels = [r"$\alpha$=0.00" + "\n(baseline)",
              r"$\alpha$=0.05", r"$\alpha$=0.10", r"$\alpha$=0.20"]

    colors = ["#4878CF", "#E24A33", "#E24A33", "#E24A33"]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    x = np.arange(len(alphas))
    bars = ax.bar(x, minADE, width=0.52, color=colors, edgecolor="black",
                  linewidth=0.5, zorder=3)
    ax.axhline(minADE[0], color="gray", linestyle="--", linewidth=0.9,
               label=f"Baseline ({minADE[0]:.3f} m)")
    ax.set_xlabel("Distance Mask Strength")
    ax.set_ylabel("minADE@6 (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(2.7, 3.2)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.set_title("Distance Mask Ablation: Masking Hurts Performance",
                 fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # Annotate delta% on top of each bar
    for i, (bar, dp) in enumerate(zip(bars, delta_pct)):
        label_str = f"{minADE[i]:.3f} m" if i == 0 else f"+{dp:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                label_str, ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_distance_ablation")


# ===================================================================
# Figure 4 -- Success vs Failure Attention Diagnosis
# ===================================================================
def figure_failure_diagnosis():
    print("Figure 4: failure diagnosis ...")

    # Data (from 1,115 targets, 150 scenes)
    # Attention metrics
    attn_labels = ["Entropy\n(bits)", "Agent\nattn (%)", "Self-\nattn",
                   "Max single-\ntoken attn"]
    success_attn = np.array([5.94, 48.8, 0.035, 0.039])
    failure_attn = np.array([5.72, 43.2, 0.049, 0.058])

    # Contextual metrics
    ctx_labels = ["Speed\n(m/s)", "Nearby agents\n(<15 m)"]
    success_ctx = np.array([0.2, 5.2])
    failure_ctx = np.array([7.2, 3.4])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # -- Left panel: attention metrics (4 subgroups) --
    # Because scales differ widely (entropy ~5-6, percentages ~40-50,
    # small floats ~0.03-0.06), we normalise to % of max in each metric
    # and show actual values as annotations.

    x = np.arange(len(attn_labels))
    width = 0.32

    # For display: use the raw values but split y-axis into grouped subplots
    # Simpler approach: side-by-side bars with dual y-axes
    # Even simpler: normalise success & failure relative to each other

    # We'll plot the raw values and use two y-axes: left for entropy &
    # agent%, right for small-scale attention weights.

    # Group 1 (left y-axis): entropy, agent%
    # Group 2 (right y-axis): self-attn, max-attn

    color_s = "#6AB572"
    color_f = "#E24A33"

    # Plot all four metrics side by side, with normalised bar heights
    # (divide each pair by the max of the pair) for visual comparison,
    # then annotate actual values.
    max_vals = np.maximum(success_attn, failure_attn)
    s_norm = success_attn / max_vals * 100
    f_norm = failure_attn / max_vals * 100

    bars_s = ax1.bar(x - width / 2, s_norm, width, color=color_s,
                     edgecolor="black", linewidth=0.5, zorder=3,
                     label="Success (Q1, ADE$\\leq$0.71 m)")
    bars_f = ax1.bar(x + width / 2, f_norm, width, color=color_f,
                     edgecolor="black", linewidth=0.5, zorder=3,
                     label="Failure (Q4, ADE$\\geq$3.32 m)")

    ax1.set_ylabel("Normalised Value (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(attn_labels, fontsize=8)
    ax1.set_ylim(0, 125)
    ax1.legend(loc="upper center", framealpha=0.9, fontsize=7.5,
               ncol=1, bbox_to_anchor=(0.5, 1.02))
    ax1.set_title("(a) Attention Metrics", fontsize=11, pad=18)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.set_axisbelow(True)

    # Annotate actual values above bars
    for i in range(len(attn_labels)):
        # Format: if value < 0.1, use 3 decimal places; else 1
        sv = success_attn[i]
        fv = failure_attn[i]
        s_str = f"{sv:.3f}" if sv < 1 else f"{sv:.1f}"
        f_str = f"{fv:.3f}" if fv < 1 else f"{fv:.1f}"
        ax1.text(x[i] - width / 2, s_norm[i] + 1.5, s_str,
                 ha="center", va="bottom", fontsize=7, color="#2E7D32")
        ax1.text(x[i] + width / 2, f_norm[i] + 1.5, f_str,
                 ha="center", va="bottom", fontsize=7, color="#C62828")

    # -- Right panel: contextual metrics --
    x2 = np.arange(len(ctx_labels))
    bars_s2 = ax2.bar(x2 - width / 2, success_ctx, width, color=color_s,
                      edgecolor="black", linewidth=0.5, zorder=3,
                      label="Success (Q1)")
    bars_f2 = ax2.bar(x2 + width / 2, failure_ctx, width, color=color_f,
                      edgecolor="black", linewidth=0.5, zorder=3,
                      label="Failure (Q4)")
    ax2.set_ylabel("Value")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ctx_labels, fontsize=9)
    ax2.set_ylim(0, 9.5)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax2.set_title("(b) Contextual Factors", fontsize=11)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.set_axisbelow(True)

    # Annotate values
    for i in range(len(ctx_labels)):
        sv = success_ctx[i]
        fv = failure_ctx[i]
        ax2.text(x2[i] - width / 2, sv + 0.15, f"{sv:.1f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color="#2E7D32")
        ax2.text(x2[i] + width / 2, fv + 0.15, f"{fv:.1f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color="#C62828")

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig_failure_diagnosis")


# ===================================================================
# Main
# ===================================================================
def main():
    print(f"Saving figures to: {FIGURES_DIR}\n")
    figure_entropy_evolution()
    figure_scene_type_comparison()
    figure_distance_ablation()
    figure_failure_diagnosis()
    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
