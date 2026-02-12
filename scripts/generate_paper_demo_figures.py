#!/usr/bin/env python3
"""Generate publication-quality demonstration figures for the paper.

Produces:
  Figure 1: BEV spatial attention overlay (main contribution figure)
  Figure 2: VRU attention deficit comparison (safety blind spot)
  Figure 3: Layer-wise attention entropy evolution (architecture insight)

All figures use synthetic attention weights to demonstrate the visualization
pipeline before training completes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D

from visualization.spatial_utils import gaussian_splat_2d, paint_polyline_2d

# Publication styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = '/tmp/paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Drawing utilities
# ============================================================================

def draw_vehicle(ax, pos, heading, length=4.5, width=2.0,
                 color='gray', edgecolor='black', alpha=0.85, zorder=10):
    """Draw an oriented vehicle rectangle on ax."""
    rect = patches.FancyBboxPatch(
        (-length/2, -width/2), length, width,
        boxstyle="round,pad=0.1",
        linewidth=1.2, edgecolor=edgecolor, facecolor=color, alpha=alpha,
        zorder=zorder,
    )
    t = transforms.Affine2D().rotate(heading).translate(pos[0], pos[1]) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


def draw_pedestrian(ax, pos, color='orange', size=80, zorder=10):
    """Draw a pedestrian marker."""
    ax.scatter(pos[0], pos[1], s=size, c=color, marker='o',
              edgecolors='black', linewidths=1.0, zorder=zorder)


def draw_cyclist(ax, pos, color='green', size=80, zorder=10):
    """Draw a cyclist marker."""
    ax.scatter(pos[0], pos[1], s=size, c=color, marker='D',
              edgecolors='black', linewidths=1.0, zorder=zorder)


def draw_traffic_light(ax, pos, state='red', size=3.0, zorder=8):
    """Draw a traffic light indicator."""
    colors = {'red': '#FF3333', 'yellow': '#FFCC00', 'green': '#33CC33'}
    c = colors.get(state, '#888888')
    rect = patches.Rectangle(
        (pos[0] - size/2, pos[1] - size/2), size, size,
        facecolor=c, edgecolor='black', linewidth=1.0,
        alpha=0.9, zorder=zorder,
    )
    ax.add_patch(rect)


def setup_bev_axes(ax, radius, xlabel='X (m)', ylabel='Y (m)'):
    """Configure axes for BEV display."""
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15, linewidth=0.5)


def make_heatmap(agents, agent_attn, lanes, lane_attn,
                 resolution=0.5, radius=35, agent_sigma=3.0, lane_width=2.5):
    """Build a spatial attention heatmap from agent and lane weights."""
    H = W = int(2 * radius / resolution)
    heatmap = np.zeros((H, W))

    for i, agent in enumerate(agents):
        gaussian_splat_2d(heatmap, agent['pos'], agent_attn[i],
                         sigma=agent_sigma, resolution=resolution, radius=radius)

    for i, lane in enumerate(lanes):
        paint_polyline_2d(heatmap, lane['centerline'], lane_attn[i],
                         width=lane_width, resolution=resolution, radius=radius)

    # Normalize
    if heatmap.max() > 0:
        p95 = np.percentile(heatmap[heatmap > 0], 95)
        heatmap = np.clip(heatmap / p95, 0, 1)
    return heatmap


# ============================================================================
# Figure 1: Main contribution — BEV spatial attention overlay
# ============================================================================

def generate_figure1():
    """Three-panel figure: (a) Scene  (b) Attention heatmap  (c) Overlay."""

    radius = 35

    # Scene definition: unprotected left turn at intersection
    agents = [
        {'pos': np.array([0.0, -20.0]),  'heading': np.pi/2,  'color': '#2196F3', 'label': 'Ego',      'type': 'vehicle'},
        {'pos': np.array([0.0, 15.0]),   'heading': -np.pi/2, 'color': '#F44336', 'label': 'Oncoming',  'type': 'vehicle'},
        {'pos': np.array([-15.0, 2.0]),  'heading': 0.0,      'color': '#9E9E9E', 'label': 'Left',      'type': 'vehicle'},
        {'pos': np.array([20.0, 5.0]),   'heading': np.pi,    'color': '#9E9E9E', 'label': 'Right',     'type': 'vehicle'},
    ]

    lanes = [
        {'centerline': np.array([[-3.0, -35.0], [-3.0, 35.0]])},   # N-S ego lane
        {'centerline': np.array([[3.0, 35.0], [3.0, -35.0]])},     # S-N oncoming
        {'centerline': np.array([[-35.0, -3.0], [35.0, -3.0]])},   # W-E
        {'centerline': np.array([[35.0, 3.0], [-35.0, 3.0]])},     # E-W (target)
    ]

    # Left-turn curve (ego's intended path)
    t_curve = np.linspace(0, np.pi/2, 20)
    curve_r = 12
    turn_x = -3 + curve_r * np.sin(t_curve)
    turn_y = -8 + curve_r * (1 - np.cos(t_curve))
    turn_lane = np.column_stack([turn_x, turn_y])

    # Attention: ego turning left
    agent_attn = np.array([0.25, 0.82, 0.40, 0.12])
    lane_attn = np.array([0.55, 0.18, 0.70, 0.08])

    heatmap = make_heatmap(agents, agent_attn, lanes, lane_attn, radius=radius)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5))

    # (a) BEV Scene
    ax = axes[0]
    setup_bev_axes(ax, radius)
    ax.set_title('(a) BEV Traffic Scene', weight='bold')

    for lane in lanes:
        cl = lane['centerline']
        ax.plot(cl[:, 0], cl[:, 1], color='#BDBDBD', linewidth=6, alpha=0.4, solid_capstyle='round')
        ax.plot(cl[:, 0], cl[:, 1], color='#757575', linewidth=1, alpha=0.6, linestyle='--')

    ax.plot(turn_x, turn_y, color='#42A5F5', linewidth=2, linestyle=':', alpha=0.7, label='Intended path')

    draw_traffic_light(ax, np.array([-8.0, 8.0]), state='green')
    draw_traffic_light(ax, np.array([8.0, -8.0]), state='red')

    for agent in agents:
        draw_vehicle(ax, agent['pos'], agent['heading'], color=agent['color'])
        offset_y = 4 if agent['label'] != 'Left' else -4
        ax.text(agent['pos'][0], agent['pos'][1] + offset_y, agent['label'],
               ha='center', va='center', fontsize=9, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none'))

    ax.legend(loc='lower right', framealpha=0.9)

    # (b) Spatial Attention Heatmap
    ax = axes[1]
    setup_bev_axes(ax, radius)
    ax.set_title('(b) Spatial Attention Heatmap', weight='bold')

    im = ax.imshow(heatmap, extent=(-radius, radius, -radius, radius),
                   origin='lower', cmap='magma', alpha=1.0, interpolation='bilinear')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention weight (normalized)')

    # (c) Overlay
    ax = axes[2]
    setup_bev_axes(ax, radius)
    ax.set_title('(c) Scene with Attention Overlay', weight='bold')

    # Background lanes
    for lane in lanes:
        cl = lane['centerline']
        ax.plot(cl[:, 0], cl[:, 1], color='#BDBDBD', linewidth=6, alpha=0.3, solid_capstyle='round')

    # Heatmap
    ax.imshow(heatmap, extent=(-radius, radius, -radius, radius),
             origin='lower', cmap='magma', alpha=0.55, interpolation='bilinear')

    # Vehicles on top
    for agent in agents:
        draw_vehicle(ax, agent['pos'], agent['heading'], color=agent['color'])

    # Predicted trajectory
    ax.plot(turn_x, turn_y, color='cyan', linewidth=2.5, linestyle='-', alpha=0.9, zorder=15)
    ax.scatter(turn_x[-1], turn_y[-1], s=60, c='cyan', marker='>', zorder=16)

    # Highlight top-2 attended agents
    sorted_idx = np.argsort(agent_attn)[::-1]
    for rank, idx in enumerate(sorted_idx[:2]):
        agent = agents[idx]
        ax.scatter(agent['pos'][0], agent['pos'][1], s=250,
                  facecolors='none', edgecolors='cyan', linewidths=2.5, zorder=20)
        dx = 5 if agent['pos'][0] >= 0 else -5
        dy = 5
        ax.annotate(f"{agent['label']}: {agent_attn[idx]:.2f}",
                   xy=agent['pos'], xytext=(agent['pos'][0]+dx, agent['pos'][1]+dy),
                   fontsize=9, color='white', weight='bold',
                   arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75, edgecolor='cyan'),
                   zorder=25)

    # Lane attention annotation
    ax.annotate(f"Target lane: {lane_attn[2]:.2f}",
               xy=(15, -3), xytext=(20, -15),
               fontsize=9, color='white', weight='bold',
               arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75, edgecolor='#FF9800'),
               zorder=25)

    cbar2 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Attention weight')

    plt.tight_layout(w_pad=1.5)
    path = os.path.join(OUTPUT_DIR, 'fig1_spatial_attention_overlay.png')
    plt.savefig(path)
    plt.close()
    print(f"[Figure 1] Saved: {path}")


# ============================================================================
# Figure 2: VRU attention deficit — safety blind spot
# ============================================================================

def generate_figure2():
    """Two-panel comparison: vehicle vs pedestrian at same distance."""

    radius = 30

    # Common lane
    lane = {'centerline': np.array([[0.0, -30.0], [0.0, 30.0]])}

    # Scenario A: Vehicle at 10 m ahead
    agents_a = [
        {'pos': np.array([0.0, -5.0]), 'heading': np.pi/2, 'color': '#2196F3', 'label': 'Ego', 'type': 'vehicle'},
        {'pos': np.array([0.0, 10.0]), 'heading': np.pi/2, 'color': '#F44336', 'label': 'Lead vehicle', 'type': 'vehicle'},
    ]
    agent_attn_a = np.array([0.20, 0.78])
    lane_attn_a = np.array([0.50])

    # Scenario B: Pedestrian at 10 m ahead
    agents_b_vehicle = [
        {'pos': np.array([0.0, -5.0]), 'heading': np.pi/2, 'color': '#2196F3', 'label': 'Ego', 'type': 'vehicle'},
    ]
    agents_b_ped = [
        {'pos': np.array([2.0, 10.0]), 'heading': 0.0, 'color': '#FF9800', 'label': 'Pedestrian', 'type': 'pedestrian'},
    ]
    agent_attn_b_ego = 0.25
    agent_attn_b_ped = 0.31  # Lower than vehicle at same distance!
    lane_attn_b = np.array([0.55])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- Panel (a): Vehicle at 10 m ---
    ax = axes[0]
    setup_bev_axes(ax, radius)
    ax.set_title('(a) Lead vehicle at 10 m: Attention = 0.78', weight='bold')

    # Lane
    cl = lane['centerline']
    ax.plot(cl[:, 0], cl[:, 1], color='#BDBDBD', linewidth=8, alpha=0.3, solid_capstyle='round')

    # Heatmap
    H = W = int(2 * radius / 0.5)
    hm = np.zeros((H, W))
    gaussian_splat_2d(hm, agents_a[0]['pos'], agent_attn_a[0], sigma=3.0, resolution=0.5, radius=radius)
    gaussian_splat_2d(hm, agents_a[1]['pos'], agent_attn_a[1], sigma=3.0, resolution=0.5, radius=radius)
    paint_polyline_2d(hm, cl, lane_attn_a[0], width=2.5, resolution=0.5, radius=radius)
    if hm.max() > 0:
        hm = np.clip(hm / np.percentile(hm[hm > 0], 95), 0, 1)

    ax.imshow(hm, extent=(-radius, radius, -radius, radius),
             origin='lower', cmap='magma', alpha=0.55, interpolation='bilinear')

    for a in agents_a:
        draw_vehicle(ax, a['pos'], a['heading'], color=a['color'])

    ax.scatter(agents_a[1]['pos'][0], agents_a[1]['pos'][1], s=300,
              facecolors='none', edgecolors='cyan', linewidths=3, zorder=20)
    ax.annotate('Attn: 0.78', xy=agents_a[1]['pos'],
               xytext=(8, 15), fontsize=11, color='white', weight='bold',
               arrowprops=dict(arrowstyle='->', color='cyan', lw=2),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor='cyan'),
               zorder=25)

    # Distance annotation
    ax.annotate('', xy=(3, 10), xytext=(3, -5),
               arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
    ax.text(5, 2.5, '10 m', fontsize=10, color='white', weight='bold',
           bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    # --- Panel (b): Pedestrian at 10 m ---
    ax = axes[1]
    setup_bev_axes(ax, radius)
    ax.set_title('(b) Pedestrian at 10 m: Attention = 0.31', weight='bold')

    # Lane
    ax.plot(cl[:, 0], cl[:, 1], color='#BDBDBD', linewidth=8, alpha=0.3, solid_capstyle='round')

    # Heatmap
    hm2 = np.zeros((H, W))
    gaussian_splat_2d(hm2, agents_b_vehicle[0]['pos'], agent_attn_b_ego, sigma=3.0, resolution=0.5, radius=radius)
    gaussian_splat_2d(hm2, agents_b_ped[0]['pos'], agent_attn_b_ped, sigma=2.0, resolution=0.5, radius=radius)
    paint_polyline_2d(hm2, cl, lane_attn_b[0], width=2.5, resolution=0.5, radius=radius)
    if hm2.max() > 0:
        hm2 = np.clip(hm2 / np.percentile(hm2[hm2 > 0], 95), 0, 1)

    ax.imshow(hm2, extent=(-radius, radius, -radius, radius),
             origin='lower', cmap='magma', alpha=0.55, interpolation='bilinear')

    draw_vehicle(ax, agents_b_vehicle[0]['pos'], agents_b_vehicle[0]['heading'],
                color=agents_b_vehicle[0]['color'])
    draw_pedestrian(ax, agents_b_ped[0]['pos'], color='#FF9800', size=120)

    ax.scatter(agents_b_ped[0]['pos'][0], agents_b_ped[0]['pos'][1], s=300,
              facecolors='none', edgecolors='#FF9800', linewidths=3, zorder=20)
    ax.annotate('Attn: 0.31', xy=agents_b_ped[0]['pos'],
               xytext=(10, 15), fontsize=11, color='white', weight='bold',
               arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor='#FF9800'),
               zorder=25)

    # Distance annotation
    ax.annotate('', xy=(5, 10), xytext=(5, -5),
               arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
    ax.text(7, 2.5, '10 m', fontsize=10, color='white', weight='bold',
           bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    # Deficit annotation
    ax.text(0, -22,
           'Attention deficit: 0.78 - 0.31 = 0.47  (60% lower)',
           ha='center', fontsize=12, color='#FF5722', weight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.85, edgecolor='#FF5722'),
           transform=ax.transData, zorder=30)

    plt.tight_layout(w_pad=2.0)
    path = os.path.join(OUTPUT_DIR, 'fig2_vru_attention_deficit.png')
    plt.savefig(path)
    plt.close()
    print(f"[Figure 2] Saved: {path}")


# ============================================================================
# Figure 3: Attention entropy evolution across layers
# ============================================================================

def generate_figure3():
    """Show how attention entropy decreases across encoder/decoder layers."""

    # Simulated entropy values (will be replaced with real data post-training)
    encoder_layers = [0, 1, 2, 3]
    decoder_layers = [0, 1, 2, 3]

    # Simulated: entropy decreases with depth (broad → focused)
    np.random.seed(42)
    encoder_entropy_mean = np.array([5.2, 4.5, 3.8, 3.1])
    encoder_entropy_std = np.array([0.4, 0.35, 0.3, 0.25])
    decoder_entropy_mean = np.array([4.8, 3.9, 3.2, 2.6])
    decoder_entropy_std = np.array([0.5, 0.4, 0.35, 0.3])

    # Simulated Gini coefficient (sparsity increases with depth)
    encoder_gini_mean = np.array([0.22, 0.35, 0.48, 0.58])
    encoder_gini_std = np.array([0.05, 0.04, 0.04, 0.03])
    decoder_gini_mean = np.array([0.28, 0.42, 0.55, 0.65])
    decoder_gini_std = np.array([0.06, 0.05, 0.04, 0.03])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Entropy
    ax = axes[0]
    ax.errorbar(encoder_layers, encoder_entropy_mean, yerr=encoder_entropy_std,
               fmt='o-', color='#1976D2', linewidth=2, capsize=4, markersize=7,
               label='Scene encoder')
    ax.errorbar(decoder_layers, decoder_entropy_mean, yerr=decoder_entropy_std,
               fmt='s--', color='#E64A19', linewidth=2, capsize=4, markersize=7,
               label='Motion decoder')
    ax.axhline(y=np.log2(96), color='gray', linestyle=':', alpha=0.5, label='Max entropy (uniform)')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Attention entropy (bits)')
    ax.set_title('(a) Attention Entropy Across Layers', weight='bold')
    ax.legend(framealpha=0.9)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(1.5, 7.5)
    ax.grid(True, alpha=0.2)

    # (b) Gini coefficient (sparsity)
    ax = axes[1]
    ax.errorbar(encoder_layers, encoder_gini_mean, yerr=encoder_gini_std,
               fmt='o-', color='#1976D2', linewidth=2, capsize=4, markersize=7,
               label='Scene encoder')
    ax.errorbar(decoder_layers, decoder_gini_mean, yerr=decoder_gini_std,
               fmt='s--', color='#E64A19', linewidth=2, capsize=4, markersize=7,
               label='Motion decoder')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Gini coefficient')
    ax.set_title('(b) Attention Sparsity Across Layers', weight='bold')
    ax.legend(framealpha=0.9)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 0.85)
    ax.grid(True, alpha=0.2)

    # Annotation
    ax.annotate('More focused', xy=(3, 0.65), fontsize=9, color='#E64A19',
               ha='center', style='italic')
    axes[0].annotate('More focused', xy=(3, 2.6), fontsize=9, color='#E64A19',
                    ha='center', style='italic')

    plt.tight_layout(w_pad=2.0)
    path = os.path.join(OUTPUT_DIR, 'fig3_attention_entropy_evolution.png')
    plt.savefig(path)
    plt.close()
    print(f"[Figure 3] Saved: {path}")


# ============================================================================
# Figure 4: Counterfactual — agent removal
# ============================================================================

def generate_figure4():
    """Two-panel: original scene vs counterfactual (oncoming vehicle removed)."""

    radius = 35

    lanes = [
        {'centerline': np.array([[-3.0, -35.0], [-3.0, 35.0]])},
        {'centerline': np.array([[3.0, 35.0], [3.0, -35.0]])},
        {'centerline': np.array([[-35.0, -3.0], [35.0, -3.0]])},
        {'centerline': np.array([[35.0, 3.0], [-35.0, 3.0]])},
    ]

    # Left-turn curve
    t_curve = np.linspace(0, np.pi/2, 20)
    turn_x = -3 + 12 * np.sin(t_curve)
    turn_y = -8 + 12 * (1 - np.cos(t_curve))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for panel_idx, (ax, title, has_oncoming) in enumerate(zip(
        axes,
        ['(a) Original: Oncoming vehicle present',
         '(b) Counterfactual: Oncoming vehicle removed'],
        [True, False],
    )):
        setup_bev_axes(ax, radius)
        ax.set_title(title, weight='bold')

        # Agents
        agents = [
            {'pos': np.array([0.0, -20.0]), 'heading': np.pi/2, 'color': '#2196F3'},
        ]
        agent_attn = [0.25]

        if has_oncoming:
            agents.append({'pos': np.array([0.0, 15.0]), 'heading': -np.pi/2, 'color': '#F44336'})
            agent_attn.append(0.82)
            lane_attn = np.array([0.55, 0.18, 0.45, 0.08])
        else:
            lane_attn = np.array([0.30, 0.10, 0.90, 0.05])  # Attention shifts to target lane!

        agent_attn = np.array(agent_attn)

        # Heatmap
        hm = make_heatmap(
            [{'pos': a['pos']} for a in agents], agent_attn,
            lanes, lane_attn, radius=radius,
        )

        # Draw
        for lane in lanes:
            cl = lane['centerline']
            ax.plot(cl[:, 0], cl[:, 1], color='#BDBDBD', linewidth=6, alpha=0.3, solid_capstyle='round')

        ax.imshow(hm, extent=(-radius, radius, -radius, radius),
                 origin='lower', cmap='magma', alpha=0.55, interpolation='bilinear')

        for a in agents:
            draw_vehicle(ax, a['pos'], a['heading'], color=a['color'])

        # Predicted trajectory
        if has_oncoming:
            # Wait — trajectory only goes a little
            short_t = t_curve[:5]
            ax.plot(-3 + 12*np.sin(short_t), -8 + 12*(1 - np.cos(short_t)),
                   color='cyan', linewidth=2.5, linestyle='-', alpha=0.6, zorder=15)
            ax.text(-3, -12, 'Prediction: WAIT', ha='center', fontsize=10, color='cyan', weight='bold',
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan'))
        else:
            # Go — full trajectory
            ax.plot(turn_x, turn_y, color='cyan', linewidth=2.5, linestyle='-', alpha=0.9, zorder=15)
            ax.scatter(turn_x[-1], turn_y[-1], s=60, c='cyan', marker='>', zorder=16)
            ax.text(turn_x[-1]+3, turn_y[-1], 'Prediction: GO', ha='left', fontsize=10,
                   color='cyan', weight='bold',
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan'))

        if has_oncoming:
            # Show high attention to oncoming
            ax.annotate('Attn: 0.82', xy=agents[1]['pos'],
                       xytext=(8, 22), fontsize=10, color='white', weight='bold',
                       arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75, edgecolor='cyan'),
                       zorder=25)
            ax.annotate('Target lane: 0.45', xy=(15, -3), xytext=(18, -15),
                       fontsize=9, color='white',
                       arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.2),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='#FF9800'),
                       zorder=25)
        else:
            # Show attention shifted to target lane
            ax.annotate('Target lane: 0.90', xy=(15, -3), xytext=(18, -15),
                       fontsize=10, color='white', weight='bold',
                       arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75, edgecolor='#FF9800'),
                       zorder=25)

            # "Removed" marker
            ax.scatter(0, 15, s=200, marker='x', c='red', linewidths=3, zorder=20)
            ax.text(0, 19, 'Removed', ha='center', fontsize=9, color='#F44336',
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor='#F44336'))

    # Bottom annotation
    fig.text(0.5, 0.01,
            'Counterfactual analysis: Removing the oncoming vehicle causes attention '
            'to shift from conflict checking (0.82) to target lane (0.45 → 0.90), '
            'and the predicted behavior changes from WAIT to GO.',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDE7', edgecolor='#FFC107', alpha=0.9))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUTPUT_DIR, 'fig4_counterfactual_agent_removal.png')
    plt.savefig(path)
    plt.close()
    print(f"[Figure 4] Saved: {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Generating publication-quality demo figures...\n")

    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("Note: These use synthetic attention weights for demonstration.")
    print("Real figures will be generated after training completes.")
