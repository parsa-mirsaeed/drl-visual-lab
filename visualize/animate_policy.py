"""Animated policy heatmap — shows how the agent's greedy policy evolves.

Each cell shows the best action arrow. Background color = max Q-value.
Produces: output/policy_heatmap.gif
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

from core.environment import GridWorld, ACTION_NAMES

OUTPUT_DIR = "output"
ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
CELL_COLORS = {
    "goal": "#238636",
    "trap": "#da3633",
    "start": "#1f6feb",
    "free": None,
}


def animate_policy(
    q_snapshots: list,
    policy_snapshots: list,
    snapshot_interval: int = 50,
    filename: str = "policy_heatmap.gif",
    fps: int = 3,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    env = GridWorld()
    G = env.GRID_SIZE

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_title("🗺️ Policy Heatmap — Greedy Actions",
                 color="#58a6ff", fontsize=14, pad=10)
    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(-0.5, G - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    episode_text = ax.text(
        0.5, -0.05, "",
        transform=ax.transAxes,
        ha="center", color="#8b949e", fontsize=11,
    )

    # Pre-build cell patches and text objects
    patches = {}
    arrows = {}
    for r in range(G):
        for c in range(G):
            rect = plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=1.5, edgecolor="#30363d", facecolor="#161b22"
            )
            ax.add_patch(rect)
            patches[(r, c)] = rect
            txt = ax.text(
                c, r, "",
                ha="center", va="center",
                fontsize=20, color="white",
            )
            arrows[(r, c)] = txt

    # Goal / trap / start labels
    ax.text(*env.GOAL[::-1], "G", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold", zorder=5)
    ax.text(*env.START[::-1], "S", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold", zorder=5)
    for t in env.TRAPS:
        ax.text(t[1], t[0], "☠", ha="center", va="center", fontsize=16, zorder=5)

    cmap = plt.cm.plasma
    n_frames = len(policy_snapshots)

    def update(frame_idx):
        policy = policy_snapshots[frame_idx]
        q_table = q_snapshots[frame_idx]  # (25, 4)
        max_q = q_table.max(axis=1)       # (25,)
        norm = Normalize(vmin=max_q.min(), vmax=max_q.max())

        for r in range(G):
            for c in range(G):
                idx = r * G + c
                t = (r, c)
                # Color by max Q-value
                if t == env.GOAL:
                    color = "#1a7f37"
                elif t in env.TRAPS:
                    color = "#6e1a18"
                elif t == env.START:
                    color = "#0c2d6b"
                else:
                    rgba = cmap(norm(max_q[idx]))
                    color = rgba
                patches[(r, c)].set_facecolor(color)

                # Arrow (skip special cells)
                if t not in env.TRAPS and t != env.GOAL:
                    arrows[(r, c)].set_text(ACTION_ARROWS[policy[idx]])
                else:
                    arrows[(r, c)].set_text("")

        ep = (frame_idx + 1) * snapshot_interval
        episode_text.set_text(f"Episode {ep}")
        return list(patches.values()) + list(arrows.values()) + [episode_text]

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False
    )
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path
