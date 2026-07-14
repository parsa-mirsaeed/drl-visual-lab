"""3D Q-value surface animation — morphing as agent learns.

Shows max Q-value per state as a 3D surface that rises and sharpens
as training progresses. Most dramatic visualization in the lab.
Produces: output/qvalue_surface.gif
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core.environment import GridWorld

OUTPUT_DIR = "output"


def animate_qvalue_surface(
    q_snapshots: list,
    snapshot_interval: int = 50,
    filename: str = "qvalue_surface.gif",
    fps: int = 4,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    env = GridWorld()
    G = env.GRID_SIZE
    x = np.arange(G)
    y = np.arange(G)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7), facecolor="#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#30363d")
    ax.yaxis.pane.set_edgecolor("#30363d")
    ax.zaxis.pane.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", alpha=0.5)

    z_global_min = min(q.max(axis=1).min() for q in q_snapshots)
    z_global_max = max(q.max(axis=1).max() for q in q_snapshots)

    surf_ref = [None]
    ep_text = fig.text(
        0.5, 0.93, "",
        ha="center", color="#58a6ff", fontsize=13, fontweight="bold"
    )

    def update(frame_idx):
        q_table = q_snapshots[frame_idx]
        max_q = q_table.max(axis=1).reshape(G, G)

        if surf_ref[0] is not None:
            surf_ref[0].remove()

        surf_ref[0] = ax.plot_surface(
            X, Y, max_q,
            cmap="plasma",
            edgecolor="none",
            alpha=0.9,
            vmin=z_global_min,
            vmax=z_global_max,
        )
        ax.set_zlim(z_global_min - 0.5, z_global_max + 0.5)
        ax.set_xlabel("Column", color="#8b949e", labelpad=6)
        ax.set_ylabel("Row", color="#8b949e", labelpad=6)
        ax.set_zlabel("Max Q-Value", color="#8b949e", labelpad=6)
        ax.set_title("Q-Value Surface", color="#c9d1d9", pad=4)

        ep = (frame_idx + 1) * snapshot_interval
        ep_text.set_text(f"🔥 Episode {ep}  —  Q-Value Surface Evolving")

        # Rotate camera for cinematic effect
        ax.view_init(elev=30, azim=45 + frame_idx * 8)

        return [surf_ref[0], ep_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(q_snapshots), interval=1000 // fps, blit=False
    )
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


def animate_agent_play(
    trajectory: list,
    filename: str = "agent_play.gif",
    fps: int = 4,
) -> str:
    """Animate the trained agent navigating the grid."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    env = GridWorld()
    G = env.GRID_SIZE

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(-0.5, G - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_title("🤖 Trained Agent — Live Play", color="#58a6ff", fontsize=13)

    ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    # Draw grid
    for r in range(G):
        for c in range(G):
            t = (r, c)
            if t == env.GOAL:
                color = "#1a7f37"
            elif t in env.TRAPS:
                color = "#6e1a18"
            elif t == env.START:
                color = "#0c2d6b"
            else:
                color = "#161b22"
            rect = plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=1, edgecolor="#30363d", facecolor=color
            )
            ax.add_patch(rect)

    ax.text(*env.GOAL[::-1], "G", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")
    ax.text(*env.START[::-1], "S", ha="center", va="center",
            fontsize=12, color="white")
    for t in env.TRAPS:
        ax.text(t[1], t[0], "☠", ha="center", va="center", fontsize=14)

    # Agent marker
    agent_dot, = ax.plot([], [], "o", color="#ffa657",
                         markersize=28, zorder=10, alpha=0.9)
    agent_arrow = ax.text(0, 0, "", ha="center", va="center",
                          fontsize=18, color="white", zorder=11)
    step_text = ax.text(
        0.5, -0.06, "",
        transform=ax.transAxes,
        ha="center", color="#8b949e", fontsize=10
    )
    reward_text = ax.text(
        0.98, 0.02, "",
        transform=ax.transAxes,
        ha="right", color="#3fb950", fontsize=11
    )

    # Trail
    trail_dots, = ax.plot([], [], "o", color="#388bfd",
                          markersize=8, alpha=0.3, zorder=9)
    trail_x: list = []
    trail_y: list = []

    def update(frame_idx):
        pos, action, reward, outcome = trajectory[frame_idx]
        r, c = pos
        trail_x.append(c)
        trail_y.append(r)
        agent_dot.set_data([c], [r])
        agent_arrow.set_position((c, r))
        agent_arrow.set_text(ACTION_ARROWS[action])
        trail_dots.set_data(trail_x[:-1], trail_y[:-1])
        step_text.set_text(f"Step {frame_idx + 1}/{len(trajectory)}  |  {outcome.upper()}")
        reward_text.set_text(f"r={reward:+.1f}")
        if outcome == "goal":
            agent_dot.set_color("#3fb950")
        elif outcome == "trap":
            agent_dot.set_color("#f85149")
        return agent_dot, agent_arrow, trail_dots, step_text, reward_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(trajectory), interval=1000 // fps, blit=False
    )
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path
