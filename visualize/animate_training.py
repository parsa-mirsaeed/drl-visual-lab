"""Animated training curves: reward, loss, and epsilon over episodes.

Produces: output/training_curve.gif
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = "output"


def animate_training(
    rewards: list,
    losses: list,
    epsilons: list,
    filename: str = "training_curve.gif",
    fps: int = 12,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    n = len(rewards)
    smooth_r = _smooth(rewards, window=20)

    fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_r = fig.add_subplot(gs[0, :])
    ax_l = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])

    _style_ax(ax_r, "Episode Reward", "Episode", "Total Reward")
    _style_ax(ax_l, "Training Loss", "Episode", "MSE Loss")
    _style_ax(ax_e, "Exploration (ε)", "Episode", "Epsilon")

    fig.suptitle(
        "🧠 DRL Training Progress",
        color="#58a6ff",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Lines
    (line_r,) = ax_r.plot([], [], color="#388bfd", lw=1.2, alpha=0.4, label="Raw")
    (line_rs,) = ax_r.plot([], [], color="#ffa657", lw=2.5, label="Smoothed (20-ep)")
    (line_l,) = ax_l.plot([], [], color="#f85149", lw=1.5)
    (line_e,) = ax_e.plot([], [], color="#3fb950", lw=2.0)

    ax_r.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    # Axis limits
    ax_r.set_xlim(0, n)
    ax_r.set_ylim(min(rewards) - 1, max(rewards) + 1)
    ax_l.set_xlim(0, n)
    ax_l.set_ylim(0, max(losses) * 1.1 + 1e-6)
    ax_e.set_xlim(0, n)
    ax_e.set_ylim(0, 1.05)

    # Animated episode counter text
    ep_text = ax_r.text(
        0.02, 0.92, "",
        transform=ax_r.transAxes,
        color="#8b949e",
        fontsize=10,
    )

    # Reward fill
    fill_ref = [None]

    step = max(1, n // 120)  # ~120 frames
    frames = list(range(0, n, step)) + [n - 1]

    def update(frame_idx):
        i = frames[frame_idx]
        xs = list(range(i + 1))
        line_r.set_data(xs, rewards[: i + 1])
        line_rs.set_data(xs, smooth_r[: i + 1])
        line_l.set_data(xs, losses[: i + 1])
        line_e.set_data(xs, epsilons[: i + 1])
        ep_text.set_text(f"Episode {i + 1}/{n}")

        # Animated fill under reward curve
        if fill_ref[0] is not None:
            fill_ref[0].remove()
        fill_ref[0] = ax_r.fill_between(
            xs, rewards[: i + 1], alpha=0.08, color="#388bfd"
        )
        return line_r, line_rs, line_l, line_e, ep_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ──────────────────────────────────────────────

def _smooth(data: list, window: int) -> list:
    result = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        result.append(float(np.mean(data[lo : i + 1])))
    return result


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#161b22")
    ax.set_title(title, color="#c9d1d9", fontsize=11, pad=6)
    ax.set_xlabel(xlabel, color="#8b949e", fontsize=9)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5)
