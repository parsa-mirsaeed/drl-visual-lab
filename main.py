"""🚀 DRL Visual Lab — main entry point.

Runs:
  1. Training (DQN on GridWorld)
  2. Animated training curves
  3. Animated policy heatmap
  4. Animated Q-value 3D surface
  5. Animated agent play

All GIFs saved to output/
"""

import time
from core.trainer import Trainer
from visualize.animate_training import animate_training
from visualize.animate_policy import animate_policy
from visualize.animate_qvalues import animate_qvalue_surface, animate_agent_play


BANNER = """
╔══════════════════════════════════════════════════════════╗
║          🧠  D R L   V I S U A L   L A B  🧠           ║
║    Deep Reinforcement Learning — from first principles   ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── 1. Training ──────────────────────────────────────────────────
    print("[1/5] 🏋️  Training DQN on GridWorld...")
    t0 = time.time()
    trainer = Trainer(n_episodes=500, seed=42)
    history = trainer.train(verbose=True)
    print(f"      Done in {time.time() - t0:.1f}s\n")

    # ── 2. Training curves animation ────────────────────────────────
    print("[2/5] 📈  Animating training curves...")
    animate_training(
        history["rewards"],
        history["losses"],
        history["epsilons"],
    )
    print()

    # ── 3. Policy heatmap animation ──────────────────────────────────
    print("[3/5] 🗺️   Animating policy heatmap...")
    animate_policy(
        history["q_snapshots"],
        history["policy_snapshots"],
    )
    print()

    # ── 4. Q-value surface animation ─────────────────────────────────
    print("[4/5] 🔥  Animating 3D Q-value surface...")
    animate_qvalue_surface(history["q_snapshots"])
    print()

    # ── 5. Trained agent play ────────────────────────────────────────
    print("[5/5] 🤖  Animating trained agent play...")
    trajectory = trainer.play_episode()
    animate_agent_play(trajectory)
    print()

    print("══════════════════════════════════════════")
    print("✅  All animations saved to output/")
    print("   training_curve.gif")
    print("   policy_heatmap.gif")
    print("   qvalue_surface.gif")
    print("   agent_play.gif")
    print("══════════════════════════════════════════")
    print("\nRun tests: pytest tests/ -v")


if __name__ == "__main__":
    main()
