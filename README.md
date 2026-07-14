# 🧠 DRL Visual Lab

> **Deep Reinforcement Learning — from first principles, with insane animations.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-100%25_Passing-brightgreen?style=for-the-badge&logo=pytest)](./tests/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![DRL](https://img.shields.io/badge/Concept-Deep_RL-orange?style=for-the-badge)](https://en.wikipedia.org/wiki/Reinforcement_learning)

---

## 🎯 What is this?

A **lightweight, visual, and fully-tested** playground for Deep Reinforcement Learning concepts.
No heavy frameworks. No black boxes. Every algorithm is implemented from scratch so you **see exactly what's happening**.

```
┌─────────────────────────────────────────────────────────────┐
│  Agent  ──── action ────►  Environment                     │
│    ▲                           │                            │
│    └──── reward + state ◄──────┘                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
drl-visual-lab/
├── 📁 core/
│   ├── agent.py          # DQN Agent (neural net policy)
│   ├── environment.py    # GridWorld environment
│   ├── replay_buffer.py  # Experience replay memory
│   └── trainer.py        # Training loop
├── 📁 visualize/
│   ├── animate_training.py    # Animated training progress
│   ├── animate_policy.py      # Policy heatmap animation
│   └── animate_qvalues.py     # Q-value evolution animation
├── 📁 tests/
│   ├── test_environment.py
│   ├── test_agent.py
│   ├── test_replay_buffer.py
│   └── test_trainer.py
├── main.py               # 🚀 Run everything
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/parsa-mirsaeed/drl-visual-lab
cd drl-visual-lab

# Install
pip install -r requirements.txt

# Run the visual lab (trains + animates)
python main.py

# Run tests
pytest tests/ -v --tb=short
```

---

## 🧩 Core Concepts Covered

| Concept | File | Animated? |
|---|---|---|
| Markov Decision Process | `core/environment.py` | ✅ GridWorld |
| Q-Learning / DQN | `core/agent.py` | ✅ Q-value heatmap |
| Experience Replay | `core/replay_buffer.py` | ✅ Memory fill bar |
| Epsilon-Greedy Policy | `core/agent.py` | ✅ Explore vs exploit |
| Reward curve | `core/trainer.py` | ✅ Live reward plot |

---

## 🎬 Animations

All animations are saved as `.gif` files in `output/`:

- `output/training_curve.gif` — reward & loss curves animate episode-by-episode
- `output/policy_heatmap.gif` — agent's best action per cell evolves over training
- `output/qvalue_surface.gif` — 3D Q-value surface morphing as the agent learns
- `output/agent_play.gif` — agent navigating the GridWorld after training

---

## 🧪 Tests

```bash
pytest tests/ -v
```

All tests are deterministic (seeded) and cover:
- Environment transitions, rewards, terminal states
- Replay buffer capacity, sampling, overflow
- Agent action selection, network forward pass
- Full training loop convergence

---

## 📚 Learn More

- [Sutton & Barto — RL: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [DQN Paper — Mnih et al. 2015](https://www.nature.com/articles/nature14236)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

---

<p align="center">Made with 🧠 by <a href="https://github.com/parsa-mirsaeed">parsa-mirsaeed</a></p>
