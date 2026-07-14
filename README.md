# 🧠 DRL Visual Lab

> **Deep Reinforcement Learning — from first principles, live in your browser.**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-GitHub_Pages-blue?style=for-the-badge)](https://parsa-mirsaeed.github.io/drl-visual-lab/)
[![Rust/WASM](https://img.shields.io/badge/🦀_Engine-Rust%2FWASM_Architecture-orange?style=for-the-badge)](./src/drl_engine.js)
[![Zero Install](https://img.shields.io/badge/📦_Zero-Install_Required-green?style=for-the-badge)](#)
[![Tests](https://img.shields.io/badge/🧪_Tests-26_Passing-brightgreen?style=for-the-badge)](./src/tests.js)

---

## 🌐 [Open the Live Demo →](https://parsa-mirsaeed.github.io/drl-visual-lab/)

No Python. No `pip install`. No setup. Just open the URL and start training.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🦀 Rust/WASM Architecture | Engine mirrors exact Rust WASM-bindgen API — drop-in replaceable |
| 🎨 Animated GridWorld | Live agent with glowing pulse, trajectory trail, flash on goal/trap |
| 🔥 Policy Heatmap | Plasma colormap + shimmer animation, updates every episode |
| 📊 Live Charts | 4 animated charts: reward, loss, ε-decay, reward distribution |
| 🌌 Particle Background | Animated neural-network-style particle field |
| ⚙️ Hyperparameter Sliders | Tune α, γ, ε, episodes, speed in real time |
| 🧪 In-Browser Test Suite | 26 tests, animated reveal, zero frameworks |
| 📄 GitHub Pages | Auto-deploys on every push to main |

---

## 🧩 Architecture

```
browser
├── index.html          # Single-page app shell
├── style.css           # Dark theme, animations, responsive
└── src/
    ├── drl_engine.js   # 🦀 DRL engine (Rust/WASM API-compatible)
    │   ├── GridWorld        (MDP environment)
    │   ├── QLearningAgent   (Q-table, Bellman updates)
    │   └── Trainer          (episode loop)
    ├── renderer.js     # Canvas: GridWorld + heatmap
    ├── charts.js       # Animated line + histogram charts
    ├── tests.js        # 26 in-browser assertions
    └── app.js          # Controller — wires everything
```

---

## 🧠 Concepts Visualised

- **MDP** — states, actions, transitions, terminal states
- **Bellman Equation** — `Q(s,a) = r + γ · max Q(s')`
- **Epsilon-Greedy** — exploration vs exploitation curve
- **Q-Table** — value landscape as a heatmap
- **Policy** — greedy action arrows per cell
- **Training curve** — reward + loss converging over episodes

---

## 🦀 Rust/WASM Note

The engine in `src/drl_engine.js` is intentionally written to mirror a
`wasm-bindgen`-compiled Rust module's API exactly. To swap in the real
Rust version:

```bash
# (optional — engine already works without this)
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32.../drl_engine.wasm --out-dir src/
```

The JS API stays identical — zero changes to `app.js`, `renderer.js`, or `tests.js`.

---

<p align="center">Made with 🦀 + 🌐 by <a href="https://github.com/parsa-mirsaeed">parsa-mirsaeed</a></p>
