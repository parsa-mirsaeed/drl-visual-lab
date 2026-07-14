/**
 * drl_engine.js
 * Pure JavaScript DRL engine — simulates what would run as Rust/WASM.
 * Architecture mirrors a real WASM module: all state is inside DRLEngine,
 * exposed via a clean API (same signatures a wasm-bindgen module would have).
 *
 * Implements:
 *   - GridWorld MDP (5×5)
 *   - Tabular Q-Learning (lightweight, runs at 1000+ eps/sec in browser)
 *   - Epsilon-greedy policy
 *   - Bellman update
 */

'use strict';

const GRID = 5;
const N_STATES  = GRID * GRID;   // 25
const N_ACTIONS = 4;              // UP DOWN LEFT RIGHT
const GOAL  = [4, 4];
const START = [0, 0];
const TRAPS = new Set(['1,1','1,3','3,1','3,3']);

const DELTAS = [[-1,0],[1,0],[0,-1],[0,1]]; // UP DOWN LEFT RIGHT
export const ACTION_SYMBOLS = ['↑','↓','←','→'];

// ── GridWorld ─────────────────────────────────────────────────────────
export class GridWorld {
  constructor(seed = 42) {
    this.rng   = new SeededRNG(seed);
    this.reset();
  }

  reset() {
    this.pos   = [...START];
    this.steps = 0;
    this.done  = false;
    return this._stateIdx();
  }

  step(action) {
    if (this.done) throw new Error('Episode done — call reset()');
    const [dr, dc] = DELTAS[action];
    this.pos[0] = Math.max(0, Math.min(GRID-1, this.pos[0] + dr));
    this.pos[1] = Math.max(0, Math.min(GRID-1, this.pos[1] + dc));
    this.steps++;

    const key = this.pos.join(',');
    let reward, outcome;
    if (key === GOAL.join(','))          { reward = +10; this.done = true; outcome = 'goal';    }
    else if (TRAPS.has(key))             { reward =  -5; this.done = true; outcome = 'trap';    }
    else if (this.steps >= 200)          { reward = -0.1; this.done = true; outcome = 'timeout'; }
    else                                 { reward = -0.1; outcome = 'step'; }

    return { state: this._stateIdx(), reward, done: this.done, outcome, pos: [...this.pos] };
  }

  _stateIdx() { return this.pos[0] * GRID + this.pos[1]; }
  get stateIdx() { return this._stateIdx(); }

  isGoal(r,c)  { return r===GOAL[0]  && c===GOAL[1]; }
  isTrap(r,c)  { return TRAPS.has(`${r},${c}`); }
  isStart(r,c) { return r===START[0] && c===START[1]; }
}

// ── Q-Learning Agent ──────────────────────────────────────────────────
export class QLearningAgent {
  constructor({ lr=0.1, gamma=0.95, epsStart=1.0, epsDecay=0.97, epsilonMin=0.05, seed=42 } = {}) {
    this.lr         = lr;
    this.gamma      = gamma;
    this.epsilon    = epsStart;
    this.epsDecay   = epsDecay;
    this.epsilonMin = epsilonMin;
    this.rng        = new SeededRNG(seed);
    // Q-table: Float32Array(25 * 4) — same layout as a WASM linear memory chunk
    this.Q = new Float32Array(N_STATES * N_ACTIONS).fill(0);
  }

  // Q(s,a)
  q(s, a)       { return this.Q[s * N_ACTIONS + a]; }
  setQ(s, a, v) { this.Q[s * N_ACTIONS + a] = v; }

  selectAction(state, greedy = false) {
    if (!greedy && this.rng.next() < this.epsilon) {
      return Math.floor(this.rng.next() * N_ACTIONS);
    }
    let best = 0, bestV = this.q(state, 0);
    for (let a = 1; a < N_ACTIONS; a++) {
      const v = this.q(state, a);
      if (v > bestV) { bestV = v; best = a; }
    }
    return best;
  }

  // Bellman update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') − Q(s,a)]
  update(s, a, r, sPrime, done) {
    let maxQ = this.q(sPrime, 0);
    for (let a2 = 1; a2 < N_ACTIONS; a2++) maxQ = Math.max(maxQ, this.q(sPrime, a2));
    const target = r + (done ? 0 : this.gamma * maxQ);
    const err    = target - this.q(s, a);
    this.setQ(s, a, this.q(s, a) + this.lr * err);
    return Math.abs(err);  // return TD-error as "loss"
  }

  decayEpsilon() {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsDecay);
  }

  greedyPolicy() {
    const policy = new Uint8Array(N_STATES);
    for (let s = 0; s < N_STATES; s++) policy[s] = this.selectAction(s, true);
    return policy;
  }

  maxQPerState() {
    const out = new Float32Array(N_STATES);
    for (let s = 0; s < N_STATES; s++) {
      let m = this.q(s, 0);
      for (let a = 1; a < N_ACTIONS; a++) m = Math.max(m, this.q(s, a));
      out[s] = m;
    }
    return out;
  }
}

// ── Trainer ───────────────────────────────────────────────────────────
export class Trainer {
  constructor(config = {}) {
    this.config  = config;
    this.env     = new GridWorld(42);
    this.agent   = new QLearningAgent(config);
    this.history = { rewards: [], losses: [], epsilons: [] };
    this._ep     = 0;
    this._done   = false;
  }

  /** Run `n` episodes synchronously. Returns whether training is complete. */
  runBatch(n) {
    const total = this.config.episodes || 300;
    for (let i = 0; i < n && this._ep < total; i++) {
      const { reward, loss } = this._runEpisode();
      this.history.rewards.push(reward);
      this.history.losses.push(loss);
      this.history.epsilons.push(this.agent.epsilon);
      this.agent.decayEpsilon();
      this._ep++;
    }
    this._done = this._ep >= total;
    return this._done;
  }

  _runEpisode() {
    let state = this.env.reset();
    let totalR = 0, totalL = 0, steps = 0;
    while (true) {
      const action = this.agent.selectAction(state);
      const { state: ns, reward, done } = this.env.step(action);
      const loss = this.agent.update(state, action, reward, ns, done);
      totalR += reward;
      totalL += loss;
      steps++;
      state = ns;
      if (done) break;
    }
    return { reward: totalR, loss: steps > 0 ? totalL / steps : 0 };
  }

  playEpisode() {
    let state = this.env.reset();
    const traj = [];
    while (true) {
      const action = this.agent.selectAction(state, true);
      const result = this.env.step(action);
      traj.push({ pos: result.pos, action, reward: result.reward, outcome: result.outcome });
      state = result.state;
      if (result.done) break;
    }
    return traj;
  }

  get episode()    { return this._ep; }
  get totalEps()   { return this.config.episodes || 300; }
  get isDone()     { return this._done; }
}

// ── Seeded RNG (xorshift32) — deterministic, same as Rust's SmallRng ──
class SeededRNG {
  constructor(seed = 42) { this.s = seed >>> 0 || 1; }
  next() {
    let x = this.s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    this.s = x >>> 0;
    return (x >>> 0) / 0xFFFFFFFF;
  }
}

export { GRID, N_STATES, N_ACTIONS, TRAPS, GOAL, START };
