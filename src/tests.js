/**
 * tests.js
 * In-browser test suite for the DRL engine.
 * No test framework needed — pure JS assertions.
 * Results rendered live into #test-results.
 */

'use strict';

import { GridWorld, QLearningAgent, Trainer, GRID, N_STATES, N_ACTIONS, TRAPS, GOAL } from './drl_engine.js';

const SUITE = [];
function test(name, fn) { SUITE.push({ name, fn }); }

// ── GridWorld Tests ───────────────────────────────────────────────────
test('GridWorld: reset returns valid state index', () => {
  const env = new GridWorld();
  const s = env.reset();
  assert(s === 0, `Expected 0, got ${s}`);
});

test('GridWorld: step DOWN moves to row 1', () => {
  const env = new GridWorld();
  env.reset();
  const { pos } = env.step(1); // DOWN
  assertEq(pos[0], 1); assertEq(pos[1], 0);
});

test('GridWorld: wall clamp — UP from start stays at (0,0)', () => {
  const env = new GridWorld();
  env.reset();
  const { pos } = env.step(0); // UP
  assertEq(pos[0], 0); assertEq(pos[1], 0);
});

test('GridWorld: wall clamp — LEFT from start stays at (0,0)', () => {
  const env = new GridWorld();
  env.reset();
  const { pos } = env.step(2); // LEFT
  assertEq(pos[0], 0); assertEq(pos[1], 0);
});

test('GridWorld: step reward is -0.1 for free cell', () => {
  const env = new GridWorld();
  env.reset();
  const { reward, outcome } = env.step(3); // RIGHT
  assertClose(reward, -0.1);
  assertEq(outcome, 'step');
});

test('GridWorld: trap gives reward -5 and done=true', () => {
  const env = new GridWorld();
  env.reset();
  env.pos = [1, 0];
  const { reward, done, outcome } = env.step(3); // RIGHT → (1,1) trap
  assertClose(reward, -5); assert(done); assertEq(outcome, 'trap');
});

test('GridWorld: goal gives reward +10 and done=true', () => {
  const env = new GridWorld();
  env.reset();
  env.pos = [4, 3];
  const { reward, done, outcome } = env.step(3); // RIGHT → (4,4)
  assertClose(reward, 10); assert(done); assertEq(outcome, 'goal');
});

test('GridWorld: step after done throws', () => {
  const env = new GridWorld();
  env.reset();
  env.pos = [4, 3]; env.step(3); // reach goal
  let threw = false;
  try { env.step(0); } catch(e) { threw = true; }
  assert(threw, 'Should throw after done');
});

test('GridWorld: isGoal / isTrap / isStart helpers', () => {
  const env = new GridWorld();
  assert(env.isGoal(4,4));  assert(!env.isGoal(0,0));
  assert(env.isTrap(1,1));  assert(!env.isTrap(0,0));
  assert(env.isStart(0,0)); assert(!env.isStart(4,4));
});

test('GridWorld: n_states === 25, n_actions === 4', () => {
  const env = new GridWorld();
  // These are module-level constants
  assertEq(N_STATES, 25); assertEq(N_ACTIONS, 4);
});

// ── QLearningAgent Tests ──────────────────────────────────────────────
test('Agent: Q-table initialised to zeros', () => {
  const a = new QLearningAgent();
  for (let s = 0; s < N_STATES; s++)
    for (let act = 0; act < N_ACTIONS; act++)
      assertClose(a.q(s, act), 0);
});

test('Agent: selectAction returns value in [0,3]', () => {
  const a = new QLearningAgent();
  for (let i = 0; i < 50; i++) {
    const act = a.selectAction(0);
    assert(act >= 0 && act < N_ACTIONS, `action ${act} out of range`);
  }
});

test('Agent: greedy action is deterministic', () => {
  const a = new QLearningAgent();
  a.setQ(0, 2, 5.0); // force action 2 to be best
  const acts = new Set();
  for (let i = 0; i < 20; i++) acts.add(a.selectAction(0, true));
  assertEq(acts.size, 1); assertEq([...acts][0], 2);
});

test('Agent: epsilon=1.0 → multiple actions sampled over 100 tries', () => {
  const a = new QLearningAgent({ epsStart: 1.0 });
  const acts = new Set();
  for (let i = 0; i < 100; i++) acts.add(a.selectAction(0));
  assert(acts.size > 1, 'Expected random exploration');
});

test('Agent: epsilon=0.0 → always greedy', () => {
  const a = new QLearningAgent({ epsStart: 0.0 });
  a.setQ(0, 3, 9.9);
  const acts = new Set();
  for (let i = 0; i < 20; i++) acts.add(a.selectAction(0));
  assertEq(acts.size, 1);
});

test('Agent: Bellman update increases Q toward target', () => {
  const a = new QLearningAgent({ lr: 1.0, gamma: 0.9, epsStart: 0 });
  // Q(0,0)=0, r=10, done=true → target=10
  a.update(0, 0, 10, 1, true);
  assert(a.q(0,0) > 0, `Q should be positive after positive reward, got ${a.q(0,0)}`);
});

test('Agent: epsilon decays after decayEpsilon()', () => {
  const a = new QLearningAgent({ epsStart: 1.0, epsDecay: 0.9 });
  const before = a.epsilon;
  a.decayEpsilon();
  assert(a.epsilon < before, 'Epsilon should decrease');
});

test('Agent: epsilon never below epsilonMin', () => {
  const a = new QLearningAgent({ epsStart: 0.05, epsDecay: 0.5, epsilonMin: 0.05 });
  for (let i = 0; i < 100; i++) a.decayEpsilon();
  assert(a.epsilon >= 0.05, `epsilon=${a.epsilon} below min`);
});

test('Agent: greedyPolicy returns array of length 25', () => {
  const a = new QLearningAgent();
  const p = a.greedyPolicy();
  assertEq(p.length, N_STATES);
  for (const act of p) assert(act >= 0 && act < N_ACTIONS);
});

test('Agent: maxQPerState returns Float32Array length 25', () => {
  const a = new QLearningAgent();
  const m = a.maxQPerState();
  assertEq(m.length, N_STATES);
});

// ── Trainer Tests ─────────────────────────────────────────────────────
test('Trainer: runBatch runs N episodes', () => {
  const t = new Trainer({ episodes: 50 });
  t.runBatch(50);
  assertEq(t.episode, 50);
  assertEq(t.history.rewards.length, 50);
});

test('Trainer: epsilon decreases over training', () => {
  const t = new Trainer({ episodes: 100 });
  t.runBatch(100);
  assert(t.agent.epsilon < 1.0, 'Epsilon should have decayed');
});

test('Trainer: playEpisode returns non-empty trajectory', () => {
  const t = new Trainer({ episodes: 100 });
  t.runBatch(100);
  const traj = t.playEpisode();
  assert(traj.length > 0, 'Trajectory should not be empty');
});

test('Trainer: trajectory last step is terminal', () => {
  const t = new Trainer({ episodes: 100 });
  t.runBatch(100);
  const traj = t.playEpisode();
  const last = traj[traj.length-1];
  assert(['goal','trap','timeout'].includes(last.outcome), `Bad outcome: ${last.outcome}`);
});

test('Trainer: after 300 episodes, max reward > 0', () => {
  const t = new Trainer({ episodes: 300 });
  t.runBatch(300);
  const mx = Math.max(...t.history.rewards);
  assert(mx > 0, `Expected some positive rewards, max=${mx}`);
});

// ── Runner ────────────────────────────────────────────────────────────
export async function runTests() {
  const container = document.getElementById('test-results');
  container.innerHTML = '';

  let pass = 0, fail = 0;

  // Header
  addItem(container, 'info', '🧪', `Running ${SUITE.length} tests…`);

  for (let i = 0; i < SUITE.length; i++) {
    const { name, fn } = SUITE[i];
    await sleep(18); // small delay for animated reveal
    try {
      fn();
      pass++;
      addItem(container, 'pass', '✅', name);
    } catch (e) {
      fail++;
      addItem(container, 'fail', '❌', `${name} — ${e.message}`);
    }
  }

  await sleep(30);
  addItem(container, fail === 0 ? 'pass' : 'fail', fail===0 ? '🎉' : '⚠️',
    `${pass}/${SUITE.length} passed${fail>0 ? ` · ${fail} failed` : ''}`);
}

function addItem(container, cls, icon, text) {
  const div = document.createElement('div');
  div.className = `test-item ${cls}`;
  div.style.animationDelay = '0ms';
  div.innerHTML = `<span class="test-icon">${icon}</span><span>${text}</span>`;
  container.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Assertion helpers ─────────────────────────────────────────────────
function assert(cond, msg = 'Assertion failed') { if (!cond) throw new Error(msg); }
function assertEq(a, b)  { if (a !== b)              throw new Error(`Expected ${b}, got ${a}`); }
function assertClose(a, b, eps=1e-9) { if (Math.abs(a-b) > eps) throw new Error(`Expected ≈${b}, got ${a}`); }
