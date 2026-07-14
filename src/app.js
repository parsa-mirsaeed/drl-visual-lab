/**
 * app.js — main controller
 * Wires UI, engine, renderer, charts together.
 */

'use strict';

import { Trainer }                         from './drl_engine.js';
import { GridRenderer, HeatmapRenderer }   from './renderer.js';
import { LineChart, HistChart }             from './charts.js';
import { runTests }                         from './tests.js';

// ── State ─────────────────────────────────────────────────────────────
let trainer   = null;
let rafId     = null;
let isTraining = false;

// ── Canvas renderers ──────────────────────────────────────────────────
const gridRen    = new GridRenderer('grid-canvas');
const heatRen    = new HeatmapRenderer('heatmap-canvas');

// ── Charts ────────────────────────────────────────────────────────────
const chartReward = new LineChart('chart-reward', 'rgb(56,139,253)', 'Reward');
const chartLoss   = new LineChart('chart-loss',   'rgb(248,81,73)',  'Loss');
const chartEps    = new LineChart('chart-eps',    'rgb(63,185,80)',  'Epsilon');
const chartDist   = new HistChart('chart-dist',   'rgb(188,140,255)');

// ── Slider bindings ───────────────────────────────────────────────────
const sliders = [
  ['lr',        'lr-val',        v => parseFloat(v).toFixed(3)],
  ['gamma',     'gamma-val',     v => parseFloat(v).toFixed(3)],
  ['eps-start', 'eps-start-val', v => parseFloat(v).toFixed(2)],
  ['eps-decay', 'eps-decay-val', v => parseFloat(v).toFixed(3)],
  ['episodes',  'episodes-val',  v => v],
  ['speed',     'speed-val',     v => v],
];
sliders.forEach(([id, valId, fmt]) => {
  const el = document.getElementById(id);
  const vEl = document.getElementById(valId);
  el.addEventListener('input', () => { vEl.textContent = fmt(el.value); });
});

// ── Helper: read config from sliders ─────────────────────────────────
function readConfig() {
  return {
    lr:        parseFloat(document.getElementById('lr').value),
    gamma:     parseFloat(document.getElementById('gamma').value),
    epsStart:  parseFloat(document.getElementById('eps-start').value),
    epsDecay:  parseFloat(document.getElementById('eps-decay').value),
    episodes:  parseInt(document.getElementById('episodes').value),
    speed:     parseInt(document.getElementById('speed').value),
  };
}

function setStatus(msg, color = '#3fb950') {
  const el = document.getElementById('status-text');
  el.textContent = msg;
  el.style.color = color;
}

// ── Train button ──────────────────────────────────────────────────────
document.getElementById('btn-train').addEventListener('click', () => {
  if (isTraining) return;
  const config = readConfig();
  trainer = new Trainer(config);
  [chartReward, chartLoss, chartEps, chartDist].forEach(c => c.reset());
  isTraining = true;
  document.getElementById('btn-train').disabled = true;
  document.getElementById('btn-play').disabled  = true;

  const heatState = { maxQ: null, policy: null, episode: '' };
  heatRen.startShimmer(() => heatState);

  let tick = 0;
  function loop() {
    if (!isTraining) return;
    const batchSize = config.speed;
    const done = trainer.runBatch(batchSize);

    // Push to charts
    const { rewards, losses, epsilons } = trainer.history;
    const n = rewards.length;
    for (let i = Math.max(0, n - batchSize); i < n; i++) {
      chartReward.push(rewards[i]);
      chartLoss.push(losses[i]);
      chartEps.push(epsilons[i]);
      chartDist.push(rewards[i]);
    }

    // Draw charts
    chartReward.draw(); chartLoss.draw(); chartEps.draw(); chartDist.draw();

    // Update grid (show agent at start during training)
    const policy = trainer.agent.greedyPolicy();
    const maxQ   = trainer.agent.maxQPerState();
    gridRen.draw([0,0], maxQ, policy, tick++);

    // Update heatmap state (shimmer loop reads this)
    heatState.maxQ    = maxQ;
    heatState.policy  = policy;
    heatState.episode = `${trainer.episode}/${trainer.totalEps}`;

    setStatus(
      `Training… ep ${trainer.episode}/${trainer.totalEps}  |  ε=${trainer.agent.epsilon.toFixed(3)}`,
      '#ffa657'
    );

    if (done) {
      isTraining = false;
      heatRen.stopShimmer();
      heatRen.draw(maxQ, policy, `${trainer.episode}/${trainer.totalEps}`);
      setStatus('✅ Training complete! Hit 🤖 Play to watch the agent.', '#3fb950');
      document.getElementById('btn-train').disabled = false;
      document.getElementById('btn-play').disabled  = false;
    } else {
      rafId = requestAnimationFrame(loop);
    }
  }

  requestAnimationFrame(loop);
});

// ── Play button ───────────────────────────────────────────────────────
document.getElementById('btn-play').addEventListener('click', () => {
  if (!trainer || isTraining) return;
  const traj = trainer.playEpisode();
  const outcome = traj[traj.length-1].outcome;
  const emoji = outcome === 'goal' ? '🎯' : outcome === 'trap' ? '💀' : '⏱️';
  setStatus(`Playing… ${traj.length} steps → ${emoji} ${outcome.toUpperCase()}`, '#bc8cff');
  document.getElementById('btn-play').disabled = true;
  gridRen.animateTrajectory(traj, () => {
    document.getElementById('btn-play').disabled = false;
    setStatus(`Episode ended: ${emoji} ${outcome.toUpperCase()} in ${traj.length} steps`, '#3fb950');
  });
});

// ── Reset button ──────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', () => {
  isTraining = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  gridRen.stopAnimation();
  heatRen.stopShimmer();
  trainer = null;
  [chartReward, chartLoss, chartEps, chartDist].forEach(c => c.reset());
  gridRen.draw();
  setStatus('Reset. Hit Train to start.', '#8b949e');
  document.getElementById('btn-train').disabled = false;
  document.getElementById('btn-play').disabled  = true;
});

// ── Tests button ──────────────────────────────────────────────────────
document.getElementById('btn-tests').addEventListener('click', () => {
  runTests();
});

// ── Animated particle background ──────────────────────────────────────
(function initBg() {
  const canvas = document.getElementById('bg-canvas');
  const ctx    = canvas.getContext('2d');
  const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
  resize();
  window.addEventListener('resize', resize);

  const N = 55;
  const pts = Array.from({ length: N }, () => ({
    x: Math.random(), y: Math.random(),
    vx: (Math.random()-.5)*0.0003,
    vy: (Math.random()-.5)*0.0003,
    r: Math.random()*2+1,
    hue: Math.random()*60+200, // blue-purple range
  }));

  function frame() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);
    for (const p of pts) {
      p.x = (p.x + p.vx + 1) % 1;
      p.y = (p.y + p.vy + 1) % 1;
      ctx.beginPath();
      ctx.arc(p.x*W, p.y*H, p.r, 0, Math.PI*2);
      ctx.fillStyle = `hsla(${p.hue},80%,65%,0.7)`;
      ctx.fill();
    }
    // Draw connections
    for (let i = 0; i < N; i++) {
      for (let j = i+1; j < N; j++) {
        const dx = (pts[i].x - pts[j].x)*W;
        const dy = (pts[i].y - pts[j].y)*H;
        const d  = Math.sqrt(dx*dx+dy*dy);
        if (d < 120) {
          ctx.beginPath();
          ctx.moveTo(pts[i].x*W, pts[i].y*H);
          ctx.lineTo(pts[j].x*W, pts[j].y*H);
          ctx.strokeStyle = `rgba(56,139,253,${0.15*(1-d/120)})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

// ── Initial render ────────────────────────────────────────────────────
gridRen.draw();
