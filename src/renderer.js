/**
 * renderer.js
 * Canvas-based renderer for GridWorld + Policy Heatmap.
 * Uses requestAnimationFrame for buttery-smooth animations.
 */

'use strict';

import { GRID, TRAPS, GOAL, START, ACTION_SYMBOLS } from './drl_engine.js';

const CELL = 72; // px per cell

// Plasma colormap (simplified 8-stop)
const PLASMA = [
  [13,8,135],[84,2,163],[139,10,165],[185,50,137],
  [219,92,104],[244,136,73],[254,188,43],[240,249,33]
];

function plasmaColor(t) {
  t = Math.max(0, Math.min(1, t));
  const n = PLASMA.length - 1;
  const i = Math.floor(t * n);
  const f = t * n - i;
  const [r1,g1,b1] = PLASMA[Math.min(i,   n)];
  const [r2,g2,b2] = PLASMA[Math.min(i+1, n)];
  return `rgb(${lerp(r1,r2,f)|0},${lerp(g1,g2,f)|0},${lerp(b1,b2,f)|0})`;
}
function lerp(a,b,t) { return a + (b-a)*t; }

// ── GridWorld renderer ────────────────────────────────────────────────
export class GridRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx    = this.canvas.getContext('2d');
    this._agentAnim = 0;  // animation tick
    this._rafId = null;
  }

  draw(agentPos = null, maxQ = null, policy = null, animTick = 0) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Compute Q normalisation range
    let qMin = Infinity, qMax = -Infinity;
    if (maxQ) {
      for (let v of maxQ) { if (v < qMin) qMin = v; if (v > qMax) qMax = v; }
    }
    const qRange = qMax - qMin || 1;

    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const x = c * CELL, y = r * CELL;
        const idx = r * GRID + c;
        const key = `${r},${c}`;

        // Background
        if (GOAL[0]===r && GOAL[1]===c) {
          ctx.fillStyle = '#1a7f37';
        } else if (TRAPS.has(key)) {
          ctx.fillStyle = '#6e1a18';
        } else if (START[0]===r && START[1]===c) {
          ctx.fillStyle = '#0c2d6b';
        } else if (maxQ) {
          const t = (maxQ[idx] - qMin) / qRange;
          ctx.fillStyle = plasmaColor(t * 0.8); // dim a bit so arrows are visible
        } else {
          ctx.fillStyle = '#161b22';
        }
        ctx.fillRect(x, y, CELL, CELL);

        // Border
        ctx.strokeStyle = '#30363d';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x, y, CELL, CELL);

        // Cell labels
        ctx.font = 'bold 18px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (GOAL[0]===r && GOAL[1]===c) {
          ctx.fillStyle = '#fff'; ctx.fillText('G', x+CELL/2, y+CELL/2);
        } else if (TRAPS.has(key)) {
          ctx.font = '22px sans-serif';
          ctx.fillText('☠', x+CELL/2, y+CELL/2);
        } else if (START[0]===r && START[1]===c && (!agentPos || agentPos[0]!==r || agentPos[1]!==c)) {
          ctx.fillStyle = '#8b949e'; ctx.fillText('S', x+CELL/2, y+CELL/2);
        }

        // Policy arrows (behind agent)
        if (policy && !TRAPS.has(key) && !(GOAL[0]===r && GOAL[1]===c)) {
          ctx.font = '20px sans-serif';
          ctx.fillStyle = 'rgba(255,255,255,0.55)';
          ctx.fillText(ACTION_SYMBOLS[policy[idx]], x+CELL/2, y+CELL/2);
        }
      }
    }

    // Agent
    if (agentPos) {
      const [ar, ac] = agentPos;
      const x = ac * CELL + CELL/2;
      const y = ar * CELL + CELL/2;
      const pulse = 1 + 0.12 * Math.sin(animTick * 0.18);
      const r = 18 * pulse;

      // Glow
      const grd = ctx.createRadialGradient(x, y, 0, x, y, r * 2);
      grd.addColorStop(0, 'rgba(255,166,87,0.6)');
      grd.addColorStop(1, 'rgba(255,166,87,0)');
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(x, y, r * 2, 0, Math.PI*2);
      ctx.fill();

      // Body
      ctx.fillStyle = '#ffa657';
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI*2);
      ctx.fill();

      // Face
      ctx.font = `${(r*1.2)|0}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('🤖', x, y);
    }
  }

  // Animated play-back of a trajectory
  animateTrajectory(trajectory, onDone) {
    let i = 0, tick = 0;
    const self = this;
    const policy = null; // hide arrows during play

    const loop = () => {
      if (i >= trajectory.length) { onDone && onDone(); return; }
      const { pos } = trajectory[i];
      self.draw(pos, null, null, tick);
      tick++;
      // Flash effect on terminal
      const outcome = trajectory[i].outcome;
      if (outcome === 'goal') { self._flash('#3fb950'); }
      else if (outcome === 'trap') { self._flash('#f85149'); }
      i++;
      self._rafId = setTimeout(() => requestAnimationFrame(loop), 220);
    };
    requestAnimationFrame(loop);
  }

  _flash(color) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.restore();
  }

  stopAnimation() {
    if (this._rafId) { clearTimeout(this._rafId); this._rafId = null; }
  }
}

// ── Policy Heatmap renderer ───────────────────────────────────────────
export class HeatmapRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx    = this.canvas.getContext('2d');
    this._tick  = 0;
    this._animating = false;
  }

  draw(maxQ, policy, episode = '') {
    const ctx  = this.ctx;
    const size = 80; // px per cell
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    let qMin = Infinity, qMax = -Infinity;
    for (let v of maxQ) { if (v<qMin) qMin=v; if (v>qMax) qMax=v; }
    const qRange = qMax - qMin || 1;

    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const x = c*size, y = r*size;
        const idx = r*GRID+c;
        const key = `${r},${c}`;

        const t = (maxQ[idx] - qMin) / qRange;
        if (GOAL[0]===r && GOAL[1]===c)      ctx.fillStyle = '#1a7f37';
        else if (TRAPS.has(key))             ctx.fillStyle = `hsl(0,60%,${15+t*10}%)`;
        else                                 ctx.fillStyle = plasmaColor(t);
        ctx.fillRect(x, y, size, size);

        // Animated shimmer on high-Q cells
        if (t > 0.75 && !TRAPS.has(key)) {
          const alpha = 0.08 + 0.06 * Math.sin(this._tick * 0.1 + idx);
          ctx.fillStyle = `rgba(255,255,255,${alpha})`;
          ctx.fillRect(x, y, size, size);
        }

        ctx.strokeStyle = 'rgba(0,0,0,0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, size, size);

        // Action arrow
        ctx.font = 'bold 26px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        if (GOAL[0]===r && GOAL[1]===c) {
          ctx.fillText('G', x+size/2, y+size/2);
        } else if (TRAPS.has(key)) {
          ctx.fillText('☠', x+size/2, y+size/2);
        } else {
          ctx.fillText(ACTION_SYMBOLS[policy[idx]], x+size/2, y+size/2);
        }

        // Q-value label
        ctx.font = '10px monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fillText(maxQ[idx].toFixed(2), x+size/2, y+size-10);
      }
    }

    // Episode badge
    if (episode) {
      ctx.fillStyle = 'rgba(13,17,23,0.75)';
      ctx.fillRect(6, 6, 130, 26);
      ctx.fillStyle = '#58a6ff';
      ctx.font = '13px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(`Episode: ${episode}`, 12, 12);
    }

    this._tick++;
  }

  startShimmer(getState) {
    this._animating = true;
    const loop = () => {
      if (!this._animating) return;
      const s = getState();
      if (s) this.draw(s.maxQ, s.policy, s.episode);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stopShimmer() { this._animating = false; }
}
