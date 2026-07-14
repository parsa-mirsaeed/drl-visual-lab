/**
 * charts.js
 * Lightweight animated chart renderer — pure Canvas, zero deps.
 * Supports: line chart with fill, bar (histogram), animated drawing.
 */

'use strict';

const COLORS = {
  reward: '#388bfd',
  smooth: '#ffa657',
  loss:   '#f85149',
  eps:    '#3fb950',
  dist:   '#bc8cff',
};

export class LineChart {
  constructor(canvasId, color = '#388bfd', label = '') {
    this.canvas = document.getElementById(canvasId);
    this.ctx    = this.canvas.getContext('2d');
    this.color  = color;
    this.label  = label;
    this.data   = [];
    this.smooth = [];
    this._tick  = 0;
  }

  push(value) {
    this.data.push(value);
    // Exponential moving average
    const alpha = 0.08;
    const prev  = this.smooth.length ? this.smooth[this.smooth.length-1] : value;
    this.smooth.push(prev + alpha * (value - prev));
  }

  draw() {
    const { canvas, ctx, data, smooth, color } = this;
    const W = canvas.width, H = canvas.height;
    const PAD = { t:10, r:10, b:28, l:44 };
    const w = W - PAD.l - PAD.r;
    const h = H - PAD.t - PAD.b;

    ctx.clearRect(0, 0, W, H);
    if (data.length < 2) return;

    let mn = Infinity, mx = -Infinity;
    for (const v of data) { if (v<mn) mn=v; if (v>mx) mx=v; }
    const range = mx - mn || 1;

    const px = (i) => PAD.l + (i / (data.length-1)) * w;
    const py = (v) => PAD.t + h - ((v - mn) / range) * h;

    // Grid lines
    ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = PAD.t + (i/4)*h;
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l+w, y); ctx.stroke();
      const label = (mx - (i/4)*(mx-mn)).toFixed(1);
      ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace';
      ctx.textAlign = 'right'; ctx.fillText(label, PAD.l-4, y+3);
    }

    // Animated fill under raw line
    const animatedN = Math.ceil(data.length * Math.min(1, (this._tick * 0.04)));
    if (animatedN > 1) {
      ctx.beginPath();
      ctx.moveTo(px(0), py(data[0]));
      for (let i = 1; i < animatedN; i++) ctx.lineTo(px(i), py(data[i]));
      ctx.lineTo(px(animatedN-1), PAD.t+h);
      ctx.lineTo(px(0), PAD.t+h);
      ctx.closePath();
      ctx.fillStyle = color.replace(')', ',0.12)').replace('rgb','rgba');
      ctx.fill();

      // Raw line
      ctx.beginPath();
      ctx.moveTo(px(0), py(data[0]));
      for (let i = 1; i < animatedN; i++) ctx.lineTo(px(i), py(data[i]));
      ctx.strokeStyle = color.replace(')', ',0.35)').replace('rgb','rgba');
      ctx.lineWidth = 1.2; ctx.stroke();

      // Smooth line
      ctx.beginPath();
      ctx.moveTo(px(0), py(smooth[0]));
      for (let i = 1; i < animatedN; i++) ctx.lineTo(px(i), py(smooth[i]));
      ctx.strokeStyle = COLORS.smooth; ctx.lineWidth = 2.2; ctx.stroke();

      // Live dot
      const last = animatedN-1;
      const dotGrd = ctx.createRadialGradient(px(last), py(smooth[last]), 0, px(last), py(smooth[last]), 8);
      dotGrd.addColorStop(0, COLORS.smooth);
      dotGrd.addColorStop(1, 'transparent');
      ctx.fillStyle = dotGrd; ctx.beginPath();
      ctx.arc(px(last), py(smooth[last]), 8, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.beginPath();
      ctx.arc(px(last), py(smooth[last]), 3, 0, Math.PI*2); ctx.fill();
    }

    this._tick++;
  }

  reset() { this.data = []; this.smooth = []; this._tick = 0; }
}

export class HistChart {
  constructor(canvasId, color = '#bc8cff') {
    this.canvas = document.getElementById(canvasId);
    this.ctx    = this.canvas.getContext('2d');
    this.color  = color;
    this.data   = [];
    this._tick  = 0;
  }

  push(v) { this.data.push(v); }

  draw() {
    const { canvas, ctx, data, color } = this;
    const W = canvas.width, H = canvas.height;
    const PAD = { t:10, r:10, b:28, l:44 };
    if (data.length < 2) return;

    const BINS = 18;
    let mn = Infinity, mx = -Infinity;
    for (const v of data) { if (v<mn) mn=v; if (v>mx) mx=v; }
    const bw = (mx - mn) / BINS || 1;
    const counts = new Array(BINS).fill(0);
    for (const v of data) {
      const i = Math.min(BINS-1, Math.floor((v - mn) / bw));
      counts[i]++;
    }
    const maxC = Math.max(...counts) || 1;

    ctx.clearRect(0, 0, W, H);
    const cellW = (W - PAD.l - PAD.r) / BINS;
    const h = H - PAD.t - PAD.b;
    const anim = Math.min(1, this._tick * 0.05);

    counts.forEach((c, i) => {
      const bh = (c / maxC) * h * anim;
      const x  = PAD.l + i * cellW;
      const y  = PAD.t + h - bh;

      // Gradient bar
      const grd = ctx.createLinearGradient(x, y, x, y+bh);
      grd.addColorStop(0, color);
      grd.addColorStop(1, color.replace(')', ',0.2)').replace('rgb','rgba'));
      ctx.fillStyle = grd;
      ctx.fillRect(x+1, y, cellW-2, bh);
    });

    this._tick++;
  }

  reset() { this.data = []; this._tick = 0; }
}
