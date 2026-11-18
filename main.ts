// gta6-twitch-ccu-forecast-01 — v2 entrypoint.
// Donor-weighted log-ratio quantile forecaster (Node 22, stdlib only).
//
// Method: for each case we forecast hours 2..24 of launch-day Twitch CCU
// from the case's hour-1 print p1. Comparable launch-day curves are
// smoothed (3h moving average on log viewers), pooled by size similarity
// to p1, kernel-weighted, and combined per hour with a weighted least
// squares location (log-ratio regressed on size gap, evaluated at the
// case's own size) plus residual-quantile spread pooled over neighbouring
// hours. Falls back to a widening persistence baseline whenever data is
// thin or anything fails.
import * as fs from "node:fs";
import * as path from "node:path";

const LEVELS: number[] = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z: number[] = [-1.645, -0.674, 0, 0.674, 1.645];
const HOUR_MS = 3600_000;
const SPREAD_SCALE = 0.9;
const KERNEL_BW = 1.0;
const BLEND = 0.5;
const MIN_POOL = 5;
const MIN_HOUR_DONORS = 3;

type Row = { hour: number; q: number[] };

type Donor = {
  lnV1: number;
  // Smoothed log-ratio per hour: smoothed ln(viewers) minus ln(v1).
  ratio: Map<number, number>;
};

function parseMs(iso: string): number {
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : NaN;
}

// Weighted quantile of (value, weight) pairs.
function wquantile(pairs: Array<[number, number]>, tau: number): number {
  const sorted = pairs.slice().sort((a, b) => a[0] - b[0]);
  if (sorted.length === 0) return 0;
  let total = 0;
  for (const [, w] of sorted) total += w;
  if (!(total > 0)) return sorted[Math.floor(sorted.length / 2)][0];
  const cut = tau * total;
  let acc = 0;
  for (const [v, w] of sorted) {
    acc += w;
    if (acc >= cut) return v;
  }
  return sorted[sorted.length - 1][0];
}

function wmedian(pairs: Array<[number, number]>): number {
  return wquantile(pairs, 0.5);
}

// Persistence baseline row for hour h: q_tau = p1 * exp(z_tau * 0.15 * sqrt(h-1)).
function persistenceRow(p1: number, h: number): number[] {
  const s = 0.15 * Math.sqrt(h - 1);
  return Z.map((z) => p1 * Math.exp(z * s));
}

function baselineHours(p1: number): Row[] {
  const rows: Row[] = [];
  for (let h = 2; h <= 24; h++) rows.push({ hour: h, q: finalize(persistenceRow(p1, h)) });
  return rows;
}

// Clamp to (0, 1e9], sort non-decreasing, round to 2 decimals.
// Sanitise BEFORE sorting: Array.sort with NaN in the mix orders
// unpredictably, so no non-finite value may reach the comparator.
function finalize(qs: number[]): number[] {
  const clean = qs.map((q) => {
    let v = q;
    if (!Number.isFinite(v)) v = 1e9;
    if (v > 1e9) v = 1e9;
    if (v <= 0) v = 0.01;
    return v;
  });
  clean.sort((a, b) => a - b);
  return clean.map((v) => {
    const r = Math.round(v * 100) / 100;
    return r <= 0 ? 0.01 : r;
  });
}

function hourOneViewers(data: any): number {
  if (Array.isArray(data.known)) {
    for (const k of data.known) {
      if (k && k.hour === 1 && typeof k.viewers === "number" && k.viewers > 0) {
        return k.viewers;
      }
    }
  }
  return 1;
}

// Build one donor's smoothed log-ratio curve. Returns null when unusable.
function buildDonor(comp: any): Donor | null {
  if (!comp || typeof comp.day_start !== "string" || !Array.isArray(comp.hourly)) return null;
  const dayStart = parseMs(comp.day_start);
  if (!Number.isFinite(dayStart)) return null;
  const pts = new Map<number, number>();
  for (const p of comp.hourly) {
    if (!p || typeof p.from !== "string" || typeof p.viewers !== "number") continue;
    if (p.viewers <= 0) continue;
    const t = parseMs(p.from);
    if (Number.isFinite(t)) pts.set(t, p.viewers);
  }
  // Hour h value = point at day_start + (h-1) hours, exact match only.
  const logVals = new Map<number, number>();
  for (let h = 1; h <= 24; h++) {
    const v = pts.get(dayStart + (h - 1) * HOUR_MS);
    if (v !== undefined && v > 0) logVals.set(h, Math.log(v));
  }
  const v1 = pts.get(dayStart);
  if (v1 === undefined || v1 < 100) return null;
  const lnV1 = Math.log(v1);
  // 3-hour moving average on log viewers, then subtract ln(v1).
  const ratio = new Map<number, number>();
  for (const h of logVals.keys()) {
    let sum = 0;
    let n = 0;
    for (const hh of [h - 1, h, h + 1]) {
      const lv = logVals.get(hh);
      if (lv !== undefined) {
        sum += lv;
        n++;
      }
    }
    if (n > 0) ratio.set(h, sum / n - lnV1);
  }
  return { lnV1, ratio };
}

function buildDonors(data: any): Donor[] {
  const out: Donor[] = [];
  if (!Array.isArray(data.comparables)) return out;
  for (const comp of data.comparables) {
    const d = buildDonor(comp);
    if (d) out.push(d);
  }
  return out;
}

// Size-limited pool: |ln(v1_donor) - ln(p1)| < threshold, widened until >= 5.
function selectPool(donors: Donor[], lnP1: number): Donor[] {
  for (const t of [1.0, 1.5, 2.0, Infinity]) {
    const pool = donors.filter((d) => Math.abs(d.lnV1 - lnP1) < t);
    if (pool.length >= MIN_POOL || t === Infinity) return pool;
  }
  return donors;
}

// Gaussian kernel weights on the size gap, blended toward uniform.
function kernelWeights(pool: Donor[], lnP1: number): number[] {
  const raw = pool.map((d) => {
    const dl = (d.lnV1 - lnP1) / KERNEL_BW;
    return Math.exp(-0.5 * dl * dl);
  });
  let mean = 0;
  for (const w of raw) mean += w;
  mean /= raw.length;
  return raw.map((w) => (1 - BLEND) * w + BLEND * mean);
}

function forecastCase(data: any): Row[] {
  const p1 = hourOneViewers(data);
  try {
    return buildForecast(data, p1);
  } catch {
    return baselineHours(p1);
  }
}

function buildForecast(data: any, p1: number): Row[] {
  const lnP1 = Math.log(p1);
  const donors = buildDonors(data);
  const pool = selectPool(donors, lnP1);
  if (pool.length === 0) return baselineHours(p1);
  const ws = kernelWeights(pool, lnP1);

  // Per-hour WLS location of the log-ratio on the size gap, at x = 0.
  const loc = new Map<number, number>();
  for (let h = 2; h <= 24; h++) {
    const pairs: Array<[number, number, number]> = []; // [ratio, weight, x]
    for (let i = 0; i < pool.length; i++) {
      const r = pool[i].ratio.get(h);
      if (r !== undefined) pairs.push([r, ws[i], pool[i].lnV1 - lnP1]);
    }
    if (pairs.length < MIN_HOUR_DONORS) continue;
    let sw = 0;
    let mx = 0;
    let my = 0;
    for (const [y, w, x] of pairs) {
      sw += w;
      mx += w * x;
      my += w * y;
    }
    mx /= sw;
    my /= sw;
    let sxx = 0;
    let sxy = 0;
    for (const [y, w, x] of pairs) {
      sxx += w * (x - mx) * (x - mx);
      sxy += w * (x - mx) * (y - my);
    }
    if (sxx < 1e-12) {
      // All donors the same size: fall back to the weighted median.
      loc.set(h, wmedian(pairs.map(([y, w]) => [y, w] as [number, number])));
    } else {
      const b = sxy / sxx;
      const a = my - b * mx;
      loc.set(h, a);
    }
  }

  const rows: Row[] = [];
  for (let h = 2; h <= 24; h++) {
    const m = loc.get(h);
    if (m === undefined) {
      rows.push({ hour: h, q: finalize(persistenceRow(p1, h)) });
      continue;
    }
    // Residuals pooled over hours h-2..h+2, each against its own location.
    const res: Array<[number, number]> = [];
    for (let hh = Math.max(2, h - 2); hh <= Math.min(24, h + 2); hh++) {
      const mhh = loc.get(hh);
      if (mhh === undefined) continue;
      let count = 0;
      for (let i = 0; i < pool.length; i++) {
        if (pool[i].ratio.has(hh)) count++;
      }
      if (count < MIN_HOUR_DONORS) continue;
      for (let i = 0; i < pool.length; i++) {
        const r = pool[i].ratio.get(hh);
        if (r !== undefined) res.push([r - mhh, ws[i]]);
      }
    }
    let use = res;
    if (use.length < 5) {
      // Too few pooled residuals: use this hour's own residuals only.
      use = [];
      for (let i = 0; i < pool.length; i++) {
        const r = pool[i].ratio.get(h);
        if (r !== undefined) use.push([r - m, ws[i]]);
      }
    }
    const qs = LEVELS.map((tau) => p1 * Math.exp(m + SPREAD_SCALE * wquantile(use, tau)));
    rows.push({ hour: h, q: finalize(qs) });
  }
  return rows;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const raw = fs.readFileSync(path.join(inputDir, file), "utf8");
    const data = JSON.parse(raw);
    const id = typeof data.id === "string" && data.id.length > 0 ? data.id : file.replace(/\.in$/, "");
    const hours = forecastCase(data);
    fs.writeFileSync(path.join(outputDir, id + ".out"), JSON.stringify({ hours }));
  }
}

main();
