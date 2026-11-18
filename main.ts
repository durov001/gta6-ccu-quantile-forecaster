// GTA VI Twitch CCU forecast — quantile forecaster (makefun/twitch-ccu-case/v1).
//
// Metode: donor-weighted log-ratio quantiles.
//  - p1 = viewers jam ke-1 (known row).
//  - Donor = comparable dengan kurva hari rilis (jam 1..24 dari hourly points
//    yang `from`-nya cocok persis dengan day_start + (h-1) jam).
//  - Pool dibatasi jarak |ln v1_donor - ln p1| < 1.0, dilebarkan (1.5, 2.0,
//    semua) sampai >= 5 donor.
//  - Bobot kernel Gaussian pada ln v1 (sigma 0.75), di-blend 0.7/0.3 dengan
//    rata-rata bobot pool.
//  - Per jam h: m_h = median tertimbang r = ln(v_h / v1_donor); spread dari
//    kuantil tertimbang residual jam h-1..h+1; q_tau = p1 * exp(m_h + Q_tau).
//  - Jam dengan < 3 donor: fallback persistence
//    q_tau = p1 * exp(z_tau * 0.15 * sqrt(h-1)).
//  - Program SELALU menulis output valid: error apa pun -> baseline persistence.
//
// Stdlib Node saja, tanpa network, deterministik (tanpa random / Date.now).

import * as fs from "node:fs";
import * as path from "node:path";

const LEVELS: number[] = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z: number[] = [-1.645, -0.674, 0, 0.674, 1.645];
const HOURS: number[] = [];
for (let h = 2; h <= 24; h++) HOURS.push(h);
const MS_PER_HOUR = 3600000;
const MAX_Q = 1e9;
const MIN_Q = 0.01;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number;
}

interface Comparable {
  name?: string;
  day_start: string;
  unlock?: string;
  platforms?: unknown;
  source?: string;
  hourly: HourlyPoint[];
}

interface KnownRow {
  hour: number;
  from?: string;
  until?: string;
  viewers: number;
  channels?: number;
  reads?: number;
}

interface CaseData {
  id: string;
  game?: { name?: string; twitch_id?: string };
  day_start?: string;
  checkpoint?: string;
  levels?: number[];
  known?: KnownRow[];
  forecast?: Array<{ hour: number; from?: string; until?: string }>;
  history?: unknown;
  history_hourly?: unknown;
  streams?: unknown;
  categories_hourly?: unknown;
  comparables?: Comparable[] | null;
  notes?: unknown;
}

interface Donor {
  v1: number;
  byHour: Map<number, number>;
  w: number;
}

interface WeightedValue {
  v: number;
  w: number;
}

// Weighted quantile: urutkan nilai, bobot kumulatif, ambil nilai pertama saat
// kumulatif >= tau * total bobot.
function weightedQuantile(pairs: WeightedValue[], tau: number): number {
  const sorted = pairs.slice().sort((a, b) => a.v - b.v);
  let total = 0;
  for (const p of sorted) total += p.w;
  const target = tau * total;
  let cum = 0;
  for (const p of sorted) {
    cum += p.w;
    if (cum >= target) return p.v;
  }
  return sorted[sorted.length - 1].v;
}

// Baseline persistence: q_tau = p1 * exp(z_tau * 0.15 * sqrt(h-1)).
function persistenceQuantiles(p1: number, h: number): number[] {
  const s = 0.15 * Math.sqrt(h - 1);
  return Z.map((z) => p1 * Math.exp(z * s));
}

// Bangun kurva hari rilis donor: jam h = viewers dari hourly point yang
// `from`-nya == day_start + (h-1) jam (cocok timestamp persis, tanpa
// interpolasi). Return null bila tidak ada jam 1 atau viewers jam 1 < 100.
function buildDonor(c: Comparable): Donor | null {
  if (!c || !Array.isArray(c.hourly)) return null;
  const startMs = Date.parse(c.day_start);
  if (!Number.isFinite(startMs)) return null;
  const byMs = new Map<number, number>();
  for (const p of c.hourly) {
    if (!p || typeof p.from !== "string") continue;
    const t = Date.parse(p.from);
    if (!Number.isFinite(t)) continue;
    if (typeof p.viewers === "number" && Number.isFinite(p.viewers)) {
      byMs.set(t, p.viewers);
    }
  }
  const byHour = new Map<number, number>();
  for (let h = 1; h <= 24; h++) {
    const v = byMs.get(startMs + (h - 1) * MS_PER_HOUR);
    if (v !== undefined) byHour.set(h, v);
  }
  const v1 = byHour.get(1);
  if (v1 === undefined || v1 < 100) return null;
  return { v1, byHour, w: 0 };
}

// Pool terbatas ukuran: ambang 1.0, lalu 1.5, 2.0, lalu semua donor,
// sampai >= 5 donor (atau habis).
function selectPool(donors: Donor[], p1: number): Donor[] {
  const lnP1 = Math.log(p1);
  const thresholds = [1.0, 1.5, 2.0];
  for (const t of thresholds) {
    const pool = donors.filter((d) => Math.abs(Math.log(d.v1) - lnP1) < t);
    if (pool.length >= 5) return pool;
  }
  return donors;
}

// Bobot kernel Gaussian pada ln v1 (sigma 0.75), blend 0.7*w + 0.3*mean(w).
function assignWeights(pool: Donor[], p1: number): void {
  const lnP1 = Math.log(p1);
  let sum = 0;
  for (const d of pool) {
    const z = (Math.log(d.v1) - lnP1) / 0.75;
    d.w = Math.exp(-0.5 * z * z);
    sum += d.w;
  }
  const mean = pool.length > 0 ? sum / pool.length : 0;
  for (const d of pool) d.w = 0.7 * d.w + 0.3 * mean;
}

// Pasangan (r, w') untuk satu jam: r = ln(v_h / v1_donor).
function hourPairs(pool: Donor[], h: number): WeightedValue[] {
  const pairs: WeightedValue[] = [];
  for (const d of pool) {
    const vh = d.byHour.get(h);
    if (vh !== undefined && vh > 0) {
      pairs.push({ v: Math.log(vh / d.v1), w: d.w });
    }
  }
  return pairs;
}

// Forecast satu jam: median tertimbang + kuantil residual tetangga.
// pairsByHour sudah dihitung sekali per case (jam 2..24) agar pasangan
// (r, w') tiap jam tidak dihitung ulang sebagai tetangga jam lain.
function forecastHour(
  p1: number,
  h: number,
  pairsByHour: Map<number, WeightedValue[]>,
): number[] {
  const pairs = pairsByHour.get(h) ?? [];
  if (pairs.length < 3) return persistenceQuantiles(p1, h);
  const mh = weightedQuantile(pairs, 0.5);

  // Residual dari jam h-1, h, h+1 (yang ada dalam 2..24).
  const residuals: WeightedValue[] = [];
  for (let hh = h - 1; hh <= h + 1; hh++) {
    if (hh < 2 || hh > 24) continue;
    const hp = pairsByHour.get(hh) ?? [];
    if (hp.length < 3) continue;
    const mhh = weightedQuantile(hp, 0.5);
    for (const p of hp) residuals.push({ v: p.v - mhh, w: p.w });
  }
  let res = residuals;
  if (res.length < 5) {
    // Terlalu sedikit residual: pakai residual jam h saja.
    res = pairs.map((p) => ({ v: p.v - mh, w: p.w }));
  }
  return LEVELS.map((tau) => p1 * Math.exp(mh + weightedQuantile(res, tau)));
}

// Clamp ke (0, 1e9], monoton tidak menurun, bulatkan ke 2 desimal.
function finalizeRow(qs: number[]): number[] {
  const clamped: number[] = [];
  let run = 0;
  for (const q of qs) {
    let v = q;
    if (!Number.isFinite(v) || v <= 0) v = MIN_Q;
    if (v > MAX_Q) v = MAX_Q;
    if (v < run) v = run;
    run = v;
    clamped.push(v);
  }
  const out: number[] = [];
  run = 0;
  for (const v0 of clamped) {
    let v = Math.round(v0 * 100) / 100;
    if (!Number.isFinite(v) || v <= 0) v = MIN_Q;
    if (v > MAX_Q) v = MAX_Q;
    if (v < run) v = run;
    run = v;
    out.push(v);
  }
  return out;
}

interface HourOut {
  hour: number;
  q: number[];
}

interface CaseOut {
  hours: HourOut[];
}

function persistenceCase(p1: number): CaseOut {
  return {
    hours: HOURS.map((h) => ({ hour: h, q: finalizeRow(persistenceQuantiles(p1, h)) })),
  };
}

function extractP1(c: CaseData): number {
  let p1 = 1000;
  if (Array.isArray(c.known) && c.known.length > 0) {
    const row1 = c.known.find((r) => r && r.hour === 1);
    const row = row1 ?? c.known[0];
    if (row && typeof row.viewers === "number" && Number.isFinite(row.viewers)) {
      p1 = row.viewers;
    }
  }
  if (!Number.isFinite(p1) || p1 < 1) p1 = 1;
  return p1;
}

function forecastCase(c: CaseData): CaseOut {
  const p1 = extractP1(c);
  const donors: Donor[] = [];
  const comps = Array.isArray(c.comparables) ? c.comparables : [];
  for (const comp of comps) {
    const d = buildDonor(comp);
    if (d) donors.push(d);
  }
  const pool = selectPool(donors, p1);
  assignWeights(pool, p1);
  const pairsByHour = new Map<number, WeightedValue[]>();
  for (let hh = 2; hh <= 24; hh++) pairsByHour.set(hh, hourPairs(pool, hh));
  return {
    hours: HOURS.map((h) => ({
      hour: h,
      q: finalizeRow(forecastHour(p1, h, pairsByHour)),
    })),
  };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outDir, { recursive: true });
  const files = fs.readdirSync(inDir).filter((f) => f.endsWith(".in")).sort();
  for (const file of files) {
    const stem = file.slice(0, -3);
    const outPath = path.join(outDir, stem + ".out");
    let payload: CaseOut;
    try {
      const raw = fs.readFileSync(path.join(inDir, file), "utf-8");
      const c = JSON.parse(raw) as CaseData;
      payload = forecastCase(c);
    } catch {
      // Last resort: baseline persistence dengan p1 yang masih bisa diselamatkan.
      let p1 = 1000;
      try {
        const raw = fs.readFileSync(path.join(inDir, file), "utf-8");
        const c = JSON.parse(raw) as CaseData;
        p1 = extractP1(c);
      } catch {
        // p1 tetap 1000.
      }
      payload = persistenceCase(p1);
    }
    fs.writeFileSync(outPath, JSON.stringify(payload));
  }
}

main();
