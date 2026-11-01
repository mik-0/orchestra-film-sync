// Trial generation, result logging, and export.

import { buildOffsets } from "./config";

export type Mode = "rating" | "trigger";

// 3-level ordinal scale (Dutch), plus "nvt" = timing irrelevant for this scene
// change (not a judgement of the offset; it has no scored value and is excluded
// from the acceptability curve).
export type Rating = "acceptabel" | "twijfelachtig" | "niet" | "nvt";
export const RATING_VALUE: Record<Rating, number | null> = {
  acceptabel: 1,
  twijfelachtig: 0,
  niet: -1,
  nvt: null,
};
export const RATING_LABEL: Record<Rating, string> = {
  acceptabel: "Acceptabel",
  twijfelachtig: "Twijfelachtig",
  niet: "Niet acceptabel",
  nvt: "Maakt niet uit",
};

export interface Trial {
  index: number; // presentation order (0-based)
  transitionKey: string; // which scene transition this trial tests
  offset: number; // cut offset relative to T, seconds (rating mode)
  repeat: number; // which repeat of this (transition, offset)
}

export interface ResultRow {
  sessionId: string;
  judge: string;
  mode: Mode;
  transition: string;
  cutTime: number; // true T, seconds
  baseAdjust: number; // live nudge applied to T, seconds
  order: number; // presentation index
  offset: number | null; // rating mode: the variant offset; trigger mode: null
  triggerOffset: number | null; // trigger mode: (trigger time - T), seconds
  rating: Rating | null;
  ratingValue: number | null;
  note: string;
  isoTime: string;
}

// Block-based rating trials: one transition at a time (music fixed per block), in
// the order given (chronological / film order), with the offsets shuffled WITHIN
// each block so the cut timing can't be anticipated.
export function buildTrials(transitionKeys: string[], magnitudes: number[], repeats: number): Trial[] {
  const offsets = buildOffsets(magnitudes);
  const trials: Trial[] = [];
  for (const transitionKey of transitionKeys) {
    const block: Trial[] = [];
    for (const offset of offsets) {
      for (let r = 0; r < repeats; r++) block.push({ index: 0, transitionKey, offset, repeat: r });
    }
    shuffle(block); // shuffle offsets within the block
    trials.push(...block);
  }
  trials.forEach((t, i) => (t.index = i));
  return trials;
}

// Self-trigger trials, blocked per transition, in the given (chronological) order.
export function buildTriggerTrials(transitionKeys: string[], runs: number): Trial[] {
  const trials: Trial[] = [];
  for (const transitionKey of transitionKeys) {
    for (let r = 0; r < runs; r++) trials.push({ index: 0, transitionKey, offset: 0, repeat: r });
  }
  trials.forEach((t, i) => (t.index = i));
  return trials;
}

export function shuffle<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

export function makeSessionId(label: string): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  // Seconds + a short random suffix guarantee uniqueness: two sessions started in
  // the same minute must NOT share an id, or the (session_id, ordering) upsert would
  // overwrite one session's rows with the other's.
  const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  const rand = Math.random().toString(36).slice(2, 6);
  return `${stamp}-${rand}_${label}`;
}

const LS_KEY = "acceptability-study-results";

export function loadSaved(): ResultRow[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as ResultRow[]) : [];
  } catch {
    return [];
  }
}

export function saveAll(rows: ResultRow[]): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(rows));
  } catch {
    // localStorage may be unavailable; export is the backup.
  }
}

export function toCSV(rows: ResultRow[]): string {
  const cols: (keyof ResultRow)[] = [
    "sessionId", "judge", "mode", "transition", "cutTime", "baseAdjust",
    "order", "offset", "triggerOffset", "rating", "ratingValue", "note", "isoTime",
  ];
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const head = cols.join(",");
  const body = rows.map((r) => cols.map((c) => esc(r[c])).join(",")).join("\n");
  return head + "\n" + body + "\n";
}

export function download(filename: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Overall acceptability curve: pool all transitions by offset. The boundary is
// where the mean rating flips from + to - on each side.
export function summariseByOffset(rows: ResultRow[]) {
  const rel = rows.filter((r) => r.mode === "rating" && r.rating);
  const byOffset = new Map<number, number[]>();
  for (const r of rel) {
    if (r.offset == null || r.ratingValue == null) continue;
    if (!byOffset.has(r.offset)) byOffset.set(r.offset, []);
    byOffset.get(r.offset)!.push(r.ratingValue);
  }
  return Array.from(byOffset.keys()).sort((a, b) => a - b).map((o) => {
    const vals = byOffset.get(o)!;
    return { offset: o, n: vals.length, mean: vals.reduce((a, b) => a + b, 0) / vals.length };
  });
}

// Transitions the judge marked as timing-irrelevant ("maakt niet uit"). Reported
// separately so they can be read as the director's "this cut needn't land sharply"
// signal, and kept out of the pooled acceptability curve.
export function timingIrrelevant(rows: ResultRow[]) {
  const byT = new Map<string, { total: number; nvt: number }>();
  for (const r of rows) {
    if (r.mode !== "rating") continue;
    const e = byT.get(r.transition) ?? { total: 0, nvt: 0 };
    e.total++;
    if (r.rating === "nvt") e.nvt++;
    byT.set(r.transition, e);
  }
  return Array.from(byT.entries())
    .filter(([, e]) => e.nvt > 0)
    .map(([transition, e]) => ({ transition, nvt: e.nvt, total: e.total }))
    .sort((a, b) => a.transition.localeCompare(b.transition));
}

// Coverage per transition (how many trials each got this session).
export function countByTransition(rows: ResultRow[]) {
  const m = new Map<string, number>();
  for (const r of rows) m.set(r.transition, (m.get(r.transition) ?? 0) + 1);
  return Array.from(m.entries()).map(([transition, n]) => ({ transition, n }))
    .sort((a, b) => a.transition.localeCompare(b.transition));
}
