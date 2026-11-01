// Firebird study configuration.
// Segment start times in conductor-video seconds, parsed from
// cue-detect-video/firebird_timeline.txt (MM:SS:FF at 25 fps).

export interface Segment {
  id: string;
  start: number; // seconds in conductor-video time
  dur: number; // segment clip duration in seconds (measured with ffprobe)
}

export const SEGMENTS: Segment[] = [
  { id: "133", start: 2.92, dur: 19.84 },
  { id: "134", start: 16.6, dur: 8.8 },
  { id: "135", start: 24.04, dur: 5.24 },
  { id: "136", start: 34.4, dur: 12.0 },
  { id: "137", start: 44.84, dur: 4.76 },
  { id: "138", start: 49.2, dur: 5.84 },
  { id: "139", start: 51.6, dur: 15.96 },
  { id: "140", start: 64.48, dur: 17.4 },
  { id: "141", start: 77.96, dur: 7.04 },
  { id: "142", start: 86.0, dur: 17.44 },
  { id: "143", start: 97.56, dur: 8.84 },
  { id: "144", start: 106.04, dur: 6.28 },
  { id: "145", start: 113.16, dur: 3.72 },
  { id: "146", start: 119.56, dur: 9.32 },
  { id: "147", start: 127.36, dur: 6.72 },
  { id: "148", start: 131.32, dur: 20.0 },
];

export interface Transition {
  key: string; // e.g. "134->135"
  label: string; // e.g. "133 -> 134"
  outgoing: string; // outgoing segment id
  incoming: string; // incoming segment id
  cutTime: number; // true cut moment (= incoming segment start), seconds
}

// Consecutive segment pairs are the 15 selectable scene transitions.
export const TRANSITIONS: Transition[] = [
  // Opening cut: from black/nothing into the first scene (133).
  { key: "start->133", label: "start → 133", outgoing: "", incoming: SEGMENTS[0].id, cutTime: SEGMENTS[0].start },
  ...SEGMENTS.slice(1).map((seg, i) => {
    const prev = SEGMENTS[i];
    return {
      key: `${prev.id}->${seg.id}`,
      label: `${prev.id} → ${seg.id}`,
      outgoing: prev.id,
      incoming: seg.id,
      cutTime: seg.start,
    };
  }),
];

// Largest |offset| any preset uses; the outgoing clip must keep moving until the
// latest possible cut (cutTime + MAX_OFFSET). Keep >= the biggest preset magnitude.
export const MAX_OFFSET = 0.8;

// If an outgoing segment is shorter than the time it must stay on screen, slow its
// playback so it plays through to (just past) the latest cut instead of freezing on
// its last frame. Returns 1.0 when no slowdown is needed.
export function outgoingPlaybackRate(t: Transition): number {
  const out = SEGMENTS.find((s) => s.id === t.outgoing);
  if (!out) return 1;
  const required = t.cutTime - out.start + MAX_OFFSET + 0.2; // on-screen time + small pad
  return out.dur >= required ? 1 : +(out.dur / required).toFixed(3);
}

// Media: served from public/media; override with VITE_MEDIA_BASE for a CDN/Blob.
const RAW_BASE = (import.meta.env.VITE_MEDIA_BASE as string | undefined) ?? "/media/";
export const MEDIA_BASE = RAW_BASE.endsWith("/") ? RAW_BASE : RAW_BASE + "/";
export const conductorSrc = () => `${MEDIA_BASE}conductor.mp4`;
export const segmentSrc = (id: string) => `${MEDIA_BASE}seg_${id}.mp4`;

// Offset magnitudes in seconds (mirrored to +/-), frame-aligned to 25 fps
// (multiples of 0.04 s). The study covers ALL transitions, with a handful of
// offsets per transition, so keep the per-transition set small (5 or 7 offsets).
export const OFFSET_PRESETS: Record<string, number[]> = {
  "5": [0, 0.16, 0.4],         // -> 5 offsets: 0, +/-0.16, +/-0.4
  "7": [0, 0.16, 0.4, 0.8],    // -> 7 offsets: 0, +/-0.16, +/-0.4, +/-0.8 (spans tight & wide boundaries)
};
export const DEFAULT_OFFSET_KEY = "7";

export const DEFAULT_REPEATS = 1;

// Fixed music window around the true cut moment T, identical every trial,
// so only the cut moment (T + offset) changes within it.
export const WINDOW_PRE = 4.0; // seconds of music before T
export const WINDOW_POST = 4.0; // seconds of music after T

export function buildOffsets(magnitudes: number[]): number[] {
  const set = new Set<number>();
  for (const m of magnitudes) {
    set.add(+m.toFixed(3));
    set.add(+(-m).toFixed(3));
  }
  return Array.from(set).sort((a, b) => a - b);
}
