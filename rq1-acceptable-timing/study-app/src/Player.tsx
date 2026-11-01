import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { conductorSrc, segmentSrc, outgoingPlaybackRate, SEGMENTS, Transition, WINDOW_PRE, WINDOW_POST } from "./config";
import { Mode } from "./study";

interface PlayerProps {
  transition: Transition;
  baseAdjust: number; // live nudge to the true cut moment T
  mode: Mode;
  offset: number; // rating mode: cut at T + offset
  trialId: number; // changes per trial -> reset
  onEnded: (triggerOffset: number | null) => void;
  onPlayStart?: () => void; // fired when playback (re)starts, so the parent can hide the rating UI
}

// Conductor video is the master clock (its AAC track is the orchestra audio).
// Film segments are slaved to it; the cut switches outgoing -> incoming at the
// cut moment. The music window [T-PRE, T+POST] is identical every trial, so only
// the cut moment (T + offset) changes within it.
export default function Player({ transition, baseAdjust, mode, offset, trialId, onEnded, onPlayStart }: PlayerProps) {
  const conductor = useRef<HTMLVideoElement>(null);
  const filmOut = useRef<HTMLVideoElement>(null);
  const filmIn = useRef<HTMLVideoElement>(null);

  const cutFired = useRef(false);
  const triggerTime = useRef<number | null>(null);
  const foStarted = useRef(false);
  const rafId = useRef<number>(0);

  const [status, setStatus] = useState<"idle" | "playing" | "ended">("idle");
  const [showIn, setShowIn] = useState(false);

  const T = transition.cutTime + baseAdjust;
  // Opening transition has no outgoing scene (cut from black into the first scene).
  const hasOut = transition.outgoing !== "";
  const outStart = useMemo(
    () => SEGMENTS.find((s) => s.id === transition.outgoing)?.start ?? 0,
    [transition],
  );
  // Slow the outgoing clip if it is shorter than its on-screen time, so it never
  // freezes before the cut.
  const outRate = useMemo(() => outgoingPlaybackRate(transition), [transition]);
  // Pre-roll, capped to the outgoing scene's real on-screen time so a short scene
  // (e.g. 138->139) starts from frame 0 once instead of playing early then snapping back.
  const pre = useMemo(() => Math.min(WINDOW_PRE, T - outStart), [T, outStart]);
  const windowStart = Math.max(0, T - pre);

  const fireCut = useCallback(() => {
    cutFired.current = true;
    setShowIn(true);
    const fo = filmOut.current, fi = filmIn.current;
    if (fo) fo.pause();
    if (fi) {
      fi.currentTime = 0;
      fi.play().catch(() => {});
    }
  }, []);

  const endTrial = useCallback(() => {
    cancelAnimationFrame(rafId.current);
    conductor.current?.pause();
    filmOut.current?.pause();
    filmIn.current?.pause();
    setStatus("ended");
    const tOff = mode === "trigger" && triggerTime.current != null
      ? +(triggerTime.current - T).toFixed(3)
      : null;
    onEnded(tOff);
  }, [mode, T, onEnded]);

  const loop = useCallback(() => {
    const c = conductor.current, fo = filmOut.current;
    if (!c) return;
    const t = c.currentTime;
    const cutTarget = mode === "rating" ? T + offset : triggerTime.current ?? Infinity;

    // Start the outgoing film only once the master clock is actually advancing,
    // so it can't race ahead during the conductor's play/seek startup latency
    // (which caused a small backward "snap" otherwise).
    if (hasOut && !cutFired.current && fo && !foStarted.current && t > windowStart + 0.04) {
      foStarted.current = true;
      fo.play().catch(() => {});
    }

    if (!cutFired.current && t >= cutTarget) {
      fireCut();
    } else if (!cutFired.current && fo && foStarted.current) {
      const desired = (t - outStart) * outRate; // slave outgoing film, scaled by its slowdown
      if (desired >= 0 && Number.isFinite(fo.duration) && desired < fo.duration - 0.05) {
        if (Math.abs(fo.currentTime - desired) > 0.15) fo.currentTime = desired;
      }
    }

    if (t >= T + WINDOW_POST) {
      endTrial();
      return;
    }
    rafId.current = requestAnimationFrame(loop);
  }, [mode, T, offset, outStart, outRate, windowStart, hasOut, fireCut, endTrial]);

  const play = useCallback(() => {
    const c = conductor.current, fo = filmOut.current, fi = filmIn.current;
    if (!c || !fo || !fi) return;
    cutFired.current = false;
    triggerTime.current = null;
    setShowIn(false);
    foStarted.current = false;
    c.currentTime = windowStart;
    if (hasOut) {
      fo.playbackRate = outRate;
      fo.currentTime = Math.max(0, (T - pre - outStart) * outRate);
      fo.pause(); // outgoing film is started in the loop, once the master clock rolls
    }
    fi.currentTime = 0;
    fi.pause();
    setStatus("playing");
    onPlayStart?.();
    c.play().catch(() => {}); // conductor (with audio) is the master clock
    cancelAnimationFrame(rafId.current);
    rafId.current = requestAnimationFrame(loop);
  }, [T, pre, outStart, outRate, windowStart, hasOut, loop, onPlayStart]);

  const triggerNow = useCallback(() => {
    const c = conductor.current;
    if (mode !== "trigger" || status !== "playing" || cutFired.current || triggerTime.current != null || !c) return;
    triggerTime.current = c.currentTime;
  }, [mode, status]);

  // Reset whenever the trial changes.
  useEffect(() => {
    cancelAnimationFrame(rafId.current);
    cutFired.current = false;
    triggerTime.current = null;
    foStarted.current = false;
    setShowIn(false);
    setStatus("idle");
    conductor.current?.pause();
    filmOut.current?.pause();
    filmIn.current?.pause();
  }, [trialId]);

  // Space plays/replays when idle/ended, or triggers the cut while playing (trigger mode).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      e.preventDefault();
      if (mode === "trigger" && status === "playing") triggerNow();
      else if (status !== "playing") play();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, status, play, triggerNow]);

  useEffect(() => () => cancelAnimationFrame(rafId.current), []);

  return (
    <div className="player">
      <div className="panes">
        <div className="pane film">
          {/* Both stacked and decoded; the cut is an instant opacity flip, not a
              display swap + cold play(), so the new scene appears without lag. */}
          <video ref={filmOut} src={hasOut ? segmentSrc(transition.outgoing) : undefined} muted playsInline preload="auto"
                 className="stacked" style={{ opacity: showIn ? 0 : 1 }} />
          <video ref={filmIn} src={segmentSrc(transition.incoming)} muted playsInline preload="auto"
                 className="stacked" style={{ opacity: showIn ? 1 : 0 }} />
          <span className="tag">Film</span>
        </div>
        <div className="pane conductor">
          <video ref={conductor} src={conductorSrc()} playsInline preload="auto" />
          <span className="tag">Dirigent + muziek</span>
        </div>
      </div>

      <div className="player-controls">
        {status === "idle" && (
          <button className="btn-primary" onClick={play}>▶ Afspelen <kbd>spatie</kbd></button>
        )}
        {status === "playing" && mode === "trigger" && (
          <button className="btn-trigger" onClick={triggerNow}>● Nu overgang <kbd>spatie</kbd></button>
        )}
        {status === "playing" && mode === "rating" && (
          <span className="hint">Kijk en luister naar de overgang.</span>
        )}
        {status === "ended" && (
          <button className="btn-secondary" onClick={play}>↻ Opnieuw afspelen <kbd>spatie</kbd></button>
        )}
      </div>
    </div>
  );
}
