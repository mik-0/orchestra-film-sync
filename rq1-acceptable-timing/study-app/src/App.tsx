import { useEffect, useState } from "react";
import Player from "./Player";
import { postTrial } from "./api";
import { OFFSET_PRESETS, DEFAULT_OFFSET_KEY, DEFAULT_REPEATS, TRANSITIONS, buildOffsets } from "./config";
import {
  Mode, Rating, RATING_LABEL, RATING_VALUE, ResultRow, Trial,
  buildTrials, buildTriggerTrials, countByTransition, download, loadSaved,
  makeSessionId, saveAll, summariseByOffset, timingIrrelevant, toCSV,
} from "./study";

type Screen = "setup" | "run" | "done";
type Phase = "play" | "rating" | "triggered";

export default function App() {
  const [screen, setScreen] = useState<Screen>("setup");

  // setup config
  const [judge, setJudge] = useState("");
  const [mode, setMode] = useState<Mode>("rating");
  const [offsetKey, setOffsetKey] = useState(DEFAULT_OFFSET_KEY);
  const [repeats, setRepeats] = useState(DEFAULT_REPEATS);
  const [runs, setRuns] = useState(1);
  const [previewKey, setPreviewKey] = useState(TRANSITIONS[0].key);

  // run state
  const [trials, setTrials] = useState<Trial[]>([]);
  const [idx, setIdx] = useState(0);
  const [phase, setPhase] = useState<Phase>("play");
  const [lastTrigger, setLastTrigger] = useState<number | null>(null);
  const [sessionId, setSessionId] = useState("");
  // True once the current trial's clip has begun playing; lets the grader score
  // mid-clip (the rating panel is shown while playing and after it ends). Reset
  // per trial. Revisited (already-answered) trials show the panel regardless.
  const [started, setStarted] = useState(false);
  // Answers recorded so far this session, keyed by trial index. Answering is
  // in-order, so the keys form a contiguous prefix [0 .. answeredCount-1] and the
  // next unanswered trial (the "frontier") is at index answeredCount.
  const [answers, setAnswers] = useState<Record<number, { rating: Rating | null; triggerOffset: number | null }>>({});

  const [results, setResults] = useState<ResultRow[]>(() => loadSaved());

  // which transitions to include this session (133->134 already done -> excluded)
  const [included, setIncluded] = useState<string[]>(
    () => TRANSITIONS.map((t) => t.key).filter((k) => k !== "133->134"),
  );
  const toggleTransition = (k: string) =>
    setIncluded((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]));

  const nOffsets = buildOffsets(OFFSET_PRESETS[offsetKey]).length;
  const ratingTotal = included.length * nOffsets * repeats;
  const triggerTotal = included.length * runs;

  function start() {
    if (included.length === 0) return;
    const keys = TRANSITIONS.map((t) => t.key).filter((k) => included.includes(k));
    const built = mode === "rating"
      ? buildTrials(keys, OFFSET_PRESETS[offsetKey], repeats)
      : buildTriggerTrials(keys, runs);
    setTrials(built);
    setIdx(0);
    setPhase("play");
    setLastTrigger(null);
    setStarted(false);
    setAnswers({});
    setSessionId(makeSessionId(mode === "rating" ? "alle-overgangen" : "trigger"));
    setScreen("run");
  }

  function transitionFor(key: string) {
    return TRANSITIONS.find((t) => t.key === key)!;
  }

  const answeredCount = Object.keys(answers).length; // contiguous prefix length
  const maxIdx = Math.min(answeredCount, trials.length - 1); // furthest reachable trial
  const allAnswered = trials.length > 0 && answeredCount === trials.length;

  // Jump to an already-answered trial (or the next unanswered frontier). Bounded so
  // you can only move within answered work, never skip ahead past the frontier.
  function goTo(target: number) {
    if (target < 0 || target >= trials.length || target > answeredCount) return;
    setIdx(target);
    setStarted(false); // freshly navigated clip hasn't played yet; revisits still show via `revisiting`
    const ans = answers[target];
    if (ans) {
      setLastTrigger(ans.triggerOffset);
      setPhase(mode === "rating" ? "rating" : "triggered");
    } else {
      setLastTrigger(null);
      setPhase("play");
    }
  }

  function recordAnswer(rating: Rating | null, triggerOffset: number | null) {
    const trial = trials[idx];
    const tr = transitionFor(trial.transitionKey);
    const wasNew = answers[trial.index] === undefined;
    const row: ResultRow = {
      sessionId, judge, mode,
      transition: tr.key,
      cutTime: tr.cutTime,
      baseAdjust: 0,
      order: trial.index,
      offset: mode === "rating" ? trial.offset : null,
      triggerOffset,
      rating,
      ratingValue: rating ? RATING_VALUE[rating] : null,
      note: "",
      isoTime: new Date().toISOString(),
    };
    // Keep exactly one local row per (session, trial); a revision replaces it.
    setResults((prev) => {
      const next = [...prev.filter((r) => !(r.sessionId === sessionId && r.order === trial.index)), row];
      saveAll(next);
      return next;
    });
    postTrial(row); // backend is idempotent per (session, ordering)
    setAnswers((prev) => ({ ...prev, [trial.index]: { rating, triggerOffset } }));

    // Answering the frontier (a new trial) advances to the next; re-answering an
    // earlier trial just updates it and stays put so the reviewer keeps their place.
    if (wasNew && idx + 1 < trials.length) {
      setIdx(idx + 1);
      setPhase("play");
      setLastTrigger(null);
      setStarted(false); // next clip must start before its rating panel returns
    } else {
      setPhase(mode === "rating" ? "rating" : "triggered");
    }
  }

  // "Maakt niet uit": the timing of THIS scene change is irrelevant. Marks the
  // current transition's remaining (unanswered) variants as nvt and jumps to the
  // next scene. Pressing it while reviewing an earlier trial just revises that one.
  function skipTransition() {
    const trial = trials[idx];
    if (!trial || mode !== "rating") return;
    if (idx < answeredCount) { recordAnswer("nvt", null); return; } // revising one earlier trial
    const key = trial.transitionKey;
    const tr = transitionFor(key);
    let end = idx;
    while (end + 1 < trials.length && trials[end + 1].transitionKey === key) end++; // block is contiguous
    const iso = new Date().toISOString();
    const newRows: ResultRow[] = [];
    const newAns: Record<number, { rating: Rating | null; triggerOffset: number | null }> = {};
    for (let k = idx; k <= end; k++) {
      const t = trials[k];
      newRows.push({
        sessionId, judge, mode: "rating", transition: key, cutTime: tr.cutTime, baseAdjust: 0,
        order: t.index, offset: t.offset, triggerOffset: null,
        rating: "nvt", ratingValue: null, note: "timing-irrelevant", isoTime: iso,
      });
      newAns[t.index] = { rating: "nvt", triggerOffset: null };
    }
    setResults((prev) => {
      const orders = new Set(newRows.map((r) => r.order));
      const next = [...prev.filter((r) => !(r.sessionId === sessionId && orders.has(r.order))), ...newRows];
      saveAll(next);
      return next;
    });
    newRows.forEach((r) => postTrial(r));
    setAnswers((prev) => ({ ...prev, ...newAns }));
    if (end + 1 < trials.length) {
      setIdx(end + 1); setPhase("play"); setStarted(false); setLastTrigger(null); // next scene
    } else {
      setIdx(end); setPhase("rating"); // skipped the final scene -> stay; Afronden appears
    }
  }

  useEffect(() => {
    if (screen !== "run" || mode !== "rating") return;
    const trial = trials[idx];
    const canRate = !!trial && (started || answers[trial.index] !== undefined);
    if (!canRate) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "1") recordAnswer("acceptabel", null);
      else if (e.key === "2") recordAnswer("twijfelachtig", null);
      else if (e.key === "3") recordAnswer("niet", null);
      else if (e.key === "4") skipTransition();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Warn on tab close/refresh while a session is in progress (browser shows its
  // own generic "leave site?" prompt; the text can't be customised).
  useEffect(() => {
    if (screen !== "run") return;
    const onUnload = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, [screen]);

  // ── Setup ──────────────────────────────────────────────────────────────────
  if (screen === "setup") {
    const previewTransition = transitionFor(previewKey);
    return (
      <div className="wrap">
        <h1>Acceptabele vertraging</h1>
        <p className="lede">
          De studie loopt door alle scène-overgangen heen. Per overgang wordt het moment
          van de filmcut een paar keer verschoven (vroeg, op tijd, laat) en geeft de
          deelnemer aan of de overgang op het juiste moment valt.
        </p>

        <div className="card grid">
          <label>Deelnemer
            <input value={judge} onChange={(e) => setJudge(e.target.value)} />
          </label>

          <label>Modus
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="rating">Beoordelen (varianten scoren)</option>
              <option value="trigger">Zelf triggeren (cut zelf inzetten)</option>
            </select>
          </label>

          {mode === "rating" ? (
            <>
              <label>Variaties per overgang
                <select value={offsetKey} onChange={(e) => setOffsetKey(e.target.value)}>
                  {Object.keys(OFFSET_PRESETS).map((k) => (
                    <option key={k} value={k}>{buildOffsets(OFFSET_PRESETS[k]).length} offsets</option>
                  ))}
                </select>
              </label>
              <label>Herhalingen
                <input type="number" min={1} max={5} value={repeats}
                       onChange={(e) => setRepeats(Math.max(1, +e.target.value))} />
                <span className="sub">{included.length} overgangen × {nOffsets} × {repeats} = {ratingTotal} proeven</span>
              </label>
            </>
          ) : (
            <label>Runs per overgang
              <input type="number" min={1} max={5} value={runs}
                     onChange={(e) => setRuns(Math.max(1, +e.target.value))} />
              <span className="sub">{included.length} overgangen × {runs} = {triggerTotal} proeven</span>
            </label>
          )}
        </div>

        <div className="card">
          <h3>Overgangen ({included.length} geselecteerd)</h3>
          <p className="sub">Elke overgang speelt als één blok met vaste muziek; de blokvolgorde is willekeurig. 133→134 staat uit (al gedaan).</p>
          <div className="transition-grid">
            {TRANSITIONS.map((t) => (
              <label key={t.key} className="tcheck">
                <input type="checkbox" checked={included.includes(t.key)} onChange={() => toggleTransition(t.key)} />
                {t.label}
              </label>
            ))}
          </div>
        </div>

        <div className="card">
          <h3>Controleer een overgang (offset 0)</h3>
          <p className="sub">Speel een willekeurige overgang af op het basismoment om te checken dat beeld en muziek kloppen.</p>
          <label style={{ maxWidth: 280 }}>Overgang om te testen
            <select value={previewKey} onChange={(e) => setPreviewKey(e.target.value)}>
              {TRANSITIONS.map((t) => (
                <option key={t.key} value={t.key}>{t.label}  (op {t.cutTime.toFixed(1)} s)</option>
              ))}
            </select>
          </label>
          <Player
            key={`preview-${previewKey}`}
            transition={previewTransition}
            baseAdjust={0}
            mode="rating"
            offset={0}
            trialId={-1}
            onEnded={() => {}}
          />
        </div>

        <button className="btn-primary big" onClick={start}>Start sessie</button>
        {results.length > 0 && (
          <p className="sub">{results.length} eerdere resultaten opgeslagen.{" "}
            <button className="link" onClick={() => download(`studie_${Date.now()}.csv`, toCSV(results), "text/csv")}>exporteer alles (CSV)</button>
          </p>
        )}
      </div>
    );
  }

  // ── Run ────────────────────────────────────────────────────────────────────
  if (screen === "run") {
    const trial = trials[idx];
    const tr = transitionFor(trial.transitionKey);
    const blocksTotal = new Set(trials.map((t) => t.transitionKey)).size;
    const blocksDone = new Set(trials.slice(0, idx + 1).map((t) => t.transitionKey)).size;
    const current = answers[trial.index]; // existing answer for this trial, if revisiting
    const revisiting = current !== undefined;
    return (
      <div className="wrap">
        <div className="runbar">
          <span>Overgang {tr.label}  ·  scene {blocksDone} / {blocksTotal}</span>
          <span>Proef {idx + 1} / {trials.length}</span>
          <button className="link" onClick={() => setScreen("done")}>Stoppen</button>
        </div>
        <div className="progress"><div style={{ width: `${(answeredCount / trials.length) * 100}%` }} /></div>

        <div className="navbar">
          <button className="btn-secondary" disabled={idx <= 0} onClick={() => goTo(idx - 1)}>← Vorige</button>
          <span className="sub">
            {revisiting ? "Je bekijkt een eerder antwoord (je kunt het wijzigen)." : "Nieuwe proef."}
            {" "}{answeredCount} / {trials.length} beantwoord
          </span>
          {allAnswered
            ? <button className="btn-primary" onClick={() => setScreen("done")}>Afronden →</button>
            : <button className="btn-secondary" disabled={idx >= maxIdx} onClick={() => goTo(idx + 1)}>Volgende →</button>}
        </div>

        {allAnswered && (
          <div className="note-done">
            Alle {trials.length} proeven beantwoord. Klik <b>Afronden →</b> om de sessie af te ronden,
            of ga met <b>← Vorige</b> terug om een antwoord te wijzigen.
          </div>
        )}

        <Player
          key={`trial-${trial.index}`}
          transition={tr}
          baseAdjust={0}
          mode={mode}
          offset={trial.offset}
          trialId={trial.index}
          onPlayStart={() => setStarted(true)}
          onEnded={(tOff) => {
            if (mode === "rating") setPhase("rating");
            else { setLastTrigger(tOff); setPhase("triggered"); }
          }}
        />

        {mode === "rating" && (started || revisiting) && (
          <div className="rating">
            <p>Hoe vond je deze overgang? <span className="sub">(je mag tijdens of na de clip beoordelen)</span>{revisiting && current.rating ? ` — huidig: ${RATING_LABEL[current.rating]}` : ""}</p>
            <div className="rating-btns">
              <button className={`r1${current?.rating === "acceptabel" ? " chosen" : ""}`} onClick={() => recordAnswer("acceptabel", null)}
                      title="De cut valt op een goed moment op de muziek.">
                <kbd>1</kbd> {RATING_LABEL.acceptabel}
              </button>
              <button className={`r2${current?.rating === "twijfelachtig" ? " chosen" : ""}`} onClick={() => recordAnswer("twijfelachtig", null)}
                      title="Twijfelgeval: niet duidelijk goed, niet duidelijk fout.">
                <kbd>2</kbd> {RATING_LABEL.twijfelachtig}
              </button>
              <button className={`r3${current?.rating === "niet" ? " chosen" : ""}`} onClick={() => recordAnswer("niet", null)}
                      title="De cut valt duidelijk op het verkeerde moment.">
                <kbd>3</kbd> {RATING_LABEL.niet}
              </button>
              <button className={`r4${current?.rating === "nvt" ? " chosen" : ""}`} onClick={skipTransition}
                      title="De timing van DEZE scène-overgang is niet belangrijk: er is geen duidelijk muzikaal moment om op te landen, dus elke offset is prima. Hiermee sla je de overige varianten van deze scène over en ga je naar de volgende.">
                <kbd>4</kbd> {RATING_LABEL.nvt} <span className="info" aria-hidden="true">ⓘ</span>
              </button>
            </div>
            <details className="help">
              <summary>Wat betekenen de knoppen?</summary>
              <ul>
                <li><b>Acceptabel / Twijfelachtig / Niet acceptabel</b>: jouw oordeel over of de cut op dít moment goed op de muziek valt.</li>
                <li><b>Maakt niet uit</b>: gebruik dit als de timing van déze scène-overgang niet belangrijk is, er is geen duidelijk muzikaal moment om op te landen. Dit is iets anders dan "twijfelachtig" (dat gaat over deze ene variant). Je slaat hiermee de resterende varianten van deze scène over.</li>
              </ul>
            </details>
            {revisiting && idx < maxIdx && (
              <p className="sub">Klaar met nakijken? Ga met <b>Volgende →</b> verder waar je was.</p>
            )}
          </div>
        )}

        {mode === "trigger" && phase === "triggered" && (
          <div className="rating">
            <p>
              {lastTrigger == null
                ? "Geen trigger genoteerd (geen cut ingezet)."
                : `Jouw timing: ${lastTrigger >= 0 ? "+" : ""}${lastTrigger.toFixed(2)} s t.o.v. het basismoment.`}
            </p>
            <button className="btn-primary" onClick={() => recordAnswer(null, lastTrigger)}>
              {revisiting ? "Bewaar" : "Volgende"}
            </button>
          </div>
        )}
      </div>
    );
  }

  // ── Done ───────────────────────────────────────────────────────────────────
  const sessionRows = results.filter((r) => r.sessionId === sessionId);
  const curve = summariseByOffset(sessionRows);
  const coverage = countByTransition(sessionRows);
  const irrelevant = timingIrrelevant(sessionRows);

  return (
    <div className="wrap">
      <h1>Sessie afgerond</h1>
      <p className="lede">Resultaten zijn lokaal opgeslagen. Download een kopie om ze te bewaren.</p>

      <div className="export">
        <button className="btn-primary" onClick={() => download(`studie_${sessionId || Date.now()}.csv`, toCSV(results), "text/csv")}>Download CSV</button>
        <button className="btn-secondary" onClick={() => download(`studie_${sessionId || Date.now()}.json`, JSON.stringify(results, null, 2), "application/json")}>Download JSON</button>
        <button className="btn-secondary" onClick={() => setScreen("setup")}>Nieuwe sessie</button>
      </div>

      {curve.length > 0 && (
        <div className="card">
          <h3>Acceptabiliteit per offset (alle overgangen samen)</h3>
          <p className="sub">Gemiddelde score per offset (acceptabel = +1, twijfelachtig = 0, niet = -1). De grens ligt waar dit van plus naar min omslaat, links te vroeg, rechts te laat. "Maakt niet uit"-overgangen tellen hier niet mee.</p>
          <table>
            <thead><tr><th>Offset (s)</th><th>n</th><th>Gemiddelde</th></tr></thead>
            <tbody>
              {curve.map((s) => (
                <tr key={s.offset}>
                  <td>{s.offset >= 0 ? "+" : ""}{s.offset.toFixed(2)}</td>
                  <td>{s.n}</td>
                  <td className={s.mean > 0.34 ? "good" : s.mean < -0.34 ? "bad" : "mid"}>{s.mean.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {irrelevant.length > 0 && (
        <div className="card">
          <h3>Overgangen waar timing "niet uitmaakt"</h3>
          <p className="sub">Door de beoordelaar gemarkeerd als timing-ongevoelig (geen scherp muzikaal moment om op te landen).</p>
          <table>
            <thead><tr><th>Overgang</th><th>nvt / totaal</th></tr></thead>
            <tbody>
              {irrelevant.map((c) => (
                <tr key={c.transition}><td>{c.transition}</td><td>{c.nvt} / {c.total}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {coverage.length > 0 && (
        <div className="card">
          <h3>Dekking per overgang</h3>
          <table>
            <thead><tr><th>Overgang</th><th>Proeven</th></tr></thead>
            <tbody>
              {coverage.map((c) => (
                <tr key={c.transition}><td>{c.transition}</td><td>{c.n}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
