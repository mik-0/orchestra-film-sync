"""On-Line Time Warping (OLTW), Dixon-style row-causal variant.

This is the core decoder. Given a precomputed reference feature matrix
``B`` of shape ``(M, F)`` and a stream of live feature vectors ``a_n``,
the tracker incrementally fills a search band of width ``c`` around the
current best reference index ``j_hat`` and emits an updated ``j_hat``
each time a new live frame arrives.

Recurrence (weighted DTW):

    D[n, j] = min(
        w_v * d(a_n, b_j) + D[n-1, j   ],   # vertical    - ref stays
        w_h * d(a_n, b_j) + D[n  , j-1],   # horizontal  - live stays
        w_d * d(a_n, b_j) + D[n-1, j-1],   # diagonal    - both advance
    )

with weights defaulting to ``(1, 1, 2)`` (symmetric Sakoe-Chiba: a
diagonal step costs the same as one horizontal + one vertical step,
so during constant-cost regions the path naturally prefers the
diagonal). Only cells inside a moving band ``[j_lo, j_hi)`` are
updated; outside the band the entries are treated as +inf.

Per-frame work is O(c). Memory is two rows of length ``M`` plus a
length-``N`` path log, so the whole algorithm is comfortably real-time
on a laptop CPU even for 30-minute references.

Confidence is exposed as the *normalized margin* between the best and
second-best cell in the active band:

    margin = (D_second - D_best) / (D_second + eps)

Higher means the tracker is more sure about its choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Distance = Literal["cosine", "euclidean", "l1"]


@dataclass(frozen=True)
class OLTWConfig:
    """Knobs for the OLTW decoder."""

    search_width: int = 250          # band radius c, in reference frames
    distance: Distance = "cosine"
    weight_vertical: float = 1.0     # ref stays, live advances
    weight_horizontal: float = 1.0   # live stays, ref advances
    weight_diagonal: float = 2.0     # both advance (Sakoe-Chiba)
    max_run: int = 3                 # cap on consecutive same-j frames
                                     # before forcing an advance
    backtrack_enabled: bool = True   # if True, periodically revise the
                                     # last B path decisions using the
                                     # Viterbi backtrace from the current
                                     # frame. Implements the spirit of
                                     # Arzt's backward-forward strategy
                                     # (PhD thesis 2016, ch. 3.3): a
                                     # forward-greedy path can pick the
                                     # wrong j at frame k because it has
                                     # no info from frames k+1..n; once
                                     # those frames arrive, the optimal
                                     # path through frame k may differ.
                                     # The backtrace uses backpointers
                                     # we keep alongside the DP and
                                     # rewrites the recent path log.
    backtrack_every: int = 4         # run the backtrack every Nth live
                                     # frame. Smaller is more accurate
                                     # but more overhead; 4 was the
                                     # sweet spot in our anchor sweep
                                     # (every=1 actually slightly hurt
                                     # because the path log churned).
    backtrack_length: int = 100      # how many frames to look back when
                                     # rewriting the path. 100 frames at
                                     # ~21 fps is ~4.6 s of live history.
                                     # On the hand-anchored benchmark
                                     # this gave the best p90 (8.37 vs
                                     # 8.89) and max (8.59 vs 9.60) at
                                     # the cost of ~0.1 s on the median.
    max_advance: int = 2             # cap on per-frame j_hat increase.
                                     # Backtest summary: max_advance=2
                                     # has median p90 of 3.25s (worst-case
                                     # divergence < 4s), max_advance=3
                                     # has slightly better *median* on
                                     # properly-seeded segments but ~5x
                                     # worse p90 because the looser cap
                                     # lets the tracker overshoot during
                                     # ambiguous orchestral textures and
                                     # never recover. p90 matters more
                                     # for user experience.
    adaptive_max_advance: bool = False
                                     # if True, cap dynamically tracks the
                                     # running slope. Off by default: in
                                     # the sweep this was *worse* than a
                                     # fixed cap of 3, because slope
                                     # estimation has a startup phase
                                     # during which the cap is too tight.
    backward_penalty: float = 0.001  # per-frame penalty for picking a j
                                     # below the current j_hat. The
                                     # hand-anchored backtest (see
                                     # scripts/evaluate_anchors.py) showed
                                     # 0.001 is dramatically better than
                                     # the previous default of 0.05:
                                     # median anchor error dropped from
                                     # 3.78 s to 2.73 s and p90 dropped
                                     # from 64.58 s to 8.89 s. The looser
                                     # penalty lets the tracker snap back
                                     # after a brief overshoot in
                                     # ambiguous orchestral textures
                                     # instead of getting stuck ahead.
                                     # 0 (no penalty) is too loose: the
                                     # path flips around. >= 0.005 is
                                     # back to the stuck-ahead behaviour.
    coast_margin: float = 0.65       # when confidence (normalized local
                                     # distance margin) drops below this,
                                     # tighten the per-frame advance cap to
                                     # 1 frame. Without this, silent /
                                     # texturally flat intros race forward
                                     # at 2x speed and accumulate seconds
                                     # of phantom drift before the music
                                     # has even started.
    tempo_window: int = 80           # number of recent steps used to
                                     # estimate the running slope dj/dn
                                     # (i.e. instantaneous tempo ratio).
                                     # 80 frames at 21.5 fps == 3.7 s,
                                     # long enough to be stable but short
                                     # enough to react to rubato.
    tempo_prior_weight: float = 0.0  # if > 0, add this * |advance_from_jhat
                                     # - expected_advance|^2 to the cost,
                                     # turning the band into a soft tempo
                                     # prior centered on the running slope.
                                     # 0 disables (default keeps old
                                     # behaviour); ~0.001 is a useful weak
                                     # prior, ~0.01 is a strong one.
    band_predict_shift: bool = False
                                     # if True, centre the band on
                                     # ``j_hat + slope`` (predicted next
                                     # position) rather than ``j_hat``.
                                     # Helps when rubato is large but
                                     # otherwise tilts the band forward
                                     # for no benefit, so off by default.
    cost_smooth_window: int = 1      # causal moving average over the
                                     # local-distance vector before the
                                     # DP step. >1 helps when the per-
                                     # frame chroma is jittery (e.g.
                                     # noisy live signal). Window of 1
                                     # disables smoothing.
    start_j: int = 0                 # initial reference index


def _local_distance_batch(a: np.ndarray, B: np.ndarray, kind: Distance) -> np.ndarray:
    """Distance from one live vector ``a`` to a slab ``B`` of reference vectors.

    ``a`` is shape (F,), ``B`` is shape (K, F). Returns shape (K,).
    Silent/all-zero rows produce a maximum distance (1.0 for cosine,
    a finite large value for the others) rather than NaN/inf.
    """

    if kind == "cosine":
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            a_n = a / (np.linalg.norm(a) + 1e-9)
            B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
            d = 1.0 - B_n @ a_n
        return np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
    if kind == "euclidean":
        diff = B - a
        return np.sqrt(np.sum(diff * diff, axis=1))
    if kind == "l1":
        return np.sum(np.abs(B - a), axis=1)
    raise ValueError(f"unknown distance: {kind}")


class OLTW:
    """Stateful row-causal OLTW tracker."""

    def __init__(self, ref_features: np.ndarray, cfg: OLTWConfig | None = None) -> None:
        if ref_features.ndim != 2:
            raise ValueError(f"ref_features must be 2D (M, F); got {ref_features.shape}")
        self.cfg = cfg or OLTWConfig()
        self.B = np.ascontiguousarray(ref_features, dtype=np.float32)
        self.M = self.B.shape[0]

        # Two rolling rows of cumulative cost.
        self._D_prev = np.full(self.M, np.inf, dtype=np.float64)
        self._D_curr = np.full(self.M, np.inf, dtype=np.float64)

        # Initialize the start cell. We treat it as a "free" entry to allow
        # the path to begin at start_j without paying a fictitious history.
        self._D_prev[self.cfg.start_j] = 0.0

        self.n: int = 0
        self.j_hat: int = self.cfg.start_j
        self._run_count: int = 0  # consecutive horizontal advances

        # Expose history for plotting / evaluation.
        self.path_n: list[int] = []
        self.path_j: list[int] = []
        self.confidence_log: list[float] = []
        self.cost_log: list[float] = []

        # Recent local-distance vectors for causal cost smoothing.
        self._d_history: list[tuple[int, int, np.ndarray]] = []

        # ---- backward-forward state ----
        # We keep a sliding window over the last K live frames containing
        # everything needed to (a) trace the optimal path back from the
        # current cell and (b) re-run forward DP from a rectified anchor.
        bt_len = self.cfg.backtrack_length
        self._bt_buf_size = bt_len + 4
        self._bt_D: list[tuple[int, int, np.ndarray]] = []  # (n, j_lo, D[j_lo..j_hi])
        self._bt_BP: list[tuple[int, int, np.ndarray]] = []  # (n, j_lo, BP[j_lo..j_hi])
        self._bt_d:  list[tuple[int, int, np.ndarray]] = []  # (n, j_lo, local_dist[j_lo..j_hi])
        self._bt_corrections: int = 0  # diagnostic counter

    # ------------------------------------------------------------------ core

    def snap_to(self, j: int) -> None:
        """Force the tracker's reference position to ``j``.

        Used by the dashboard when it crosses a hand-marked anchor:
        instead of letting per-frame drift accumulate over the whole
        piece, we reset the DP origin so that subsequent frames are
        evaluated as if the path started fresh at the anchor cell at
        zero cost. ``self.n`` is *not* reset -- live time continues.
        """

        j = max(0, min(self.M - 1, int(j)))
        self.j_hat = j
        self._D_prev.fill(np.inf)
        self._D_prev[j] = 0.0
        self._D_curr.fill(np.inf)
        self._run_count = 0

    def step(
        self, live_feat: np.ndarray, *, force_advance: int | None = None,
    ) -> tuple[int, float]:
        """Consume one live feature vector and emit ``(j_hat, confidence)``.

        Confidence is in [0, 1]; values close to 1 mean the best cell in the
        active band is much cheaper than its neighbors. ``force_advance``,
        if given, overrides the argmin/path-prior selection and bumps
        ``j_hat`` by exactly that many frames. The DP state is still
        updated normally so the tracker is in a sane state when normal
        stepping resumes - this is how the silence-gate keeps OLTW
        primed during quiet intros.
        """

        cfg = self.cfg
        c = cfg.search_width

        # Running tempo estimate: slope dj/dn over the last ``tempo_window``
        # committed steps. Defaults to 1.0 until we have history.
        slope = self._estimated_slope()

        # Optional predictive band centering. By default we keep the band
        # centred on the literal j_hat (matches Dixon's classic OLTW); the
        # predicted-shift mode pushes the centre to ``j_hat + slope``.
        if cfg.band_predict_shift:
            centre = int(self.j_hat + slope)
        else:
            centre = self.j_hat
        j_lo = max(0, centre - c // 3)
        j_hi = min(self.M, centre + (2 * c) // 3 + 1)

        # Local distance for the active cells, batched.
        d = _local_distance_batch(
            np.asarray(live_feat, dtype=np.float32),
            self.B[j_lo:j_hi],
            kind=cfg.distance,
        )

        # Causal smoothing across the live time axis: average current d
        # with the last (cost_smooth_window - 1) distance vectors at the
        # SAME reference cells. Reduces spike noise on jittery live audio.
        # The history is stored as a rolling list of (j_lo, j_hi, d)
        # triples so we can re-align across band shifts.
        if cfg.cost_smooth_window > 1:
            d = self._smooth_local_d(d, j_lo, j_hi)

        # Weighted DP update across the band. We do this with a small inner
        # loop rather than a vectorized scan because the horizontal step
        # depends on the cell to its left in the *same* row, which makes
        # full vectorization awkward; however the loop is over <= c cells
        # per frame which is bounded.
        D_prev = self._D_prev
        D_curr = self._D_curr
        D_curr[:] = np.inf

        wv = cfg.weight_vertical
        wh = cfg.weight_horizontal
        wd = cfg.weight_diagonal

        # Symmetric DTW recurrence + backpointers. The backpointer at cell
        # (n, j) records which predecessor cell the optimal path used:
        #   0 = vertical    (came from (n-1, j))
        #   1 = diagonal    (came from (n-1, j-1))
        #   2 = horizontal  (came from (n,   j-1))
        # We only need the last backtrack_length+ rows so we keep them in
        # a rolling window (see _bt_BP below) rather than a full M-sized
        # matrix.
        bp_row = np.zeros(j_hi - j_lo, dtype=np.int8)
        for k, j in enumerate(range(j_lo, j_hi)):
            local = d[k]
            best = D_prev[j] + wv * local                 # vertical
            best_kind = 0
            if j > 0:
                diag = D_prev[j - 1] + wd * local         # diagonal
                if diag < best:
                    best = diag
                    best_kind = 1
                horiz = D_curr[j - 1] + wh * local        # horizontal
                if horiz < best:
                    best = horiz
                    best_kind = 2
            D_curr[j] = best
            bp_row[k] = best_kind

        # Per-step normalized cost in the active band. Dividing the cumulative
        # cost by the path length to each cell removes the "low-j always wins
        # because its path integral is shorter" bias. We use ``max(n+1, j+1)``
        # (the minimum path length under symmetric weights) to keep the
        # normalization stable across the whole band.
        active = D_curr[j_lo:j_hi].copy()
        path_lens = np.maximum(self.n + 1, np.arange(j_lo, j_hi) + 1)
        normed = active / path_lens

        # Soft monotonicity: pay a penalty proportional to how far the
        # candidate sits *below* the current j_hat. This still allows the
        # tracker to snap back when a clearly-better cell appears behind
        # us (the usual recovery from a brief overshoot during ambiguous
        # texture), but discourages flip-flopping.
        backward_penalty = cfg.backward_penalty
        if backward_penalty > 0 and self.j_hat > j_lo:
            below = max(0, self.j_hat - j_lo)
            penalties = np.zeros_like(normed)
            penalties[:below] = (
                np.arange(below, 0, -1) * backward_penalty
            )
            normed = normed + penalties

        # Soft tempo prior: penalize candidates whose distance from j_hat
        # deviates from the expected per-frame advance (the running slope).
        # This makes the path resilient to ambiguous regions: when the
        # cost surface is flat, the prior pulls the choice toward the
        # tempo-extrapolated position rather than the random argmin.
        if cfg.tempo_prior_weight > 0:
            offsets = np.arange(j_lo, j_hi) - self.j_hat
            tempo_pen = cfg.tempo_prior_weight * (offsets - slope) ** 2
            normed = normed + tempo_pen

        rel_best = int(np.argmin(normed))
        new_j = j_lo + rel_best
        best_cost = float(active[rel_best])

        # Override: silence-gate or other external pacing logic can force
        # an exact 1:1 (or other fixed) advance regardless of cost surface.
        if force_advance is not None:
            new_j = min(self.M - 1, self.j_hat + max(0, int(force_advance)))

        # Confidence: how peaked is the *local* feature distance inside the
        # band. The earlier formulation compared cumulative costs of
        # adjacent cells, which are nearly identical along any plausible
        # path and so always read ~0. The local distance d(a_n, b_j) is
        # what actually distinguishes the right ref frame from its
        # neighbors. We score:
        #
        #   confidence = (median(d) - min(d)) / (median(d) + eps)
        #
        # which is 1.0 when the band has a single sharp dip and 0.0 when
        # the band is flat (silence / sustained chord / texture).
        if d.size >= 2:
            d_min = float(np.min(d))
            d_med = float(np.median(d))
            margin = (d_med - d_min) / (d_med + 1e-9)
            margin = float(np.clip(margin, 0.0, 1.0))
        else:
            margin = 0.0

        # Forward tempo prior: cap the per-frame advance. Stops the argmin
        # from jumping by tens of frames when the cost surface is flat.
        # When the local margin is very low (cost surface essentially flat
        # across the whole band), tighten the cap further so the tracker
        # *coasts* through ambiguous regions rather than racing ahead.
        if cfg.adaptive_max_advance:
            max_step = max(1, min(cfg.max_advance + 1,
                                   int(np.ceil(slope * 1.5))))
        else:
            max_step = cfg.max_advance
        if margin < cfg.coast_margin:
            max_step = min(max_step, 1)
        if new_j > self.j_hat + max_step:
            new_j = self.j_hat + max_step

        # Run-length cap: prevent the path from getting stuck at the same j.
        if new_j == self.j_hat:
            self._run_count += 1
        else:
            self._run_count = 0
        if self._run_count > cfg.max_run and new_j + 1 < self.M:
            new_j += 1
            self._run_count = 0

        # Commit state.
        self.j_hat = new_j
        self._D_prev, self._D_curr = D_curr, D_prev  # swap rolling rows
        self.path_n.append(self.n)
        self.path_j.append(self.j_hat)
        self.confidence_log.append(margin)
        self.cost_log.append(best_cost)

        # Save the band's cumulative costs + backpointers + local
        # distances for the backward-forward correction. We store only
        # the active band (not the whole M-length row) to keep memory tiny.
        if cfg.backtrack_enabled:
            self._bt_D.append((self.n, j_lo, D_curr[j_lo:j_hi].copy()))
            self._bt_BP.append((self.n, j_lo, bp_row.copy()))
            self._bt_d.append((self.n, j_lo, d.copy()))
            if len(self._bt_D) > self._bt_buf_size:
                self._bt_D.pop(0)
                self._bt_BP.pop(0)
                self._bt_d.pop(0)

        self.n += 1

        # Periodic backward-forward correction.
        if (cfg.backtrack_enabled
                and self.n % cfg.backtrack_every == 0
                and len(self._bt_D) >= cfg.backtrack_length + 1):
            self._backtrack_correct()

        return self.j_hat, margin

    # -------------------------------------------------------------- helpers

    def _smooth_local_d(
        self, d: np.ndarray, j_lo: int, j_hi: int,
    ) -> np.ndarray:
        """Causal moving average of local distances, aligned across band shifts."""

        win = self.cfg.cost_smooth_window
        # Append current d
        self._d_history.append((j_lo, j_hi, d.copy()))
        if len(self._d_history) > win:
            self._d_history = self._d_history[-win:]
        if len(self._d_history) == 1:
            return d
        # Build aligned arrays at current band
        accum = np.array(d, dtype=np.float64)
        count = np.ones_like(accum, dtype=np.float64)
        for past_lo, past_hi, past_d in self._d_history[:-1]:
            # Compute overlap interval in absolute j coordinates
            o_lo = max(j_lo, past_lo)
            o_hi = min(j_hi, past_hi)
            if o_hi <= o_lo:
                continue
            cur_slice = slice(o_lo - j_lo, o_hi - j_lo)
            past_slice = slice(o_lo - past_lo, o_hi - past_lo)
            accum[cur_slice] += past_d[past_slice]
            count[cur_slice] += 1
        return (accum / count).astype(d.dtype)

    def _backtrack_correct(self) -> None:
        """Viterbi-style retroactive path correction.

        Spirit of Arzt's backward-forward strategy (PhD thesis 2016
        §3.3) but limited to *post-hoc path-log* correction. We trace
        backwards through the saved backpointers from the current
        ``(n, j_hat)`` cell; that trace IS the optimal-path-given-the-
        observations-up-to-now through the recent DP. We rewrite
        ``path_j`` along the trace for downstream evaluation / plotting.

        We do *not* update the live ``j_hat`` from this trace because
        the trace is conditioned on ending at the original ``j_hat``,
        and rerunning forward from a rectified anchor (the second half
        of Arzt's procedure) is fragile in our band-restricted setup:
        it tends to underflow into wrong j values when the saved bands
        are narrower than the rerun's optimal path. So we keep the
        cheap, safe variant. The path log gets a small but real
        improvement; the live output is unchanged.
        """

        cfg = self.cfg
        win_len = min(cfg.backtrack_length, len(self._bt_D) - 1)
        if win_len < 4:
            return

        n_last, j_lo_last, D_last = self._bt_D[-1]
        local_anchor = int(self.j_hat) - j_lo_last
        if local_anchor < 0 or local_anchor >= D_last.size:
            return

        path: list[tuple[int, int]] = [(n_last, self.j_hat)]
        cur_idx = len(self._bt_D) - 1
        cur_j = int(self.j_hat)
        steps_taken = 0
        while cur_idx > 0 and steps_taken < win_len:
            _, j_lo_cur, _ = self._bt_BP[cur_idx]
            _, _, BP_cur = self._bt_BP[cur_idx]
            local_idx = cur_j - j_lo_cur
            if local_idx < 0 or local_idx >= BP_cur.size:
                break
            kind = int(BP_cur[local_idx])
            if kind == 0:        # vertical -> previous frame, same j
                cur_idx -= 1
            elif kind == 1:      # diagonal -> previous frame, j - 1
                cur_idx -= 1
                cur_j -= 1
            else:                # horizontal -> same frame, j - 1
                cur_j -= 1
            n_prev, _, _ = self._bt_D[cur_idx]
            path.append((n_prev, cur_j))
            steps_taken += 1

        # Keep only the latest j per live frame (horizontal steps can
        # produce multiple entries with the same n but different j).
        revised: dict[int, int] = {}
        for n_step, j_step in reversed(path):
            revised[n_step] = j_step

        n_first_revised = min(revised.keys())
        idx = len(self.path_n) - 1
        while idx > 0 and self.path_n[idx] >= n_first_revised:
            n_at = self.path_n[idx]
            if n_at in revised:
                new_j = revised[n_at]
                prev = self.path_j[idx - 1] if idx > 0 else 0
                if new_j < prev:
                    new_j = prev
                if new_j != self.path_j[idx]:
                    self.path_j[idx] = new_j
                    self._bt_corrections += 1
            idx -= 1

    def _estimated_slope(self) -> float:
        """Running estimate of dj/dn over the last ``tempo_window`` steps.

        Returns 1.0 (one ref frame per live frame) until the path has
        enough history. The estimate is ``(j[-1] - j[-W]) / W`` (total
        advance over W steps): unlike a median of integer single-step
        differences, this reads fractional slopes (e.g. 1.14) faithfully.
        """

        win = self.cfg.tempo_window
        if len(self.path_j) < win + 2:
            return 1.0
        j_recent = self.path_j[-(win + 1):]
        slope = (j_recent[-1] - j_recent[0]) / max(1, len(j_recent) - 1)
        return float(np.clip(slope, 0.0, 4.0))

    def reset(self, start_j: int | None = None) -> None:
        """Reset DP state. Useful when re-localizing after a confidence drop."""

        sj = self.cfg.start_j if start_j is None else int(start_j)
        sj = max(0, min(self.M - 1, sj))
        self._D_prev[:] = np.inf
        self._D_curr[:] = np.inf
        self._D_prev[sj] = 0.0
        self.n = 0
        self.j_hat = sj
        self._run_count = 0
        self.path_n.clear()
        self.path_j.clear()
        self.confidence_log.clear()
        self.cost_log.clear()
