"""Online HMM forward-filter tracker for cross-performance audio alignment.

A Viterbi-style alternative to OLTW with three differences worth the
extra arithmetic:

  1. **Probabilistic emissions.**  ``b_j(x_t) = exp(-beta * d(a_t, b_j))``
     is a proper likelihood, so the cumulative cost has a probabilistic
     interpretation and the posterior can be normalised.

  2. **Tempo-aware transitions.**  ``p(j' | j) ~ N(j' - j; psi, sigma^2)``
     where ``psi`` is the running tempo estimate from the Kalman filter.
     This is the *anticipatory* part from Cont 2010: instead of a hard
     "advance by 0/1/2" choice, the model has a smooth preference for
     advancing by the current expected step size, with variance widening
     when the tempo estimate is itself uncertain.

  3. **Implicit duration prior.**  Because the transition kernel is
     centred on ``psi``, staying in the same state is increasingly
     unlikely as time passes -- this is the Markovian approximation to
     the survival distribution ``d_j(u) = exp(-u / psi)`` in the paper.
     We don't explicitly carry duration as a state variable (which would
     need a true semi-Markov chain) but the marginal effect is the same
     for our 1-frame-per-state setup.

The DP is run in negative-log-probability space (i.e. as a min-plus
recursion, exactly like OLTW's cumulative cost). The MAP path is the
argmin of the cumulative-cost vector at each step, restricted to the
moving search band so per-frame work stays O(c) where c is the band
width.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from online_dtw.tempo import KalmanTempo, KalmanTempoConfig


Distance = Literal["cosine", "euclidean", "l1"]


@dataclass(frozen=True)
class HMMConfig:
    """Knobs for the HMM forward-filter tracker."""

    search_width: int = 400          # band radius c in reference frames
    distance: Distance = "cosine"
    beta: float = 8.0                # emission temperature.
                                     # b_j = exp(-beta * d). Larger means
                                     # the likelihood is more peaked
                                     # around the closest ref frame.
    sigma_floor: float = 0.8         # minimum transition std (frames).
                                     # Even with a confident tempo, allow
                                     # ~0.8 frames of slack so the path
                                     # can absorb micro-rubato.
    sigma_scale: float = 8.0         # maps sqrt(Kalman variance) to the
                                     # transition std. sigma is computed
                                     # as max(sigma_floor, sigma_scale *
                                     # kalman.std). Larger means the
                                     # tempo prior is looser.
    backward_penalty: float = 0.001  # extra cost per frame for picking a
                                     # j strictly less than current j_hat.
                                     # Same role as in OLTW: lets the
                                     # path snap back from brief
                                     # overshoots, without flapping.
    max_advance: int = 4             # cap on per-frame j_hat increase.
                                     # 4 is a bit looser than OLTW's 2
                                     # because the HMM transitions are
                                     # already soft-constrained by psi;
                                     # we just want a safety net.
    min_advance: int = 0             # minimum advance per frame. 0 = can
                                     # stay put. 1 = strictly monotonic.
    start_j: int = 0                 # initial reference index (seed)
    emission_buffer_frames: int = 0  # 0 = single-frame emission (default).
                                     # >0 = integrate emission for each
                                     # band candidate over the last N live
                                     # frames, in the style of Otsuka's
                                     # particle filter (Otsuka et al. 2011,
                                     # EURASIP JASP; the L=2.5 s "audio
                                     # buffer matching" trick). Each
                                     # emission_nll[j] becomes the mean of
                                     # per-frame distances over the buffer
                                     # rather than a single frame's value.
                                     # Costs O(L*C*F) per step; worth it
                                     # on long-form pieces where per-frame
                                     # distance noise compounds into drift.
                                     # Reasonable values: 20-50 frames
                                     # (about 1-2.3 s at hop=1024/sr=22050).
    skip_mass: float = 1e-5          # uniform "global jump" probability
                                     # mass added to every transition,
                                     # in the style of Nakamura's outer-
                                     # product HMM (matchmaker
                                     # outer_product_hmm.py:195). Without
                                     # it the tracker can get pinned at
                                     # a single ref frame indefinitely
                                     # once the cumulative cost peaks
                                     # there: the local transition cost
                                     # to advance is larger than the
                                     # emission improvement at any near
                                     # neighbour, so stay wins forever.
                                     # The skip mass provides a constant
                                     # escape probability to *any* ref
                                     # state, so a sufficiently good
                                     # match anywhere in the band can
                                     # outcompete the stuck state.
                                     # 0 = disabled (pre-fix behaviour).


def _local_distance(a: np.ndarray, B: np.ndarray, kind: Distance) -> np.ndarray:
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


def _batched_distance(A: np.ndarray, B: np.ndarray, kind: Distance) -> np.ndarray:
    """Batched distance: A shape (L, F), B shape (C, F). Returns (L, C).

    Used by the buffered-emission path so the HMM can integrate over a
    sliding window of recent live frames against the current band without
    a Python loop over L.
    """
    if kind == "cosine":
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
            d = 1.0 - A_n @ B_n.T
        return np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
    if kind == "euclidean":
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))
    if kind == "l1":
        return np.sum(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    raise ValueError(f"unknown distance: {kind}")


class HMMTracker:
    """Online forward-filter (Viterbi) tracker over reference frames.

    Public API mirrors :class:`online_dtw.oltw.OLTW`: ``step(live_feat)``
    returns ``(j_hat, confidence)`` and exposes ``path_n``, ``path_j``,
    ``confidence_log`` for plotting.
    """

    def __init__(
        self, ref_features: np.ndarray, cfg: HMMConfig | None = None,
        kalman: KalmanTempo | None = None,
    ) -> None:
        if ref_features.ndim != 2:
            raise ValueError(f"ref must be 2D (M, F), got {ref_features.shape}")
        self.cfg = cfg or HMMConfig()
        self.B = np.ascontiguousarray(ref_features, dtype=np.float32)
        self.M = self.B.shape[0]
        self.kalman = kalman or KalmanTempo(KalmanTempoConfig())

        # Cumulative negative-log-probability over reference states. We
        # initialise everything to +inf except the seed cell, which is
        # the standard "delta function start" for a Viterbi recursion.
        self._D_prev = np.full(self.M, np.inf, dtype=np.float64)
        self._D_curr = np.full(self.M, np.inf, dtype=np.float64)
        self._D_prev[self.cfg.start_j] = 0.0

        self.j_hat: int = int(self.cfg.start_j)
        self.n: int = 0

        # Sliding window of recent live feature vectors for the
        # buffered-emission path. Empty when emission_buffer_frames=0;
        # otherwise keeps the most recent up-to-N (F,) vectors so step()
        # can integrate emission over the buffer against the current band.
        self._live_buf: list[np.ndarray] = []

        # Optional external slope prior. When set (e.g. by an onset
        # matcher running alongside the tracker), this overrides the
        # Kalman's tempo estimate as the transition mean for the
        # NEXT step only. The worker is expected to refresh it
        # whenever it has a new onset-derived slope.
        self._external_slope_prior: float | None = None

        # Path log + per-frame stats for downstream eval / plotting.
        self.path_n: list[int] = []
        self.path_j: list[int] = []
        self.confidence_log: list[float] = []
        self.cost_log: list[float] = []
        self.tempo_log: list[float] = []   # extra: emit tracked tempo

    def set_slope_prior(self, slope: float | None) -> None:
        """Override the transition mean for subsequent steps."""
        self._external_slope_prior = slope

    # --------------------------------------------------------------- core

    def step(self, live_feat: np.ndarray) -> tuple[int, float]:
        cfg = self.cfg
        c = cfg.search_width

        # Active band for the *new* state j. We bias forward (1/4 behind
        # j_hat, 3/4 ahead) since the path is expected to advance.
        j_lo = max(0, self.j_hat - c // 4)
        j_hi = min(self.M, self.j_hat + (3 * c) // 4 + 1)
        active_size = j_hi - j_lo

        # Tempo from the Kalman filter, unless an external (onset-
        # anchored) slope estimate is overriding it for this step.
        psi = float(self.kalman.psi)
        if self._external_slope_prior is not None:
            psi = float(self._external_slope_prior)
        sigma = max(cfg.sigma_floor, cfg.sigma_scale * float(self.kalman.std))

        # ----- emission: log-likelihood of each ref frame given live_feat
        live_now = np.asarray(live_feat, dtype=np.float32)
        if cfg.emission_buffer_frames > 0:
            # Buffered emission: integrate distance over the last
            # ``emission_buffer_frames`` live vectors against the current
            # band. This is the "audio-buffer matching" trick from
            # Otsuka et al. 2011; each emission becomes a robust local
            # average instead of a single-frame value, which suppresses
            # the per-frame distance noise that compounds into cumulative
            # tempo drift on long pieces.
            self._live_buf.append(live_now)
            if len(self._live_buf) > cfg.emission_buffer_frames:
                self._live_buf.pop(0)
            live_block = np.stack(self._live_buf, axis=0)              # (L, F)
            d_mat = _batched_distance(live_block, self.B[j_lo:j_hi],
                                       kind=cfg.distance)              # (L, C)
            d = d_mat.mean(axis=0)
        else:
            d = _local_distance(live_now, self.B[j_lo:j_hi], kind=cfg.distance)
        # neg-log b_j = beta * d  (up to an additive constant we drop)
        emission_nll = cfg.beta * d.astype(np.float64)

        # ----- transition: for each candidate j (in band), compute the
        # min over predecessor i of [predecessor_cost + transition_cost].
        # The transition is parameterised by the advance a = j - i:
        # cost = (a - psi)^2 / (2 sigma^2)  (Gaussian on advance).
        #
        # Restrict a to [0, max_advance] so the DP itself is monotonic
        # (the path cannot retreat in one step). Backward recovery from
        # an overshoot is handled separately by the post-hoc
        # backward_penalty applied to the cumulative cost array, which
        # is the same trick OLTW uses.
        a_min = max(0, int(np.floor(psi - 3.0 * sigma)))
        a_max = min(cfg.max_advance, int(np.ceil(psi + 3.0 * sigma)))
        if a_max <= a_min:
            a_max = min(cfg.max_advance, a_min + 1)

        # Build the transition cost vector for each offset a.
        offsets = np.arange(a_min, a_max + 1)
        trans_cost = ((offsets - psi) ** 2) / (2.0 * sigma * sigma)

        # Per-cell DP:
        #   D_curr[j] = emission_nll[j - j_lo]
        #             + min over a in offsets of (D_prev[j - a] + trans_cost[a])
        # We do this with a vectorised inner loop over offsets.
        D_prev = self._D_prev
        D_curr = self._D_curr
        D_curr.fill(np.inf)

        # Stack predecessor candidates for each j, then take the min.
        # candidates[k, j] = D_prev[j - offsets[k]] + trans_cost[k]
        # j ranges over [j_lo, j_hi).
        candidates = np.full((offsets.size, active_size), np.inf, dtype=np.float64)
        for k, a in enumerate(offsets):
            i_lo, i_hi = j_lo - a, j_hi - a
            # Bounds-clip to [0, M)
            valid_lo = max(i_lo, 0)
            valid_hi = min(i_hi, self.M)
            if valid_hi <= valid_lo:
                continue
            slice_dst_lo = valid_lo - i_lo
            slice_dst_hi = slice_dst_lo + (valid_hi - valid_lo)
            candidates[k, slice_dst_lo:slice_dst_hi] = D_prev[valid_lo:valid_hi] + trans_cost[k]

        best_pred = candidates.min(axis=0)        # (active_size,)

        # Global skip-mass escape: every cell in the band gets a
        # ``floor'' predecessor cost = (best previous cell anywhere) +
        # -log(skip_mass / M). For ``skip_mass = 1e-5`` and M ~ 3000,
        # this floor is ~19.5 nats, equivalent to ~100 frames of normal
        # accumulated cost. So in the steady state it does nothing; it
        # only activates when a band cell would otherwise have an
        # impossibly large best_pred relative to the global front, which
        # is exactly the absorbing-state stall.
        if cfg.skip_mass > 0:
            skip_cost = -float(np.log(cfg.skip_mass / float(self.M)))
            global_best_prev = float(D_prev.min())
            if np.isfinite(global_best_prev):
                best_pred = np.minimum(best_pred, global_best_prev + skip_cost)

        D_active = emission_nll + best_pred
        D_curr[j_lo:j_hi] = D_active

        # ----- pick the new j_hat -----
        # We use the raw cumulative cost rather than OLTW's path-length
        # normalisation. The HMM transition kernel allows advances up to
        # ``max_advance`` per frame, which means a "racing" path reaches
        # high j after few steps. Dividing cumulative cost by path
        # length (which is ~j for high-j paths) produces the wrong bias:
        # the racing path has lower per-step cost and wins. Without the
        # normalisation, the emission + transition cost balance handles
        # this correctly: racing pays a quadratic transition penalty.
        cost = D_active.copy()

        # Soft-monotonicity / backward penalty.
        if cfg.backward_penalty > 0 and self.j_hat > j_lo:
            below = max(0, self.j_hat - j_lo)
            penalties = np.zeros_like(cost)
            penalties[:below] = np.arange(below, 0, -1) * cfg.backward_penalty
            cost = cost + penalties

        rel_best = int(np.argmin(cost))
        new_j = j_lo + rel_best
        best_cost = float(D_active[rel_best])

        # Confidence: how peaked is the local *emission* around the
        # chosen j? We use the same "median-vs-min" recipe that worked
        # for OLTW. Small d => the picked frame has a sharp dip in the
        # likelihood landscape; uniform d => the choice is arbitrary.
        if d.size >= 2:
            d_min = float(np.min(d))
            d_med = float(np.median(d))
            margin = (d_med - d_min) / (d_med + 1e-9)
            margin = float(np.clip(margin, 0.0, 1.0))
        else:
            margin = 0.0

        # Optional minimum advance (forces strict monotonicity if you want
        # it). Default 0 = can stay put. The HMM tempo prior already
        # discourages standing still when psi >> 0, so we leave it free.
        if cfg.min_advance > 0:
            new_j = max(new_j, self.j_hat + cfg.min_advance)
        # Hard cap on per-frame advance (safety against pathological cases)
        if new_j > self.j_hat + cfg.max_advance:
            new_j = self.j_hat + cfg.max_advance
        new_j = min(self.M - 1, new_j)

        # ----- commit state -----
        self.j_hat = new_j
        self._D_prev, self._D_curr = D_curr, D_prev
        self.path_n.append(self.n)
        self.path_j.append(self.j_hat)
        self.confidence_log.append(margin)
        self.cost_log.append(best_cost)
        self.tempo_log.append(psi)
        self.n += 1

        # ----- feed the Kalman filter with our latest path step -----
        self.kalman.update(self.j_hat, margin)
        return self.j_hat, margin

    # ---------------------------------------------------------------- helpers

    def reset(self, start_j: int | None = None) -> None:
        sj = self.cfg.start_j if start_j is None else int(start_j)
        sj = max(0, min(self.M - 1, sj))
        self._D_prev[:] = np.inf
        self._D_curr[:] = np.inf
        self._D_prev[sj] = 0.0
        self.j_hat = sj
        self.n = 0
        self.path_n.clear()
        self.path_j.clear()
        self.confidence_log.clear()
        self.cost_log.clear()
        self.tempo_log.clear()
        self._live_buf.clear()
        self.kalman.reset()
