"""Kalman-filter tempo tracker for cross-performance alignment.

What it does
------------
Maintains a smoothed estimate of the running *slope* psi = d(ref) / d(live)
in units of reference-frames per live-frame. Observations come from the
tracker's recent path: each step emits a (j_hat, confidence) pair and the
filter updates a running tempo estimate.

Why a Kalman filter
-------------------
Earlier attempts used a sliding-window mean / median of integer step sizes
(see commit history). Those have two failure modes:

  * an integer-difference median always reads exactly 1 even when the true
    slope is, say, 1.14;
  * a windowed mean is fragile when the OLTW path has occasional bad
    frames (a runaway forward jump biases the running mean for the whole
    window).

A 1D Kalman filter solves both: the state is a real-valued slope, the
observation is a noisy slope estimate from the recent path, and the
observation variance is *adaptive* via the tracker's confidence -- when
the path is on solid ground, we trust the observation more; when it's
flat-cost ambiguous, we fall back on the prior.

Inspired by the Cont 2010 phase-coupled tempo agent (Antescofo): in their
formulation the tempo is a self-sustaining oscillator with attentional
focus kappa. We don't have explicit beats so we cannot run the full
Von-Mises phase coupling, but the *spirit* (predict + correct + adaptive
variance) is the same; for our continuous reference, a scalar Kalman
filter is the right small-cousin model.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KalmanTempoConfig:
    init_slope: float = 1.0          # ref-frames per live-frame at startup
    init_var: float = 0.25           # initial state variance (slope^2)
    process_var: float = 1e-5        # how much slope is allowed to drift
                                     # between steps (random walk noise)
    obs_var_floor: float = 0.001     # minimum observation variance --
                                     # avoids divide-by-zero when conf=1
    obs_var_ceiling: float = 1.0     # maximum observation variance --
                                     # used when conf=0 (effectively
                                     # ignores the observation)
    obs_window: int = 24             # how many recent steps the slope
                                     # observation averages over. Larger
                                     # means a smoother tempo estimate
                                     # but slower reaction to genuine
                                     # tempo changes.
    slope_floor: float = 0.3         # clamp slope to >=this. 0.3 is a
                                     # very slow performance ratio - if
                                     # the tracker thinks ref is moving
                                     # less than 30 % of live's rate
                                     # something is very wrong.
    slope_ceiling: float = 2.0       # clamp slope to <=this. Above 2x
                                     # tempo difference is implausible
                                     # for two performances of the same
                                     # piece.
    confidence_floor: float = 0.05   # ignore observations whose
                                     # confidence falls below this. The
                                     # Kalman simply skips the update
                                     # rather than absorbing garbage.


class KalmanTempo:
    """1D Kalman filter on path slope.

    Usage::

        kt = KalmanTempo(KalmanTempoConfig())
        for j_hat, conf in tracker.step_stream():
            psi = kt.update(j_hat, conf)   # smoothed slope estimate
    """

    def __init__(self, cfg: KalmanTempoConfig | None = None) -> None:
        self.cfg = cfg or KalmanTempoConfig()
        self.psi: float = float(self.cfg.init_slope)   # state estimate
        self.var: float = float(self.cfg.init_var)     # state variance
        self._j_hist: list[int] = []                   # recent j_hat values

    # ----- prediction (called when no observation is available) -----

    def predict(self) -> float:
        """Apply one step of the random-walk prediction without observing.

        Inflates the state variance by the process noise. Useful at frame
        boundaries where the underlying tracker hasn't emitted yet.
        """

        self.var += self.cfg.process_var
        return self.psi

    # ----- combined predict + correct -----

    def update(self, j_hat: int, confidence: float) -> float:
        """Push a new (j_hat, confidence) and return the smoothed slope.

        Confidence is in [0, 1]; higher means we should trust the
        observation more (lower observation variance).
        """

        cfg = self.cfg
        self._j_hist.append(int(j_hat))
        if len(self._j_hist) > cfg.obs_window + 1:
            self._j_hist.pop(0)

        # Predict step: state stays put, variance inflates a bit.
        var_pred = self.var + cfg.process_var

        # Need at least a few history points to take a slope reading.
        n = len(self._j_hist) - 1
        c = float(np.clip(confidence, 0.0, 1.0))
        if n >= max(4, cfg.obs_window // 4) and c >= cfg.confidence_floor:
            # Observed slope = mean per-step advance over the window.
            obs_slope = (self._j_hist[-1] - self._j_hist[0]) / n

            # Adaptive observation variance: high confidence -> trust the
            # reading; low confidence -> back off, lean on the prior.
            R = cfg.obs_var_floor + (cfg.obs_var_ceiling - cfg.obs_var_floor) * (1.0 - c) ** 2

            # Standard 1D Kalman update.
            K = var_pred / (var_pred + R)
            self.psi = self.psi + K * (obs_slope - self.psi)
            self.var = (1.0 - K) * var_pred
        else:
            # No usable observation yet (or too low confidence); keep the
            # prior with inflated variance.
            self.var = var_pred

        # Clamp to a sensible range.
        self.psi = float(np.clip(self.psi, cfg.slope_floor, cfg.slope_ceiling))
        return self.psi

    # ----- bookkeeping -----

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 0.0)))

    def reset(self, slope: float | None = None) -> None:
        cfg = self.cfg
        self.psi = float(cfg.init_slope if slope is None else slope)
        self.var = float(cfg.init_var)
        self._j_hist.clear()
