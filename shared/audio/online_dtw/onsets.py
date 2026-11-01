"""Streaming onset detection and an onset-anchored slope estimator.

This is the online half of a two-stage alignment scheme: dense
features (chroma / lse / learned embeddings) drive the per-frame DP,
but a sparse, robust **onset slope** from peaks in spectral flux
provides a tempo prior that the DP cannot easily derive on its own.

Causality is strict on the LIVE signal:

* Reference onsets are precomputed offline -- the full ref audio is
  always available before streaming starts, so we use librosa's
  vectorised peak picker over the whole file.
* Live onsets emerge causally from a streaming spectral-flux peak
  picker that holds at most 1 feature-frame of lookahead (needed to
  confirm a local maximum). At hop = 1024 / sr = 22050 that's a
  ~46 ms publish delay, which is fine for tempo estimation but is
  noted explicitly so you can swap in a smaller hop for tighter
  latency budgets.

The slope estimator matches each newly-emitted live onset to the
closest unmatched reference onset above the previously-matched ref
position, within a band around the running ``j_hat``. From the last
``K`` matched pairs it fits a robust slope (median of consecutive
pair slopes). The tracker can then weight its forward transition
toward that slope; nothing else about the DP changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator
import numpy as np
import librosa


# ---- streaming onset strength (matches librosa.onset.onset_strength) -------

class StreamingOnsetStrength:
    """Causal streaming version of ``librosa.onset.onset_strength``.

    Maintains a one-frame mel-spectrogram history so that each new
    audio chunk produces ``onset_strength`` values whose numerical
    scale matches librosa's offline output. Use this instead of the
    generic FeatureExtractor 'flux' feature when you want peak-
    picking thresholds that line up with librosa's defaults.
    """

    def __init__(
        self, sr: int = 22050, n_fft: int = 2048, hop_length: int = 1024,
        n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None,
    ) -> None:
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else float(sr) / 2.0
        self._buf = np.zeros(0, dtype=np.float32)
        self._prev_log_mel: np.ndarray | None = None  # (n_mels,)

    def process(self, audio_chunk: np.ndarray) -> Iterator[float]:
        """Yield one onset_strength scalar per emitted frame."""

        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"expected mono chunk, got shape {chunk.shape}")
        self._buf = np.concatenate([self._buf, chunk])
        if len(self._buf) < self.n_fft:
            return
        n_frames = 1 + (len(self._buf) - self.n_fft) // self.hop
        if n_frames <= 0:
            return

        signal = self._buf[: self.n_fft + (n_frames - 1) * self.hop]
        S = np.abs(librosa.stft(
            y=signal, n_fft=self.n_fft, hop_length=self.hop, center=False,
        )) ** 2
        mel = librosa.feature.melspectrogram(
            S=S, sr=self.sr, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax, n_fft=self.n_fft,
            hop_length=self.hop,
        )
        # top_db=None: see features.py:_lse for why. librosa's default
        # 80 dB clipping uses the input array's local max, so per-chunk
        # streaming output differs from offline on quiet bins.
        log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)

        for k in range(log_mel.shape[1]):
            col = log_mel[:, k]
            if self._prev_log_mel is None:
                diff = np.zeros_like(col)
            else:
                diff = np.maximum(0.0, col - self._prev_log_mel)
            self._prev_log_mel = col.copy()
            yield float(diff.mean())

        consumed = n_frames * self.hop
        self._buf = self._buf[consumed:]


# ---- reference: offline -----------------------------------------------------

def extract_ref_onsets(
    ref_audio: np.ndarray, sr: int, hop_length: int = 1024,
    delta: float = 0.07, wait: int = 3,
) -> np.ndarray:
    """Onset frame indices for the full reference recording.

    Returns an array of frame indices into a feature stream computed
    with ``hop_length`` and ``sr``. ``delta`` is the onset threshold
    above the local median flux; ``wait`` is the minimum frame gap
    between consecutive onsets. Defaults are tuned for orchestral
    audio; for thinner textures lower ``delta`` to ~0.04.
    """

    o_env = librosa.onset.onset_strength(
        y=ref_audio, sr=sr, hop_length=hop_length, center=False,
    )
    return librosa.onset.onset_detect(
        onset_envelope=o_env, sr=sr, hop_length=hop_length,
        delta=delta, wait=wait, units="frames", normalize=False,
    )


# ---- live: causal streaming -------------------------------------------------

@dataclass
class CausalOnsetDetector:
    """Streaming peak picker over an onset-strength curve.

    Matches librosa's offline ``peak_pick`` algorithm with
    ``post_max=1`` and ``post_avg=0``: a peak is confirmed when the
    candidate frame is strictly greater than the next frame (so 1
    feature-frame of lookahead, ~46 ms at hop 1024 / sr 22050) AND
    exceeds the trailing average by ``delta``. Use with the
    :class:`StreamingOnsetStrength` extractor whose numerical scale
    matches ``librosa.onset.onset_strength``.
    """

    delta: float = 0.5            # peak must exceed running mean by this
    wait: int = 3                 # min gap between emitted onsets
    pre_avg: int = 22             # frames of trailing mean for threshold
    pre_max: int = 22             # frames of trailing max for "is local max"

    _hist: list[float] = field(default_factory=list)
    _frame_counter: int = -1
    _candidate_frame: int = -1
    _candidate_value: float = 0.0
    _candidate_active: bool = False
    _last_emitted: int = -10**9

    def ingest_flux(self, flux: float) -> int | None:
        self._frame_counter += 1
        self._hist.append(float(flux))
        keep = max(self.pre_avg, self.pre_max) + 4
        if len(self._hist) > keep:
            self._hist = self._hist[-keep:]

        emitted: int | None = None

        # Confirm/reject the deferred candidate against the just-arrived
        # frame: the candidate must be strictly greater than the next
        # frame to count as a local max (1-frame post-confirmation).
        if self._candidate_active:
            if (self._candidate_value > flux
                    and self._candidate_frame - self._last_emitted >= self.wait):
                emitted = self._candidate_frame
                self._last_emitted = self._candidate_frame
            self._candidate_active = False

        # Could the just-arrived frame be a candidate? It must exceed
        # the trailing mean by delta AND be the max in the trailing
        # window (no need for post-side info on max).
        if len(self._hist) >= 2:
            tail_avg = self._hist[-(self.pre_avg + 1):-1]
            tail_max = self._hist[-(self.pre_max + 1):-1]
            avg = float(np.mean(tail_avg)) if tail_avg else 0.0
            mx = float(np.max(tail_max)) if tail_max else 0.0
            if flux > avg + self.delta and flux >= mx:
                self._candidate_active = True
                self._candidate_frame = self._frame_counter
                self._candidate_value = flux
        return emitted


# ---- slope estimator: matches live onsets to ref onsets ---------------------

@dataclass
class OnsetSlopeEstimator:
    """Streaming slope estimate from onset-sequence matching.

    Matching is decoupled from the tracker's ``j_hat`` to avoid a
    positive-feedback loop where a drifting tracker pulls the slope
    estimate in the same wrong direction. Instead, each new live
    onset is matched to the closest reference onset within a
    *predictive* window around (last_matched_ref + slope * (now -
    last_live)). The slope is bootstrapped at 1.0 and refined from
    the most-recent ``history`` matched pairs (median, robust).

    A starting anchor ``(seed_live_frame, seed_ref_frame)`` lets the
    estimator begin matching from a known good position so the
    first few onsets are immediately near the right ref frame.
    """

    ref_onset_frames: np.ndarray
    search_window: int = 80      # +- ref frames around predicted position
    history: int = 8
    fallback_slope: float = 1.0
    seed_live_frame: int = 0
    seed_ref_frame: int = 0

    _matched: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Seed with the start anchor so the first onset has a basis.
        self._matched.append((int(self.seed_live_frame),
                              int(self.seed_ref_frame)))

    def observe_live_onset(self, live_frame: int, current_j_hat: int = 0) -> int | None:
        """Match a new live onset to a ref onset based on predicted
        timing (not on j_hat). Returns the matched ref frame, or
        ``None`` if no plausible candidate exists in the window."""

        rf = self.ref_onset_frames
        last_live, last_ref = self._matched[-1]
        if live_frame <= last_live:
            return None
        slope = self.slope()
        predicted_ref = last_ref + slope * (live_frame - last_live)
        lo = predicted_ref - self.search_window
        hi = predicted_ref + self.search_window
        mask = (rf >= max(last_ref + 1, lo)) & (rf <= hi)
        cand = rf[mask]
        if cand.size == 0:
            return None
        # Closest ref onset to the predicted time.
        best = int(cand[np.argmin(np.abs(cand - predicted_ref))])
        self._matched.append((int(live_frame), best))
        return best

    def slope(self) -> float:
        if len(self._matched) < 2:
            return self.fallback_slope
        pairs = self._matched[-(self.history + 1):]
        slopes = []
        for (a_lt, a_rt), (b_lt, b_rt) in zip(pairs, pairs[1:]):
            dl = b_lt - a_lt
            if dl <= 0:
                continue
            slopes.append((b_rt - a_rt) / dl)
        if not slopes:
            return self.fallback_slope
        return float(np.median(slopes))

    @property
    def n_matched(self) -> int:
        # Subtract the synthetic seed pair.
        return max(0, len(self._matched) - 1)
