"""High-level streaming tracker tying features + OLTW + confidence together.

Typical usage::

    from online_dtw import OnlineTracker, FeatureConfig
    from online_dtw.audio_io import load_mono

    cfg = FeatureConfig(feature="chroma")
    ref_audio, sr = load_mono("reference.wav", target_sr=cfg.sr)
    tracker = OnlineTracker(ref_audio, cfg)

    for chunk in stream_live_audio():
        for t_ref, conf in tracker.update(chunk):
            print(f"now at {t_ref:.2f} s, conf={conf:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator
import numpy as np

from online_dtw.features import FeatureExtractor, FeatureConfig, extract_offline
from online_dtw.oltw import OLTW, OLTWConfig
from online_dtw.hmm import HMMTracker, HMMConfig
from online_dtw.tempo import KalmanTempo, KalmanTempoConfig
from online_dtw.localizer import SubsequenceLocalizer, LocalizerConfig


@dataclass
class TrackerConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    oltw: OLTWConfig = field(default_factory=OLTWConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    kalman: KalmanTempoConfig = field(default_factory=KalmanTempoConfig)
    backend: str = "oltw"          # "oltw" | "hmm". Default keeps the
                                   # classic Dixon-style decoder; switch
                                   # to "hmm" for the tempo-aware Viterbi
                                   # forward filter (slower per frame but
                                   # more robust on cross-performance
                                   # rubato).
    localize: bool = False
    localize_buffer_sec: float = 4.0
    confidence_smoothing: int = 5  # frames
    silence_rms: float = 0.0       # sample-RMS threshold below which the
                                   # live signal is treated as silent.
                                   # When > 0 the tracker forces a 1:1
                                   # advance during silent periods,
                                   # which was historically a band-aid
                                   # for naive 1:1 seeding overshooting
                                   # in silent intros. With proper
                                   # seeding (offline DTW or anchors)
                                   # this gate actually hurts on
                                   # cross-performance audio, because the
                                   # silence durations of the two
                                   # performances are not equal -- so we
                                   # default to OFF. Set to ~0.005 if you
                                   # are forced to seed naively at t=0.


class OnlineTracker:
    """Reference-audio follower with a streaming ``update`` API."""

    def __init__(self, ref_audio: np.ndarray, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.feat_cfg = self.cfg.feature

        ref_feats = extract_offline(ref_audio, self.feat_cfg).T  # (M, F)
        self._ref_feats = ref_feats
        self.M = ref_feats.shape[0]

        # Build the chosen backend. Both expose the same step()/path API.
        backend = self.cfg.backend.lower()
        if backend == "oltw":
            self._inner = OLTW(ref_feats, self.cfg.oltw)
        elif backend == "hmm":
            kalman = KalmanTempo(self.cfg.kalman)
            # Mirror the OLTW seed onto the HMM if the caller set it.
            hmm_cfg = self.cfg.hmm
            if self.cfg.oltw.start_j and not hmm_cfg.start_j:
                hmm_cfg = HMMConfig(**{**hmm_cfg.__dict__,
                                        "start_j": self.cfg.oltw.start_j})
            self._inner = HMMTracker(ref_feats, hmm_cfg, kalman=kalman)
        else:
            raise ValueError(
                f"unknown backend {backend!r}; expected 'oltw' or 'hmm'"
            )
        self._extractor = FeatureExtractor(self.feat_cfg)

        self._localizer: SubsequenceLocalizer | None = None
        self._localized = not self.cfg.localize
        self._localize_buffer: list[np.ndarray] = []
        if self.cfg.localize:
            self._localizer = SubsequenceLocalizer(
                ref_feats, LocalizerConfig(distance=self.cfg.oltw.distance)
            )

        # Trailing-mean smoothing for the displayed confidence.
        self._conf_window: list[float] = []

    # ------------------------------------------------------------------ accessors

    @property
    def reference_duration(self) -> float:
        return self.M * self.feat_cfg.hop_sec

    @property
    def fps(self) -> float:
        return self.feat_cfg.fps

    @property
    def hop_sec(self) -> float:
        return self.feat_cfg.hop_sec

    @property
    def reference_features(self) -> np.ndarray:
        return self._ref_feats

    # ----------------------------------------------------------------- streaming

    def update(self, audio_chunk: np.ndarray) -> Iterator[tuple[float, float]]:
        """Process one audio chunk, yield ``(t_ref_seconds, confidence)`` per frame.

        The yielded estimates are emitted as soon as each feature frame is
        ready; one chunk of audio may produce many or zero alignment events
        depending on chunk size relative to the hop.
        """

        chunk_rms = float(np.sqrt(np.mean(np.asarray(audio_chunk, dtype=np.float32) ** 2))) \
            if audio_chunk.size else 0.0
        chunk_silent = chunk_rms < self.cfg.silence_rms

        for live_feat in self._extractor.process(audio_chunk):
            if not self._localized:
                self._localize_buffer.append(live_feat)
                buffer_secs = len(self._localize_buffer) * self.hop_sec
                if buffer_secs >= self.cfg.localize_buffer_sec:
                    A = np.stack(self._localize_buffer, axis=0)
                    j_start, _score = self._localizer.localize(A)
                    self._inner.reset(start_j=j_start)
                    self._localized = True
                    for fb in self._localize_buffer:
                        j_hat, conf = self._inner.step(fb)
                        yield self._emit(j_hat, conf)
                    self._localize_buffer.clear()
                continue

            # During silence the OLTW cost surface is uninformative and
            # the natural argmin races forward. force_advance=1 keeps the
            # DP fed but skips the argmin. The HMM has its own tempo
            # prior so it doesn't need that band-aid; if the kwarg is not
            # supported we just call step() normally.
            force = 1 if chunk_silent else None
            try:
                j_hat, conf = self._inner.step(live_feat, force_advance=force)
            except TypeError:
                j_hat, conf = self._inner.step(live_feat)
            yield self._emit(j_hat, conf if force is None else 0.0)

    # ------------------------------------------------------------------ internal

    def _emit(self, j_hat: int, conf_raw: float) -> tuple[float, float]:
        self._conf_window.append(conf_raw)
        if len(self._conf_window) > self.cfg.confidence_smoothing:
            self._conf_window.pop(0)
        smooth = float(np.mean(self._conf_window))
        return j_hat * self.hop_sec, smooth

    # ------------------------------------------------------------------ history

    def history(self) -> dict[str, np.ndarray]:
        """Snapshot of all per-frame state, useful for plotting."""

        n_arr = np.asarray(self._inner.path_n, dtype=np.int64)
        j_arr = np.asarray(self._inner.path_j, dtype=np.int64)
        out = {
            "live_time": n_arr * self.hop_sec,
            "ref_time": j_arr * self.hop_sec,
            "live_frame": n_arr,
            "ref_frame": j_arr,
            "confidence": np.asarray(self._inner.confidence_log, dtype=np.float64),
            "cost": np.asarray(self._inner.cost_log, dtype=np.float64),
        }
        # The HMM also exposes a per-frame tempo trace - useful for plots.
        if hasattr(self._inner, "tempo_log"):
            out["tempo"] = np.asarray(self._inner.tempo_log, dtype=np.float64)
        return out
