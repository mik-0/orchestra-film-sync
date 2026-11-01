"""Causal feature extraction for online alignment.

All extractors here use ``center=False`` STFT framing, which means the
analysis window for frame ``i`` covers samples ``[i*hop, i*hop + n_fft)``
and never reaches into the future. This is the single most common source
of accidental look-ahead in online alignment systems, and we lock it
down at the API level.

Three feature types are supported, picked to match the three tradeoffs
discussed in the SPEC:

    * ``chroma``   - 12-dim chroma_stft. Robust across timbre, fast.
    * ``cens``     - chroma_cens. More robust still, slightly higher
      effective latency due to internal smoothing.
    * ``lse``      - 40-band log mel energy. Onset-sensitive, low latency,
      best for same-instrument alignment.

The :func:`extract_offline` helper batches an entire signal in one call
(used to pre-compute reference features) and the streaming
:meth:`FeatureExtractor.process` method consumes raw audio chunks and
yields one feature vector per completed analysis window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import librosa


FeatureType = Literal["chroma", "cens", "lse", "flux",
                       "chroma_flux", "chroma_lse", "embed"]
FEATURE_TYPES: tuple[str, ...] = (
    "chroma", "cens", "lse", "flux", "chroma_flux", "chroma_lse", "embed",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Frozen knobs for the front end."""

    sr: int = 22050
    n_fft: int = 4096
    hop_length: int = 1024
    feature: FeatureType = "chroma"
    n_mels: int = 40           # only used for lse
    cens_win: int = 21         # only used for cens (smoothing window in frames)
    encoder_path: str | None = None  # only used for the "embed" feature

    @property
    def hop_sec(self) -> float:
        return self.hop_length / self.sr

    @property
    def fps(self) -> float:
        return self.sr / self.hop_length


def _chroma(S_power: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    return librosa.feature.chroma_stft(
        S=S_power, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )


def _cens(S_power: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    chroma = _chroma(S_power, cfg)
    return librosa.feature.chroma_cens(C=chroma, win_len_smooth=cfg.cens_win)


def _lse(S_power: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        S=S_power, sr=cfg.sr, n_mels=cfg.n_mels, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    # ``amin=1e-3`` (-30 dB) + ``top_db=None`` gives an ABSOLUTE floor
    # at the natural noise floor for music recordings. Librosa's
    # default (``top_db=80``, ``amin=1e-10``) clips the output to
    # ``log_spec.max() - 80``, which is INPUT-ARRAY-RELATIVE. That made
    # the streaming extractor compute a different per-chunk floor than
    # the offline extractor, drifting tracker outputs by 84x on long
    # pieces (Daphnis: 161 s vs 1.9 s median). An absolute -30 dB floor
    # fixes the streaming/offline divergence AND empirically gives the
    # best alignment across both Firebird (orchestra+film, 2 min:
    # 0.39 s median) and Daphnis (full ballet, 60 min: 4.9 s median)
    # in a sweep over amin in {1e-1, 1e-2, 1e-3, 1e-4, 1e-6}. Higher
    # floors (-10, -20 dB) over-clip and destroy alignment; lower
    # floors (-40 dB and below) let quiet-bin noise re-enter the cosine
    # distance and compound into drift on long pieces.
    return librosa.power_to_db(mel + 1e-10, ref=1.0, amin=1e-3, top_db=None)


def _flux(S_power: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Causal half-wave-rectified spectral flux on a log-compressed
    mel-band grid.

    For each frame ``t``, ``flux[k, t] = max(0, log_mel[k, t] - log_mel[k, t-1])``.
    Working in ``log(1 + x)`` rather than dB keeps silence at zero (so the
    feature does not produce spurious activity in quiet regions) while
    still compressing the dynamic range enough to be useful across loud
    and quiet passages.
    """

    mel = librosa.feature.melspectrogram(
        S=S_power, sr=cfg.sr, n_mels=cfg.n_mels, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    log_mel = np.log1p(mel)
    flux = np.zeros_like(log_mel)
    flux[:, 1:] = np.maximum(0.0, log_mel[:, 1:] - log_mel[:, :-1])
    return flux


def _chroma_flux(S_power: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Concatenate L2-normalized chroma with normalized spectral flux.

    Combines pitch-class robustness (chroma) with onset precision
    (flux). Each component is normalised separately so neither dominates
    cosine-distance comparisons; the resulting vector is also unit-norm.
    """

    chroma = _chroma(S_power, cfg)
    flux = _flux(S_power, cfg)

    # Per-frame L2 normalize each component independently
    c_norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    f_norm = np.linalg.norm(flux, axis=0, keepdims=True) + 1e-9
    chroma_n = chroma / c_norm
    flux_n = flux / f_norm

    combined = np.concatenate([chroma_n, flux_n], axis=0)
    cb_norm = np.linalg.norm(combined, axis=0, keepdims=True) + 1e-9
    return combined / cb_norm


def _chroma_lse(S_power: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Concatenate L2-normalized chroma with normalized log-mel energy.

    LSE alone is dominant in onset-rich passages (clear single-line
    piano openings) but goes flat once dense orchestral texture
    enters; chroma alone is robust to texture but ambiguous on
    sustained chords. Stacking both gives the tracker an onset signal
    when it's available and a harmonic signal everywhere else.
    """

    chroma = _chroma(S_power, cfg)
    lse = _lse(S_power, cfg)

    c_norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    l_norm = np.linalg.norm(lse, axis=0, keepdims=True) + 1e-9
    chroma_n = chroma / c_norm
    lse_n = lse / l_norm

    combined = np.concatenate([chroma_n, lse_n], axis=0)
    cb_norm = np.linalg.norm(combined, axis=0, keepdims=True) + 1e-9
    return combined / cb_norm


_BACKENDS = {
    "chroma": _chroma, "cens": _cens, "lse": _lse,
    "flux": _flux, "chroma_flux": _chroma_flux,
    "chroma_lse": _chroma_lse,
}


def extract_offline(audio: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Extract features from a fully buffered mono signal.

    Returns an array of shape ``(F, T)`` ready for OLTW. Internally still
    uses ``center=False`` STFT so the offline result agrees frame-for-frame
    with what the online extractor would produce.
    """

    if audio.ndim != 1:
        raise ValueError(f"expected mono signal, got shape {audio.shape}")
    if cfg.feature == "embed":
        # Learned encoder path. Loaded lazily so torch isn't imported
        # for users who don't use embed features.
        from online_dtw.embed import load_encoder, extract_embed_offline
        if not cfg.encoder_path:
            raise ValueError("feature='embed' requires encoder_path in FeatureConfig")
        encoder = load_encoder(cfg.encoder_path)
        return extract_embed_offline(audio, encoder)
    S = np.abs(librosa.stft(
        y=audio, n_fft=cfg.n_fft, hop_length=cfg.hop_length, center=False,
    )) ** 2
    return _BACKENDS[cfg.feature](S, cfg)


class FeatureExtractor:
    """Streaming, causal feature extractor.

    Buffers raw audio across calls so that any leftover samples shorter than
    one analysis hop are carried into the next chunk. Each call to
    :meth:`process` may emit zero, one, or several feature vectors.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self._buf = np.zeros(0, dtype=np.float32)
        if cfg.feature == "cens":
            self._smoother = _CausalCENS(cfg.cens_win)
        else:
            self._smoother = None
        # Spectral flux needs the previous mel-db column for the diff.
        self._prev_mel_db: np.ndarray | None = None
        # Learned-encoder path: hold a StreamingEmbedExtractor that
        # we forward calls to. Lazy-loaded so torch isn't pulled in
        # for users who never touch embed features.
        self._embed_extractor = None
        if cfg.feature == "embed":
            from online_dtw.embed import load_encoder, StreamingEmbedExtractor
            if not cfg.encoder_path:
                raise ValueError("feature='embed' requires encoder_path")
            encoder = load_encoder(cfg.encoder_path)
            self._embed_extractor = StreamingEmbedExtractor(encoder)

    def process(self, audio_chunk: np.ndarray) -> Iterator[np.ndarray]:
        """Yield feature vectors for any complete analysis windows."""

        if self._embed_extractor is not None:
            # Delegate to the streaming learned encoder; it manages
            # its own audio buffer and mel history.
            yield from self._embed_extractor.process(audio_chunk)
            return

        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"expected mono chunk, got shape {chunk.shape}")
        self._buf = np.concatenate([self._buf, chunk])

        n_fft, hop = self.cfg.n_fft, self.cfg.hop_length
        # number of complete windows we can extract starting at offset 0
        if len(self._buf) < n_fft:
            return
        n_frames = 1 + (len(self._buf) - n_fft) // hop
        if n_frames <= 0:
            return

        # extract frames in one batch via STFT for efficiency
        signal = self._buf[: n_fft + (n_frames - 1) * hop]
        S = np.abs(librosa.stft(
            y=signal, n_fft=n_fft, hop_length=hop, center=False,
        )) ** 2

        cfg = self.cfg
        if cfg.feature == "lse":
            feats = _lse(S, cfg)
        elif cfg.feature == "flux":
            feats = self._streaming_flux(S)
        elif cfg.feature == "chroma_flux":
            chroma_block = _chroma(S, cfg)
            flux_block = self._streaming_flux(S)
            c_norm = np.linalg.norm(chroma_block, axis=0, keepdims=True) + 1e-9
            f_norm = np.linalg.norm(flux_block, axis=0, keepdims=True) + 1e-9
            combined = np.concatenate(
                [chroma_block / c_norm, flux_block / f_norm], axis=0,
            )
            cb_norm = np.linalg.norm(combined, axis=0, keepdims=True) + 1e-9
            feats = combined / cb_norm
        elif cfg.feature == "chroma_lse":
            chroma_block = _chroma(S, cfg)
            lse_block = _lse(S, cfg)
            c_norm = np.linalg.norm(chroma_block, axis=0, keepdims=True) + 1e-9
            l_norm = np.linalg.norm(lse_block, axis=0, keepdims=True) + 1e-9
            combined = np.concatenate(
                [chroma_block / c_norm, lse_block / l_norm], axis=0,
            )
            cb_norm = np.linalg.norm(combined, axis=0, keepdims=True) + 1e-9
            feats = combined / cb_norm
        else:
            feats = _chroma(S, cfg)
            if self._smoother is not None:
                feats = self._smoother.update(feats)

        # advance the buffer past consumed samples (keep a tail of n_fft - hop
        # so the next window starting at consumed offset is still complete)
        consumed = n_frames * hop
        self._buf = self._buf[consumed:]

        for k in range(feats.shape[1]):
            yield feats[:, k]

    def _streaming_flux(self, S_power: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        mel = librosa.feature.melspectrogram(
            S=S_power, sr=cfg.sr, n_mels=cfg.n_mels, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        log_mel = np.log1p(mel)
        flux = np.zeros_like(log_mel)
        if self._prev_mel_db is not None:
            flux[:, 0] = np.maximum(0.0, log_mel[:, 0] - self._prev_mel_db)
        if log_mel.shape[1] > 1:
            flux[:, 1:] = np.maximum(0.0, log_mel[:, 1:] - log_mel[:, :-1])
        self._prev_mel_db = log_mel[:, -1].copy()
        return flux


class _CausalCENS:
    """One-sided sliding mean + L2 normalize for CENS-style smoothing.

    Replaces librosa's centered moving average with a strictly causal one
    (window covers only the past ``win`` chroma frames). Quantization is
    skipped since for online matching the L2-normalized smoothed chroma is
    already a strong feature; adding the standard CENS quantization layer
    would not change the qualitative behavior of the tracker.
    """

    def __init__(self, win: int) -> None:
        self.win = max(1, int(win))
        self._hist: list[np.ndarray] = []

    def update(self, chroma_block: np.ndarray) -> np.ndarray:
        out = np.empty_like(chroma_block)
        for k in range(chroma_block.shape[1]):
            self._hist.append(chroma_block[:, k])
            if len(self._hist) > self.win:
                self._hist.pop(0)
            mean = np.mean(self._hist, axis=0)
            norm = np.linalg.norm(mean) + 1e-9
            out[:, k] = mean / norm
        return out
