"""Self-supervised feature encoder trained on a single reference.

Goal: replace hand-engineered features (chroma / LSE) with a small
neural embedding that maps musically-equivalent moments to similar
vectors regardless of recording surface (orchestra, hall, mic,
tuning). The encoder is trained per-preset on the **reference**
audio ONLY, with strong augmentations to teach invariance. The
live recording is never seen during training -- this is the
strictest setup: at inference time the encoder is a frozen
function applied causally to the live mel-spectrogram stream.

Architecture: a 3-layer MLP on a flattened (ctx x n_mels) mel
window -- tiny (~96K params), trains in a few seconds on CPU.

Training: SimCLR-style contrastive (NT-Xent) loss over pairs
``(window_t, aug(window_t))`` with all other in-batch windows as
negatives. Augmentations on the mel-spec:

  * SpecAugment frequency mask (zero out random mel bands)
  * SpecAugment time mask (zero out random frames)
  * Additive Gaussian noise on log-mel
  * Small temporal shift (1-3 frames, simulates mic latency)
  * Gain perturbation (multiplicative on the energy)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa


@dataclass
class EncoderConfig:
    n_mels: int = 40
    ctx: int = 16            # frames of context per window
    embed_dim: int = 32
    hidden: int = 128
    sr: int = 22050
    n_fft: int = 4096
    hop_length: int = 1024
    fmin: float = 0.0
    fmax: float | None = None


class FrameEncoder(nn.Module):
    """MLP encoder: (B, ctx, n_mels) -> (B, embed_dim), L2-normed."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.ctx * cfg.n_mels
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.flatten(1))
        return F.normalize(z, dim=-1)


# ---- mel-spectrogram (shared between training and inference) ---------------

def log_mel(audio: np.ndarray, cfg: EncoderConfig) -> np.ndarray:
    """``(n_mels, T)`` log-mel-spectrogram in dB. Causal: center=False."""
    fmax = cfg.fmax if cfg.fmax is not None else cfg.sr / 2.0
    S = np.abs(librosa.stft(
        y=audio, n_fft=cfg.n_fft, hop_length=cfg.hop_length, center=False,
    )) ** 2
    mel = librosa.feature.melspectrogram(
        S=S, sr=cfg.sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=fmax,
        n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )
    # top_db=None: see features.py:_lse for the explanation. librosa's
    # default 80 dB clipping is computed against the input array's local
    # max, which differs between offline and streaming chunks.
    return librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)


def windowize(log_mel_t: np.ndarray, ctx: int) -> np.ndarray:
    """Slide a causal window of ``ctx`` frames over a log-mel stream.

    Output shape ``(T_windows, ctx, n_mels)`` where ``T_windows =
    T - ctx + 1``. The window for output index ``i`` is the input
    columns ``[i, i + ctx - 1]`` (rightmost = current frame), which
    is the natural choice for streaming: a feature published at
    time ``t`` should depend only on frames ``[t - ctx + 1, t]``.
    """
    n_mels, T = log_mel_t.shape
    if T < ctx:
        raise ValueError(f"need at least {ctx} frames, got {T}")
    n_win = T - ctx + 1
    out = np.empty((n_win, ctx, n_mels), dtype=np.float32)
    for i in range(n_win):
        out[i] = log_mel_t[:, i:i + ctx].T
    return out


# ---- augmentations (numpy on log-mel windows) ------------------------------

def _spec_augment(
    windows: np.ndarray, *,
    f_mask_max: int = 8, t_mask_max: int = 3, noise_std: float = 1.5,
    gain_db: float = 3.0, shift_max: int = 2, rng: np.random.Generator,
) -> np.ndarray:
    """SimCLR-style augmentations applied to a batch of mel windows."""
    out = windows.copy()
    B, ctx, n_mels = out.shape
    for b in range(B):
        w = int(rng.integers(0, f_mask_max + 1))
        if w > 0:
            f0 = int(rng.integers(0, max(1, n_mels - w + 1)))
            out[b, :, f0:f0 + w] = 0.0
    for b in range(B):
        w = int(rng.integers(0, t_mask_max + 1))
        if w > 0:
            t0 = int(rng.integers(0, max(1, ctx - w + 1)))
            out[b, t0:t0 + w, :] = 0.0
    if noise_std > 0:
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    if gain_db > 0:
        g = rng.uniform(-gain_db, gain_db, size=(B, 1, 1)).astype(np.float32)
        out = out + g
    if shift_max > 0:
        for b in range(B):
            s = int(rng.integers(-shift_max, shift_max + 1))
            if s > 0:
                out[b, s:] = out[b, :ctx - s]
                out[b, :s] = out[b, 0:1]
            elif s < 0:
                s = -s
                out[b, :ctx - s] = out[b, s:]
                out[b, ctx - s:] = out[b, -1:]
    return out


# ---- training --------------------------------------------------------------

def nt_xent_loss(
    z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """SimCLR NT-Xent. ``z_a, z_b`` are L2-normalised, shape ``(B, D)``.
    Positive: a[i] -- b[i]. Negatives: all other a, b in the batch."""
    B = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)
    sim = z @ z.t() / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, pos_idx)


def train_encoder(
    ref_audio: np.ndarray, cfg: EncoderConfig | None = None, *,
    epochs: int = 60, batch_size: int = 256,
    lr: float = 1e-3, temperature: float = 0.1,
    seed: int = 0, log_every: int = 10,
    pitch_shifts: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    time_stretches: tuple[float, ...] = (0.92, 1.0, 1.08),
    device: str | torch.device = "cpu",
) -> FrameEncoder:
    """Train the encoder on a single reference recording.

    The reference is the ONLY audio source used. Audio-domain
    augmentations (pitch shift / time stretch) produce multiple
    "views" of the reference that approximate how a different
    performance / recording might sound; the per-window spec
    augmentations add finer noise. The contrastive loss makes the
    encoder discriminative about temporal position while ignoring
    these surface variations. Live audio is never touched.
    """
    cfg = cfg or EncoderConfig()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ---- build multiple log-mel views of the reference ----
    # Each (pitch_shift, time_stretch) pair produces an aligned
    # log-mel stream that the encoder should treat as equivalent
    # to the canonical one. Time-stretching is implemented on the
    # audio first, then mel; after stretch the i-th frame in the
    # stretched view corresponds to roughly the (i / stretch)-th
    # frame of the canonical view, which we use to align positives.
    print(f"  building augmented views of the reference "
          f"({len(ref_audio)/cfg.sr:.0f}s, "
          f"{len(pitch_shifts)} pitch x {len(time_stretches)} stretch)...")
    canonical = log_mel(ref_audio, cfg)
    mean = canonical.mean(axis=1, keepdims=True)
    std = canonical.std(axis=1, keepdims=True) + 1e-6
    canonical = (canonical - mean) / std
    base_windows = windowize(canonical, cfg.ctx)
    N = base_windows.shape[0]
    print(f"  {N} canonical windows, ctx={cfg.ctx}")

    # Precompute aligned augmented windows so positives at frame i
    # share an unambiguous "anchor frame" with the canonical view.
    aug_views = [base_windows]
    for p in pitch_shifts:
        for ts in time_stretches:
            if abs(p) < 1e-6 and abs(ts - 1.0) < 1e-6:
                continue
            aug = ref_audio
            if abs(p) > 1e-6:
                aug = librosa.effects.pitch_shift(
                    aug, sr=cfg.sr, n_steps=p,
                )
            if abs(ts - 1.0) > 1e-6:
                aug = librosa.effects.time_stretch(aug, rate=ts)
            aug_lm = log_mel(aug, cfg)
            aug_lm = (aug_lm - mean) / std
            # Map each canonical frame i to aug frame round(i * ts):
            # after time_stretch(rate=ts), aug has T_aug = T / ts frames,
            # and frame k_aug corresponds to canonical frame k_aug * ts.
            n_aug_frames = aug_lm.shape[1]
            aug_win_count = n_aug_frames - cfg.ctx + 1
            if aug_win_count <= 0:
                continue
            aug_wins = windowize(aug_lm, cfg.ctx)
            # Resample to match canonical window count: for canonical
            # window i, take aug window at index round(i / ts) if it
            # exists, else clamp.
            mapped = np.empty_like(base_windows)
            for i in range(N):
                src = min(int(round(i / ts)), aug_win_count - 1)
                mapped[i] = aug_wins[src]
            aug_views.append(mapped)

    # Stack views: shape (V, N, ctx, n_mels). At training time we
    # pick two random views per anchor for the positive pair.
    aug_arr = np.stack(aug_views, axis=0).astype(np.float32)
    V = aug_arr.shape[0]
    print(f"  {V} aligned views available per anchor frame")
    windows = base_windows  # only used for shape compatibility

    encoder = FrameEncoder(cfg).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    encoder.train(True)

    for ep in range(epochs):
        order = rng.permutation(N)
        losses = []
        for k in range(0, N, batch_size):
            batch_idx = order[k:k + batch_size]
            B = batch_idx.size
            if B < 8:
                continue
            # Pick two distinct views per anchor for the positive
            # pair -- the encoder must produce close embeddings for
            # the same musical moment seen under different
            # surface conditions (pitch, tempo, mel-mask, noise).
            view_a = rng.integers(0, V, size=B)
            view_b = rng.integers(0, V, size=B)
            x_a = aug_arr[view_a, batch_idx]
            x_b = aug_arr[view_b, batch_idx]
            # Layer on spec-augment for extra robustness.
            x_a = _spec_augment(x_a, rng=rng)
            x_b = _spec_augment(x_b, rng=rng)
            ta = torch.from_numpy(x_a).to(device)
            tb = torch.from_numpy(x_b).to(device)
            z_a = encoder(ta)
            z_b = encoder(tb)
            loss = nt_xent_loss(z_a, z_b, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"    epoch {ep + 1:>3d}/{epochs}  loss={np.mean(losses):.4f}")

    encoder.mean = torch.from_numpy(mean.astype(np.float32))
    encoder.std  = torch.from_numpy(std.astype(np.float32))
    encoder.train(False)
    return encoder


# ---- save / load -----------------------------------------------------------

def save_encoder(encoder: FrameEncoder, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cfg": encoder.cfg.__dict__,
        "state_dict": encoder.state_dict(),
        "mean": encoder.mean,
        "std":  encoder.std,
    }, path)


def load_encoder(path: Path | str, device: str | torch.device = "cpu") -> FrameEncoder:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = EncoderConfig(**blob["cfg"])
    enc = FrameEncoder(cfg).to(device)
    enc.load_state_dict(blob["state_dict"])
    enc.mean = blob["mean"]
    enc.std  = blob["std"]
    enc.train(False)
    return enc


# ---- offline + streaming inference -----------------------------------------

def extract_embed_offline(
    audio: np.ndarray, encoder: FrameEncoder,
) -> np.ndarray:
    """Run the trained encoder over the whole audio. Returns
    ``(embed_dim, T_emb)`` so it slots into the existing feature-
    array conventions (axis 0 = features, axis 1 = time)."""
    cfg = encoder.cfg
    lm = log_mel(audio, cfg)
    mean = encoder.mean.numpy() if hasattr(encoder.mean, "numpy") else encoder.mean
    std  = encoder.std.numpy()  if hasattr(encoder.std, "numpy")  else encoder.std
    lm = (lm - mean) / (std + 1e-6)
    wins = windowize(lm, cfg.ctx)
    with torch.no_grad():
        z = encoder(torch.from_numpy(wins))
    pad = np.repeat(z[:1].cpu().numpy(), cfg.ctx - 1, axis=0)
    full = np.concatenate([pad, z.cpu().numpy()], axis=0)
    return full.T.astype(np.float32)


class StreamingEmbedExtractor:
    """Causal streaming embedder. One embedding per analysis frame;
    each embedding for time ``t`` depends only on mel columns
    ``[t - ctx + 1, t]``."""

    def __init__(self, encoder: FrameEncoder) -> None:
        self.encoder = encoder
        self.cfg = encoder.cfg
        self._buf = np.zeros(0, dtype=np.float32)
        self._mel_hist: list[np.ndarray] = []
        self._mean = encoder.mean.numpy() if hasattr(encoder.mean, "numpy") else encoder.mean
        self._std  = encoder.std.numpy()  if hasattr(encoder.std, "numpy")  else encoder.std

    def process(self, audio_chunk: np.ndarray):
        cfg = self.cfg
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        self._buf = np.concatenate([self._buf, chunk])
        if len(self._buf) < cfg.n_fft:
            return
        n_frames = 1 + (len(self._buf) - cfg.n_fft) // cfg.hop_length
        if n_frames <= 0:
            return
        signal = self._buf[: cfg.n_fft + (n_frames - 1) * cfg.hop_length]
        S = np.abs(librosa.stft(
            y=signal, n_fft=cfg.n_fft, hop_length=cfg.hop_length, center=False,
        )) ** 2
        mel = librosa.feature.melspectrogram(
            S=S, sr=cfg.sr, n_mels=cfg.n_mels, fmin=cfg.fmin,
            fmax=cfg.fmax if cfg.fmax is not None else cfg.sr / 2.0,
            n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        )
        log = librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)
        log = (log - self._mean) / (self._std + 1e-6)

        for k in range(log.shape[1]):
            col = log[:, k]
            self._mel_hist.append(col)
            if len(self._mel_hist) > cfg.ctx + 4:
                self._mel_hist = self._mel_hist[-(cfg.ctx + 4):]
            if len(self._mel_hist) >= cfg.ctx:
                win = np.stack(self._mel_hist[-cfg.ctx:], axis=0)
            else:
                pad = np.repeat(
                    self._mel_hist[0][None, :], cfg.ctx - len(self._mel_hist),
                    axis=0,
                )
                win = np.concatenate(
                    [pad, np.stack(self._mel_hist, axis=0)], axis=0,
                )
            with torch.no_grad():
                z = self.encoder(torch.from_numpy(win[None, ...]))
            yield z[0].cpu().numpy().astype(np.float32)

        consumed = n_frames * cfg.hop_length
        self._buf = self._buf[consumed:]
