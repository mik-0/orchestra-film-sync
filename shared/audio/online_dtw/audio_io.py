"""Audio loading helpers.

Wraps ``librosa.load`` with cache support so that re-running the demo
scripts does not re-decode the same large MP3 every time. Caches are
stored next to the source file as ``.npz`` with the resampled mono signal.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import librosa


def load_mono(path: str | Path, target_sr: int, *, use_cache: bool = True) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at ``target_sr``.

    Parameters
    ----------
    path : str | Path
        Source audio file (any format librosa/audioread can decode).
    target_sr : int
        Sample rate to resample to.
    use_cache : bool
        If True, write a sidecar ``.<sr>.npz`` and read it back on
        subsequent calls. Saves ~30 s on repeat runs of long mp3s.
    """

    path = Path(path)
    cache = path.with_suffix(path.suffix + f".{target_sr}.npz")
    if use_cache and cache.exists():
        data = np.load(cache)
        return data["y"].astype(np.float32), int(data["sr"])

    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    y = y.astype(np.float32)
    if use_cache:
        np.savez(cache, y=y, sr=sr)
    return y, sr
