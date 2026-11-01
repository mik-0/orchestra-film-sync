"""Online Dynamic Time Warping for real-time audio-to-audio alignment.

The library implements a strict-causal On-Line Time Warping (OLTW) tracker
in the style of Dixon (2005), with a few practical extensions:

    * Causal feature extraction (chroma / CENS / log-spectral energy).
    * Bounded-band frontier DP with O(c) work per live frame.
    * Local-margin confidence output.
    * Optional subsequence-DTW startup localizer for mid-piece entry.

The high-level entry point is :class:`online_dtw.tracker.OnlineTracker`,
which takes a reference audio file at construction time and exposes a
streaming :meth:`update` method that consumes mono audio chunks and
returns the current estimated reference timestamp.
"""

from online_dtw.features import FeatureExtractor, FeatureConfig, FEATURE_TYPES
from online_dtw.oltw import OLTW, OLTWConfig
from online_dtw.hmm import HMMTracker, HMMConfig
from online_dtw.tempo import KalmanTempo, KalmanTempoConfig
from online_dtw.tracker import OnlineTracker, TrackerConfig
from online_dtw.localizer import SubsequenceLocalizer, LocalizerConfig

__all__ = [
    "FeatureExtractor",
    "FeatureConfig",
    "FEATURE_TYPES",
    "OLTW",
    "OLTWConfig",
    "HMMTracker",
    "HMMConfig",
    "KalmanTempo",
    "KalmanTempoConfig",
    "OnlineTracker",
    "TrackerConfig",
    "SubsequenceLocalizer",
    "LocalizerConfig",
]
