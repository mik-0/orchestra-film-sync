"""Saveable / loadable dashboard configurations.

A ``Preset`` captures everything the live dashboard needs to align a
specific audio pair: the audio paths, the feature/OLTW knobs that
were tuned for that pair, and the streaming defaults.

Presets are stored as TOML files in ``online-dtw/configs/<name>.toml``.
Audio + anchor paths inside the TOML may be absolute or relative to
the project root (``online-dtw/``).

Typical use::

    preset = Preset.load("yuja_fedorova")
    preset.audio.live = Path("/some/other/recording.mp3")  # CLI override
    preset.save("yuja_fedorova_alt")                       # round-trip
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _resolve(p: str | Path | None) -> Path | None:
    if p is None or p == "":
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(p)))
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in path: {p}")
    path = Path(expanded)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _relativize(p: Path | None) -> str | None:
    if p is None:
        return None
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


@dataclass
class AudioCfg:
    live: Path
    ref: Path
    anchors: Path | None = None


@dataclass
class FeatureCfg:
    name: str = "chroma_lse"
    sr: int = 22050
    hop: int = 1024
    n_fft: int = 4096
    # Only used when name == "embed". Relative to project root,
    # or absolute. Point at a .pt produced by scripts/train_encoder.py.
    encoder_path: str | None = None


@dataclass
class OLTWCfg:
    search_width: int = 400
    weight_vertical: float = 2.0
    weight_horizontal: float = 1.0
    weight_diagonal: float = 2.0
    silence_rms: float = 0.001
    max_advance: int = 2
    coast_margin: float = 0.65
    backward_penalty: float = 0.001


@dataclass
class HMMCfg:
    search_width: int = 400
    beta: float = 32.0
    sigma_floor: float = 0.8
    sigma_scale: float = 8.0
    max_advance: int = 4
    min_advance: int = 0
    backward_penalty: float = 0.001
    skip_mass: float = 1e-5      # global skip mass for absorbing-state
                                 # escape; 0 disables the fix. See
                                 # online_dtw/hmm.py:HMMConfig.skip_mass.
    emission_buffer_frames: int = 0   # 0 = single-frame emission. >0 to
                                      # average emission over a sliding
                                      # window of recent live frames; see
                                      # online_dtw/hmm.py for details.


@dataclass
class TrimCfg:
    enabled: bool = True
    pad_sec: float = 1.0


@dataclass
class StreamCfg:
    start_sec: float = 0.0
    max_seconds: float = 240.0
    chunk_sec: float = 0.10
    backend: str = "oltw"
    play: str = "live"
    cost_resolution: int = 600


@dataclass
class OnsetPriorCfg:
    """Onset-anchored slope prior for the HMM transition kernel.

    Off by default: in the current implementation, slope-from-matched-
    onsets feeds back into the matching window (next prediction uses
    current slope), so a single bad match biases all subsequent ones
    and the slope drifts. A proper fix is a small online onset-DTW
    that aligns the live-onset sequence to the ref-onset sequence
    with optional skips; without that, leaving it off gives better
    numbers than turning it on. The code is kept for future work.
    """

    enabled: bool = False
    # Peak threshold on the onset_strength curve (matches librosa
    # scale -- power_to_db, mean across mel bands). Peak must exceed
    # trailing mean by this much. Tune higher for noisier audio.
    delta: float = 0.5
    # min frames between consecutive emitted onsets
    wait: int = 3
    # +- band (in ref frames) around the predicted ref position
    # where a new live onset is allowed to match a ref onset
    search_window: int = 80
    # number of recent matched onset pairs used to compute the slope
    history: int = 8


@dataclass
class Preset:
    """A complete dashboard configuration."""

    name: str = "custom"
    description: str = ""
    audio: AudioCfg = field(default_factory=lambda: AudioCfg(
        live=PROJECT_ROOT / "live.mp3",
        ref=PROJECT_ROOT / "ref.mp3",
    ))
    feature: FeatureCfg = field(default_factory=FeatureCfg)
    oltw: OLTWCfg = field(default_factory=OLTWCfg)
    hmm: HMMCfg = field(default_factory=HMMCfg)
    onset_prior: OnsetPriorCfg = field(default_factory=OnsetPriorCfg)
    trim: TrimCfg = field(default_factory=TrimCfg)
    stream: StreamCfg = field(default_factory=StreamCfg)

    # ----- I/O ------------------------------------------------------

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Preset":
        """Load a preset by name (looked up in ``configs/<name>.toml``)
        or by explicit path."""

        path = Path(name_or_path)
        if not path.is_file():
            path = CONFIGS_DIR / f"{name_or_path}.toml"
        if not path.is_file():
            raise FileNotFoundError(
                f"preset not found: tried {name_or_path!r} and "
                f"{CONFIGS_DIR / f'{name_or_path}.toml'}"
            )
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls._from_dict(data, source_path=path)

    @classmethod
    def _from_dict(cls, data: dict, source_path: Path | None = None) -> "Preset":
        audio_raw = data.get("audio", {})
        audio = AudioCfg(
            live=_resolve(audio_raw["live"]),
            ref=_resolve(audio_raw["ref"]),
            anchors=_resolve(audio_raw.get("anchors")),
        )
        feature = FeatureCfg(**data.get("feature", {}))
        oltw = OLTWCfg(**data.get("oltw", {}))
        hmm = HMMCfg(**data.get("hmm", {}))
        onset_prior = OnsetPriorCfg(**data.get("onset_prior", {}))
        trim = TrimCfg(**data.get("trim", {}))
        stream = StreamCfg(**data.get("stream", {}))
        return cls(
            name=data.get("name", source_path.stem if source_path else "custom"),
            description=data.get("description", ""),
            audio=audio, feature=feature, oltw=oltw, hmm=hmm,
            onset_prior=onset_prior, trim=trim, stream=stream,
        )

    def save(self, name_or_path: str | Path) -> Path:
        """Write this preset to disk and return the file path."""

        path = Path(name_or_path)
        if not path.suffix:
            path = CONFIGS_DIR / f"{name_or_path}.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_toml())
        return path

    def to_toml(self) -> str:
        """Serialise to TOML. Hand-rolled to avoid the tomli_w
        dependency; the schema is small and known."""

        lines: list[str] = []

        def emit_str(k: str, v: str) -> None:
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{k} = "{escaped}"')

        def emit_kv(k: str, v: Any) -> None:
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            elif isinstance(v, str):
                emit_str(k, v)
            elif v is None:
                emit_str(k, "")
            else:
                emit_str(k, str(v))

        emit_str("name", self.name)
        if self.description:
            emit_str("description", self.description)
        lines.append("")

        lines.append("[audio]")
        emit_str("live", _relativize(self.audio.live) or "")
        emit_str("ref",  _relativize(self.audio.ref)  or "")
        if self.audio.anchors is not None:
            emit_str("anchors", _relativize(self.audio.anchors) or "")
        lines.append("")

        for section, obj in (("feature",      self.feature),
                             ("oltw",         self.oltw),
                             ("hmm",          self.hmm),
                             ("onset_prior",  self.onset_prior),
                             ("trim",         self.trim),
                             ("stream",       self.stream)):
            lines.append(f"[{section}]")
            for k, v in asdict(obj).items():
                emit_kv(k, v)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # ----- runtime overrides ----------------------------------------

    def with_overrides(self, **kwargs: Any) -> "Preset":
        """Return a new preset with any non-None overrides applied.

        Keys use dotted paths into the nested config, e.g.
        ``audio.live``, ``oltw.weight_vertical``, ``feature.name``.
        ``None`` values are ignored so CLI flags that were not set
        leave the preset untouched.
        """

        new = replace(self,
                      audio=replace(self.audio),
                      feature=replace(self.feature),
                      oltw=replace(self.oltw),
                      hmm=replace(self.hmm),
                      onset_prior=replace(self.onset_prior),
                      trim=replace(self.trim),
                      stream=replace(self.stream))
        for key, value in kwargs.items():
            if value is None:
                continue
            section, _, field_name = key.partition(".")
            target = getattr(new, section)
            setattr(target, field_name, value)
        return new


def list_presets() -> list[Path]:
    """All ``configs/*.toml`` files, sorted by name."""

    if not CONFIGS_DIR.exists():
        return []
    return sorted(CONFIGS_DIR.glob("*.toml"))
