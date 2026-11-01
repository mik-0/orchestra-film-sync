"""Install and verify the pinned published tracker implementations."""

from __future__ import annotations

# Resolve shared imports from this public repository layout.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
for _layout_root in _LayoutPath(__file__).resolve().parents:
    if (_layout_root / "shared" / "project_paths.py").is_file():
        _layout_sys.path.insert(0, str(_layout_root / "shared"))
        break
else:
    raise RuntimeError("Cannot locate shared/project_paths.py")
from project_paths import ROOT as _REPO_ROOT, AUDIO_ROOT as _AUDIO_ROOT, activate as _activate_layout, script_path as _script_path
_activate_layout()


import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = _AUDIO_ROOT
CACHE = PROJECT_ROOT / ".cache-published-baselines"
MATCHMAKER_VERSION = "0.3.0"
CONCERTCUE_URL = "https://github.com/matthewcaren/web-score-following.git"
CONCERTCUE_COMMIT = "2876aacae008aa75a1512b70e3aca0bca653b4fa"
CONCERTCUE_ARCHIVE_SHA256 = (
    "b41766371e3035ee4ec086f22848cce726b9c569618e95c1ce15dacbfb258bcf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout.strip():
        print(completed.stdout.rstrip())
    return completed.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def python_path(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def install_matchmaker(cache: Path) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required to install the pinned Matchmaker environment")
    venv = cache / "matchmaker-venv"
    python = python_path(venv)
    if not python.exists():
        run([uv, "venv", "--python", "3.12", str(venv)])
    run([uv, "pip", "install", "--python", str(python), f"pymatchmaker=={MATCHMAKER_VERSION}"])
    run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as m; "
                f"assert m.version('pymatchmaker') == '{MATCHMAKER_VERSION}'; "
                "print('pymatchmaker', m.version('pymatchmaker'))"
            ),
        ]
    )
    return python


def install_concertcue(cache: Path) -> Path:
    checkout = cache / "web-score-following"
    if not checkout.exists():
        run(["git", "clone", "--filter=blob:none", CONCERTCUE_URL, str(checkout)])
    run(["git", "fetch", "origin", CONCERTCUE_COMMIT], cwd=checkout)
    run(["git", "checkout", "--detach", CONCERTCUE_COMMIT], cwd=checkout)
    revision = run(["git", "rev-parse", "HEAD"], cwd=checkout).strip()
    if revision != CONCERTCUE_COMMIT:
        raise SystemExit(f"wrong ConcertCue revision: {revision}")
    archive = checkout / "web/public/python/libaudio_rt.zip"
    actual = sha256(archive)
    if actual != CONCERTCUE_ARCHIVE_SHA256:
        raise SystemExit(
            "ConcertCue tracker archive hash mismatch: "
            f"expected {CONCERTCUE_ARCHIVE_SHA256}, got {actual}"
        )
    print(f"verified {archive}: {actual}")
    return archive


def main() -> None:
    args = parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    python = install_matchmaker(cache)
    archive = install_concertcue(cache)
    print("published baselines ready")
    print(f"Matchmaker Python: {python}")
    print(f"ConcertCue archive: {archive}")


if __name__ == "__main__":
    main()
