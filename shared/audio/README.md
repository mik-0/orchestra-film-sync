# Shared audio package

This directory contains pyproject.toml, requirements.txt, recording presets, and the online_dtw Python package. Install it with `pip install -e shared/audio` from the repository root. Add `[sync]` for Sync Toolbox.

## Required context

Prototype HMM/OLTW modules remain because existing package imports use them. The published comparison explicitly selects external implementations in RQ3. Preset prototype fields do not define the published decoder settings.
