# Orchestra–film synchronisation

Research code accompanying *Audio following and conductor cues for orchestra–film synchronisation*, a TU Delft master's thesis by Tin Yau Mike Wu.

The source is organised in the same order as the thesis research questions. Each RQ has its own README and subfolders for its application, experiments or analysis. Dependencies used by several RQs live in `shared/`.

```text
rq1-acceptable-timing/
  study-app/                 Rating application
  analysis/                  Acceptance curves and diagnostics
rq2-conductor-cues/
  features/                  Pose signals and cue/non-cue comparison
  detector/                  Clustering, training and held-out export
rq3-audio-tracking/
  alignment/                 Offline reference preparation
  benchmark/                 Three published audio followers
  analysis/                  Failures, audio level and restart tests
rq4-cue-corrections/
  experiments/               Annotated and detected cue corrections
  analysis/                  Hold timing and result checks
rq5-transition-evaluation/
  analysis/                  Transition costs and sensitivity
shared/
  audio/                     Shared Python package and configurations
  analysis/                  Shared analysis helpers
  project_paths.py           Repository paths and cross-RQ imports
```

| Research question | Start here |
| --- | --- |
| RQ1: Acceptable timing | [RQ1 README](rq1-acceptable-timing/README.md) |
| RQ2: Characterising conductor cues | [RQ2 README](rq2-conductor-cues/README.md) |
| RQ3: Audio-tracking failures | [RQ3 README](rq3-audio-tracking/README.md) |
| RQ4: What video adds | [RQ4 README](rq4-cue-corrections/README.md) |
| RQ5: Transition-aware evaluation | [RQ5 README](rq5-transition-evaluation/README.md) |

## Scope

This repository contains the experiment code. Recordings, participant responses, trained models, pose caches and saved experiment outputs are excluded. See [INPUTS.md](INPUTS.md) for required inputs and [RUNNING.md](RUNNING.md) for setup instructions.

The thesis comparison uses Matchmaker 0.3.0 for Arzt and Dixon, and a pinned ConcertCue revision. Prototype HMM and OLTW modules retained in the shared package are dependencies of its existing imports, not the three published benchmark methods. For automatic cue evaluation, use the predictions from held-out blocks described under RQ2. The scripts for training on all annotations are also included.

No project-wide software licence has been selected. Third-party code is obtained separately and retains its own licences; see [THIRD_PARTY.md](THIRD_PARTY.md).
