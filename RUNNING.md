# Running the selected source

These setup commands have not been run on the reorganised code. Use Python 3.12, Node.js with npm, and FFmpeg. Keep the projects in their own environments.

## Published audio followers

From the repository root, install `uv`, then prepare the pinned external implementations:

```sh
python3 rq3-audio-tracking/benchmark/setup_published_baselines.py
uv pip install --python shared/audio/.cache-published-baselines/matchmaker-venv/bin/python -e 'shared/audio[sync]'
export THESIS_AUDIO_DIR="/path/to/audio"
export FIREBIRD_MEDIA_DIR="/path/to/firebird-media"
```

`setup_published_baselines.py` downloads Matchmaker 0.3.0 and ConcertCue at the revision/hash recorded in the script. It does not download the study recordings.

The single-pair benchmark entry point is `rq3-audio-tracking/benchmark/run_published_tracker.py`. It requires `--preset`, an approved offline `--reference`, an explicit `--method`, and an `--output` path. `sync_anchors.py` creates candidate offline alignments; candidate outputs are not automatically reviewed or interchangeable with the thesis references.

The large-recording wrapper is `run_completion_probe.py`. The analysis for all three followers also imports its Arzt memory-management helper. Its output-directory restriction and provisional metadata remain in the source; consult it before using it for a new benchmark.

Final analysis entry points:

- `analyse_all_followers.py`: failure summaries and restart replays for all three methods.
- `summarize_all_followers.py`: consolidate those outputs.
- `reanalyse_shared_audio_level.py`: use the same recorded audio-level series across methods.
- `run_all_follower_cue_corrections.py`: apply the saved cue protocol to all three methods.
- `transition_score_sensitivity.py`: vary the scoring parameters.

These scripts require the saved experiment inputs listed in [INPUTS.md](INPUTS.md).

## Conductor cues

```sh
python3 -m venv rq2-conductor-cues/detector/.venv
rq2-conductor-cues/detector/.venv/bin/python -m pip install -r rq2-conductor-cues/detector/requirements.txt
export FIREBIRD_VIDEO="/path/to/conductor-video.mov"
rq2-conductor-cues/detector/.venv/bin/python rq2-conductor-cues/detector/recache_keypoints.py --video "$FIREBIRD_VIDEO" --device cpu
rq2-conductor-cues/detector/.venv/bin/python rq2-conductor-cues/detector/export_visual_blocked_detector.py
```

`recache_keypoints.py` creates the pose cache from your video. Its default model is recorded in the script; model weights are downloaded separately. The included annotations refer to the original Firebird recording, so replace them when using different footage.

`cue_characterisation_analysis.py` requires `rq2-conductor-cues/features/analysis_results/analysis_v2.json` and the pose cache. `analyze_cues.py` generates the video project's initial pose cache; `analyze_cues_v2.py` creates the feature comparison. The saved manual labels, in the timeline's segment order, are:

```text
1,0,0,0,1,1,0,1,1,1,0,1,0,0,0,1
```

Pass these with `--labels` when reproducing that original comparison. Without explicit labels, the script has a heuristic-labelling path that is not a substitute for the saved manual labels.

For a model trained on all available annotations, run `compute_lda_mask.py` followed by `live_detector.py`. Use the blocked export above for the thesis automatic-cue evaluation.

## Acceptability application

```sh
cd rq1-acceptable-timing/study-app
SRC="/path/to/firebird-media" ./scripts/transcode.sh
npm ci
npm run dev
```

The UI saves locally and can export JSON/CSV. No participant responses are supplied. The optional API source requires your own database and environment configuration; `.env.example` contains placeholders only. No database schema or hosting configuration is included. The endpoint accepts trial POSTs without authentication and records the client IP. Add authentication and review the data collected before hosting it.

The analysis scripts in `rq1-acceptable-timing/analysis/` expect the original session format, but the original session files are excluded. They cannot reproduce the thesis ratings without separately available, authorised data.
