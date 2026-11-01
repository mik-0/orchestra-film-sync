# Inputs that are not shipped

The source release excludes:

- All conductor, film and reference recordings, including media embedded in demos, screenshots and thumbnails.
- Raw acceptability-study responses.
- Trained model files, pose caches, audio features and downloaded model weights.
- Frozen tracker traces, reviewed alignment artifacts, experiment outputs and earlier reports.
- Private repository history, thesis source/PDFs, review comments, meeting transcripts, local environment files and deployment credentials.

The following small research labels are included because they define the procedures: the 22 cue timestamps and function labels, the film-transition timeline, and the reviewed cue/transition pairing file. These contain annotations, not the underlying media or participant responses.

## Final-analysis dependencies

| Entry point | Required external/generated inputs |
| --- | --- |
| `run_published_tracker.py` | Local audio, reviewed alignment JSON/dense arrays, and a matching reviewed-alignment manifest where required by the loader. |
| `analyse_all_followers.py` | `shared/audio/output/benchmark-completion-20261025/completed_comparison.json`, its referenced benchmark JSON/trace files, reviewed references, and audio for restarts. |
| `run_all_follower_cue_corrections.py` | Original `visual_all_cue_resets_v1` detector, landmarks and path arrays; corrected `visual_all_cue_resets_v2_causal_holds/results.json`; and per-method Firebird feature caches. |
| `cue_characterisation_analysis.py` | Video features in `analysis_v2.json` and the cue-classify pose cache. |
| Acceptability plots | Authorised session JSON in the original schema. |

Cloning this repository does not provide these inputs. Some can be regenerated from recordings and the included annotations. Reproducing the thesis results also requires the reviewed alignments, saved traces, software versions and original sessions. The analysis scripts expect saved files under `shared/audio/output/`.

## Reusing saved results

When supplying saved results, update file paths to match your local recordings and outputs.
