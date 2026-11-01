# RQ4: What video adds

How does using visible conductor cues affect film-transition timing compared with audio-only tracking?

## Contents

- [experiments/](experiments/README.md): Annotated-cue and automatic-cue corrections for the published followers.
- [analysis/](analysis/README.md): Cut confirmation/hold timing and validation of saved outputs.

## Inputs

RQ2 held-out detections, annotated cue positions, reviewed reference alignment, RQ3 feature caches and saved protocol artifacts.

## Outputs

Corrected position traces, applied/ignored cue events, transition times and summaries. The all-method runner calls the scoring function in RQ5.

## Running and limitations

Use run_all_follower_cue_corrections.py for the final three-method experiment. The older all-cue and paired-cue scripts are retained because they define the protocol and dependencies. Their results are not substitutes for the final three-method run.

See [setup instructions](../RUNNING.md), [excluded inputs](../INPUTS.md) and [shared dependencies](../shared/README.md).
