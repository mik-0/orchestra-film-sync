# RQ3: Audio-tracking failures

Where in the tested performances does audio-only following degrade, and how are its errors associated with recorded signal level?

## Contents

- [alignment/](alignment/README.md): Candidate offline alignments and reference checks.
- [benchmark/](benchmark/README.md): Pinned Arzt, Dixon and ConcertCue runners and implementation validation.
- [analysis/](analysis/README.md): Sustained failures, signal-level association and diagnostic restarts.

## Inputs

Audio recordings, reviewed offline alignments and the frozen comparison artifacts used by the final analysis.

## Outputs

Tracker traces, failure episodes, audio-level summaries and restart results. Shared intermediate outputs retain the names expected by downstream scripts under shared/audio/output.

## Running and limitations

Install the shared audio package and pinned published followers through the commands in RUNNING.md. The final analysis expects saved inputs listed in INPUTS.md; those inputs are not bundled.

See [setup instructions](../RUNNING.md), [excluded inputs](../INPUTS.md) and [shared dependencies](../shared/README.md).
