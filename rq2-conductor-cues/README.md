# RQ2: Characterising conductor cues

Which movement features characterise conductor cues? How consistently do movement patterns recur across cues?

## Contents

- [features/](features/README.md): Eight movement signals and the selected cue/non-cue transition comparison.
- [detector/](detector/README.md): Cue features, clustering, model training and the blocked held-out detector export.

## Inputs

Conductor footage, the included cue/timeline labels and locally generated pose caches.

## Outputs

Feature comparisons, clustering results, trained models and cue detections for RQ4 from held-out blocks.

## Running and limitations

The blocked held-out export is the relevant detector input for RQ4. Training a model on all annotations is a separate operation. No pose caches or trained models are shipped.

See [setup instructions](../RUNNING.md), [excluded inputs](../INPUTS.md) and [shared dependencies](../shared/README.md).
