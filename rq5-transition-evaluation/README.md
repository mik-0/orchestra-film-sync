# RQ5: Transition-aware evaluation

How should transition error be scored when early and late offsets may be tolerated differently and statistics over the whole performance may hide large misses?

## Contents

- [analysis/](analysis/README.md): Transition-cost calculation, parameter sensitivity and summaries.

## Inputs

RQ1 timing limits and RQ4 transition-error rows. Full transition datasets are excluded.

## Outputs

Costs by condition, unreached-target counts and sensitivity to the catastrophic-error threshold and additional penalty.

## Running and limitations

`score_transitions.py` contains the `costs()` function used by the RQ4 experiment. The RQ4 script tries each combination of scoring parameters and saves the results. `transition_score_sensitivity.py` expects the older result format and needs different input from the final three-method experiment.

See [setup instructions](../RUNNING.md), [excluded inputs](../INPUTS.md) and [shared dependencies](../shared/README.md).
