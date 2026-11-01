# Transition scoring

`score_transitions.py` provides `costs()`, used by the RQ4 experiment for all three followers. `transition_score_sensitivity.py` runs the earlier scoring analysis and expects its original result format. `summarize_visual_contribution.py` contains summary helpers.

## Required context

Transitions that were never reached are counted separately because they have no firing time. The RQ4 script calls `costs()` for each combination of scoring parameters.
