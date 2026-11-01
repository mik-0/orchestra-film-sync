"""Transition costs used by the all-three-follower cue experiment."""
import numpy as np


def costs(rows,names,threshold=1.,penalty=1.):
    result={}
    for name in names:
        values=[r['conditions'][name]['signed_error_s'] for r in rows]
        reached=[v for v in values if v is not None]
        def cost(e):
            if -.60<=e<=.33:return 0.
            if abs(e)>=threshold:return 1.+penalty
            tolerance=.60 if e<0 else .33
            return (abs(e)-tolerance)/(threshold-tolerance)
        result[name]={'mean_cost':float(np.mean([cost(e) for e in reached])) if len(reached)==len(values) else None,
                      'mean_cost_reached_only':float(np.mean([cost(e) for e in reached])) if reached else None,
                      'unreached_targets':len(values)-len(reached),
                      'catastrophes':sum(abs(e)>=threshold for e in reached)}
    return result
