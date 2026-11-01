"""Repository locations and import paths for the RQ-organised source release."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
AUDIO_ROOT = ROOT / "shared/audio"
SCRIPT_PATHS = {'analyse_all_followers.py': 'rq3-audio-tracking/analysis/analyse_all_followers.py',
 'analyze_audio_failures.py': 'shared/analysis/analyze_audio_failures.py',
 'audio_failure_analysis.py': 'shared/analysis/audio_failure_analysis.py',
 'audio_failure_threshold_sensitivity.py': 'rq3-audio-tracking/analysis/audio_failure_threshold_sensitivity.py',
 'audit_alignment_reference.py': 'rq3-audio-tracking/alignment/audit_alignment_reference.py',
 'audit_all_follower_cues.py': 'rq4-cue-corrections/analysis/audit_all_follower_cues.py',
 'audit_all_follower_results.py': 'rq3-audio-tracking/analysis/audit_all_follower_results.py',
 'compare_published_trackers.py': 'rq3-audio-tracking/benchmark/compare_published_trackers.py',
 'evaluate_anchors.py': 'rq3-audio-tracking/alignment/evaluate_anchors.py',
 'reanalyse_audio_energy.py': 'rq3-audio-tracking/analysis/reanalyse_audio_energy.py',
 'reanalyse_shared_audio_level.py': 'rq3-audio-tracking/analysis/reanalyse_shared_audio_level.py',
 'recompute_visual_hold_times.py': 'rq4-cue-corrections/analysis/recompute_visual_hold_times.py',
 'run_all_follower_cue_corrections.py': 'rq4-cue-corrections/experiments/run_all_follower_cue_corrections.py',
 'run_completion_probe.py': 'rq3-audio-tracking/benchmark/run_completion_probe.py',
 'run_published_tracker.py': 'rq3-audio-tracking/benchmark/run_published_tracker.py',
 'setup_published_baselines.py': 'rq3-audio-tracking/benchmark/setup_published_baselines.py',
 'summarize_all_followers.py': 'rq3-audio-tracking/analysis/summarize_all_followers.py',
 'summarize_visual_contribution.py': 'rq5-transition-evaluation/analysis/summarize_visual_contribution.py',
 'sync_anchors.py': 'rq3-audio-tracking/alignment/sync_anchors.py',
 'transition_score_sensitivity.py': 'rq5-transition-evaluation/analysis/transition_score_sensitivity.py',
 'validate_all_follower_decoders.py': 'rq3-audio-tracking/benchmark/validate_all_follower_decoders.py',
 'visual_all_cue_resets.py': 'rq4-cue-corrections/experiments/visual_all_cue_resets.py',
 'visual_paired_cue_resets.py': 'rq4-cue-corrections/experiments/visual_paired_cue_resets.py'}
IMPORT_DIRS = ['shared/audio',
 'rq1-acceptable-timing/analysis',
 'rq2-conductor-cues/detector',
 'rq2-conductor-cues/features',
 'rq3-audio-tracking/alignment',
 'rq3-audio-tracking/analysis',
 'rq3-audio-tracking/benchmark',
 'rq4-cue-corrections/analysis',
 'rq4-cue-corrections/experiments',
 'rq5-transition-evaluation/analysis',
 'shared/analysis']

def activate():
    """Make the single copies of shared and cross-RQ modules importable."""
    for directory in IMPORT_DIRS:
        path = str(ROOT / directory)
        if path not in sys.path:
            sys.path.append(path)


def script_path(name):
    """Find a relocated entry script when recording provenance or invoking it."""
    return ROOT / SCRIPT_PATHS[name]
