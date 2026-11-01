# Published follower comparison

`setup_published_baselines.py` installs pinned external implementations. `run_published_tracker.py` runs one published method. `run_completion_probe.py` provides the storage/cleanup wrapper used for completion runs. `compare_published_trackers.py` combines results.

## Required context

Dependencies and configurations live under shared/audio. The completion wrapper retains its historical output restriction and metadata. `validate_all_follower_decoders.py` is existing validation code; it has not been rerun in this export.
