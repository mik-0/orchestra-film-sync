# Cue analysis and detection

`recache_keypoints.py` creates the detector pose cache. `cue_characterisation_analysis.py` audits feature separation and clustering. `export_visual_blocked_detector.py` produces the held-out predictions for RQ4.

## Required context

Install requirements.txt here. `compute_lda_mask.py` followed by `live_detector.py` trains the separate all-data model. Models, caches and media must be generated or supplied locally. `combined_features.py`, `pca_rich.py` and `rf_v2_rich.py` support the retained feature/training code.
