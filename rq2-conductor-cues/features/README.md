# Movement features

`analyze_cues.py --process-every 2` reproduces the descriptive analysis's YOLOv8n-pose sampling: every second frame is processed and intermediate frames are interpolated. `analyze_cues_v2.py` compares movement features at selected transition moments. `firebird_timeline.txt` supplies the timeline.

## Required context

Install this directory's requirements. Supply the manual label sequence documented in RUNNING.md; the heuristic-label path is not the original manual comparison. Both scripts require conductor footage.
