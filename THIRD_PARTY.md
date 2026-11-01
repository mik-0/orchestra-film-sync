# External implementations and dependencies

No third-party source checkout or model weight is bundled in this snapshot.

- Matchmaker: `pymatchmaker==0.3.0`, installed by `rq3-audio-tracking/benchmark/setup_published_baselines.py`.
- ConcertCue: <https://github.com/matthewcaren/web-score-following>, revision `2876aacae008aa75a1512b70e3aca0bca653b4fa`. The setup script verifies the tracker archive checksum before use.
- Sync Toolbox supplies the offline alignment implementation used by the alignment scripts.
- Ultralytics supplies YOLO pose estimation; the model files are external downloads.
- Python and browser dependencies are listed in each project's requirements or package manifest.

The upstream packages and recordings retain their respective licences and permissions. This snapshot does not grant redistribution rights to the study media. No new licence is assigned to the project code here.
