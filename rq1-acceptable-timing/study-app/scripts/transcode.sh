#!/bin/bash
# Transcode the Firebird ProRes segments + conductor video into compact,
# browser-playable MP4 for the study app. Output: public/media/.
#
# Usage:
#   SRC="/path/to/firebird-media" ./scripts/transcode.sh
set -euo pipefail

: "${SRC:?Set SRC to the directory containing your authorised media files}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/public/media"
mkdir -p "$OUT"

echo "[transcode] conductor (video + orchestra audio, the master clock)..."
ffmpeg -y -loglevel error -i "$SRC/FIREBIRD CONDUCTOR BEELD_AUDIO.mov" \
  -vf scale=640:-2 -c:v libx264 -crf 28 -preset veryfast -pix_fmt yuv420p \
  -c:a aac -b:a 128k -movflags +faststart "$OUT/conductor.mp4"

for n in 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148; do
  echo "[transcode] segment $n (film, muted)..."
  ffmpeg -y -loglevel error -i "$SRC/$n.mov" -an \
    -vf scale=854:-2 -c:v libx264 -crf 27 -preset veryfast -pix_fmt yuv420p \
    -movflags +faststart "$OUT/seg_$n.mp4"
done

echo "[transcode] done -> $OUT"
ls -lh "$OUT" | awk '{print $5, $9}'
