#!/usr/bin/env bash
# Write data/captures_manifest.sha256: the SHA256 of every raw input the benchmark reads from the data bundle (the
# config/bench.yaml captures' photos, videos and hints, and the magicplan export), so anyone can check that a
# downloaded bundle holds exactly these files: shasum -a 256 -c data/captures_manifest.sha256
# Re-run whenever a capture's input files change (for example after stripping GPS), then re-run scan-bench.
set -euo pipefail
cd "$(dirname "$0")/.."
find captures/home01_photo_a/photos captures/home01_photo_b/photos captures/home01_video_d/video \
     captures/home01_video_e/video captures/home01_video_e/hints.yaml captures/magicplan \
     -type f ! -name .DS_Store -print0 \
  | LC_ALL=C sort -z | xargs -0 shasum -a 256 > data/captures_manifest.sha256
echo "data/captures_manifest.sha256: $(wc -l < data/captures_manifest.sha256 | tr -d ' ') files"
