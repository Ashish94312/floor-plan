#!/bin/zsh
# E22 sweep: one video_keyframes run per (capture, k), strictly one GPU run at a time (16 GB machine).
# Waits until no other video experiment runs (another session may be running some) before each run.
# Usage: scripts/experiments/e22_sweep.sh [section.key=value ...] home01_video_b:16 home01_video_a:24 ...
#   section.key=value args go to every run (video_keyframes.py names the output out_k<k>_<value>).
cd "$(dirname "$0")/../.."
extra=(${@:#*:*}); jobs=(${(M)@:#*:*})
suffix=""; for a in $extra; do suffix+="_${a#*=}"; done
idle() { while pgrep -f "experiments/video_(keyframes|focal)" >/dev/null; do sleep 10; done; }
for job in $jobs; do
  cap=${job%%:*}; k=${job##*:}; out=captures/$cap/out_k$k$suffix
  idle
  echo "== $cap k$k$suffix start $(date +%H:%M:%S)"
  uv run python scripts/experiments/video_keyframes.py captures/$cap $k $extra > captures/$cap/e22_run_k$k$suffix.log 2>&1
  grep -E '^\{"k"|Traceback|Error' captures/$cap/e22_run_k$k$suffix.log | cut -c1-700
  grep -E 'walls within|calibration|area:|ceiling:|adjacency' $out/eval.md
  python3 -c "import json;g=json.load(open('$out/debug/geometry.json'));print({r:(v['views'],v.get('peak_accel_gb')) for r,v in g['geometry']['runs'].items()}, g['warnings'][:1])"
  echo "== $cap k$k$suffix end $(date +%H:%M:%S)"
done
