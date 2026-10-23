#!/bin/sh
# Run the SPECULA LTAO recorder as independent chunks (different seeds), skipping chunks
# whose slopes file already exists, so an interrupted campaign can simply be restarted.
# Usage: run_chunks.sh WORKROOT NCHUNKS [chunk_time_s] [time_step_s]
ROOT=$1; N=$2; T=${3:-100}; DT=${4:-0.25}
HERE=$(dirname "$0")
mkdir -p "$ROOT"
PY=${PYTHON:-python}
for k in $(seq 1 "$N"); do
  W="$ROOT/chunk$k"
  if ls "$W"/slopes_*.npz >/dev/null 2>&1; then echo "chunk $k done"; continue; fi
  rm -rf "$W"
  $PY "$HERE/run_specula.py" "$W" "$T" "$DT" $((100 + k)) > "$ROOT/chunk$k.log" 2>&1 \
    && echo "chunk $k finished" || echo "chunk $k FAILED"
done
echo ALL_DONE
