#!/bin/bash
# Runs the resumable commands listed in a queue file, one after another, but starts a new command only while less
# than MAX_START minutes have passed, so that the whole call ends well inside the two-hour limit of a background job.
# Commands already finished return at once (they skip done tasks), so the same queue can simply be called again.
# Usage: run_queue.sh QUEUE_FILE MAX_START_MINUTES
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
Q=$1; MAX=${2:-45}
T0=$(date +%s)
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  el=$(( ($(date +%s) - T0) / 60 ))
  if [ "$el" -ge "$MAX" ]; then echo "TIME BOX REACHED after $el min; next: $line"; exit 0; fi
  echo "[$(date +%H:%M)] $line"
  eval "$line"
done < "$Q"
echo "QUEUE FINISHED $(date +%H:%M)"
