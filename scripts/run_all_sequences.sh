#!/usr/bin/env bash
# Run stereo SLAM on every KITTI odometry training sequence (00-10) in
# parallel, writing results/<SEQ>.png and results/<SEQ>.mp4 for each.
#
# Each process is pinned to a single BLAS thread so that 11 concurrent
# jobs don't oversubscribe the machine's ~12 cores.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results results/logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="$ROOT/.venv/bin/python"
SEQUENCES=(00 01 02 03 04 05 06 07 08 09 10)

declare -a pids
start_time=$(date +%s)
for seq in "${SEQUENCES[@]}"; do
    log="results/logs/${seq}.log"
    "$PY" src/stereo_slam.py --sequence "$seq" >"$log" 2>&1 &
    pid=$!
    pids+=("$pid:$seq")
    echo "launched seq $seq (pid $pid) -> $log"
done

echo "waiting for ${#pids[@]} jobs..."
failed=0
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"
    seq="${entry##*:}"
    if wait "$pid"; then
        echo "[OK]  seq $seq finished"
    else
        echo "[ERR] seq $seq failed — see results/logs/${seq}.log"
        failed=$((failed + 1))
    fi
done

end_time=$(date +%s)
echo
echo "all jobs done in $((end_time - start_time)) s"
echo "results:"
ls -la results/*.png results/*.mp4 2>/dev/null || true

exit $failed
