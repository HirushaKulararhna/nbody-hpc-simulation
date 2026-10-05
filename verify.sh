#!/usr/bin/env bash
# ============================================================================
# verify.sh - Accuracy check for the OpenMP and Hybrid N-Body benchmarks.
#
# Course: EE7218 / EC7207 - High Performance Computing
#
# Usage: bash verify.sh [N] [STEPS]
#
# Both benchmarks compute their own in-memory serial baseline and report an
# RMSE per thread count, so this checks EVERY thread count in the sweep - not
# just one configuration. Exits 0 only if every RMSE is below 1e-10.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N=${1:-1000}
STEPS=${2:-50}
THRESHOLD="1e-10"

echo "=============================================="
echo " N-Body Accuracy Verification"
echo " N=$N STEPS=$STEPS threshold RMSE < $THRESHOLD"
echo "=============================================="
echo ""
echo "Building..."
gcc -O2 -fopenmp -o openmp_nbody openmp_nbody.c -lm || { echo "FAIL: openmp_nbody build error"; exit 1; }
mpicc -O2 -fopenmp -o hybrid_nbody hybrid_nbody.c -lm || { echo "FAIL: hybrid_nbody build error"; exit 1; }
echo ""

# Check every rmse column entry in a CSV produced by the benchmarks.
# $1 = label, $2 = csv path, $3 = 1-based index of the rmse column
check_csv() {
    local name="$1" csv="$2" col="$3"
    if [ ! -f "$csv" ]; then
        echo "  [$name] FAIL (no $csv produced)"
        return 1
    fi
    local bad
    bad=$(awk -F, -v c="$col" -v t="$THRESHOLD" \
          'NR>1 && ($c+0) >= (t+0) { print NR": rmse="$c }' "$csv")
    local worst
    worst=$(awk -F, -v c="$col" 'NR>1 { if (($c+0) > m) m = $c+0 } END { printf "%.3e", m }' "$csv")
    if [ -n "$bad" ]; then
        echo "  [$name] FAIL  (worst RMSE = $worst >= $THRESHOLD)"
        echo "$bad" | sed 's/^/      /'
        return 1
    fi
    local rows
    rows=$(awk -F, 'NR>1' "$csv" | wc -l)
    echo "  [$name] PASS  ($rows configs checked, worst RMSE = $worst < $THRESHOLD)"
    return 0
}

echo "Running OpenMP sweep (up to 4 threads)..."
./openmp_nbody "$N" "$STEPS" 4 1 || { echo "FAIL: openmp_nbody run error"; exit 1; }
echo ""

echo "Running Hybrid sweep (MPI=2, up to 2 threads)..."
mpirun --oversubscribe --bind-to none --mca mpi_yield_when_idle 1 \
    -np 2 ./hybrid_nbody "$N" "$STEPS" 2 1 \
    || { echo "FAIL: hybrid_nbody run error"; exit 1; }
echo ""

echo "=============================================="
echo " Accuracy Check"
echo "=============================================="
OVERALL=0
# scaling.csv:        threads,time_s,speedup,efficiency_pct,rmse           -> col 5
# hybrid_scaling.csv: ranks,threads,workers,time_s,speedup,efficiency_pct,rmse -> col 7
check_csv "OpenMP" scaling.csv 5        || OVERALL=1
check_csv "Hybrid" hybrid_scaling.csv 7 || OVERALL=1

echo ""
if [ $OVERALL -eq 0 ]; then
    echo "ALL CHECKS PASSED"
else
    echo "SOME CHECKS FAILED"
fi
exit $OVERALL
