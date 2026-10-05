#!/usr/bin/env bash
# ============================================================================
# run_all.sh - Build all three N-Body implementations and run the full
#              scaling sweep, collecting output into results.txt and printing
#              a combined summary table.
#
# Course: EE7218 / EC7207 - High Performance Computing
#
# Usage: bash run_all.sh [N] [STEPS] [maxThreads] [reps]
#        e.g. bash run_all.sh 2000 50 8 1
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N=${1:-2000}
STEPS=${2:-50}
MAXTHR=${3:-8}
REPS=${4:-1}
RANK_COUNTS=(1 2 4 8)
CORES=$(nproc)

RESULTS="results.txt"
: > "$RESULTS"

# Tee everything to results.txt as we go.
log() { echo "$@" | tee -a "$RESULTS"; }
run_logged() { "$@" 2>&1 | tee -a "$RESULTS"; }

log "=============================================="
log " N-Body HPC Scaling Benchmark"
log " N=$N STEPS=$STEPS maxThreads=$MAXTHR reps=$REPS"
log "=============================================="
log ""

echo "[1/3] Compiling serial_nbody.c ..."
gcc -O2 -o serial_nbody serial_nbody.c -lm || { echo "BUILD FAILED: serial_nbody"; exit 1; }
echo "[2/3] Compiling openmp_nbody.c ..."
gcc -O2 -fopenmp -o openmp_nbody openmp_nbody.c -lm || { echo "BUILD FAILED: openmp_nbody"; exit 1; }
echo "[3/3] Compiling hybrid_nbody.c ..."
mpicc -O2 -fopenmp -o hybrid_nbody hybrid_nbody.c -lm || { echo "BUILD FAILED: hybrid_nbody"; exit 1; }
echo "Build complete."
echo ""

# --- Fixed-size serial reference (also writes serial_result.bin) ------------
log "=============================================="
log " Serial reference program (fixed N=1000)"
log "=============================================="
run_logged ./serial_nbody
log ""

# --- OpenMP scaling sweep ---------------------------------------------------
log "=============================================="
log " OpenMP scaling sweep"
log "=============================================="
run_logged ./openmp_nbody "$N" "$STEPS" "$MAXTHR" "$REPS"
log ""

# --- Hybrid scaling sweep, one pass per MPI rank count ----------------------
rm -f hybrid_scaling_np*.csv
for P in "${RANK_COUNTS[@]}"; do
    if (( N % P != 0 )); then
        log "Skipping MPI=$P: N=$N is not divisible by $P"
        continue
    fi
    log "=============================================="
    log " Hybrid scaling sweep - MPI ranks = $P"
    log "=============================================="
    # --bind-to none is essential for hybrid runs: OpenMPI's default binds each
    # rank to a restricted CPU set, which both caps omp_get_max_threads() and
    # forces a rank's OpenMP threads to contend for the same pinned core.
    run_logged mpirun --oversubscribe --bind-to none --mca mpi_yield_when_idle 1 \
        -np "$P" ./hybrid_nbody "$N" "$STEPS" "$MAXTHR" "$REPS"
    # Each invocation truncates hybrid_scaling.csv, so keep a per-rank copy.
    [ -f hybrid_scaling.csv ] && mv hybrid_scaling.csv "hybrid_scaling_np${P}.csv"
    log ""
done

# --- Combined summary -------------------------------------------------------
log "======================================================================"
log " SUMMARY  (N=$N, STEPS=$STEPS, best of $REPS)"
log "======================================================================"
{
    printf "%-10s %-16s %-9s %-12s %-10s %-12s %-10s\n" \
           "Method" "Config" "Workers" "Time(s)" "Speedup" "Efficiency" "RMSE"
    printf -- "----------------------------------------------------------------------------------\n"

    if [ -f scaling.csv ]; then
        awk -F, 'NR>1 {
            if ($1==0) printf "%-10s %-16s %-9d %-12.4f %-10.2f %-12s %-10s\n", "Serial","-",1,$2,1.0,"100.0%","-";
            else       printf "%-10s %-16s %-9d %-12.4f %-10.2f %-12s %-10.2e\n", "OpenMP","T="$1,$1,$2,$3,sprintf("%.1f%%",$4),$5;
        }' scaling.csv
    fi

    for f in hybrid_scaling_np*.csv; do
        [ -e "$f" ] || continue
        awk -F, -v cores="$CORES" 'NR>1 && $1!=0 {
            printf "%-10s %-16s %-9d %-12.4f %-10.2f %-12s %-10.2e%s\n", \
                   "Hybrid","MPI="$1",OMP="$2,$3,$4,$5,sprintf("%.1f%%",$6),$7, \
                   ($3 > cores ? "  <- oversubscribed" : "");
        }' "$f"
    done
} | tee -a "$RESULTS"

log ""
log "Full log: $RESULTS"
log "CSVs: scaling.csv, hybrid_scaling_np{1,2,4,8}.csv"
