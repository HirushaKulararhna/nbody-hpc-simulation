#!/usr/bin/env bash
# ============================================================================
# bench_report.sh - Produce the full benchmark evidence log used in the
#                   project report (Serial / OpenMP / Hybrid MPI+OpenMP).
#
# Course: EE7218 / EC7207 - High Performance Computing
#
# Usage: bash bench_report.sh
# Output: report_evidence.txt  (verbatim terminal output for the report)
# ============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIZES=(1000 2000 4000)
STEPS=100
MAXTHR=8
REPS=2
RANKS=(1 2 4)
LOG=report_evidence.txt

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# N-Body Benchmark Evidence"
echo "# host cores : $(nproc)"
echo "# gcc        : $(gcc -dumpversion)"
echo "# mpirun     : $(mpirun --version 2>&1 | head -1)"
echo "# STEPS=$STEPS  reps=$REPS (best-of)"
echo "############################################################"
echo ""

echo "=== BUILD ==="
gcc -O2 -o serial_nbody serial_nbody.c -lm            && echo "  serial_nbody  OK"
gcc -O2 -fopenmp -o openmp_nbody openmp_nbody.c -lm   && echo "  openmp_nbody  OK"
mpicc -O2 -fopenmp -o hybrid_nbody hybrid_nbody.c -lm && echo "  hybrid_nbody  OK"
echo ""

echo "############################################################"
echo "# 1. SERIAL reference program (fixed N=1000, STEPS=100)"
echo "############################################################"
./serial_nbody
echo ""

for N in "${SIZES[@]}"; do
    echo "############################################################"
    echo "# 2. OPENMP scaling sweep  -  N=$N"
    echo "############################################################"
    echo "\$ ./openmp_nbody $N $STEPS $MAXTHR $REPS"
    ./openmp_nbody "$N" "$STEPS" "$MAXTHR" "$REPS"
    cp -f scaling.csv "csv_omp_N${N}.csv" 2>/dev/null
    echo ""
done

for N in "${SIZES[@]}"; do
    for P in "${RANKS[@]}"; do
        (( N % P != 0 )) && continue
        echo "############################################################"
        echo "# 3. HYBRID MPI+OpenMP sweep  -  N=$N, MPI ranks=$P"
        echo "############################################################"
        echo "\$ mpirun --bind-to none -np $P ./hybrid_nbody $N $STEPS 4 $REPS"
        mpirun --oversubscribe --bind-to none --mca mpi_yield_when_idle 1 \
               -np "$P" ./hybrid_nbody "$N" "$STEPS" 4 "$REPS"
        cp -f hybrid_scaling.csv "csv_hyb_N${N}_np${P}.csv" 2>/dev/null
        echo ""
    done
done

echo "############################################################"
echo "# 4. ACCURACY VERIFICATION"
echo "############################################################"
bash verify.sh 1000 100
echo ""
echo "############################################################"
echo "# Evidence log written to $LOG"
echo "############################################################"
