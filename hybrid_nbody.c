/*
 * hybrid_nbody.c - Hybrid MPI + OpenMP scaling benchmark for the O(N^2)
 *                  gravitational N-Body simulation.
 *
 * Course: EE7218 / EC7207 - High Performance Computing
 *
 * Build: mpicc -O2 -fopenmp -o hybrid_nbody hybrid_nbody.c -lm
 * Run:   mpirun --bind-to none -np <ranks> ./hybrid_nbody [N] [STEPS] [maxThreads] [reps]
 *        e.g. mpirun --bind-to none -np 4 ./hybrid_nbody 4000 100 8 3
 *
 * --bind-to none matters: OpenMPI binds each rank to a restricted CPU set by
 * default, which caps what omp_get_max_threads() reports (so the default
 * thread sweep silently shrinks) and makes a rank's OpenMP threads contend
 * for one pinned core. Always pass it when running this hybrid benchmark.
 *
 * For a fixed MPI rank count (set by mpirun -np), runs the pure serial version
 * first (baseline), then the hybrid version at OMP 1,2,4,8,... threads per
 * rank, and reports for each:
 *      time, speedup = T_serial / T_p, efficiency = speedup / (ranks*threads),
 *      RMSE of positions against the serial run held in memory.
 * Also writes hybrid_scaling.csv for plotting in the report.
 *
 * Parallel decomposition:
 *   - Every rank initializes the SAME N particles (same srand(42) seed), so
 *     every rank starts with a full, identical copy of the system.
 *   - Rank r owns the contiguous chunk [r*(N/P), (r+1)*(N/P)) of particles.
 *   - Each step a rank computes forces for its own chunk only, reading ALL N
 *     positions (it holds a full local copy), then integrates its own chunk.
 *   - MPI_Allgather redistributes updated positions so every rank again holds
 *     a complete, consistent array before the next step.
 *   - Velocities and masses are never communicated: a rank only ever needs the
 *     velocities of the particles it owns, and masses never change.
 *   - All size-N arrays are heap allocated, never fixed-size globals/stack.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define DT         0.01
#define SOFTENING  1e-9
#define G          1.0

/* Structure-of-arrays layout: lets MPI_Allgather operate directly on the
 * contiguous position arrays without a derived datatype. */
typedef struct {
    double *x, *y, *z;
    double *vx, *vy, *vz;
    double *mass;
} Bodies;

static int bodies_alloc(Bodies *b, int n) {
    b->x    = malloc(sizeof(double) * n);
    b->y    = malloc(sizeof(double) * n);
    b->z    = malloc(sizeof(double) * n);
    b->vx   = malloc(sizeof(double) * n);
    b->vy   = malloc(sizeof(double) * n);
    b->vz   = malloc(sizeof(double) * n);
    b->mass = malloc(sizeof(double) * n);
    return b->x && b->y && b->z && b->vx && b->vy && b->vz && b->mass;
}

static void bodies_free(Bodies *b) {
    free(b->x); free(b->y); free(b->z);
    free(b->vx); free(b->vy); free(b->vz);
    free(b->mass);
}

/* Identical rand() call order to serial_nbody.c / openmp_nbody.c so every
 * version starts from the same state. */
static void init_particles(Bodies *b, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        b->x[i]  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        b->y[i]  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        b->z[i]  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        b->vx[i] = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        b->vy[i] = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        b->vz[i] = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        b->mass[i] = (double)rand() / RAND_MAX * 9.0 + 1.0;
    }
}

/* ---------------- serial (baseline, no OpenMP, no MPI) ---------------- */
static double run_serial(int n, int steps, int reps, Bodies *out) {
    Bodies b;
    if (!bodies_alloc(&b, n)) return -1.0;
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    double *az = malloc(sizeof(double) * n);
    if (!ax || !ay || !az) { bodies_free(&b); free(ax); free(ay); free(az); return -1.0; }
    double best = 1e300;

    for (int r = 0; r < reps; r++) {
        init_particles(&b, n);
        double t0 = MPI_Wtime();
        for (int step = 0; step < steps; step++) {
            for (int i = 0; i < n; i++) {
                double xi = b.x[i], yi = b.y[i], zi = b.z[i];
                double axi = 0.0, ayi = 0.0, azi = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j == i) continue;
                    double dx = b.x[j] - xi;
                    double dy = b.y[j] - yi;
                    double dz = b.z[j] - zi;
                    double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    double f = G * b.mass[j] * invDist3;
                    axi += f * dx; ayi += f * dy; azi += f * dz;
                }
                ax[i] = axi; ay[i] = ayi; az[i] = azi;
            }
            for (int i = 0; i < n; i++) {
                b.vx[i] += ax[i] * DT;
                b.vy[i] += ay[i] * DT;
                b.vz[i] += az[i] * DT;
                b.x[i]  += b.vx[i] * DT;
                b.y[i]  += b.vy[i] * DT;
                b.z[i]  += b.vz[i] * DT;
            }
        }
        double t = MPI_Wtime() - t0;
        if (t < best) best = t;
    }

    memcpy(out->x, b.x, sizeof(double) * n);
    memcpy(out->y, b.y, sizeof(double) * n);
    memcpy(out->z, b.z, sizeof(double) * n);
    bodies_free(&b);
    free(ax); free(ay); free(az);
    return best;
}

/* ---------------- hybrid MPI + OpenMP version ---------------- */
static double run_hybrid(int n, int steps, int reps, int threads,
                         int rank, int size, Bodies *out) {
    int local_n = n / size;
    int start = rank * local_n;

    Bodies b;
    if (!bodies_alloc(&b, n)) return -1.0;
    double *ax = malloc(sizeof(double) * local_n);
    double *ay = malloc(sizeof(double) * local_n);
    double *az = malloc(sizeof(double) * local_n);
    if (!ax || !ay || !az) { bodies_free(&b); free(ax); free(ay); free(az); return -1.0; }

    omp_set_num_threads(threads);
    double best = 1e300;

    for (int r = 0; r < reps; r++) {
        init_particles(&b, n);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int step = 0; step < steps; step++) {
            /* Forces for this rank's chunk, reading all N positions. */
            #pragma omp parallel for schedule(static)
            for (int li = 0; li < local_n; li++) {
                int i = start + li;
                double xi = b.x[i], yi = b.y[i], zi = b.z[i];
                double axi = 0.0, ayi = 0.0, azi = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j == i) continue;
                    double dx = b.x[j] - xi;
                    double dy = b.y[j] - yi;
                    double dz = b.z[j] - zi;
                    double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    double f = G * b.mass[j] * invDist3;
                    axi += f * dx; ayi += f * dy; azi += f * dz;
                }
                ax[li] = axi; ay[li] = ayi; az[li] = azi;
            }

            /* Integrate this rank's own chunk in place. */
            #pragma omp parallel for schedule(static)
            for (int li = 0; li < local_n; li++) {
                int i = start + li;
                b.vx[i] += ax[li] * DT;
                b.vy[i] += ay[li] * DT;
                b.vz[i] += az[li] * DT;
                b.x[i]  += b.vx[i] * DT;
                b.y[i]  += b.vy[i] * DT;
                b.z[i]  += b.vz[i] * DT;
            }

            MPI_Allgather(MPI_IN_PLACE, local_n, MPI_DOUBLE, b.x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgather(MPI_IN_PLACE, local_n, MPI_DOUBLE, b.y, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgather(MPI_IN_PLACE, local_n, MPI_DOUBLE, b.z, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        double local_t = MPI_Wtime() - t0;

        /* A rep costs what its SLOWEST rank costs, so reduce per rep and keep
         * the best rep - not each rank's own best, which could come from a
         * different rep on every rank. */
        double rep_t;
        MPI_Allreduce(&local_t, &rep_t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rep_t < best) best = rep_t;
    }

    memcpy(out->x, b.x, sizeof(double) * n);
    memcpy(out->y, b.y, sizeof(double) * n);
    memcpy(out->z, b.z, sizeof(double) * n);
    bodies_free(&b);
    free(ax); free(ay); free(az);
    return best;
}

static double rmse(const Bodies *a, const Bodies *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = a->x[i] - b->x[i];
        double dy = a->y[i] - b->y[i];
        double dz = a->z[i] - b->z[i];
        s += dx * dx + dy * dy + dz * dz;
    }
    return sqrt(s / (3.0 * n));
}

int main(int argc, char **argv) {
    /* Hybrid codes must negotiate a thread level rather than use plain
     * MPI_Init. All MPI calls here run on the main thread outside OpenMP
     * parallel regions, so FUNNELED is the level actually required. */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n      = (argc > 1) ? atoi(argv[1]) : 4000;
    int steps  = (argc > 2) ? atoi(argv[2]) : 100;
    int maxthr = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();
    int reps   = (argc > 4) ? atoi(argv[4]) : 3;
    if (n < 2) n = 2;
    if (steps < 1) steps = 1;
    if (maxthr < 1) maxthr = 1;
    if (reps < 1) reps = 1;

    if (provided < MPI_THREAD_FUNNELED && rank == 0) {
        fprintf(stderr, "Warning: MPI provided thread level %d, below MPI_THREAD_FUNNELED (%d).\n",
                provided, MPI_THREAD_FUNNELED);
    }

    if (n % size != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: N (%d) must be divisible by the number of MPI ranks (%d).\n"
                    "       Pick a rank count that evenly divides N and try again.\n",
                    n, size);
        }
        MPI_Finalize();
        return 1;
    }

    /* thread counts to test: 1,2,4,8,... up to maxthr, plus maxthr itself */
    int list[32], cnt = 0;
    for (int t = 1; t <= maxthr && cnt < 31; t *= 2) list[cnt++] = t;
    if (cnt == 0 || list[cnt - 1] != maxthr) list[cnt++] = maxthr;

    Bodies ref, res;
    if (!bodies_alloc(&ref, n) || !bodies_alloc(&res, n)) {
        fprintf(stderr, "malloc failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("N = %d, STEPS = %d, MPI ranks = %d, best of %d run(s), cores available = %d\n\n",
               n, steps, size, reps, omp_get_max_threads());
    }

    /* Serial baseline runs on rank 0 alone so it measures true single-core
     * cost; the other ranks idle at the barrier below. */
    double ts = 0.0;
    if (rank == 0) {
        ts = run_serial(n, steps, reps, &ref);
        if (ts < 0.0) { fprintf(stderr, "malloc failed in serial baseline\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        printf("Serial baseline time: %.4f s\n\n", ts);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&ts, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    FILE *csv = NULL;
    if (rank == 0) {
        printf("%-9s %-9s %-12s %-10s %-12s %-10s\n",
               "Threads", "Workers", "Time(s)", "Speedup", "Efficiency", "RMSE");
        printf("---------------------------------------------------------------------\n");
        printf("%-9s %-9d %-12.4f %-10.2f %-12s %-10s\n", "serial", 1, ts, 1.00, "100.0%", "-");

        csv = fopen("hybrid_scaling.csv", "w");
        if (csv) {
            fprintf(csv, "ranks,threads,workers,time_s,speedup,efficiency_pct,rmse\n");
            fprintf(csv, "0,0,1,%.6f,1.0,100.0,0.0\n", ts);
        }
    }

    for (int k = 0; k < cnt; k++) {
        int t = list[k];
        double tp = run_hybrid(n, steps, reps, t, rank, size, &res);
        if (tp < 0.0) { fprintf(stderr, "malloc failed on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

        if (rank == 0) {
            int workers = size * t;
            double sp = ts / tp;
            double eff = 100.0 * sp / workers;
            double e = rmse(&ref, &res, n);
            char effs[16];
            snprintf(effs, sizeof effs, "%.1f%%", eff);
            printf("%-9d %-9d %-12.4f %-10.2f %-12s %-10.2e\n", t, workers, tp, sp, effs, e);
            if (csv) fprintf(csv, "%d,%d,%d,%.6f,%.4f,%.2f,%.3e\n",
                             size, t, workers, tp, sp, eff, e);
        }
    }

    if (rank == 0 && csv) { fclose(csv); printf("\nWrote hybrid_scaling.csv\n"); }

    bodies_free(&ref);
    bodies_free(&res);
    MPI_Finalize();
    return 0;
}
