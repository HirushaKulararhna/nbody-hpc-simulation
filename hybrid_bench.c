

C
CCCC
brid_bench.c - Instrumented MPI + OpenMP scaling benchmark for the
 *                  O(N^2) gravitational N-Body simulation.
 *
 * Build: mpicc -O2 -fopenmp -o hybrid_bench hybrid_bench.c -lm
 * Run:   mpirun -np <ranks> --bind-to none ./hybrid_bench [N] [STEPS] [maxThreads] [reps]
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
#define BASE_FILE  "serial_baseline.bin"
#define CSV_FILE   "hybrid_scaling.csv"

static void init_particles(double *x, double *y, double *z,
                           double *vx, double *vy, double *vz,
                           double *mass, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        x[i]  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        y[i]  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        z[i]  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        vx[i] = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        vy[i] = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        vz[i] = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        mass[i] = (double)rand() / RAND_MAX * 9.0 + 1.0;
    }
}

static double run_serial(int n, int steps, int reps,
                         double *ox, double *oy, double *oz) {
    double *x  = malloc(sizeof(double) * n), *y  = malloc(sizeof(double) * n);
    double *z  = malloc(sizeof(double) * n), *vx = malloc(sizeof(double) * n);
    double *vy = malloc(sizeof(double) * n), *vz = malloc(sizeof(double) * n);
    double *m  = malloc(sizeof(double) * n), *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n), *az = malloc(sizeof(double) * n);
    double best = 1e300;

    for (int r = 0; r < reps; r++) {
        init_particles(x, y, z, vx, vy, vz, m, n);
        double t0 = MPI_Wtime();
        for (int step = 0; step < steps; step++) {
            for (int i = 0; i < n; i++) {
                double xi = x[i], yi = y[i], zi = z[i];
                double axi = 0.0, ayi = 0.0, azi = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j == i) continue;
                    double dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
                    double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    double f = G * m[j] * invDist3;
                    axi += f * dx; ayi += f * dy; azi += f * dz;
                }
                ax[i] = axi; ay[i] = ayi; az[i] = azi;
            }
            for (int i = 0; i < n; i++) {
                vx[i] += ax[i] * DT; vy[i] += ay[i] * DT; vz[i] += az[i] * DT;
                x[i]  += vx[i] * DT; y[i]  += vy[i] * DT; z[i]  += vz[i] * DT;
            }
        }
        double t = MPI_Wtime() - t0;
        if (t < best) best = t;
    }

    memcpy(ox, x, sizeof(double) * n);
    memcpy(oy, y, sizeof(double) * n);
    memcpy(oz, z, sizeof(double) * n);
    free(x); free(y); free(z); free(vx); free(vy); free(vz);
    free(m); free(ax); free(ay); free(az);
    return best;
}

static int load_baseline(int n, int steps, double *t,
                         double *x, double *y, double *z) {
    FILE *f = fopen(BASE_FILE, "rb");
    if (!f) return 0;
    int hn = 0, hs = 0; double ht = 0.0;
    int ok = (fread(&hn, sizeof(int), 1, f) == 1) &&
             (fread(&hs, sizeof(int), 1, f) == 1) &&
             (fread(&ht, sizeof(double), 1, f) == 1) &&
             hn == n && hs == steps &&
             (fread(x, sizeof(double), n, f) == (size_t)n) &&
             (fread(y, sizeof(double), n, f) == (size_t)n) &&
             (fread(z, sizeof(double), n, f) == (size_t)n);
    fclose(f);
    if (ok) *t = ht;
    return ok;
}

static void save_baseline(int n, int steps, double t,
                          const double *x, const double *y, const double *z) {
    FILE *f = fopen(BASE_FILE, "wb");
    if (!f) return;
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&steps, sizeof(int), 1, f);
    fwrite(&t, sizeof(double), 1, f);
    fwrite(x, sizeof(double), n, f);
    fwrite(y, sizeof(double), n, f);
    fwrite(z, sizeof(double), n, f);
    fclose(f);
}

typedef struct { double total, comp, wait, comm; } Timing;

static Timing run_hybrid(int n, int steps, int threads, int reps,
                         int rank, int size, int local_n, int start,
                         double *ox, double *oy, double *oz) {
    double *x  = malloc(sizeof(double) * n), *y  = malloc(sizeof(double) * n);
    double *z  = malloc(sizeof(double) * n), *vx = malloc(sizeof(double) * n);
    double *vy = malloc(sizeof(double) * n), *vz = malloc(sizeof(double) * n);
    double *m  = malloc(sizeof(double) * n);
    double *ax = malloc(sizeof(double) * local_n);
    double *ay = malloc(sizeof(double) * local_n);
    double *az = malloc(sizeof(double) * local_n);
    if (!x || !y || !z || !vx || !vy || !vz || !m || !ax || !ay || !az) {
        fprintf(stderr, "Error: malloc failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    omp_set_num_threads(threads);
    Timing best = { 1e300, 0, 0, 0 };

    for (int r = 0; r < reps; r++) {
        init_particles(x, y, z, vx, vy, vz, m, n);
        double comp = 0.0, wait = 0.0, comm = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int step = 0; step < steps; step++) {
            double a = MPI_Wtime();

            #pragma omp parallel for schedule(static)
            for (int li = 0; li < local_n; li++) {
                int i = start + li;
                double xi = x[i], yi = y[i], zi = z[i];
                double axi = 0.0, ayi = 0.0, azi = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j == i) continue;
                    double dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
                    double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    double f = G * m[j] * invDist3;
                    axi += f * dx; ayi += f * dy; azi += f * dz;
                }
                ax[li] = axi; ay[li] = ayi; az[li] = azi;
            }

            #pragma omp parallel for schedule(static)
            for (int li = 0; li < local_n; li++) {
                int i = start + li;
                vx[i] += ax[li] * DT; vy[i] += ay[li] * DT; vz[i] += az[li] * DT;
                x[i]  += vx[i] * DT;  y[i]  += vy[i] * DT;  z[i]  += vz[i] * DT;
            }

            double b = MPI_Wtime();
            MPI_Barrier(MPI_COMM_WORLD);
            double c = MPI_Wtime();

            MPI_Allgather(MPI_IN_PLACE, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgather(MPI_IN_PLACE, local_n, MPI_DOUBLE, y, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgather(MPI_IN_PLACE, local_n, MPI_DOUBLE, z, local_n, MPI_DOUBLE, MPI_COMM_WORLD);

            double d = MPI_Wtime();
            comp += b - a; wait += c - b; comm += d - c;
        }

        double total = MPI_Wtime() - t0;
        double gtotal;
        MPI_Allreduce(&total, &gtotal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (gtotal < best.total) {
            best.total = gtotal; best.comp = comp; best.wait = wait; best.comm = comm;
        }
    }

    memcpy(ox, x, sizeof(double) * n);
    memcpy(oy, y, sizeof(double) * n);
    memcpy(oz, z, sizeof(double) * n);
    free(x); free(y); free(z); free(vx); free(vy); free(vz);
    free(m); free(ax); free(ay); free(az);
    return best;
}

static double rmse3(const double *x, const double *y, const double *z,
                    const double *rx, const double *ry, const double *rz, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - rx[i], dy = y[i] - ry[i], dz = z[i] - rz[i];
        s += dx * dx + dy * dy + dz * dz;
    }
    return sqrt(s / (3.0 * n));
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    setvbuf(stdout, NULL, _IOLBF, 0);

    int n      = (argc > 1) ? atoi(argv[1]) : 4000;
    int steps  = (argc > 2) ? atoi(argv[2]) : 100;
    int maxthr = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();
    int reps   = (argc > 4) ? atoi(argv[4]) : 3;
    if (n < 2) n = 2;
    if (steps < 1) steps = 1;
    if (maxthr < 1) maxthr = 1;
    if (maxthr > 64) maxthr = 64;
    if (reps < 1) reps = 1;

    if (n % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N (%d) must be divisible by rank count (%d).\n", n, size);
        MPI_Finalize();
        return 1;
    }

    int local_n = n / size;
    int start = rank * local_n;

    double *rx = NULL, *ry = NULL, *rz = NULL;
    double t_serial = 0.0;

    if (rank == 0) {
        rx = malloc(sizeof(double) * n);
        ry = malloc(sizeof(double) * n);
        rz = malloc(sizeof(double) * n);
        if (load_baseline(n, steps, &t_serial, rx, ry, rz)) {
            printf("Serial baseline loaded from %s: %.4f s\n", BASE_FILE, t_serial);
        } else {
            printf("Computing serial baseline (N=%d, STEPS=%d)... ", n, steps);
            fflush(stdout);
            t_serial = run_serial(n, steps, reps, rx, ry, rz);
            save_baseline(n, steps, t_serial, rx, ry, rz);
            printf("%.4f s  (cached in %s)\n", t_serial, BASE_FILE);
        }
        printf("\nMPI ranks = %d, particles per rank = %d, best of %d run(s)\n",
               size, local_n, reps);
        printf("%-8s %-9s %-10s %-10s %-9s %-9s %-9s %-10s %-10s\n",
               "Ranks", "Threads", "Total(s)", "Comp(s)", "Wait(s)", "Comm(s)",
               "Comm%", "Speedup", "Effic.");
        printf("--------------------------------------------------------------"
               "-------------------------\n");
        printf("%-8s %-9s %-10.4f %-10.4f %-9s %-9s %-9s %-10.2f %-10s\n",
               "serial", "-", t_serial, t_serial, "-", "-", "-", 1.00, "100.0%");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double *ox = malloc(sizeof(double) * n);
    double *oy = malloc(sizeof(double) * n);
    double *oz = malloc(sizeof(double) * n);

    int list[32], cnt = 0;
    for (int t = 1; t <= maxthr; t *= 2) list[cnt++] = t;
    if (list[cnt - 1] != maxthr) list[cnt++] = maxthr;

    FILE *csv = NULL;
    if (rank == 0) {
        int fresh = 1;
        FILE *probe = fopen(CSV_FILE, "rb");
        if (probe) { fseek(probe, 0, SEEK_END); fresh = (ftell(probe) == 0); fclose(probe); }
        csv = fopen(CSV_FILE, "a");
        if (csv && fresh)
            fprintf(csv, "ranks,threads,cores,total_s,comp_s,wait_s,comm_s,"
                         "comm_pct,speedup,efficiency_pct,rmse\n");
    }

    for (int k = 0; k < cnt; k++) {
        int thr = list[k];
        if (rank == 0)
            fprintf(stderr, "[running: %d ranks x %d threads]\n", size, thr);
        Timing tm = run_hybrid(n, steps, thr, reps, rank, size, local_n, start, ox, oy, oz);

        double gcomp, gwait, gcomm, mincomp;
        MPI_Reduce(&tm.comp, &gcomp,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tm.wait, &gwait,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tm.comm, &gcomm,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tm.comp, &mincomp, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            int cores = size * thr;
            double sp = t_serial / tm.total;
            double eff = 100.0 * sp / cores;
            double cpct = 100.0 * (gcomm + gwait) / tm.total;
            double e = rmse3(ox, oy, oz, rx, ry, rz, n);
            printf("%-8d %-9d %-10.4f %-10.4f %-9.4f %-9.4f %-8.1f%% %-10.2f %-9.1f%%\n",
                   size, thr, tm.total, gcomp, gwait, gcomm, cpct, sp, eff);
            if (cnt > 1 && k == cnt - 1 && size > 1)
                printf("   (rank compute spread: fastest %.4f s vs slowest %.4f s"
                       " -> imbalance %.1f%%)\n",
                       mincomp, gcomp, 100.0 * (gcomp - mincomp) / gcomp);
            if (csv)
                fprintf(csv, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.4f,%.2f,%.3e\n",
                        size, thr, cores, tm.total, gcomp, gwait, gcomm, cpct, sp, eff, e);
            if (e > 1e-9)
                printf("   WARNING: RMSE %.3e is larger than expected\n", e);
        }
    }

    if (rank == 0) {
        if (csv) { fclose(csv); printf("\nAppended results to %s\n", CSV_FILE); }
        free(rx); free(ry); free(rz);
    }
    free(ox); free(oy); free(oz);
    MPI_Finalize();
    return 0;
}