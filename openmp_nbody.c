/*
 * nbody_bench.c - Serial vs OpenMP scaling benchmark for the O(N^2)
 *                 gravitational N-Body simulation.
 *
 * Course: EE7218 / EC7207 - High Performance Computing
 *
 * Build: gcc -O2 -fopenmp -o nbody_bench nbody_bench.c -lm
 * Run:   ./nbody_bench [N] [STEPS] [maxThreads] [reps]
 *        e.g. ./nbody_bench 4000 100 8 3
 *
 * Runs the pure serial version first (baseline), then the OpenMP version at
 * 1,2,4,8,... threads, and reports for each:
 *      time, speedup = T_serial / T_p, efficiency = speedup / p,
 *      RMSE of positions against the serial run held in memory.
 * Also writes scaling.csv for plotting in the report.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define DT         0.01
#define SOFTENING  1e-9
#define G          1.0

typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double mass;
} Particle;

/* Identical to serial_nbody.c so both versions start from the same state. */
static void init_particles(Particle *p, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        p[i].x  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        p[i].y  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        p[i].z  = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        p[i].vx = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        p[i].vy = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        p[i].vz = ((double)rand() / RAND_MAX - 0.5) * 5.0;
        p[i].mass = (double)rand() / RAND_MAX * 9.0 + 1.0;
    }
}

/* ---------------- serial (baseline, no OpenMP at all) ---------------- */
static double run_serial(int n, int steps, int reps, Particle *out) {
    Particle *p = malloc(sizeof(Particle) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    double *az = malloc(sizeof(double) * n);
    double best = 1e300;

    for (int r = 0; r < reps; r++) {
        init_particles(p, n);
        double t0 = omp_get_wtime();
        for (int step = 0; step < steps; step++) {
            for (int i = 0; i < n; i++) {
                double xi = p[i].x, yi = p[i].y, zi = p[i].z;
                double axi = 0.0, ayi = 0.0, azi = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j == i) continue;
                    double dx = p[j].x - xi;
                    double dy = p[j].y - yi;
                    double dz = p[j].z - zi;
                    double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    double f = G * p[j].mass * invDist3;
                    axi += f * dx; ayi += f * dy; azi += f * dz;
                }
                ax[i] = axi; ay[i] = ayi; az[i] = azi;
            }
            for (int i = 0; i < n; i++) {
                p[i].vx += ax[i] * DT;
                p[i].vy += ay[i] * DT;
                p[i].vz += az[i] * DT;
                p[i].x  += p[i].vx * DT;
                p[i].y  += p[i].vy * DT;
                p[i].z  += p[i].vz * DT;
            }
        }
        double t = omp_get_wtime() - t0;
        if (t < best) best = t;
    }

    memcpy(out, p, sizeof(Particle) * n);
    free(p); free(ax); free(ay); free(az);
    return best;
}

/* ---------------- OpenMP version ---------------- */
static double run_omp(int n, int steps, int reps, int threads, Particle *out) {
    Particle *p = malloc(sizeof(Particle) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    double *az = malloc(sizeof(double) * n);
    double best = 1e300;

    omp_set_num_threads(threads);

    for (int r = 0; r < reps; r++) {
        init_particles(p, n);
        double t0 = omp_get_wtime();
        for (int step = 0; step < steps; step++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                double xi = p[i].x, yi = p[i].y, zi = p[i].z;
                double axi = 0.0, ayi = 0.0, azi = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j == i) continue;
                    double dx = p[j].x - xi;
                    double dy = p[j].y - yi;
                    double dz = p[j].z - zi;
                    double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    double f = G * p[j].mass * invDist3;
                    axi += f * dx; ayi += f * dy; azi += f * dz;
                }
                ax[i] = axi; ay[i] = ayi; az[i] = azi;
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                p[i].vx += ax[i] * DT;
                p[i].vy += ay[i] * DT;
                p[i].vz += az[i] * DT;
                p[i].x  += p[i].vx * DT;
                p[i].y  += p[i].vy * DT;
                p[i].z  += p[i].vz * DT;
            }
        }
        double t = omp_get_wtime() - t0;
        if (t < best) best = t;
    }

    memcpy(out, p, sizeof(Particle) * n);
    free(p); free(ax); free(ay); free(az);
    return best;
}

static double rmse(const Particle *a, const Particle *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = a[i].x - b[i].x;
        double dy = a[i].y - b[i].y;
        double dz = a[i].z - b[i].z;
        s += dx * dx + dy * dy + dz * dz;
    }
    return sqrt(s / (3.0 * n));
}

int main(int argc, char **argv) {
    int n       = (argc > 1) ? atoi(argv[1]) : 4000;
    int steps   = (argc > 2) ? atoi(argv[2]) : 100;
    int maxthr  = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();
    int reps    = (argc > 4) ? atoi(argv[4]) : 3;
    if (n < 2) n = 2;
    if (steps < 1) steps = 1;
    if (maxthr < 1) maxthr = 1;
    if (reps < 1) reps = 1;

    /* thread counts to test: 1,2,4,8,... up to maxthr, plus maxthr itself */
    int list[32], cnt = 0;
    for (int t = 1; t <= maxthr; t *= 2) list[cnt++] = t;
    if (list[cnt - 1] != maxthr) list[cnt++] = maxthr;

    Particle *ref = malloc(sizeof(Particle) * n);
    Particle *res = malloc(sizeof(Particle) * n);
    if (!ref || !res) { fprintf(stderr, "malloc failed\n"); return 1; }

    printf("N = %d, STEPS = %d, best of %d run(s), cores available = %d\n\n",
           n, steps, reps, omp_get_max_threads());

    double ts = run_serial(n, steps, reps, ref);
    printf("Serial baseline time: %.4f s\n\n", ts);

    printf("%-9s %-12s %-10s %-12s %-10s\n",
           "Threads", "Time(s)", "Speedup", "Efficiency", "RMSE");
    printf("---------------------------------------------------------\n");
    printf("%-9s %-12.4f %-10.2f %-12s %-10s\n", "serial", ts, 1.00, "100.0%", "-");

    FILE *csv = fopen("scaling.csv", "w");
    if (csv) fprintf(csv, "threads,time_s,speedup,efficiency_pct,rmse\n");
    if (csv) fprintf(csv, "0,%.6f,1.0,100.0,0.0\n", ts);

    for (int k = 0; k < cnt; k++) {
        int t = list[k];
        double tp = run_omp(n, steps, reps, t, res);
        double sp = ts / tp;
        double eff = 100.0 * sp / t;
        double e = rmse(ref, res, n);
        char effs[16];
        snprintf(effs, sizeof effs, "%.1f%%", eff);
        printf("%-9d %-12.4f %-10.2f %-12s %-10.2e\n", t, tp, sp, effs, e);
        if (csv) fprintf(csv, "%d,%.6f,%.4f,%.2f,%.3e\n", t, tp, sp, eff, e);
    }

    if (csv) { fclose(csv); printf("\nWrote scaling.csv\n"); }
    free(ref); free(res);
    return 0;
}