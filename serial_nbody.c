/*
 * serial_nbody.c - Single-threaded baseline O(N^2) gravitational N-Body simulation.
 *
 * Course: EE7218 / EC7207 - High Performance Computing
 *
 * Build:  gcc -O2 -o serial_nbody serial_nbody.c -lm
 * Run:    ./serial_nbody
 *
 * Produces serial_result.bin, the reference used by openmp_nbody / hybrid_nbody
 * to compute RMSE accuracy against this baseline.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N            1000
#define STEPS        100
#define DT           0.01
#define SOFTENING    1e-9
#define G            1.0
#define RESULT_FILE  "serial_result.bin"

typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double mass;
} Particle;

/* Deterministic initialization: same srand(42) seed and call order are
 * replicated exactly in openmp_nbody.c and hybrid_nbody.c so that all three
 * versions start from bit-identical initial conditions. */
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

int main(void) {
    Particle *p = malloc(sizeof(Particle) * N);
    double *ax = malloc(sizeof(double) * N);
    double *ay = malloc(sizeof(double) * N);
    double *az = malloc(sizeof(double) * N);
    if (!p || !ax || !ay || !az) {
        fprintf(stderr, "Error: malloc failed\n");
        return 1;
    }

    init_particles(p, N);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < STEPS; step++) {
        /* Phase 1: compute acceleration on every particle from every other
         * particle, reading the position array only (no writes yet). */
        for (int i = 0; i < N; i++) {           // loop over particles to update
            double xi = p[i].x, yi = p[i].y, zi = p[i].z;
            double axi = 0.0, ayi = 0.0, azi = 0.0;
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                double dx = p[j].x - xi;    // compute distance vector
                double dy = p[j].y - yi;
                double dz = p[j].z - zi;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;  
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                double f = G * p[j].mass * invDist3;
                axi += f * dx;
                ayi += f * dy;
                azi += f * dz;
            }
            ax[i] = axi;             // store acceleration for particle i
            ay[i] = ayi;
            az[i] = azi;
        }

        /* Phase 2: integrate velocity and position (semi-implicit Euler).
         * Kept as a separate pass so phase 1 never reads a position that has
         * already been updated in the current step. */
        for (int i = 0; i < N; i++) {     // integrate velocity and  update the position
            p[i].vx += ax[i] * DT;
            p[i].vy += ay[i] * DT;
            p[i].vz += az[i] * DT;
            p[i].x  += p[i].vx * DT;
            p[i].y  += p[i].vy * DT;
            p[i].z  += p[i].vz * DT;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);            // end timer
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    FILE *f = fopen(RESULT_FILE, "wb");   //
    if (!f) {
        fprintf(stderr, "Error: could not open %s for writing\n", RESULT_FILE);
        return 1;
    }
    for (int i = 0; i < N; i++) {              // write final particle positions to file
        fwrite(&p[i].x, sizeof(double), 1, f);
        fwrite(&p[i].y, sizeof(double), 1, f);
        fwrite(&p[i].z, sizeof(double), 1, f);
    }
    fclose(f);

    printf("Serial time: %.4f seconds\n", elapsed); // print elapsed time

    free(p);
    free(ax);
    free(ay);
    free(az);
    return 0;
}
