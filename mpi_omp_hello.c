/*
 * mpi_omp_hello.c - smallest possible MPI + OpenMP program.
 *
 * If THIS fails with "libgomp: Thread creation failed", the problem is the
 * MPI/OpenMP environment, not the N-Body benchmark.
 *
 * Build: mpicc -O2 -fopenmp -o mpi_omp_hello mpi_omp_hello.c
 * Run:   mpirun -np 2 --bind-to none ./mpi_omp_hello 2
 */
 #define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    setvbuf(stdout, NULL, _IOLBF, 0);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int want = (argc > 1) ? atoi(argv[1]) : 2;

    /* How many CPUs is this rank actually allowed to run on? If Open MPI
     * bound the rank to a single core, this prints 1 and every OpenMP
     * thread will fight over that one core. */
    cpu_set_t mask;
    int allowed = -1;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
        allowed = CPU_COUNT(&mask);

    printf("rank %d/%d: cpus allowed = %d, omp_get_max_threads = %d\n",
           rank, size, allowed, omp_get_max_threads());

    omp_set_num_threads(want);
    printf("rank %d: requesting %d threads...\n", rank, want);

    #pragma omp parallel
    {
        #pragma omp single
        printf("rank %d: parallel region OK, team size = %d\n",
               rank, omp_get_num_threads());
    }

    printf("rank %d: done\n", rank);
    MPI_Finalize();
    return 0;
}