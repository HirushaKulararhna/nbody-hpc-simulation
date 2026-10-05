# N-Body Gravitational Simulation — HPC Performance Analysis

**Course:** EE7218 / EC7207 — High Performance Computing
**Simulation size:** N = 1000 particles, STEPS = 100, DT = 0.01, SOFTENING = 1e-9
**Test platform:** Ubuntu 24.04 (WSL2), 16-core host, GCC 13.3.0, OpenMPI 4.1.6

---

## 1. Introduction

The N-Body problem asks: given N particles, each with a position, velocity, and
mass, how does the system evolve under mutual gravitational attraction? Every
particle pulls on every other particle according to Newton's law of gravitation

```
F_ij = G * m_i * m_j * (r_j - r_i) / |r_j - r_i|^3
```

and the naive ("direct summation") algorithm computes this force for **every
pair** of particles at **every time step**. For N particles this is O(N²) work
per step, and over STEPS time steps the total work is O(N² * STEPS). At
N = 1000 and STEPS = 100 that is already 10⁸ pairwise interactions — small
enough to run in well under a second on a modern core, but the same O(N²)
scaling means doubling N quadruples the work per step. This makes the N-Body
problem an ideal, embarrassingly-parallel benchmark for comparing shared-memory
(OpenMP), distributed-memory (MPI), and hybrid parallelization strategies: the
per-particle force computation is independent of every other particle's
*update*, so different particles can safely be assigned to different threads
or processes, as long as everyone reads a consistent snapshot of positions.

A small softening term (`SOFTENING = 1e-9`) is added inside the distance
calculation to avoid division-by-zero when two particles happen to coincide,
without materially affecting the physics at the separations produced by this
benchmark's initial conditions.

---

## 2. Parallelization Strategy

### 2.1 Serial (baseline)

```
for step in 1..STEPS:
    for i in 0..N-1:                 <- single thread, one core
        F(i) = sum over all j != i of gravity(i, j)
    for i in 0..N-1:
        v[i] += F(i)/m[i] * dt
        x[i] += v[i] * dt
```
All N particles, all N² pairwise force terms, one core, one instruction
stream. This is the reference for both correctness (RMSE) and speedup.

### 2.2 OpenMP (shared memory, single node)

```
                    ┌───────────────────────────────────────────┐
                    │           Particles 0 .. N-1               │
                    │   (all resident in one shared address      │
                    │    space — every thread can read all of it)│
                    └───────────────────────────────────────────┘
                                       │
                     #pragma omp parallel for schedule(static)
                                       │
        ┌──────────────┬──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
   Thread 0        Thread 1       Thread 2       Thread 3   ...  Thread T-1
   i = 0..N/T-1    i = N/T..      i = 2N/T..     i = 3N/T..      i = (T-1)N/T..N-1
                   2N/T-1         3N/T-1         4N/T-1
        │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼
   writes F[i] for its own contiguous block only — no shared writes, no locks
        │              │              │              │              │
        └──────────────┴──────────────┴──────────────┴──────────────┘
                                       │
                     #pragma omp parallel for schedule(static)
                            (integrate v, x — same partition)
```
Every thread has full read access to the position array (it lives in one
process's shared memory) but only ever *writes* the force/velocity/position
entries for the contiguous block of particles it owns — so there is no data
race and no synchronization needed inside the parallel region.

### 2.3 Hybrid MPI + OpenMP (distributed memory, multiple nodes/ranks)

```
   Rank 0 (OMP: T threads)     Rank 1 (OMP: T threads)     Rank P-1 (OMP: T threads)
  ┌───────────────────────┐  ┌───────────────────────┐   ┌───────────────────────┐
  │ full copy of ALL N     │  │ full copy of ALL N     │   │ full copy of ALL N     │
  │ particle positions     │  │ particle positions     │...│ particle positions     │
  │ (read-only this step)  │  │ (read-only this step)  │   │ (read-only this step)  │
  │                        │  │                        │   │                        │
  │ owns chunk:            │  │ owns chunk:             │   │ owns chunk:             │
  │ [0, N/P)               │  │ [N/P, 2N/P)             │   │ [(P-1)N/P, N)           │
  │                        │  │                        │   │                        │
  │ #pragma omp parallel   │  │ #pragma omp parallel   │   │ #pragma omp parallel   │
  │ for schedule(static)   │  │ for schedule(static)   │   │ for schedule(static)   │
  │  -> T threads split    │  │  -> T threads split    │   │  -> T threads split    │
  │     the local chunk    │  │     the local chunk    │   │     the local chunk    │
  └───────────┬───────────┘  └───────────┬───────────┘   └───────────┬───────────┘
              │                          │                           │
              └──────────────┬───────────┴─────────────┬─────────────┘
                              ▼                         ▼
                     MPI_Allgather(positions) — every rank sends its updated
                     chunk and receives everyone else's, so all P ranks again
                     hold a complete, consistent position array for the next step
```
Two levels of parallelism: MPI splits the N particles across P ranks/processes
(potentially on different machines), and within each rank, OpenMP splits that
rank's local chunk across T threads. `MPI_Allgather` is the only
communication per step — it synchronizes positions so every rank's next force
computation sees a globally consistent snapshot.

---

## 3. Accuracy Results

`verify.sh` compares OpenMP and Hybrid final positions against the serial
baseline (`serial_result.bin`) using RMSE over all `(x, y, z)` components,
after N = 1000 particles and 100 steps.

| Version | Configuration | RMSE vs Serial | Threshold (1e-10) |
|---|---|---|---|
| OpenMP | T = 4 | `0.00e+00` | PASS |
| Hybrid | MPI = 2, OMP = 2 | `0.00e+00` | PASS |

**Why RMSE is exactly zero, not merely small (~1e-14):** both parallel
versions parallelize *across particles* (the outer `i` loop), while the inner
summation over `j = 0..N-1` for a given particle `i` is always executed in the
same order, by a single thread, whether that thread is running under OpenMP
or inside an MPI rank. Since IEEE-754 floating-point addition is only
sensitive to the *order* of operations (not to which core executes them),
every per-particle force sum is bit-for-bit identical to the serial version.
The only way to introduce non-zero RMSE here would be to reorder the `j`
summation itself (e.g. a tree/SIMD reduction under `-ffast-math`, or splitting
a single particle's neighbor list across threads) — which this implementation
deliberately avoids. In a typical HPC report you would expect RMSE around
`1e-13`–`1e-15` (floating-point noise from reordering); an exact `0.00e+00`
here reflects that the chosen work-partitioning strategy (partition by
*particle*, not by *pairwise term*) preserves summation order exactly.

---

## 4. Timing Results

Measured with `run_all.sh` (N = 1000, STEPS = 100, 16 physical cores available):

| Method | Config | Time (s) | Speedup vs Serial |
|---|---|---|---|
| Serial | – | 0.3485 | 1.00× |
| OpenMP | T=1 | 0.3685 | 0.95× |
| OpenMP | T=2 | 0.2207 | 1.58× |
| OpenMP | T=4 | 0.1360 | 2.56× |
| OpenMP | T=8 | 0.0957 | 3.64× |
| Hybrid | MPI=2, OMP=2 | 0.1221 | 2.85× |
| Hybrid | MPI=4, OMP=2 | 0.1002 | 3.48× |
| Hybrid | MPI=2, OMP=4 | 0.1061 | 3.28× |
| Hybrid | MPI=4, OMP=4 | 0.1432 | 2.43× |

*(Raw per-run output is in `results.txt`, regenerated each time `run_all.sh`
is executed.)*

---

## 5. Speedup Analysis — Why Speedup Is Less Than Ideal

At this problem size, none of the parallel configurations come close to
*linear* speedup (e.g. 8 threads → 8×). Three effects explain the gap:

**Amdahl's Law.** Every run pays a fixed, non-parallelizable cost: process/
thread startup (`omp_set_num_threads`, `MPI_Init`), initialization of the
particle arrays, and I/O at the end. If a fraction *s* of the total work is
strictly serial, the maximum possible speedup with *T* workers is bounded by
`1 / (s + (1-s)/T)` — which flattens out as *T* grows, no matter how well the
parallel portion scales. This is visible directly in the data: going from
T=4 to T=8 only takes OpenMP from 2.56× to 3.64× (not 5.12×), because the
serial fraction dominates more heavily as the parallel work per thread
shrinks.

**Thread/process launch and scheduling overhead.** At T=1, OpenMP is
*slower* than serial (0.95×) — the overhead of entering a parallel region
and creating a thread team outweighs any benefit when there is nothing to
parallelize across. This overhead is roughly constant per thread but is a
larger fraction of the *total* runtime as N (and hence per-thread work)
shrinks, which is also why the gains taper off at higher thread counts here.

**Communication overhead (Hybrid only).** Every step, Hybrid pays for an
`MPI_Allgather` of the full position array across all ranks — cost that
OpenMP-only never pays. This is why Hybrid MPI=4,OMP=4 (0.1432s) is
*slower* than MPI=4,OMP=2 (0.1002s) even though it uses twice as many total
worker threads: with 16 total OS threads/processes active on a 16-core
machine, the system is fully oversubscribed by background/VM overhead, and
the extra OpenMP threads inside each rank increase contention for shared
caches and memory bandwidth at the same time as MPI is repeating its
`Allgather` synchronization barrier every single step. This is the classic
HPC tradeoff: more parallel workers reduce *compute* time per worker but
increase *synchronization and communication* time, and at some point the
two curves cross and adding workers makes things worse.

**Memory bandwidth.** The force kernel is compute-light but memory-read-heavy
(each pairwise interaction reads two particles' `{x,y,z,mass}` from an array
that, for N=1000, does not fit in L1/L2 cache). As thread count grows, all
threads on the same socket compete for the same memory bus, so speedup
saturates well below the core count — a signature of a **memory-bandwidth-
bound**, rather than compute-bound, workload. This is also why realistic
N-Body codes at scale switch to tree algorithms (Barnes-Hut, O(N log N)) or
particle-mesh methods to reduce both compute *and* memory traffic per step.

---

## 6. When to Use OpenMP vs Hybrid

| Scenario | Recommended approach |
|---|---|
| Single machine, all cores on one socket/node | **OpenMP only** — no communication overhead, simplest to reason about and debug |
| Problem fits in one node's RAM but you want to use all cores efficiently | **OpenMP only**, tuned thread count (often < max cores, per Amdahl above) |
| Problem/data too large for one node's memory | **Hybrid** — MPI distributes the working set across nodes, OpenMP still exploits each node's cores |
| Multi-node cluster available | **Hybrid** — one MPI rank per node (or per NUMA domain), OpenMP threads within each rank; avoids running one MPI rank per core, which would multiply communication overhead unnecessarily |
| Very small N, single node | **Serial or OpenMP with few threads** — parallel overhead exceeds the benefit, as seen directly in the T=1 OpenMP result (0.95×) |

The general rule of thumb this data supports: **use exactly enough MPI ranks
to span the machines/nodes you have, and fill each rank's cores with OpenMP
threads** — not more MPI ranks than nodes, since each additional rank adds
another participant to the `Allgather` communication step for no additional
memory-distribution benefit on a single machine.

---

## 7. Conclusion

All three implementations (serial, OpenMP, hybrid MPI+OpenMP) compute
identical physics — verified by an exact `0.00e+00` RMSE against the serial
baseline — while achieving up to **3.64× speedup** (OpenMP, 8 threads) and
**3.48× speedup** (Hybrid, MPI=4/OMP=2) over the single-threaded baseline at
N=1000. Neither approach reaches ideal linear speedup, and the data shows why:
fixed serial overhead (Amdahl's Law), thread/process launch cost, MPI
communication (`Allgather`) overhead, and shared memory-bandwidth contention
all eat into the theoretical maximum. The hybrid model's main advantage is
not raw speed at this single-node problem size, but **scalability across
multiple nodes** — a capability OpenMP alone does not have. For production
N-Body codes at much larger N, the next steps would be increasing N to make
per-thread work dominate fixed overhead, and replacing the O(N²) direct-sum
kernel with an O(N log N) Barnes-Hut or particle-mesh method to relieve the
memory-bandwidth bottleneck identified in Section 5.
