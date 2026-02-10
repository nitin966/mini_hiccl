# Mini HiCCL

**High-performance Collective Communications Library** -- a C++ template-based library for collective communication primitives built on MPI with OpenMP-accelerated computation kernels.

## Overview

Mini HiCCL provides a modular, layered architecture for expressing and executing collective communication patterns in distributed-memory systems. It separates communication scheduling from execution, allowing fine-grained control over broadcast, reduce, allgather, allreduce, and other collective operations.

Key design goals:

- **Template-based generics** -- works with any MPI-compatible data type (`float`, `double`, `int`, etc.)
- **Composable primitives** -- combine communication and computation into unified `Command` objects
- **Epoch-based scheduling** -- group operations into epochs separated by fences for dependency management
- **Configurable topology** -- multi-level hierarchy, striping, ring, and pipelining parameters
- **Performance measurement** -- built-in benchmarking with warmup iterations, timing statistics, and throughput reporting

## Architecture

```
hiccl.h                  Main header -- orchestrates includes, defines enums and globals
  |
  +-- CommBench/
  |     commbench.h      MPI type wrappers, memory management, utility functions
  |     commbench.cpp    Global variable definitions (myid, numproc, comm_mpi)
  |
  +-- source/
        broadcast.h      BROADCAST<T> struct -- fan-out from one sender to many receivers
        reduce.h         REDUCE<T> struct -- fan-in from many senders to one receiver
        compute.h        Compute<T> class -- OpenMP-parallelized reduction kernel
        comm_primitive.h CommPrimitive<T> class -- point-to-point MPI send/recv
        command.h        Command<T> class -- composite of CommPrimitive + Compute
        comm.h           Comm<T> class -- high-level collective operation orchestrator
```

### Component Summary

| Component | Role |
|---|---|
| `BROADCAST<T>` | Describes a one-to-many data distribution pattern. Supports sending to a specific set of receivers, all ranks, or all-except-sender. |
| `REDUCE<T>` | Describes a many-to-one data gathering pattern. Supports receiving from a specific set of senders, all ranks, or all-except-receiver. |
| `Compute<T>` | Executes element-wise reduction (summation) across multiple input buffers using an OpenMP-parallelized kernel. Includes benchmarking infrastructure. |
| `CommPrimitive<T>` | Manages point-to-point MPI communication tasks. Handles send/recv scheduling with self-copy optimization when sender and receiver are the same rank. |
| `Command<T>` | Composite pattern that pairs a `CommPrimitive` with a `Compute` unit, launching both together and providing unified measurement. |
| `Comm<T>` | Top-level orchestrator that manages epochs of broadcast/reduce operations with configurable hierarchy, striping, ring topology, and pipeline depth. |

### Dependency Order

Headers must be included in a specific order due to template dependencies:

1. `CommBench/commbench.h` (MPI globals and utilities)
2. `HiCCL` enums and namespace globals
3. `broadcast.h`, `reduce.h` (data structures)
4. `compute.h` (computation engine)
5. `comm_primitive.h` (communication engine)
6. `command.h` (composite of comm + compute)
7. `comm.h` (top-level orchestrator)

This is handled automatically by including `hiccl.h`.

## Prerequisites

- **C++ compiler** with C++11 support or later
- **MPI implementation** (OpenMPI, MPICH, Intel MPI, or similar)
- **OpenMP** support (for the parallel reduction kernel)

## Building

There is no build system provided. Compile directly with an MPI-aware compiler:

```bash
mpicxx -fopenmp -o hiccl_test main.cpp CommBench/commbench.cpp
```

For optimized builds:

```bash
mpicxx -O2 -fopenmp -o hiccl_test main.cpp CommBench/commbench.cpp
```

## Running

The test program requires an MPI launcher. Some tests need at least 2 processes:

```bash
# Run with 2 processes (recommended minimum)
mpirun -np 2 ./hiccl_test

# Run with 4 processes
mpirun -np 4 ./hiccl_test
```

With a single process, `CommPrimitive` and `Command` tests are skipped gracefully.

## Usage

### Include the Library

```cpp
#include "hiccl.h"

// Declare external globals (defined in commbench.cpp)
extern int myid;
extern int numproc;
extern int printid;
extern MPI_Comm comm_mpi;
```

### Compute (Element-wise Reduction)

```cpp
HiCCL::Compute<float> compute;

// Allocate input/output buffers
float *in1, *in2, *out;
CommBench::allocate(in1, count);
CommBench::allocate(in2, count);
CommBench::allocate(out, count);

// Register a reduction task on a specific rank
compute.add({in1, in2}, out, count, /*compid=*/HiCCL::myid);

// Execute
compute.start();
compute.wait();

// Print statistics
compute.report();
```

### Point-to-Point Communication

```cpp
HiCCL::CommPrimitive<float> comm(CommBench::MPI);

// Schedule a send from rank 0 to rank 1
comm.add(sendbuf, /*soffset=*/0, recvbuf, /*roffset=*/0, count, /*sendid=*/0, /*recvid=*/1);

// Execute all scheduled communications
comm.start();
comm.wait();
```

### Composite Command (Communication + Computation)

```cpp
HiCCL::CommPrimitive<float> comm_prim(CommBench::MPI);
comm_prim.add(sendbuf, 0, recvbuf, 0, count, 0, 1);

HiCCL::Compute<float> compute;
compute.add({in1, in2}, out, count, HiCCL::myid);

// Combine into a single command
HiCCL::Command<float> cmd(&comm_prim, &compute);
cmd.start();
cmd.wait();
```

### High-Level Collective Operations

```cpp
HiCCL::Comm<float> coll;

// Configure topology
coll.set_hierarchy({4, 2}, {CommBench::MPI, CommBench::MPI});
coll.set_numstripe(2);
coll.set_pipedepth(4);

// Add broadcast and reduce operations
coll.add_bcast(sendbuf, 0, recvbuf, 0, count, /*sender=*/0, HiCCL::all);
coll.add_fence();  // Epoch barrier
coll.add_reduce(sendbuf, 0, recvbuf, 0, count, HiCCL::all, /*receiver=*/0);

// Initialize and run
coll.init();
coll.run();
```

## Test Program

`main.cpp` exercises the core components in sequence:

1. **Compute test** -- Reduces 3 input buffers of 100 floats each, verifies element-wise sum
2. **CommPrimitive test** -- Sends data from rank 0 to rank 1 and performs a self-copy on rank 0, verifies received values
3. **Command test** -- Combines communication (rank 0 to rank 1) with computation (2-buffer reduction), verifies both outputs independently

Each test includes correctness verification with pass/fail reporting and is synchronized with `MPI_Barrier` calls.

## Project Status

This is an early-stage research implementation. The low-level primitives (`CommPrimitive`, `Compute`, `Command`) are fully functional. The high-level `Comm` class provides the scheduling and configuration interface, but `init()` and `run()` are currently stubs pending the implementation of collective algorithm decomposition (e.g., ring allreduce, recursive halving).

## Supported Collective Patterns

Defined in the `HiCCL::collective` enum:

- `gather`
- `scatter`
- `broadcast`
- `reduce`
- `alltoall`
- `allgather`
- `reducescatter`
- `allreduce`

## Supported Communication Backends

Defined in the `CommBench::library` enum:

- `MPI` -- Standard MPI point-to-point (currently implemented)
- `IPC` / `IPC_get` -- Inter-process communication (planned)
- `XCCL` -- Accelerator collective communication library (planned)
