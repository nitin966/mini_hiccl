// hiccl.h
#ifndef HICCL_H
#define HICCL_H

// 1. Core MPI/CommBench globals and utilities MUST be included first.
#include "CommBench/commbench.h"

// 2. Standard C++ library includes used widely throughout.
#include <vector>
#include <list>
#include <algorithm>
#include <cstddef>
#include <pthread.h> // For multithreading (e.g., Comm::run_async)
#include <iostream>  // For std::cout/std::flush
#include <typeinfo>  // For typeid(T).name()

// 3. HiCCL Enums (defined here, as they are fundamental and used by other HiCCL components like BROADCAST/REDUCE)
namespace HiCCL {
    enum pattern {all, others};
    enum collective {dummy, gather, scatter, broadcast, reduce, alltoall, allgather, reducescatter, allreduce};
}

// 4. HiCCL Global Variables (static references to global ones for direct access within HiCCL namespace)
//    These MUST be defined AFTER CommBench/commbench.h is included and global variables are available.
namespace HiCCL { // Continue HiCCL namespace

    static int printid = ::printid; // Referencing the global 'printid'
    static const MPI_Comm &comm_mpi = ::comm_mpi; // Referencing the global 'comm_mpi'
    static const int &numproc = ::numproc; // Referencing the global 'numproc'
    static const int &myid = ::myid;       // Referencing the global 'myid'

    static size_t buffsize = 0;
    static size_t recycle = 0;
    static size_t reuse = 0;

} // End HiCCL namespace (for initial globals/enums)

// 5. Include HiCCL component headers (ORDER IS CRUCIAL based on dependencies)
//    These will extend the HiCCL namespace.
//    - BROADCAST/REDUCE must be defined before Comm uses them in its vectors.
//    - Compute must be defined before Command/CommPrimitive use it.
//    - CommPrimitive must be defined before Command uses it.
//    - Command must be defined before Comm uses it.
//    - Comm is last among the core classes.
#include "source/broadcast.h"      // BROADCAST struct definition (uses HiCCL::pattern)
#include "source/reduce.h"         // REDUCE struct definition (uses HiCCL::pattern)
#include "source/compute.h"        // Compute class definition (uses HiCCL globals)
#include "source/comm_primitive.h" // CommPrimitive class definition (uses HiCCL globals, CommBench::MPI_Type)
#include "source/command.h"        // Command class definition (uses CommPrimitive, Compute)
#include "source/comm.h"           // Comm class definition (uses BROADCAST, REDUCE, Command, CommPrimitive)

// 6. Placeholder for top-level functions (from bench.h), declared in HiCCL namespace
namespace HiCCL { // Re-open HiCCL namespace to extend it

    template <typename T, typename CommType>
    void validate(T *sendbuf_d, T *recvbuf_d, size_t count, int patternid, int root, CommType &comm) {
        if (myid == printid) {
            printf("HiCCL::validate (stub) called for type %s, count %zu, pattern %d.\n", typeid(T).name(), count, patternid);
            fflush(stdout);
        }
    }

    template <typename T>
    void measure(int warmup, int numiter, size_t count, Comm<T> &comm) { // This `Comm<T>` is now the actual class
        if (myid == printid) {
            printf("HiCCL::measure (stub) called for type %s, warmup %d, numiter %d, count %zu.\n", typeid(T).name(), warmup, numiter, count);
            fflush(stdout);
        }
    }

} // namespace HiCCL

#endif // HICCL_H
