// source/command.h
#ifndef HICCL_COMMAND_H
#define HICCL_COMMAND_H

// IMPORTANT: hiccl.h MUST be included first for global variables
#include "hiccl.h" // Includes CommBench/commbench.h, source/compute.h, source/comm_primitive.h
#include <pthread.h> // For pthread_t (needed for potential async operations)

namespace HiCCL {

template <typename T>
class Command {
public:
    CommPrimitive<T> *comm_primitive = nullptr;
    Compute<T> *compute = nullptr;

    Command(CommPrimitive<T> *comm_p, Compute<T> *comp_p)
        : comm_primitive(comm_p), compute(comp_p) {}

    ~Command() {} // No change here

    void start() {
        if (comm_primitive) { comm_primitive->start(); }
        if (compute) { compute->start(); }
    }

    void wait() {
        if (comm_primitive) { comm_primitive->wait(); }
        if (compute) { compute->wait(); }
    }

    void measure(int warmup, int numiter, size_t count) {
        int total_num_comm = 0;
        int total_num_comp = 0;
        if (comm_primitive) { total_num_comm = comm_primitive->numcomm; }
        if (compute) { total_num_comp = compute->numcomp; }

        MPI_Allreduce(MPI_IN_PLACE, &total_num_comm, 1, MPI_INT, MPI_SUM, comm_mpi);
        MPI_Allreduce(MPI_IN_PLACE, &total_num_comp, 1, MPI_INT, MPI_SUM, comm_mpi);

        if (total_num_comm > 0) {
            if (myid == printid) {
                if (total_num_comp > 0) printf("COMMAND TYPE: COMMUNICATION + COMPUTATION\n");
                else printf("COMMAND TYPE: COMMUNICATION\n");
            }
            if (total_num_comp > 0) {
                compute->measure(warmup, numiter);
            }
        } else if (total_num_comp > 0) {
            if (myid == printid) printf("COMMAND TYPE: COMPUTATION\n");
            compute->measure(warmup, numiter);
        }
    }

    // --- CRITICAL CHANGE: Make report() minimal to bypass hang ---
    void report() {
        // These Allreduces are crucial and MUST be called by ALL ranks.
        int has_comm = (comm_primitive && comm_primitive->numcomm > 0) ? 1 : 0;
        int has_comp = (compute && compute->numcomp > 0) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &has_comm, 1, MPI_INT, MPI_SUM, comm_mpi);
        MPI_Allreduce(MPI_IN_PLACE, &has_comp, 1, MPI_INT, MPI_SUM, comm_mpi);

        // --- Only print minimal info to check participation, no nested report calls ---
        if (myid == printid) {
            printf("Command Report (Comm: %d, Comp: %d) (Minimal):\n", has_comm, has_comp);
            printf("\n"); // Ensure a newline
            fflush(stdout); // Aggressive flush
        }
    }
};

} // namespace HiCCL

#endif // HICCL_COMMAND_H
