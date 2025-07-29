// source/compute.h
#ifndef HICCL_COMPUTE_H
#define HICCL_COMPUTE_H

// IMPORTANT: hiccl.h MUST be included first for global variables
#include "hiccl.h" // Includes CommBench/commbench.h

#include <vector>
#include <cstdio>
#include <numeric>
#include <omp.h>

namespace HiCCL {

template <typename T>
void reduce_kernel_cpu(T *output, size_t count, T **input, int numinput) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        T acc = 0;
        for (int in = 0; in < numinput; ++in) {
            acc += input[in][i];
        }
        output[i] = acc;
    }
}

template <typename T>
class Compute {
public:
    int numcomp = 0;
    std::vector<std::vector<T*>> inputbuf;
    std::vector<T*> outputbuf;
    std::vector<size_t> count;
    std::vector<T**> inputbuf_d;

    Compute() {}

    void add(std::vector<T*> inputbuf_vec, T* outputbuf_ptr, size_t count_val, int compid) {
        if (myid == compid) {
            this->inputbuf.push_back(inputbuf_vec);
            this->outputbuf.push_back(outputbuf_ptr);
            this->count.push_back(count_val);

            T **input_ptr_array_on_host;
            CommBench::allocate(input_ptr_array_on_host, inputbuf_vec.size());
            CommBench::memcpyH2D(input_ptr_array_on_host, inputbuf_vec.data(), inputbuf_vec.size());
            this->inputbuf_d.push_back(input_ptr_array_on_host);

            numcomp++;
        }
    }

    void start() {
        for (int comp = 0; comp < numcomp; ++comp) {
            reduce_kernel_cpu<T>(outputbuf[comp], count[comp], inputbuf_d[comp], inputbuf[comp].size());
        }
    }

    void wait() {}

    void report() {
        std::vector<int> numcomp_all(numproc);
        MPI_Allgather(&numcomp, 1, MPI_INT, numcomp_all.data(), 1, MPI_INT, comm_mpi);

        int numinput_local = 0;
        for (int comp = 0; comp < numcomp; ++comp) {
            numinput_local += inputbuf[comp].size();
        }
        std::vector<int> numinput_all(numproc);
        MPI_Allgather(&numinput_local, 1, MPI_INT, numinput_all.data(), 1, MPI_INT, comm_mpi);

        if (myid == printid) {
            printf("Compute Report:\n");
            printf("  numcomp (total compute tasks across all processes): ");
            for (int p = 0; p < numproc; ++p) {
                printf("%d(%d) ", numcomp_all[p], numinput_all[p]);
            }
            printf("\n");
            printf("\n");
        }
    }

    void measure(int warmup, int numiter) {
        size_t count_total = 0;
        for(int comp = 0; comp < numcomp; comp++)
          count_total += count[comp] * (inputbuf[comp].size() + 1);
        MPI_Allreduce(MPI_IN_PLACE, &count_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);

        if(myid == printid) {
            printf("Measure Reduction Kernel\n");
            printf("%d warmup iterations (in order)\n", warmup);
        }
        for (int iter = -warmup; iter < numiter; iter++) {
            MPI_Barrier(comm_mpi);
            double time = MPI_Wtime();

            this->start();

            double start_time_overhead = MPI_Wtime() - time;
            this->wait();
            time = MPI_Wtime() - time;

            MPI_Allreduce(MPI_IN_PLACE, &start_time_overhead, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
            MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);

            if(iter < 0) {
                if(myid == printid)
                    printf("warmup: startup %.2e, total: %e\n", start_time_overhead, time);
            }
            else {
                times[iter] = time;
            }
        }
        std::sort(times, times + numiter, [](const double & a, const double & b) -> bool {return a < b;});

        if(myid == printid) {
            printf("%d measurement iterations (sorted):\n", numiter);
            for(int iter = 0; iter < numiter; iter++) {
                printf("time: %.4e", times[iter]);
                if(iter == 0) printf(" -> min\n");
                else if(iter == numiter / 2) printf(" -> median\n");
                else if(iter == numiter - 1) printf(" -> max\n");
                else printf("\n");
            }
            printf("\n");
            double minTime = times[0];
            double medTime = times[numiter / 2];
            double maxTime = times[numiter - 1];
            double avgTime = 0;
            for(int iter = 0; iter < numiter; iter++)
                avgTime += times[iter];
            avgTime /= numiter;

            double data_bytes = (double)count_total * sizeof(T);
            printf("Total data processed (approx): "); CommBench::print_data(data_bytes); printf("\n");
            printf("minTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", minTime * 1e6, minTime / data_bytes * 1e12, data_bytes / minTime / 1e9);
            printf("medTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", medTime * 1e6, medTime / data_bytes * 1e12, data_bytes / medTime / 1e9);
            printf("maxTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data_bytes * 1e12, data_bytes / maxTime / 1e9);
            printf("avgTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data_bytes * 1e12, data_bytes / avgTime / 1e9);
            printf("\n");
        }
    }
    double times[100];

    ~Compute() {
        for (T** ptr_array : inputbuf_d) {
            CommBench::free(ptr_array);
        }
        inputbuf_d.clear();
    }
};

} // namespace HiCCL

#endif // HICCL_COMPUTE_H
