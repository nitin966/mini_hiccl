// source/comm_primitive.h
#ifndef HICCL_COMM_PRIMITIVE_H
#define HICCL_COMM_PRIMITIVE_H

// IMPORTANT: hiccl.h MUST be included first for global variables
#include "hiccl.h" // Includes CommBench/commbench.h

#include <vector>
#include <list>
#include <cstdio>
#include <numeric>
#include <algorithm>

namespace HiCCL {

template <typename T>
class CommPrimitive {
public:
    CommBench::library lib;

    int numcomm = 0;
    std::vector<T*> sendbuf;
    std::vector<size_t> sendoffset;
    std::vector<T*> recvbuf;
    std::vector<size_t> recvoffset;
    std::vector<size_t> count;
    std::vector<int> sendid;
    std::vector<int> recvid;

    int numcompute = 0;
    std::vector<std::vector<T*>> inputbuf_comp;
    std::vector<T*> outputbuf_comp;
    std::vector<size_t> numreduce_comp;
    std::vector<int> compid;

    CommPrimitive(CommBench::library library_type) : lib(library_type) {}

    ~CommPrimitive() {
        // No custom memory deallocation needed here yet
    }

    void add(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t cnt, int sid, int rid) {
        this->sendbuf.push_back(sbuf);
        this->sendoffset.push_back(soffset);
        this->recvbuf.push_back(rbuf);
        this->recvoffset.push_back(roffset);
        this->count.push_back(cnt);
        this->sendid.push_back(sid);
        this->recvid.push_back(rid);
        numcomm++;
    }

    void add(std::vector<T*> input_buffers, T* output_buffer, size_t num_elements_reduce, int compute_id) {
        this->inputbuf_comp.push_back(input_buffers);
        this->outputbuf_comp.push_back(output_buffer);
        this->numreduce_comp.push_back(num_elements_reduce);
        this->compid.push_back(compute_id);
        numcompute++;
    }

    void start() {
        for (int i = 0; i < numcomm; ++i) {
            bool is_sender = (myid == sendid[i]);
            bool is_receiver = (myid == recvid[i]);
            int mpi_tag = 0;

            if (is_sender && is_receiver) {
                if (sendbuf[i] + sendoffset[i] != recvbuf[i] + recvoffset[i]) {
                     CommBench::memcpyD2D(recvbuf[i] + recvoffset[i], sendbuf[i] + sendoffset[i], count[i]);
                }
            } else if (is_sender) {
                MPI_Send(sendbuf[i] + sendoffset[i], count[i], CommBench::MPI_Type<T>::value(), recvid[i], mpi_tag, comm_mpi);
            } else if (is_receiver) {
                MPI_Recv(recvbuf[i] + recvoffset[i], count[i], CommBench::MPI_Type<T>::value(), sendid[i], mpi_tag, comm_mpi, MPI_STATUS_IGNORE);
            }
        }
    }

    void wait() {
        // Blocking MPI operations are complete in start().
    }

    void report() {
        if (myid == printid) {
            printf("CommPrimitive Report (Lib: "); CommBench::print_lib(this->lib); printf("):\n");
            printf("  Communication Tasks: %d\n", numcomm);
            if (numcomm > 0) {
                size_t total_comm_data_bytes = 0;
                for (int i = 0; i < numcomm; ++i) {
                    total_comm_data_bytes += count[i] * sizeof(T);
                }
                printf("  Total Communication Data: "); CommBench::print_data(total_comm_data_bytes); printf("\n");

                if (numproc <= 64) {
                    printf("  Communication Matrix (rows=recv, cols=send):\n");
                    std::vector<std::vector<int>> matrix(numproc, std::vector<int>(numproc, 0));
                    for(int i = 0; i < numcomm; ++i) {
                        matrix[recvid[i]][sendid[i]]++;
                    }
                    for(int recv_p = 0; recv_p < numproc; ++recv_p) {
                        for(int send_p = 0; send_p < numproc; ++send_p) {
                            if (matrix[recv_p][send_p] > 0) {
                                printf("%d ", matrix[recv_p][send_p]);
                            } else {
                                printf(". ");
                            }
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
            }
            printf("  Computation Tasks: %d\n", numcompute);
            if (numcompute > 0) {
                size_t total_input_data_bytes = 0;
                size_t total_output_data_bytes = 0;
                for (int i = 0; i < numcompute; ++i) {
                    total_input_data_bytes += numreduce_comp[i] * sizeof(T) * inputbuf_comp[i].size();
                    total_output_data_bytes += numreduce_comp[i] * sizeof(T);
                }
                printf("  Total Input Data for Comp: "); CommBench::print_data(total_input_data_bytes); printf("\n");
                printf("  Total Output Data for Comp: "); CommBench::print_data(total_output_data_bytes); printf("\n");
            }
            printf("\n");
        }
    }
};

} // namespace HiCCL

#endif // HICCL_COMM_PRIMITIVE_H
