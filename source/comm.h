// source/comm.h
#ifndef HICCL_COMM_H
#define HICCL_COMM_H

// IMPORTANT: hiccl.h MUST be included first for global variables and enums
#include "hiccl.h" // Includes CommBench/commbench.h, source/compute.h, source/comm_primitive.h, source/command.h, source/broadcast.h, source/reduce.h

#include <vector>
#include <list>
#include <cstdio>
#include <numeric>
#include <algorithm>

namespace HiCCL {

template <typename T>
class Comm {
public:
    std::vector<std::vector<BROADCAST<T>>> bcast_epoch;
    std::vector<std::vector<REDUCE<T>>> reduce_epoch;
    int numepoch = 0;

    std::vector<int> hierarchy = {numproc};
    std::vector<CommBench::library> library = {CommBench::MPI};
    int numstripe = 1;
    int ringnodes = 1;
    int pipedepth = 1;

    T *sendbuf = nullptr;
    T *recvbuf = nullptr;
    size_t sendcount = 0;
    size_t recvcount = 0;

    std::vector<std::list<Command<T>>> command_batch;
    std::vector<std::list<CommPrimitive<T>*>> coll_batch;

    void set_hierarchy(std::vector<int> hierarchy_val, std::vector<CommBench::library> library_val) {
        if(hierarchy_val.size() != library_val.size()) {
            if(myid == printid)
                printf("Error: Hierarchy and library vectors must have the same size!\n");
            return;
        } else {
            this->hierarchy = hierarchy_val;
            this->library = library_val;
        }
    }
    void set_pipedepth(int pipedepth_val) { // FIX: Corrected typo 'pipedepedth_val'
        this->pipedepth = pipedepth_val;
    }
    void set_numstripe(int numstripe_val) {
        this->numstripe = numstripe_val;
    }
    void set_ringnodes(int ringnodes_val) {
        this->ringnodes = ringnodes_val;
    }
    void set_endpoints(T *sendbuf_ptr, size_t sendcount_val, T *recvbuf_ptr, size_t recvcount_val) {
        this->sendbuf = sendbuf_ptr;
        this->sendcount = sendcount_val;
        this->recvbuf = recvbuf_ptr;
        this->recvcount = recvcount_val;
    }

    void print_parameters() {
        if(myid == printid) {
            printf("**************** HiCCL PARAMETERS\n");
            printf("%zu-level hierarchy:\n", hierarchy.size());
            for(size_t i = 0; i < hierarchy.size(); i++) {
                printf("  level %zu factor: %d library: ", i, hierarchy[i]);
                CommBench::print_lib(library[i]);
                if(i == 0 && hierarchy[0] == numproc && library[0] == CommBench::MPI)
                    printf(" (default)\n");
                else
                    printf("\n");
            }
            printf("numstripe: %d", numstripe);
            if(numstripe == 1) printf(" (default)\n");
            else printf("\n");
            printf("ringnodes: %d", ringnodes);
            if(ringnodes == 1) printf(" (default)\n");
            else printf("\n");
            printf("pipedepth: %d", pipedepth);
            if(pipedepth == 1) printf(" (default)\n");
            else printf("\n");
            printf("sendbuf: %p, sendcount %zu", (void*)sendbuf, sendcount);
            if(sendbuf == nullptr) printf(" (default)\n");
            else printf("\n");
            printf("recvbuf: %p, recvcount %zu", (void*)recvbuf, recvcount);
            if(recvbuf == nullptr) printf(" (default)\n");
            else printf("\n");
            printf("*********************************\n");
            fflush(stdout);
        }
    }

    Comm() {
        add_fence();
    }

    ~Comm() {
        for (auto& command_list : command_batch) {
            for (auto& cmd : command_list) {
                if (cmd.comm_primitive) {
                    delete cmd.comm_primitive;
                }
                if (cmd.compute) {
                    delete cmd.compute;
                }
            }
        }
        command_batch.clear();

        for (auto& coll_list : coll_batch) {
            for (auto& coll_ptr : coll_list) {
                if (coll_ptr) {
                    delete coll_ptr;
                }
            }
        }
        coll_batch.clear();
    }

    void add_fence() {
        bcast_epoch.push_back(std::vector<BROADCAST<T>>());
        reduce_epoch.push_back(std::vector<REDUCE<T>>());
        if(myid == printid) {
            printf("Add epoch %d\n", numepoch);
            fflush(stdout);
        }
        numepoch++;
    }

    // --- Corrected add_bcast overloads ---
    void add_bcast(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t count_val, int sid, std::vector<int> &rids) {
        bcast_epoch.back().push_back(BROADCAST<T>(sbuf, soffset, rbuf, roffset, count_val, sid, rids));
        if (myid == printid) { printf("add_bcast (vector rids) called\n"); fflush(stdout); }
    }
    void add_bcast(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t count_val, int sid, int rid) {
        bcast_epoch.back().push_back(BROADCAST<T>(sbuf, soffset, rbuf, roffset, count_val, sid, rid));
        if (myid == printid) { printf("add_bcast (single rid) called\n"); fflush(stdout); }
    }
    void add_bcast(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t count_val, int sid, HiCCL::pattern recv_pattern) { // Use HiCCL::pattern
        bcast_epoch.back().push_back(BROADCAST<T>(sbuf, soffset, rbuf, roffset, count_val, sid, recv_pattern));
        if (myid == printid) { printf("add_bcast (pattern) called\n"); fflush(stdout); }
    }

    // --- Corrected add_reduce overloads ---
    void add_reduce(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t count_val, std::vector<int> &sids, int rid) {
        reduce_epoch.back().push_back(REDUCE<T>(sbuf, soffset, rbuf, roffset, count_val, sids, rid));
        if (myid == printid) { printf("add_reduce (vector sids) called\n"); fflush(stdout); }
    }
    void add_reduce(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t count_val, int sid, int rid) {
        reduce_epoch.back().push_back(REDUCE<T>(sbuf, soffset, rbuf, roffset, count_val, sid, rid));
        if (myid == printid) { printf("add_reduce (single sid) called\n"); fflush(stdout); }
    }
    void add_reduce(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t count_val, HiCCL::pattern send_pattern, int rid) { // CORRECTED: Added 'int rid' parameter back
        reduce_epoch.back().push_back(REDUCE<T>(sbuf, soffset, rbuf, roffset, count_val, send_pattern, rid));
        if (myid == printid) { printf("add_reduce (pattern) called\n"); fflush(stdout); }
    }

    void init() {
        if(myid == printid) {
            printf("\nFINAL PARAMETERS (from Comm::init)\n");
            print_parameters();
            fflush(stdout);
        }
        if(myid == printid) {
            printf("Comm::init (stub): No collective algorithms implemented yet.\n");
            fflush(stdout);
        }
    }

    void run() {
        if(myid == printid) { printf("Comm::run (stub) called.\n"); fflush(stdout); }
    }

    void run(T *sendbuf_param, T *recvbuf_param) {
        if(myid == printid) { printf("Comm::run (overload stub) called.\n"); fflush(stdout); }
        CommBench::memcpyD2D(this->sendbuf, sendbuf_param, sendcount);
        run();
        CommBench::memcpyD2D(recvbuf_param, this->recvbuf, recvcount);
    }

    static void* run_async(void* arg) {
        Comm<T> *test = (Comm<T>*) arg;
        test->run();
        pthread_exit(NULL);
        return NULL;
    }

    pthread_t thread;

    void start() {
        // pthread_create(&thread, NULL, Comm<T>::run_async, this);
    }
    void wait() {
        // pthread_join(thread, NULL);
    }
};

} // namespace HiCCL

#endif // HICCL_COMM_H
