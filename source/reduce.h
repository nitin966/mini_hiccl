// source/reduce.h
#ifndef HICCL_REDUCE_H
#define HICCL_REDUCE_H

// IMPORTANT: hiccl.h MUST be included first for 'pattern' enum visibility
#include "hiccl.h" // Includes CommBench/commbench.h and defines HiCCL::pattern

#include <vector>   // For std::vector

namespace HiCCL {

template <typename T>
struct REDUCE {
    T* sendbuf;
    size_t sendoffset;
    T* recvbuf;
    size_t recvoffset;
    size_t count;
    std::vector<int> sendids; // Note: vector of sender IDs
    int recvid;

    // Constructor 1: Takes a vector of sender IDs directly
    REDUCE(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t cnt, std::vector<int> &sids, int rid)
        : sendbuf(sbuf), sendoffset(soffset), recvbuf(rbuf), recvoffset(roffset), count(cnt), sendids(sids), recvid(rid) {
    }

    // Constructor 2: Takes a single sender ID (or special values for 'all'/'others')
    REDUCE(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t cnt, int sendid_single, int rid)
        : sendbuf(sbuf), sendoffset(soffset), recvbuf(rbuf), recvoffset(roffset), count(cnt), recvid(rid) {
        for(int i = 0; i < numproc; i++) { // numproc is from HiCCL's globals
            if(sendid_single == numproc)
                sendids.push_back(i);
            else if(sendid_single == -1) {
                if(i != recvid)
                    sendids.push_back(i);
            }
            else
                if(i == sendid_single)
                    sendids.push_back(i);
        }
    }

    // Constructor 3: Takes a 'pattern' enum (all or others)
    REDUCE(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t cnt, HiCCL::pattern send_pattern_enum, int rid) // Use HiCCL::pattern
        : sendbuf(sbuf), sendoffset(soffset), recvbuf(rbuf), recvoffset(roffset), count(cnt), recvid(rid) {
        int sendid_internal = (send_pattern_enum == HiCCL::pattern::others ? -1 : numproc);
        for(int i = 0; i < numproc; i++) {
            if(sendid_internal == numproc)
                sendids.push_back(i);
            else if(sendid_internal == -1) {
                if(i != recvid)
                    sendids.push_back(i);
            }
            else
                if(i == sendid_internal)
                    sendids.push_back(i);
        }
    }

    void report() {
        if (myid == printid) {
            printf("REDUCE (stub) report: count %zu\n", count);
            fflush(stdout);
        }
    }
};

} // namespace HiCCL

#endif // HICCL_REDUCE_H
