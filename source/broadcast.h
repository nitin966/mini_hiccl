// source/broadcast.h
#ifndef HICCL_BROADCAST_H
#define HICCL_BROADCAST_H

// IMPORTANT: hiccl.h MUST be included first for 'pattern' enum visibility
#include "hiccl.h" // Includes CommBench/commbench.h and defines HiCCL::pattern

#include <vector>   // For std::vector

namespace HiCCL {

template <typename T>
struct BROADCAST { // BROADCAST is a struct in your original code
    T* sendbuf;
    size_t sendoffset;
    T* recvbuf;
    size_t recvoffset;
    size_t count;
    int sendid;
    std::vector<int> recvids; // Note: vector of receiver IDs

    // Constructor 1: Takes a vector of receiver IDs directly
    BROADCAST(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t cnt, int sid, std::vector<int> &rids)
        : sendbuf(sbuf), sendoffset(soffset), recvbuf(rbuf), recvoffset(roffset), count(cnt), sendid(sid), recvids(rids) {
    }

    // Constructor 2: Takes a single receiver ID (or special values for 'all'/'others')
    // This constructor populates the recvids vector based on the single 'recvid_single' parameter.
    BROADCAST(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t cnt, int sid, int recvid_single)
        : sendbuf(sbuf), sendoffset(soffset), recvbuf(rbuf), recvoffset(roffset), count(cnt), sendid(sid) {
        // Logic to populate recvids based on recvid_single
        for(int i = 0; i < numproc; i++) { // numproc is from HiCCL's globals
            if(recvid_single == numproc) // Special value indicating 'all' processes
                recvids.push_back(i);
            else if(recvid_single == -1) { // Special value indicating 'others' (all except sender)
                if(i != sendid)
                    recvids.push_back(i);
            }
            else // A specific single receiver ID
                if(i == recvid_single)
                    recvids.push_back(i);
        }
    }

    // Constructor 3: Takes a 'pattern' enum (all or others)
    // This is sugar for Constructor 2, using enum for clarity.
    BROADCAST(T *sbuf, size_t soffset, T *rbuf, size_t roffset, size_t cnt, int sid, HiCCL::pattern recv_pattern_enum) // Use HiCCL::pattern
        : sendbuf(sbuf), sendoffset(soffset), recvbuf(rbuf), recvoffset(roffset), count(cnt), sendid(sid) {
        int recvid_internal = (recv_pattern_enum == HiCCL::pattern::others ? -1 : numproc); // Use HiCCL::pattern
        // Re-use the logic from Constructor 2 to populate recvids
        for(int i = 0; i < numproc; i++) {
            if(recvid_internal == numproc)
                recvids.push_back(i);
            else if(recvid_internal == -1) {
                if(i != sendid)
                    recvids.push_back(i);
            }
            else
                if(i == recvid_internal)
                    recvids.push_back(i);
        }
    }

    // Report method (simplified for now, actual implementation is complex)
    void report() {
        if (myid == printid) { // myid and printid are from HiCCL globals
            printf("BROADCAST (stub) report: count %zu\n", count);
            fflush(stdout);
        }
    }
};

} // namespace HiCCL

#endif // HICCL_BROADCAST_H
