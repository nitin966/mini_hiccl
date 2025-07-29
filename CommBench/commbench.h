// CommBench/commbench.h
#ifndef COMMBENCH_H
#define COMMBENCH_H

#include <cstdio>   // For printf
#include <string>   // For std::string (used by print_lib)
#include <mpi.h>    // Essential for MPI types and functions
#include <cstddef>  // For size_t (guaranteed for array sizes)
#include <type_traits> // For std::is_same (used in MPI_Type default specialization)

// --- GLOBAL VARIABLES (OUTSIDE OF ANY NAMESPACE) ---
extern int myid;
extern int numproc;
extern int printid;
extern MPI_Comm comm_mpi;

namespace CommBench {

    enum library {
        dummy, IPC, IPC_get, MPI, XCCL, numlib
    };

    inline void print_data(size_t bytes) {
        if (bytes < 1024) {
            printf("%zu B", bytes);
        } else if (bytes < 1024 * 1024) {
            printf("%.2f KB", (double)bytes / 1024.0);
        } else if (bytes < 1024 * 1024 * 1024) {
            printf("%.2f MB", (double)bytes / (1024.0 * 1024.0));
        } else {
            printf("%.2f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
        }
    }

    inline void print_lib(library lib) {
        switch (lib) {
            case dummy:    printf("Dummy"); break;
            case IPC:      printf("IPC"); break;
            case IPC_get:  printf("IPC_get"); break;
            case MPI:      printf("MPI"); break;
            case XCCL:     printf("XCCL"); break;
            case numlib:   printf("NUM_LIB"); break;
        }
    }

    template<typename T> struct MPI_Type {
        static MPI_Datatype value() {
            static_assert(std::is_same<T, void>::value, "Unsupported MPI type. Please add specialization for your type.");
            return MPI_DATATYPE_NULL;
        }
    };
    template<> struct MPI_Type<char> { static MPI_Datatype value() { return MPI_CHAR; } };
    template<> struct MPI_Type<int> { static MPI_Datatype value() { return MPI_INT; } };
    template<> struct MPI_Type<float> { static MPI_Datatype value() { return MPI_FLOAT; } };
    template<> struct MPI_Type<double> { static MPI_Datatype value() { return MPI_DOUBLE; } };
    template<> struct MPI_Type<long> { static MPI_Datatype value() { return MPI_LONG; } };
    template<> struct MPI_Type<long long> { static MPI_Datatype value() { return MPI_LONG_LONG; } };
    template<> struct MPI_Type<unsigned char> { static MPI_Datatype value() { return MPI_UNSIGNED_CHAR; } };
    template<> struct MPI_Type<unsigned int> { static MPI_Datatype value() { return MPI_UNSIGNED; } };
    template<> struct MPI_Type<unsigned long long> { static MPI_Datatype value() { return MPI_UNSIGNED_LONG_LONG; } };
    template<> struct MPI_Type<size_t> { static MPI_Datatype value() { return MPI_UNSIGNED_LONG; } };
    template<> struct MPI_Type<bool> { static MPI_Datatype value() { return MPI_C_BOOL; } };


    template <typename T> inline void allocate(T*& ptr, size_t count) { ptr = new T[count]; }
    template <typename T> inline void free(T*& ptr) { if (ptr) { delete[] ptr; ptr = nullptr; } }
    template <typename T> inline void memcpyH2D(T* dst, const T* src, size_t count) { if (dst && src && count > 0) { for (size_t i = 0; i < count; ++i) { dst[i] = src[i]; } } }
    template <typename T> inline void memcpyD2H(T* dst, const T* src, size_t count) { if (dst && src && count > 0) { for (size_t i = 0; i < count; ++i) { dst[i] = src[i]; } } }
    template <typename T> inline void memcpyD2D(T* dst, const T* src, size_t count) { if (dst && src && count > 0) { for (size_t i = 0; i < count; ++i) { dst[i] = src[i]; } } }
    inline void setup_gpu() {}
    inline void report_memory() {}

} // namespace CommBench

#endif // COMMBENCH_H
