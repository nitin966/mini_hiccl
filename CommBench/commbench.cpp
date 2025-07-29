// CommBench/commbench.cpp
#include "commbench.h"

// --- GLOBAL VARIABLE DEFINITIONS (OUTSIDE OF ANY NAMESPACE) ---
int myid = 0;
int numproc = 1;
int printid = 0;
MPI_Comm comm_mpi = MPI_COMM_WORLD;
