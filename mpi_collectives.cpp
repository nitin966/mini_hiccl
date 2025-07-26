#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 5;
    std::vector<int> data(N);

    // Each rank sets data = rank + 1
    for (int i = 0; i < N; i++) {
        data[i] = rank + 1;
    }
    std::cout << "Before Bcast, Rank " << rank << ": ";
    for (auto x : data) std::cout << x << " ";
    std::cout << std::endl;

    // Broadcast from rank 0
    MPI_Bcast(data.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << "After Bcast, Rank " << rank << ": ";
    for (auto x : data) std::cout << x << " ";
    std::cout << std::endl;

    // Perform Allreduce (sum)
    std::vector<int> reduced(N);
    MPI_Allreduce(data.data(), reduced.data(), N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::cout << "After Allreduce, Rank " << rank << ": ";
    for (auto x : reduced) std::cout << x << " ";
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}

