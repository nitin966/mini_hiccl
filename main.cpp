// main.cpp
#include "hiccl.h" // Includes hiccl.h, which in turn includes all other necessary headers

#include <iostream> // For std::cout
#include <vector>   // For std::vector (needed for basic data initialization later)
#include <numeric>  // For std::iota (to easily fill arrays)
#include <algorithm> // For std::min

// These are *declarations* of the global variables that are *defined* in CommBench/commbench.cpp.
// They are in the global scope. MPI_Init will modify them.
extern int myid;
extern int numproc;
extern int printid;
extern MPI_Comm comm_mpi;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);

    if (HiCCL::myid == HiCCL::printid) {
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "MPI Rank (HiCCL::myid): " << HiCCL::myid << std::endl;
        std::cout << "Total MPI Processes (HiCCL::numproc): " << HiCCL::numproc << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Sync after initial setup
    if (HiCCL::myid == HiCCL::printid) {
        printf("\n--- MPI Barrier Sync Point 1 ---\n");
        fflush(stdout);
    }


    // --- Test HiCCL::Compute class ---
    HiCCL::Compute<float> my_compute_unit;
    size_t reduction_count = 100;
    float* input1_cpu; float* input2_cpu; float* input3_cpu;
    CommBench::allocate(input1_cpu, reduction_count);
    CommBench::allocate(input2_cpu, reduction_count);
    CommBench::allocate(input3_cpu, reduction_count);
    for (size_t i = 0; i < reduction_count; ++i) {
        input1_cpu[i] = (float)i + 1.0f;
        input2_cpu[i] = (float)i * 2.0f;
        input3_cpu[i] = (float)i * 0.5f;
    }
    float* output_cpu_compute;
    CommBench::allocate(output_cpu_compute, reduction_count);
    std::vector<float*> inputs_for_task = {input1_cpu, input2_cpu, input3_cpu};
    my_compute_unit.add(inputs_for_task, output_cpu_compute, reduction_count, HiCCL::myid);
    if (HiCCL::myid == HiCCL::printid) {
        std::cout << "\n--- Running Compute Task (Previous Test) ---" << std::endl;
        std::cout << std::flush;
    }
    my_compute_unit.start();
    my_compute_unit.wait();
    my_compute_unit.report();
    if (HiCCL::myid == HiCCL::printid) {
        std::cout << "First 5 output elements (compute): " << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, reduction_count); ++i) {
            printf("  output[%zu] = %.2f\n", i, output_cpu_compute[i]);
        }
        std::cout << "--- End Compute Test (Previous Test) ---" << std::endl;
        fflush(stdout);
    }
    CommBench::free(input1_cpu);
    CommBench::free(input2_cpu);
    CommBench::free(input3_cpu);
    CommBench::free(output_cpu_compute);

    MPI_Barrier(MPI_COMM_WORLD); // Sync after Compute test
    if (HiCCL::myid == HiCCL::printid) {
        printf("\n--- MPI Barrier Sync Point 2 ---\n");
        fflush(stdout);
    }


    // --- Test HiCCL::CommPrimitive class (existing test, renamed) ---
    if (HiCCL::numproc >= 2) {
        size_t comm_count = 10;
        float* send_buffer;
        float* recv_buffer;
        CommBench::allocate(send_buffer, comm_count);
        CommBench::allocate(recv_buffer, comm_count);
        if (HiCCL::myid == 0) { for (size_t i = 0; i < comm_count; ++i) { send_buffer[i] = (float)(i + 100.0f); } }
        if (HiCCL::myid == 1) { for (size_t i = 0; i < comm_count; ++i) { recv_buffer[i] = -1.0f; } }
        if (HiCCL::myid == 0) { for (size_t i = 0; i < comm_count; ++i) { recv_buffer[i] = -2.0f; } }

        HiCCL::CommPrimitive<float> my_comm_primitive_unit(CommBench::MPI);
        my_comm_primitive_unit.add(send_buffer, 0, recv_buffer, 0, comm_count, 0, 1);
        my_comm_primitive_unit.add(send_buffer, 0, recv_buffer, 0, comm_count, 0, 0);

        if (HiCCL::myid == HiCCL::printid) {
            std::cout << "\n--- Running CommPrimitive Task (Previous Test) ---" << std::endl;
            std::cout << std::flush;
        }
        my_comm_primitive_unit.report();
        my_comm_primitive_unit.start();
        my_comm_primitive_unit.wait();

        // Print verification for CommPrimitive:
        if (HiCCL::myid == 1) {
            bool pass = true;
            std::cout << "Rank 1 received data (first 5, CommPrimitive): " << std::endl;
            for (size_t i = 0; i < std::min((size_t)5, comm_count); ++i) {
                printf("  recv_buffer[%zu] = %.2f (expected: %.2f)\n", i, recv_buffer[i], (float)(i + 100.0f));
                if (recv_buffer[i] != (float)(i + 100.0f)) { pass = false; }
            }
            if (pass) { std::cout << "Rank 1: COMMUNICATION PASSED!" << std::endl; } else { std::cout << "Rank 1: COMMUNICATION FAILED!" << std::endl; }
            std::cout << std::flush;
        }
        if (HiCCL::myid == 0) {
            bool pass = true;
            std::cout << "Rank 0 self-communicated data (first 5, CommPrimitive): " << std::endl;
            for (size_t i = 0; i < std::min((size_t)5, comm_count); ++i) {
                printf("  recv_buffer[%zu] = %.2f (expected: %.2f)\n", i, recv_buffer[i], (float)(i + 100.0f));
                if (recv_buffer[i] != (float)(i + 100.0f)) { pass = false; }
            }
            if (pass) { std::cout << "Rank 0: SELF-COMMUNICATION PASSED!" << std::endl; } else { std::cout << "Rank 0: SELF-COMMUNICATION FAILED!" << std::endl; }
            std::cout << std::flush;
        }
        if (HiCCL::myid == HiCCL::printid) {
            std::cout << "--- End CommPrimitive Test (Previous Test) ---" << std::endl;
            fflush(stdout);
        }
        CommBench::free(send_buffer);
        CommBench::free(recv_buffer);
    } else if (HiCCL::myid == HiCCL::printid) {
        std::cout << "\nSkipping CommPrimitive test: Need at least 2 processes (currently " << HiCCL::numproc << ").\n" << std::endl;
        std::cout << std::flush;
    }

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD); // Sync after CommPrimitive test
    if (HiCCL::myid == HiCCL::printid) {
        printf("\n--- MPI Barrier Sync Point 3 ---\n");
        fflush(stdout);
    }


    // --- NEW: Test HiCCL::Command class ---
    if (HiCCL::numproc >= 2) {
        // Allocate buffers etc. (always needed)
        size_t command_test_count = 20;
        float* command_send_buf;
        float* command_recv_buf;
        CommBench::allocate(command_send_buf, command_test_count);
        CommBench::allocate(command_recv_buf, command_test_count);
        if (HiCCL::myid == 0) { for (size_t i = 0; i < command_test_count; ++i) command_send_buf[i] = (float)i * 100.0f; }
        if (HiCCL::myid == 1) { for (size_t i = 0; i < command_test_count; ++i) command_recv_buf[i] = -1.0f; }

        // Allocate CommPrimitive and Compute on the stack for testing this specific hang.
        // This avoids issues with new/delete/destructor interactions across processes.
        HiCCL::CommPrimitive<float> command_comm_primitive_stack(CommBench::MPI); // Stack allocated
        command_comm_primitive_stack.add(command_send_buf, 0, command_recv_buf, 0, command_test_count, 0, 1);

        float* comp_input1; float* comp_input2;
        CommBench::allocate(comp_input1, command_test_count);
        CommBench::allocate(comp_input2, command_test_count);
        for (size_t i = 0; i < command_test_count; ++i) { comp_input1[i] = (float)i; comp_input2[i] = (float)i * 2; }
        float* comp_output;
        CommBench::allocate(comp_output, command_test_count);

        HiCCL::Compute<float> command_compute_unit_stack; // Stack allocated
        command_compute_unit_stack.add({comp_input1, comp_input2}, comp_output, command_test_count, HiCCL::myid);

        // Use addresses of stack-allocated objects for the Command constructor
        HiCCL::Command<float> my_command(&command_comm_primitive_stack, &command_compute_unit_stack);

        // -----------------------------------------------------------------------
        // ALL BLOCKS ARE UNCOMMENTED IN THIS VERSION
        // -----------------------------------------------------------------------

        // Block 1: Basic print and sync
        if (HiCCL::myid == HiCCL::printid) {
            std::cout << "\n--- Running HiCCL::Command Test - Block 1 (Minimal) ---\n" << std::endl;
            std::cout << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // -----------------------------------------------------------------------


        // Block 2: Try reporting (using the minimal Command::report())
        if (HiCCL::myid == HiCCL::printid) {
            std::cout << "\n--- Running HiCCL::Command Test - Block 2 (Reporting) ---\n" << std::endl;
            std::cout << std::flush;
        }
        my_command.report(); // This now uses the minimal report() in Command.h
        MPI_Barrier(MPI_COMM_WORLD); // Sync after reporting
        if (HiCCL::myid == HiCCL::printid) {
            printf("\n--- MPI Barrier Sync Point 4 (after report) ---\n");
            fflush(stdout);
        }
        // -----------------------------------------------------------------------


        // Block 3: Try starting and waiting for the command
        if (HiCCL::myid == HiCCL::printid) {
            std::cout << "\n--- Running HiCCL::Command Test - Block 3 (Start/Wait) ---\n" << std::endl;
            std::cout << std::flush;
        }
        my_command.start();
        my_command.wait();
        MPI_Barrier(MPI_COMM_WORLD); // Sync after command start/wait
        if (HiCCL::myid == HiCCL::printid) {
            printf("\n--- MPI Barrier Sync Point 5 (after start/wait) ---\n");
            fflush(stdout);
        }
        // -----------------------------------------------------------------------

        // Block 4: Try communication verification (Rank 1 specific)
        if (HiCCL::myid == 1) {
            bool pass = true;
            std::cout << "Rank 1 received data (first 5, Command comm): " << std::endl;
            for (size_t i = 0; i < std::min((size_t)5, command_test_count); ++i) {
                printf("  recv_buffer[%zu] = %.2f (expected: %.2f)\n", i, command_recv_buf[i], (float)(i * 100.0f));
                if (command_recv_buf[i] != (float)(i * 100.0f)) { pass = false; }
            }
            if (pass) { std::cout << "Rank 1: COMMAND COMMUNICATION PASSED!" << std::endl; } else { std::cout << "Rank 1: COMMAND COMMUNICATION FAILED!" << std::endl; }
            std::cout << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD); // Sync after verification part 1
        if (HiCCL::myid == HiCCL::printid) {
            printf("\n--- MPI Barrier Sync Point 6 (after Rank 1 verify) ---\n");
            fflush(stdout);
        }
        // -----------------------------------------------------------------------

        // Block 5: Try computation verification and final output (Rank 0 specific)
        if (HiCCL::myid == HiCCL::printid) {
            std::cout << "Command Computation Output (first 5, expected: i + i*2): " << std::endl;
            for (size_t i = 0; i < std::min((size_t)5, command_test_count); ++i) {
                printf("  comp_output[%zu] = %.2f (expected: %.2f)\n", i, comp_output[i], (float)i * 3.0f);
            }
            bool pass = true;
            for (size_t i = 0; i < command_test_count; ++i) {
                 if (comp_output[i] != (float)i * 3.0f) { pass = false; break; }
            }
            if (pass) { std::cout << "COMMAND COMPUTATION PASSED!" << std::endl; } else { std::cout << "COMMAND COMPUTATION FAILED!" << std::endl; }
            std::cout << "--- End HiCCL::Command Test ---" << std::endl;
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD); // Sync after final verification part
        if (HiCCL::myid == HiCCL::printid) {
            printf("\n--- MPI Barrier Sync Point 7 (End Command Test) ---\n");
            fflush(stdout);
        }
        // -----------------------------------------------------------------------


        // Clean up memory and delete Command components (always needed)
        CommBench::free(command_send_buf);
        CommBench::free(command_recv_buf);
        // Stack allocated, no delete: delete command_comm_primitive;
        CommBench::free(comp_input1);
        CommBench::free(comp_input2);
        CommBench::free(comp_output);
        // Stack allocated, no delete: delete command_compute_unit;

    } else if (HiCCL::myid == HiCCL::printid) {
        std::cout << "\nSkipping Command test: Need at least 2 processes (currently " << HiCCL::numproc << ").\n" << std::endl;
        std::cout << std::flush;
    }

    MPI_Finalize();
    return 0;
}
