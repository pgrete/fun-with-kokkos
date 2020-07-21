#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "Kokkos_Layout.hpp"
#include "space_instances.hpp"

using std::cout;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace = Kokkos::HostSpace;
using DevExecSpace = Kokkos::DefaultExecutionSpace;
using ScratchMemSpace = DevExecSpace::scratch_memory_space;
using MyView = Kokkos::View<double ****, Kokkos::LayoutRight, DevMemSpace>;

constexpr int nghost = 2;

auto main(int argc, char *argv[]) -> int {
  int num_threads = 1;  // number of host threads to use
  int num_streams = 1;  // number of (Cuda) streams to use
  int num_cycles = 100; // number of cycles to run
  int num_stages = 2;   // number of stages per cycles
  int size_mesh = 256;  // size of the entire mesh
  int size_block = 128; // size of blocks the create the mesh
  int num_vars = 7;     // number of variables to work on

  // Read command line arguments.
  for (int i = 1; i < argc; i++) {

    if ((strcmp(argv[i], "-num_threads") == 0)) {
      num_threads = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "-num_streams") == 0)) {
      num_streams = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "-num_cycles") == 0)) {
      num_cycles = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "-num_stages") == 0)) {
      num_stages = atoi(argv[++i]);
    } else {
      cout << "Wrong arguments" << std::endl;
      exit(1);
    }
  }

  // sanity checks
  if ((num_streams < num_threads) || (num_streams % num_threads != 0)) {
    cout << "Please use a multiple of num_threads for num_streams" << std::endl;
    exit(1);
  }

  int num_blocks = // total number of blocks
      (size_mesh / size_block) * (size_mesh / size_block) * (size_mesh / size_block);

  Kokkos::initialize(argc, argv);
  {
    // Allocate data
    std::vector<MyView> views;
    views.reserve(num_blocks);
    for (auto i = 0; i < num_blocks; i++) {
      views.emplace_back("view_" + std::to_string(i), num_vars, size_block + 2 * nghost,
                         size_block + 2 * nghost, size_block + 2 * nghost);
    }

    std::vector<DevExecSpace> exec_spaces;

    if (num_streams > 1) {
      for (auto n = 0; n < num_streams; n++) {
        exec_spaces.push_back(SpaceInstance<DevExecSpace>::create());
      }
    } else {
      exec_spaces.push_back(DevExecSpace());
    }

  int complete_cnt = 0;
  auto f = [&](int thread_id, int num_threads) {
    while (complete_cnt != num_blocks) {
      for (auto i = 0; i < num_blocks; ++i) {
        // Workaround to ensure that the same thread works on the same MeshBlocks within
        // a cycle. Trying to circumvent problem in setting scratch pad memory, which is
        // not thread safe.
        if (i % num_threads != thread_id) {
          continue;
        }

        // in principle here's a call to a task list with multiple tasks for each block
        // now it's just a single simple kernel

            cout << "[" << thread_id << "] taking care of block " << i << std::endl << std::flush;
              Kokkos::atomic_increment(&complete_cnt);
      }
    }
  };

for (auto cycle = 0; cycle < num_cycles; cycle++) {
    cout << "Staring cycle " << cycle << std::endl;
  for (auto stage = 0; stage < num_stages; stage++) {
#ifdef KOKKOS_ENABLE_OPENMP
  // using a fixed number of partitions (= nthreads) with each partition of size 1,
  // i.e., one thread per partition and this thread is the master thread
  Kokkos::OpenMP::partition_master(f, num_threads, 1);
#else
  f(0, 1);
#endif
Kokkos::fence();
complete_cnt = 0;
  }
}
    if (num_streams > 1) {
      for (auto n = 0; n < num_streams; n++) {
        SpaceInstance<DevExecSpace>::destroy(exec_spaces[n]);
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
