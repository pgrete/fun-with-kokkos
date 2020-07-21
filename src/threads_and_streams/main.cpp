#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "Kokkos_Layout.hpp"
#include "Kokkos_Parallel.hpp"
#include "space_instances.hpp"

using std::cout;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace = Kokkos::HostSpace;
using DevExecSpace = Kokkos::DefaultExecutionSpace;
using ScratchMemSpace = DevExecSpace::scratch_memory_space;

using View4D = Kokkos::View<double ****, Kokkos::LayoutRight, DevMemSpace>;
using ScratchPad2D = Kokkos::View<double **, Kokkos::LayoutRight, ScratchMemSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using team_policy = Kokkos::TeamPolicy<>;
using team_mbr_t = Kokkos::TeamPolicy<>::member_type;

auto main(int argc, char *argv[]) -> int {
  int num_threads = 1;         // number of host threads to use
  int num_streams = 1;         // number of (Cuda) streams to use
  int num_cycles = 100;        // number of cycles to run
  int num_stages = 2;          // number of stages per cycles
  int size_mesh = 256;         // size of the entire mesh
  int size_block = 128;        // size of blocks the create the mesh
  const int nghost = 2;        // number of ghost zones
  const int num_vars = 7;      // number of variables to work on
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM

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

  const int num_blocks = // total number of blocks
      (size_mesh / size_block) * (size_mesh / size_block) * (size_mesh / size_block);

  Kokkos::initialize(argc, argv);
  {
    // Allocate data
    std::vector<View4D> views;
    for (auto i = 0; i < num_blocks; i++) {
      views.push_back(View4D("view_" + std::to_string(i), num_vars,
                             size_block + 2 * nghost, size_block + 2 * nghost,
                             size_block + 2 * nghost));
    }

    std::vector<DevExecSpace> exec_spaces;

    if (num_streams > 1) {
      for (auto n = 0; n < num_streams; n++) {
        exec_spaces.push_back(SpaceInstance<DevExecSpace>::create());
      }
    } else {
      exec_spaces.push_back(DevExecSpace());
    }

    auto f = [&](int thread_id, int num_threads) {
      for (auto i = 0; i < num_blocks; ++i) {
        // Workaround to ensure that the same thread works on the same MeshBlocks within
        // a cycle. Trying to circumvent problem in setting scratch pad memory, which is
        // not thread safe.
        if (i % num_threads != thread_id) {
          continue;
        }
        cout << "Thread " << thread_id << " uses stream " << i % num_streams
             << " for: block " << i << std::endl
             << std::flush;

        // in principle here's a call to a task list with multiple tasks for each block
        // now it's just a single simple kernel
        size_t scratch_size_in_bytes = ScratchPad2D::shmem_size(num_vars, size_block);
        auto this_block = views[i];
        Kokkos::parallel_for(
            "simple kernel",
            team_policy(exec_spaces[i % num_streams], size_block * size_block,
                        Kokkos::AUTO)
                .set_scratch_size(scratch_level,
                                  Kokkos::PerTeam(2 * scratch_size_in_bytes)),
            KOKKOS_LAMBDA(team_mbr_t team_member) {
              const int k = team_member.league_rank() / size_block + nghost;
              const int j = team_member.league_rank() % size_block + nghost;

              ScratchPad2D scratch_l(team_member.team_scratch(scratch_level), num_vars,
                                     size_block);
              ScratchPad2D scratch_r(team_member.team_scratch(scratch_level), num_vars,
                                     size_block);

              Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(team_member, nghost, size_block + nghost),
                  [&](const int i) {
                    // do sth with the vars
                    for (auto n = 0; n < num_vars; n++) {
                      scratch_l(n, i) = this_block(n, k, j, i - 1);
                      scratch_r(n, i) = this_block(n, k, j, i + 1);
                    }
                  });
              // Sync all threads in the team so that scratch memory is consistent
              team_member.team_barrier();
              Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(team_member, nghost, size_block + nghost),
                  [&](const int i) {
                    // do sth with the vars
                    for (auto n = 0; n < num_vars; n++) {
                      this_block(n, k, j, i) =
                          scratch_l(n, i) + scratch_r(n, i) + scratch_l(0, i);
                    }
                  });
            });
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
