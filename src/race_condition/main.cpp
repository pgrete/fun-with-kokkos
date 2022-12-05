#include <Kokkos_Core.hpp>
#include <iostream>

using std::cout;

auto main(int argc, char *argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {

    size_t num_b = 26 * 4;
    Kokkos::View<bool *> nonzero_flags("nonzero_flags", num_b);

    size_t num_idx = 16;

    Kokkos::View<double **> work("work", num_b, num_idx);

    Kokkos::parallel_for(
        "set flags", Kokkos::TeamPolicy<>(num_b, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
          const int b = member.league_rank();

          nonzero_flags(b) = false;

          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, num_idx),
                               [&](const int idx) {
                                 //  const Real &val = bnd_info(b).var(t, u, v, k, j, i);
                                 work(b, idx) = 1.0;
                                 if (std::abs(work(b, idx)) >= 0.0) {
                                   nonzero_flags(b) = true;
                                 }
                               });
        });

    auto nonzero_flags_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nonzero_flags);

    for (auto b = 0; b < num_b; b++) {
      if (!nonzero_flags_h(b)) {
        std::cerr << "HEEEEEEEEEEEEEELP!!!\n";
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
