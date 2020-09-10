#include "KokkosExp_MDRangePolicy.hpp"
#include "Kokkos_Core_fwd.hpp"
#include "Kokkos_ExecPolicy.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

constexpr int N = 8;

auto main(int argc, char *argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double[N], Kokkos::CudaSpace> dev_view("dev_view");
    Kokkos::View<double[N], Kokkos::CudaHostPinnedSpace> host_pinned_view("host_pinned_view");

    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(Kokkos::Cuda(), 0, N), KOKKOS_LAMBDA(const int i) {
          dev_view(i) = 1.0;
          host_pinned_view(i) = dev_view(i);
        });
    Kokkos::fence();
    for (int i = 0; i < N; i++)
      std::cout << host_pinned_view(i);
  }
  Kokkos::finalize();

  return 0;
}
