#include <Kokkos_Core.hpp>
#include <iostream>

using std::cout;

auto main(int argc, char *argv[]) -> int {
  Kokkos::initialize(argc, argv);
  { Kokkos::print_configuration(cout); }
  Kokkos::finalize();

  return 0;
}
