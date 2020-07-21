
cmake_minimum_required(VERSION 3.10)
project(fun-with-kokkos LANGUAGES CXX)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Kokkos recommendatation resulting in not using default GNU extensions
set(CMAKE_CXX_EXTENSIONS OFF)

set(Kokkos_CXX_STANDARD 14 CACHE STRING "CXX standard" FORCE)

if (Kokkos_ENABLE_CUDA)
  set (Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL
       "Enable lambda expressions in CUDA" FORCE)
  # considered experimental for Cuda <11
  set (Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL
       "Enable relaxed constexpressions in CUDA" FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos Kokkos)

add_subdirectory(src)
