
add_executable(hello
  hello.cpp
)

target_link_libraries(hello PRIVATE Kokkos::kokkos)

add_executable(host_pinned_memory
  host_pinned_memory/main.cpp
)

target_link_libraries(host_pinned_memory PRIVATE Kokkos::kokkos)
