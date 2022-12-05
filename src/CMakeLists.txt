
add_executable(hello
  hello.cpp
)

target_link_libraries(hello PRIVATE Kokkos::kokkos)

add_executable(host_pinned_memory
  host_pinned_memory/main.cpp
)

add_executable(race_condition
  race_condition/main.cpp
)

target_link_libraries(host_pinned_memory PRIVATE Kokkos::kokkos)
target_link_libraries(race_condition PRIVATE Kokkos::kokkos)
