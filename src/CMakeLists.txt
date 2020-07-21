
add_executable(hello
  hello.cpp
)

target_link_libraries(hello PRIVATE Kokkos::kokkos)

add_executable(threads_and_streams
  threads_and_streams/main.cpp
)

target_link_libraries(threads_and_streams PRIVATE Kokkos::kokkos)

