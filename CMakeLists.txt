cmake_minimum_required(VERSION 3.19)

project(Vector LANGUAGES CXX)

add_subdirectory(vector)

find_package(benchmark CONFIG REQUIRED)

if (benchmark_FOUND)
    add_executable(Bench)

    target_sources(Bench PRIVATE bench/bench.cpp)
    target_link_libraries(Bench PRIVATE vector)
    target_link_libraries(Bench PRIVATE benchmark::benchmark)
endif()