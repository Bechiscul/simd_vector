cmake_minimum_required(VERSION 3.19)

add_library(vector INTERFACE)
target_sources(vector PUBLIC vector.hpp)
target_include_directories(vector INTERFACE ..)
target_compile_features(vector INTERFACE cxx_std_20)