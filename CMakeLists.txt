cmake_minimum_required(VERSION 3.9)

project(cks)

enable_language(CUDA)

include_directories(src)

add_subdirectory(src bin)