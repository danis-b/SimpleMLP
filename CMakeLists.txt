cmake_minimum_required(VERSION 3.10)
project(MyProject CXX)

# Set compiler
set(CMAKE_CXX_COMPILER clang++)

# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3 -fsanitize=address,undefined -Wall -Wextra -Werror")

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add the MLP static library
add_library(mlp_lib STATIC mlp.cpp)

# Source files for the executable
set(EXEC_SOURCES main.cpp)

# Add the executable
add_executable(my_code.x ${EXEC_SOURCES})
target_link_libraries(my_code.x mlp_lib)

# Add the pybind11 module
pybind11_add_module(mlp_module mlp_module.cpp)
target_link_libraries(mlp_module PRIVATE mlp_lib)