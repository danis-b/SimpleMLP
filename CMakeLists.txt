cmake_minimum_required(VERSION 3.10)
project(MyProject CXX)

# Set compiler
set(CMAKE_CXX_COMPILER clang++)

# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3 -fsanitize=address,undefined -Wall -Wextra -Werror")

# Source files
set(SOURCES my_code.cpp)

# Add the executable
add_executable(my_code.x ${SOURCES})

# Include functions.h dependency tracking
# set_source_files_properties(my_code.cpp PROPERTIES OBJECT_DEPENDS functions.h)