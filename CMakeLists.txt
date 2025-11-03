cmake_minimum_required(VERSION 3.25.0)

project(tether_io_ai_research_task VERSION 0.1.0 LANGUAGES CXX)

# Define all available fetaures and taregets
option(TARGET_VULKAN_NATIVE "Windows desktop with Vulkan backend" OFF)
option(ENABLE_LLAMA_CPP "Enable LLAMA-CPP Interop" OFF)
add_compile_definitions(RESOURCE_DIR="${RESOURCE_DIR}")

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Search which target is selected
include(cmake/FindTarget.cmake)

# Load settings and dependencies for selected target
include(cmake/LoadTarget.cmake)

# Create executables
add_executable(app src/main.cpp)

set(TETHER_IO_TARGETS app)
set(TETHER_IO_LLAMA_TARGETS)

add_executable(example_binmatmul examples/binmatmull.cpp)
list(APPEND TETHER_IO_TARGETS example_binmatmul)

if(ENABLE_LLAMA_CPP)
    add_executable(example_llama_cpp_interop examples/llama-cpp-interop.cpp)
    list(APPEND TETHER_IO_TARGETS example_llama_cpp_interop)
    list(APPEND TETHER_IO_LLAMA_TARGETS example_llama_cpp_interop)
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(binmatmul_sandbox_tests tests/binmatmul_sandbox_tests.cpp)
    list(APPEND TETHER_IO_TARGETS binmatmul_sandbox_tests)

    add_test(NAME binmatmul_sandbox COMMAND binmatmul_sandbox_tests)
endif()

# Link everything needed by targets
include(cmake/LinkTarget.cmake)
