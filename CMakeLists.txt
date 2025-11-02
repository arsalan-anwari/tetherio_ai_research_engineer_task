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

# Create new executable
add_executable(app src/main.cpp)

# Link everything needed by target to executable
include(cmake/LinkTarget.cmake)