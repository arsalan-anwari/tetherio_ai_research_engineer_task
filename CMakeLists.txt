cmake_minimum_required(VERSION 3.25.0)

project(tether_io_ai_research_task VERSION 0.1.0 LANGUAGES CXX)

# Define all available fetaures and taregets
option(TARGET_WIN32 "Windows desktop x64 | x86" OFF)

# Search which target is selected
include(cmake/FindTarget.cmake)

# Load settings and dependencies for selected target
include(cmake/LoadTarget.cmake)

# Create new executable
add_executable(app src/main.cpp)

# Link everything needed by target to executable
include(cmake/LinkTarget.cmake)