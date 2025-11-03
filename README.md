# Tether IO AI Research Harness

## Overview
This repository contains a C++20 research harness for experimenting with low-precision matrix multiplication pipelines. It provides a Vulkan-backed compute context, a CPU reference implementation, and a sandbox runner that validates 1-bit (binary) GEMM results across data domains and matrix shapes. The same building blocks can be promoted into larger inference systems; an example integration with `llama.cpp` is included to show how the Vulkan backend can replace the default GGML matmul path.

The codebase is organised around:
- **Config-driven kernels** sourced from `res/kernels`, described in `index.json`, and compiled to SPIR-V via `shaderc`.
- **Device drivers** (`compute_context`) that manage Vulkan resources and expose upload/download helpers.
- **Algorithms** that wrap individual kernels for CPU or GPU execution.
- **Sandbox harnesses** that generate input tensors, launch device kernels, and check results against the CPU reference implementation.
- **Optional llama.cpp adapter** that wires the Vulkan binary matmul into GGML.

## Highlights
- Binary matrix multiplication pipeline with CPU reference and Vulkan execution path.
- Configurable kernel metadata (`res/settings.json` + `res/kernels/vk/index.json`) that controls recompilation and parameter shapes.
- Regression tests that sweep matrix sizes and data distributions to ensure numerical parity.
- Examples that demonstrate standalone GPU launches and llama.cpp integration.
- CMake presets and vcpkg manifest for reproducible builds.

## Requirements
- CMake **3.25** or newer.
- A C++20 toolchain (Visual Studio 2022, Clang 16+, or GCC 12+).
- [vcpkg](https://github.com/microsoft/vcpkg) cloned locally with `VCPKG_ROOT` pointing at the clone.
- Vulkan SDK 1.3+ (driver + headers). Install from [LunarG](https://vulkan.lunarg.com/) and ensure `VULKAN_SDK` is set.
- GPU with Vulkan compute support (desktop-class dGPU or iGPU).
- (Optional) `git` submodules initialized for `third_party/llama-cpp` if you plan to enable llama integration.

## Quick Start
```powershell
git clone https://github.com/<your-org>/tetherio_ai_research_engineer_task.git
cd tetherio_ai_research_engineer_task
git submodule update --init --recursive

# Make vcpkg discoverable by CMake presets
setx VCPKG_ROOT "C:\path\to\vcpkg"
```

On Unix-like environments the commands are identical except for the `VCPKG_ROOT` export:
```bash
export VCPKG_ROOT=$HOME/vcpkg
```

## Building
The repository ships with CMake presets that target Windows x64 with Vulkan:
```powershell
cmake --preset windows-x64-vulkan
cmake --build --preset windows-x64-vulkan
```

Configuration defaults:
- `BUILD_TESTING=ON`
- `ENABLE_LLAMA_CPP=OFF`
- `RESOURCE_DIR` points at `<repo>/res`
- Manifest feature `windows-x64-vulkan` pulls `fmt`, `nlohmann-json`, `vulkan`, and `shaderc` through vcpkg.

To enable the llama.cpp interoperability example, pass an override at configure time:
```powershell
cmake --preset windows-x64-vulkan -DENABLE_LLAMA_CPP=ON
cmake --build --preset windows-x64-vulkan
```
Make sure the `third_party/llama-cpp` submodule is populated before toggling the flag.

Advanced users can duplicate `windows-x64-vulkan` in `CMakePresets.json` to target different triplets, toolchains, or build types.

## Running the Sandbox and Examples
After building, binaries reside under `build/windows-x64-vulkan/Release` (or the preset's output directory):
```powershell
# Run the default sandbox harness
build/windows-x64-vulkan/Release/app.exe

# Standalone GPU example with larger matrices
build/windows-x64-vulkan/Release/example_binmatmul.exe

# llama.cpp integration (requires ENABLE_LLAMA_CPP=ON and model assets)
build/windows-x64-vulkan/Release/example_llama_cpp_interop.exe
```

All executables rely on `RESOURCE_DIR` being set by CMake; they expect kernels and configuration files under `res/`.

## Tests
The sandbox regression test exercises a sweep of matrix sizes and value domains:
```powershell
ctest --preset windows-x64-vulkan
```
or execute the binary directly:
```powershell
build/windows-x64-vulkan/Release/binmatmul_sandbox_tests.exe
```

CI or local workflows can consume the test logs to detect numerical regressions when kernels or kernel metadata change.

## Repository Layout
- `src/main.cpp` - Entry point that runs the binary matmul sandbox.
- `include/tether_io/` - Core headers for types, config parsing, compute contexts, algorithms, and sandbox orchestration.
- `examples/binmatmull.cpp` - Verbose walkthrough of GPU binary matmul, showcasing manual buffer management.
- `examples/llama-cpp-interop.cpp` - Registers the Vulkan backend with llama.cpp (guarded by `ENABLE_LLAMA_CPP`).
- `tests/binmatmul_sandbox_tests.cpp` - Regression sweep verifying GPU vs. CPU parity.
- `res/settings.json` - Global configuration that selects the kernel family and output format.
- `res/kernels/vk/` - GLSL compute shaders (`*.comp.glsl`) and their compiled SPIR-V binaries (`bin/*.spv`) referenced by `index.json`.
- `res/models/` - Placeholder directory for GGML/GGUF assets used by the llama example.
- `res/docs/` - Background reading and research notes.
- `cmake/` - Helper scripts that discover targets, load dependencies, and wire link options.
- `third_party/llama-cpp/` - Git submodule for llama.cpp when interoperability is enabled.

## Configuration and Kernel Assets
- `res/settings.json` selects the active kernel backend and the format compiled by the toolchain.
- `res/kernels/vk/index.json` enumerates available compute shaders. Each entry contains the GLSL source, expected parameter block size, versioning information, and a `recompile` flag that the Vulkan driver honours when loading SPIR-V.
- `RESOURCE_DIR` is injected at compile time by CMake so binaries can resolve configuration and shader files without relying on the working directory.
- If you update any GLSL shader, rebuild to regenerate the SPIR-V artifacts under `res/kernels/vk/bin`. `shaderc` (via vcpkg) is used at build time to perform compilation.

## Troubleshooting
- Verify `cmake --preset windows-x64-vulkan` reports the correct `VCPKG_ROOT` and finds `vulkan` + `shaderc`. Missing packages usually indicate an unset environment variable or a stale vcpkg baseline.
- Ensure the Vulkan SDK is installed and that `vulkaninfo` enumerates a device capable of compute queues.
- When enabling llama.cpp, confirm the GGML model file (`res/models/tiny-llama.gguf` or similar) exists before launching the interoperability example.

## Next Steps
- Extend `sandbox<sandbox_algorithm::binmatmul, ...>` with additional algorithms (e.g., fill or multiply) by following the pattern in `include/tether_io/sanbox.hpp`.
- Add new kernels to `res/kernels/<backend>` and update `index.json` to benchmark alternative implementations.
- Integrate the Vulkan backend into larger inference workloads by adapting the adapter in `include/tether_io/integration`.
