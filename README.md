# tetherio_ai_research_engineer_task

This repository is now wired to build a small C++ harness that links against **llama.cpp** and **Vulkan** using **CMake**, **Ninja**, and **vcpkg**. The goal is to provide a ready-to-extend skeleton for future targets (desktop, mobile, different GPU backends, etc.).

## Prerequisites

- CMake ≥ 3.25
- Ninja (make sure it is on your `PATH`)
- A recent C++20 compiler (MSVC, clang, or gcc)
- `vcpkg` cloned locally. Set the `VCPKG_ROOT` environment variable to the clone location so the presets can pick it up:

  ```powershell
  setx VCPKG_ROOT "C:\path\to\vcpkg"
  ```

  ```bash
  export VCPKG_ROOT=$HOME/vcpkg
  ```

## vcpkg manifest & baseline

The repository contains a `vcpkg.json` manifest with the required dependencies (`llama.cpp` with the Vulkan feature and the Vulkan SDK loader/headers). Update the `builtin-baseline` once you have `vcpkg` available:

```powershell
cd $env:VCPKG_ROOT
.\vcpkg.exe x-update-baseline --add-initial-baseline --vcpkg-root "c:\Users\arsalan.anwari_nscal\Workspaces\tether_io\tetherio_ai_research_engineer_task"
```

You can also run the command from any shell; the important part is to write back the resolved commit hash into `vcpkg.json`.

If you need alternative registries or overlays, add them to `vcpkg-configuration.json`.

## Configuring, building, and testing

This project is configured via `CMakePresets.json` and defaults to the Ninja generator plus `vcpkg` toolchain integration.

1. **Configure** (example: Windows x64 release):

   ```powershell
   cmake --preset windows-x64-release
   ```

2. **Build**:

   ```powershell
   cmake --build --preset windows-x64-release
   ```

3. **Test** (runs the dummy self-test that exercises llama.cpp & Vulkan symbols):

   ```powershell
   ctest --preset windows-x64-release
   ```

The resulting executable is placed in `build/windows-x64-release/llama_demo`.

## Switching targets

- To switch to a different vcpkg triplet, duplicate one of the presets in `CMakePresets.json` and override `VCPKG_TARGET_TRIPLET`.
- You can set additional cache variables (e.g., `LLAMA_CUBLAS=ON`) inside the preset for specialized builds.
- `vcpkg` manifest features can be enabled by editing `vcpkg.json` (e.g., `llama.cpp[cublas]` for CUDA).

When cross-compiling, remember to configure the appropriate host tools triplet (`VCPKG_HOST_TRIPLET`) and point to any overlay triplets if required.

## Project layout

- `src/main.cpp`: Minimal executable that initialises the llama backend and prints Vulkan header information.
- `CMakeLists.txt`: Targets, dependencies, and test wiring.
- `CMakePresets.json`: Ready-made Ninja presets for Windows & Linux.
- `vcpkg.json`: Manifest dependencies for llama.cpp + Vulkan.

Extend `src/main.cpp` or add additional libraries/executables as your research experiments grow.
