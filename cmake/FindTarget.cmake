if (DEFINED VCPKG_MANIFEST_FEATURES)

  # For now only one setting is used, but in future multiple can be selected by the user.
  if (VCPKG_MANIFEST_FEATURES STREQUAL "windows-x64-vulkan")
      set(TARGET_VULKAN_NATIVE ON CACHE BOOL "Windows desktop with Vulkan backend" FORCE)
      add_compile_definitions(TARGET_VULKAN_NATIVE)

      if(ENABLE_LLAMA_CPP)
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
      endif()

  endif()

else()

  # If no feature flags are used used the default of opengl_window_context
  message(ERROR "VCPKG_MANIFEST_FEATURES is not set in CMakePresets.json!")

endif()

if(ENABLE_LLAMA_CPP)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

  add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/llama-cpp)

  add_compile_definitions(ENABLE_LLAMA_CPP)

endif()