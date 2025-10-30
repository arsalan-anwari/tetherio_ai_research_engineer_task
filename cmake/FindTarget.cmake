if (DEFINED VCPKG_MANIFEST_FEATURES)

  # For now only one setting is used, but in future multiple can be selected by the user.
  if (VCPKG_MANIFEST_FEATURES STREQUAL "windows-x64-vulkan")
      set(TARGET_VULKAN_NATIVE ON CACHE BOOL "Windows desktop with Vulkan backend" FORCE)
      add_compile_definitions(TARGET_VULKAN_NATIVE)
  endif()

  if (VCPKG_MANIFEST_FEATURES STREQUAL "windows-x64-llama-vulkan")
      set(TARGET_LLAMA_VULKAN ON CACHE BOOL "Windows desktop with llama backend using Vulkan driver" FORCE)
      add_compile_definitions(TARGET_LLAMA_VULKAN)
  endif()

else()

  # If no feature flags are used used the default of opengl_window_context
  message(ERROR "VCPKG_MANIFEST_FEATURES is not set in CMakePresets.json!")

endif()