if (DEFINED VCPKG_MANIFEST_FEATURES)

  # For now only one setting is used, but in future multiple can be selected by the user.
  if (VCPKG_MANIFEST_FEATURES STREQUAL "win32")
      set(TARGET_WIN32 ON CACHE BOOL "Windows desktop x64 | x86" FORCE)
      add_compile_definitions(TARGET_WIN32)
  endif()

else()

  # If no feature flags are used used the default of opengl_window_context
  message(ERROR "VCPKG_MANIFEST_FEATURES is not set in CMakePresets.json!")

endif()