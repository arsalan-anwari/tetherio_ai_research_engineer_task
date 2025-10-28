find_package(fmt REQUIRED)

if(TARGET_WIN32)
    find_package(llama REQUIRED)
    find_package(Vulkan REQUIRED)
endif()