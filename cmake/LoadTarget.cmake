find_package(fmt REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

if(TARGET_VULKAN_NATIVE)
    find_package(Vulkan REQUIRED)
    find_package(unofficial-shaderc CONFIG REQUIRED)
endif()

if(TARGET_LLAMA_VULKAN)
    find_package(llama REQUIRED)
    find_package(Vulkan REQUIRED)
    find_package(unofficial-shaderc CONFIG REQUIRED)
endif()
