find_package(fmt REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

if(ENABLE_LLAMA_CPP)
    find_package(llama CONFIG REQUIRED)
endif()

if(TARGET_VULKAN_NATIVE)
    find_package(Vulkan REQUIRED)
    find_package(unofficial-shaderc CONFIG REQUIRED)
endif()
