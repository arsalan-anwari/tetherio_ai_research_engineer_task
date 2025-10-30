if(TARGET_VULKAN_NATIVE)
    target_link_libraries(app
        PRIVATE
            Vulkan::Vulkan
            unofficial::shaderc::shaderc
    )

    target_compile_definitions(app PRIVATE USE_SHADERC=1)
endif()

if(TARGET_LLAMA_VULKAN)
    target_link_libraries(app
        PRIVATE
            llama
            Vulkan::Vulkan
            unofficial::shaderc::shaderc
    )

    target_compile_definitions(app PRIVATE USE_SHADERC=1)
endif()

# Univeral libraries

target_link_libraries(app PRIVATE nlohmann_json::nlohmann_json fmt::fmt)

target_include_directories(app
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(app PRIVATE RESOURCE_DIR=\"${RESOURCE_DIR}\")