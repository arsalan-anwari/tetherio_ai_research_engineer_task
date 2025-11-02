if(TARGET_VULKAN_NATIVE)
    target_link_libraries(app
        PRIVATE
            Vulkan::Vulkan
            unofficial::shaderc::shaderc
    )

    target_compile_definitions(app PRIVATE USE_SHADERC=1)
endif()

if(ENABLE_LLAMA_CPP)
    target_link_libraries(app
        PRIVATE
            llama
    )

endif()

# Univeral libraries

target_link_libraries(app PRIVATE nlohmann_json::nlohmann_json fmt::fmt)

target_include_directories(app
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

if(ENABLE_LLAMA_CPP)

target_include_directories(app
    PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/llama-cpp/include
)

endif()

target_compile_definitions(app PRIVATE RESOURCE_DIR=\"${RESOURCE_DIR}\")