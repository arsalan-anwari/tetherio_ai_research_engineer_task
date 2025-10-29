if(TARGET_WIN32)
    target_link_libraries(app
        PRIVATE
            llama
            Vulkan::Vulkan
            unofficial::shaderc::shaderc
    )

    target_compile_definitions(app PRIVATE USE_SHADERC=1)

endif()