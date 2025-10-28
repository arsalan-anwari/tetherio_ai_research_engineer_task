if(TARGET_WIN32)
    target_link_libraries(app
        PRIVATE
            llama
            Vulkan::Vulkan
    )
endif()