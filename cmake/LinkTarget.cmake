if(NOT DEFINED TETHER_IO_TARGETS)
    set(TETHER_IO_TARGETS app)
endif()

foreach(_target IN LISTS TETHER_IO_TARGETS)
    if(TARGET_VULKAN_NATIVE)
        target_link_libraries(${_target}
            PRIVATE
                Vulkan::Vulkan
                unofficial::shaderc::shaderc
        )

        target_compile_definitions(${_target} PRIVATE USE_SHADERC=1)
    endif()

    target_link_libraries(${_target} PRIVATE nlohmann_json::nlohmann_json fmt::fmt)

    target_include_directories(${_target}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )

    target_compile_definitions(${_target} PRIVATE RESOURCE_DIR=\"${RESOURCE_DIR}\")
endforeach()

if(ENABLE_LLAMA_CPP)
    foreach(_llama_target IN LISTS TETHER_IO_LLAMA_TARGETS)
        if(TARGET ${_llama_target})
            target_link_libraries(${_llama_target}
                PRIVATE
                    llama
            )

            target_include_directories(${_llama_target}
                PRIVATE
                    ${CMAKE_SOURCE_DIR}/third_party/llama-cpp/include
            )
        endif()
    endforeach()
endif()
