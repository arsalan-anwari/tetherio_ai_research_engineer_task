#pragma once

#ifdef TARGET_VULKAN_NATIVE

#include <iostream>

#include <vulkan/vulkan.hpp>

#include "../types.hpp"

namespace tether_io{

    template<> struct device_buffer<device_driver::vulkan_native>{
        VkBuffer buff_handle; VkDeviceMemory memory_handle; usize size;
    };

    template<> struct kernel<device_driver::vulkan_native>{};

    struct vulkan_native_driver {
        void test(){ std::cout << "TARGET_VULKAN_NATIVE" << std::endl; };
    };

    template<> struct device_driver_impl<device_driver::vulkan_native>{
        using type = vulkan_native_driver;
    };
    
}; // nameapce tether_io

#endif // TARGET_VULKAN_NATIVE