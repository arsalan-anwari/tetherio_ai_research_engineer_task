#pragma once

#ifdef TARGET_VULKAN_NATIVE

#include <iostream>
#include <expected>
#include <ranges>

#include <vulkan/vulkan.hpp>

#include "../types.hpp"

namespace tether_io{

    template<> struct device_buffer<device_driver::vulkan_native>{
        VkBuffer buff_handle{}; 
        VkDeviceMemory memory_handle{}; 
        VkFence lock{};
        usize size_bytes{};
    };

    template<> struct kernel<device_driver::vulkan_native>{};

    struct vulkan_native_driver {
        
        auto init(api_version<u32> vk_version, cstr app_name) -> std::expected<void, device_error> {
            VkApplicationInfo app_cfg{VK_STRUCTURE_TYPE_APPLICATION_INFO}; 
            app_cfg.apiVersion = VK_MAKE_API_VERSION(vk_version.version, vk_version.major, vk_version.minor, vk_version.patch);
            app_cfg.pApplicationName = app_name.data();

            VkInstanceCreateInfo instance_cfg{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; 
            instance_cfg.pApplicationInfo=&app_cfg;

            // Create instance with settings
            if (vkCreateInstance(&instance_cfg, nullptr, &instance) != VK_SUCCESS) { 
                //std::cout << "vkCreateInstance" << std::endl;
                return std::unexpected { device_error::could_not_create_instance };
            }

            // Find all available devices
            uint32_t device_count=0; 
            vkEnumeratePhysicalDevices(instance, &device_count, nullptr); 
            if (!device_count){ 
                //std::cout << "vkEnumeratePhysicalDevices" << std::endl;
                return std::unexpected { device_error::no_available_devices };
            }
            
            // Store device pointer to all available devices. 
            devices.resize(device_count);
            vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

            return {};
        }

        auto set_device(device_select preferred_type) -> std::expected<void, device_error>{
            
            switch (preferred_type){
                case device_select::first_compute_capable : {
                    if (!find_first_computable_deivce()){ 
                        //std::cout << "!find_first_computable_deivce()" << std::endl;
                        return std::unexpected { device_error::no_available_devices };
                    }

                    if(!create_device()){
                        //std::cout << "!create_device()" << std::endl;
                        return std::unexpected { device_error::could_not_create_selected_device };
                    }

                    break;
                }
                default: { return std::unexpected { device_error::no_available_devices }; }
            };

            return {};

        };

        auto allocate(
            usize size_bytes, alloc_method method
        ) -> std::expected<device_buffer<device_driver::vulkan_native>, device_error> {
            
            device_buffer<device_driver::vulkan_native> buff;
            buff.size_bytes = size_bytes;

            switch(method){
                case alloc_method::base : {
                    if(!create_buffer_default(buff)){
                        //std::cout << "!create_buffer_default()" << std::endl;
                        return std::unexpected { device_error::could_not_create_buffer };
                    }
                    // if(!create_buffer_lock(buff)){
                    //     std::cout << "!create_buffer_lock()" << std::endl;
                    //     return std::unexpected { device_error::could_not_create_buffer };
                    // }
                    break;
                }
                default : { return std::unexpected{ device_error::alloc_failed }; }

            }
            
            return buff;
        };
        
        template<typename T>
        auto upload(
            device_buffer<device_driver::vulkan_native>& dest, 
            std::span<T> src, 
            upload_method method
        ) -> std::expected<void, device_error> {
            
            switch(method){
                case upload_method::sync : {
                    if(!upload_buffer_sync(dest, src)){
                        return std::unexpected{ device_error::upload_failed };
                    }
                    break;
                }
                default: {return std::unexpected{ device_error::upload_failed }; }

            }

            return {};

        };
        
        template<typename T>
        auto download(
            std::span<T> dest,
            device_buffer<device_driver::vulkan_native>& src, 
            download_method method
        ) -> std::expected<void, device_error> {
            
            switch(method){
                case download_method::sync: {
                    if(!download_buffer_sync(dest, src)){
                        return std::unexpected{ device_error::download_failed };
                    }
                    break;
                }
                default: { return std::unexpected{ device_error::download_failed }; }
            };

            return {};

        }

        auto register_kernel(
            kernel_config krnl_opts, std::vector<device_buffer<device_driver::vulkan_native>>& buffers
        ) -> std::expected<kernel<device_driver::vulkan_native>, device_error> {
            kernel<device_driver::vulkan_native> krnl;


            return krnl;
        };

        void signal(device_buffer<device_driver::vulkan_native>& buff){
            vkResetFences(device_handle, 1, &buff.lock);

            VkSubmitInfo si;
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 0;

            vkQueueSubmit(queue_handle, 1, &si, buff.lock);
        }

        void kill(device_buffer<device_driver::vulkan_native>& buff){
            vkDeviceWaitIdle(device_handle);
            vkDestroyDescriptorPool(device_handle, dpool, nullptr);
            vkDestroyPipeline(device_handle, pipeline, nullptr);
            vkDestroyPipelineLayout(device_handle, pipeLayout, nullptr);
            vkDestroyDescriptorSetLayout(device_handle, dsetLayout, nullptr);
            vkDestroyBuffer(device_handle, buff.buff_handle, nullptr); vkFreeMemory(device_handle, buff.memory_handle, nullptr);
            vkDestroyCommandPool(device_handle, cpool, nullptr);
            vkDestroyDevice(device_handle, nullptr);
            vkDestroyInstance(instance, nullptr);
        }

    private:
        VkDescriptorSetLayout dsetLayout{};
        VkPipelineLayout pipeLayout{};
        VkPipeline pipeline{};
        VkDescriptorPool dpool{};
        VkCommandPool cpool{};

        // App context
        VkInstance instance{};
        
        // Device
        std::vector<VkPhysicalDevice> devices;
        VkPhysicalDevice device{};
        VkDevice device_handle{};

        // Queues
        u32 queue_family = 0;
        VkQueue queue_handle{};

        auto find_first_computable_deivce() -> bool {
            //std::cout << "devices.size() = " << devices.size() << std::endl;
            for (auto dev : devices){
                // Get list of all available queues of device
                u32 queue_family_count=0; 
                vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, nullptr);
                
                //std::cout << "queue_family_count = " << queue_family_count << std::endl;

                // Get properties of these queues.
                std::vector<VkQueueFamilyProperties> queue_family_props(queue_family_count); 
                vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, queue_family_props.data());
                
                // Loop through all properties and look if the compute flag is set
                for(u32 i = 0; i < queue_family_count; ++i){
                    //std::cout << "queue_family_props[" << i << "].queueFlags = " << queue_family_props[i].queueFlags << std::endl;
                    if(queue_family_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT){
                        //std::cout << "queue_family_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT" << std::endl;
                        device = dev;
                        queue_family = i;
                        break;
                    }
                }
                
                if (device){
                    //std::cout << "if (device)" << std::endl;
                    return true;
                };
            }
            return false;
        };

        auto find_memory_type_index(u32 type_bits, VkMemoryPropertyFlags req) -> std::expected<u32, device_error> {
             VkPhysicalDeviceMemoryProperties mp{}; 
             vkGetPhysicalDeviceMemoryProperties(device, &mp);
             for(u32 i = 0; i < mp.memoryTypeCount; ++i){
                if (
                    (type_bits & (1u << i)) && 
                    (mp.memoryTypes[i].propertyFlags & req) == req
                ) { return i; }
             }

             return std::unexpected { device_error::could_not_create_buffer };
        }

        auto create_device() -> bool { 

            f32 queue_priority = 1.0f;

            // Configure queue settings
            VkDeviceQueueCreateInfo queue_cfg{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; 
            queue_cfg.queueFamilyIndex=queue_family; 
            queue_cfg.queueCount=1; 
            queue_cfg.pQueuePriorities=&queue_priority;

            // Configure device settings
            VkDeviceCreateInfo device_cfg{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; 
            device_cfg.queueCreateInfoCount=1; 
            device_cfg.pQueueCreateInfos=&queue_cfg;

            // Create device and assign to handle based on device settings
            if (vkCreateDevice(device, &device_cfg, nullptr, &device_handle) != VK_SUCCESS){ 
                //std::cout << "!vkCreateDevice()" << std::endl;
                return false;
            }

            // Create queue and assign to handle based on queue settings
            vkGetDeviceQueue(device_handle, queue_family, 0, &queue_handle);

            return true;
        }

        auto create_buffer_default(device_buffer<device_driver::vulkan_native>& buff) -> bool {
            //std::cout << "create_buffer_default" << std::endl;

            // Specify settings of the buffer to be created and shared
            VkBufferCreateInfo buffer_cfg{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            //std::cout << "buff.size_bytes = " << buff.size_bytes << std::endl;
            buffer_cfg.size = buff.size_bytes; 
            buffer_cfg.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; 
            buffer_cfg.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            
            // Create buffer on active devices
            if (vkCreateBuffer(device_handle, &buffer_cfg, nullptr, &buff.buff_handle) != VK_SUCCESS){
                //std::cout << "!vkCreateBuffer()" << std::endl;
                return false;
            } 
           
            // Specify how to configure memory and how to allocate
            VkMemoryRequirements memory_cfg; 
            VkMemoryAllocateInfo alloc_cfg{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};

            vkGetBufferMemoryRequirements(device_handle, buff.buff_handle, &memory_cfg);
            //std::cout << "!vkGetBufferMemoryRequirements()" << std::endl;

            alloc_cfg.allocationSize = memory_cfg.size;
            auto memory_type_idx = find_memory_type_index(
                memory_cfg.memoryTypeBits, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            );
            //std::cout << "find_memory_type_index()" << std::endl;

            if (!memory_type_idx.has_value()) {
                //std::cout << "!memory_type_idx.has_value()" << std::endl;
                return false;
            }

            //std::cout << "memory_type_idx.value() = " << memory_type_idx.value() << std::endl;

            alloc_cfg.memoryTypeIndex = memory_type_idx.value();

            // Allocate memmory and assign to buffer memory handle
            if (vkAllocateMemory(device_handle, &alloc_cfg, nullptr, &buff.memory_handle) != VK_SUCCESS){
                //std::cout << "!vkAllocateMemory()" << std::endl;
                return false;
            } 

            // Assign memory handle to buffer handle
            if (vkBindBufferMemory(device_handle, buff.buff_handle, buff.memory_handle, 0) != VK_SUCCESS ){
                //std::cout << "!vkBindBufferMemory()" << std::endl;
                return false;
            }

            return true;
        }
        
        auto create_buffer_lock(device_buffer<device_driver::vulkan_native>& buff) -> bool {
            VkFenceCreateInfo fence_cfg{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fence_cfg.flags = 0; //unsignaled

            if (vkCreateFence(device_handle, &fence_cfg, nullptr, &buff.lock) != VK_SUCCESS)
                return false;

            return true;
        }

        template<typename T>
        auto upload_buffer_sync(
            device_buffer<device_driver::vulkan_native>& dest, 
            std::span<T> src
        ) -> bool {

            if (src.size_bytes() > dest.size_bytes)
                return false;

            void* upload_handle = nullptr;
            
            // Bind device buffer and map to upload handle
            vkMapMemory(device_handle, dest.memory_handle, 0, src.size_bytes(), 0, &upload_handle);
            
            // Copy host buffer to device buffer using upload handle
            std::memcpy(upload_handle, src.data(), src.size_bytes());

            // Unbind device buffer
            vkUnmapMemory(device_handle, dest.memory_handle);

            return true;
        };
        
        template<typename T>
        auto download_buffer_sync(
            std::span<T> dest,
            device_buffer<device_driver::vulkan_native>& src
        ) -> bool {
            if (dest.size_bytes() > src.size_bytes)
                return false;

            // // Wait for kernel to finish using the data
            // vkWaitForFences(device_handle, 1, &src.lock, VK_TRUE, 1'000'000'000ull);

            void* download_handle = nullptr;
            
            // Bind device buffer and map to upload handle
            vkMapMemory(device_handle, src.memory_handle, 0, src.size_bytes, 0, &download_handle); 
            
            // Copy device buffer to host buffer using download handle
            std::memcpy(dest.data(), download_handle, dest.size_bytes()); 
            
            // Unbind device buffer
            vkUnmapMemory(device_handle, src.memory_handle);

            return true;
        }

    }; // vulkan_native_driver


    template<> struct device_driver_impl<device_driver::vulkan_native>{
        using type = vulkan_native_driver;
    }; // device_driver_impl<device_driver::vulkan_native>
    
}; // nameapce tether_io

#endif // TARGET_VULKAN_NATIVE