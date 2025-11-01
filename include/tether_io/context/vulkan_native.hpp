#pragma once

#ifdef TARGET_VULKAN_NATIVE

#include <iostream>
#include <expected>
#include <initializer_list>
#include <fstream>
#include <span>
#include <vector>
#include <array>
#include <limits>
#include <cstring>

#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp>

#include "../types.hpp"

namespace tether_io{

    template<> struct device_buffer<device_driver::vulkan_native>{
        VkBuffer buff_handle{}; 
        VkDeviceMemory memory_handle{}; 
        usize size_bytes{};
    };

    template<> struct kernel<device_driver::vulkan_native>{
        VkFence lock{};
        VkPipeline pipeline{};
        VkPipelineLayout pipeline_layout{};
        VkDescriptorSetLayout descriptor_layout{};
        VkDescriptorPool descriptor_pool{};
        VkDescriptorSet descriptor{};
        VkCommandBuffer command_buffer{};
    };

    struct vulkan_native_driver {
        
        auto init(version<u32> vk_version, cstr app_name) -> std::expected<void, device_error> {
            api_version = vk_version;
            
            VkApplicationInfo app_cfg{VK_STRUCTURE_TYPE_APPLICATION_INFO}; 
            app_cfg.apiVersion = VK_MAKE_API_VERSION(vk_version.variant, vk_version.major, vk_version.minor, vk_version.patch);
            app_cfg.pApplicationName = app_name.data();

            VkInstanceCreateInfo instance_cfg{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; 
            instance_cfg.pApplicationInfo=&app_cfg;

            // Create instance with settings
            if (vkCreateInstance(&instance_cfg, nullptr, &instance) != VK_SUCCESS) { 
                //std::cout << "vkCreateInstance" << std::endl;
                return std::unexpected { device_error::could_not_create_instance };
            }

            // Find all available devices
            u32 device_count=0; 
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
            kernel_config& krnl_opts, 
            vec3<u32> workgroup_size,
            std::initializer_list<device_buffer<device_driver::vulkan_native>> buffers
        ) -> std::expected<kernel<device_driver::vulkan_native>, device_error> {
            kernel<device_driver::vulkan_native> krnl;

            if (!is_valid_workgroup_size(workgroup_size)){
                return std::unexpected{device_error::could_not_register_kernel};
            }

            if (krnl_opts.recompile){
                switch(krnl_opts.format){
                    case kernel_format::glsl : {
                        if (krnl_opts.type_version != api_version){
                            return std::unexpected{device_error::shader_version_or_type_not_supported};
                        }

                        auto shader_bin = compile_glsl_to_spv(krnl_opts);
                        if(!shader_bin.has_value()) return std::unexpected{shader_bin.error()};
                        
                        auto res = register_spv_to_pipeline(krnl_opts, buffers, shader_bin.value(), krnl, workgroup_size);
                        if(!res.has_value()) return std::unexpected{res.error()};

                        break;
                    }
                    default : { return std::unexpected{device_error::could_not_register_kernel}; }
                }
            }

            return krnl;
        };

        template<class_type KernelParams>
        auto launch_kernel(
            kernel<device_driver::vulkan_native>& task, 
            vec3<u32> workgroup_size,
            std::initializer_list<device_buffer<device_driver::vulkan_native>> buffers,
            launch_method method,
            KernelParams kernel_params
        ) -> std::expected<void, device_error> {
            
            
            if (!is_valid_workgroup_size(workgroup_size)){
                return std::unexpected{device_error::could_not_register_kernel};
            }

            switch(method){
                case launch_method::sync: {
                    if(!update_descriptor_sets(task, buffers)){
                        return std::unexpected{device_error::could_not_update_descriptors}; 
                    }

                    if(!dispatch_kernel_to_command_buffer(task, workgroup_size, kernel_params)){
                        return std::unexpected{device_error::could_not_dispatch_kernel_to_command_buffer}; 
                    }

                    break;
                }
                default: { return std::unexpected{device_error::launch_failed}; }
            }
            
            return {};
        };

        auto wait_for_kernel(
            kernel<device_driver::vulkan_native>& task, usize time_out
        ) -> std::expected<void, device_error> {
            if(vkWaitForFences(device_handle, 1, &task.lock, VK_TRUE, time_out) != VK_SUCCESS){
                return std::unexpected{device_error::kernel_timout_reached};
            }
            
            vkDestroyFence(device_handle, task.lock, nullptr);
            task.lock = VK_NULL_HANDLE;

            return {};
        }

        auto destroy_kernel(kernel<device_driver::vulkan_native>& task) -> void {
            if (task.lock != VK_NULL_HANDLE){
                vkDestroyFence(device_handle, task.lock, nullptr);
                task.lock = VK_NULL_HANDLE;
            }

            if (task.descriptor_pool != VK_NULL_HANDLE){
                vkDestroyDescriptorPool(device_handle, task.descriptor_pool, nullptr);
                task.descriptor_pool = VK_NULL_HANDLE;
            }

            if (task.pipeline != VK_NULL_HANDLE){
                vkDestroyPipeline(device_handle, task.pipeline, nullptr);
                task.pipeline = VK_NULL_HANDLE;
            }

            if (task.pipeline_layout != VK_NULL_HANDLE){
                vkDestroyPipelineLayout(device_handle, task.pipeline_layout, nullptr);
                task.pipeline_layout = VK_NULL_HANDLE;
            }

            if (task.descriptor_layout != VK_NULL_HANDLE){
                vkDestroyDescriptorSetLayout(device_handle, task.descriptor_layout, nullptr);
                task.descriptor_layout = VK_NULL_HANDLE;
            }

            if (task.command_buffer != VK_NULL_HANDLE){
                vkFreeCommandBuffers(device_handle, command_pool, 1, &task.command_buffer);
                task.command_buffer = VK_NULL_HANDLE;
            }

            task.descriptor = VK_NULL_HANDLE;
        }

        void exit(std::initializer_list<device_buffer<device_driver::vulkan_native>> buffs){
            vkDeviceWaitIdle(device_handle);
            
            for(auto& buff: buffs){
                vkDestroyBuffer(device_handle, buff.buff_handle, nullptr); 
                vkFreeMemory(device_handle, buff.memory_handle, nullptr);
            }
            
            if (command_pool != VK_NULL_HANDLE){
                vkDestroyCommandPool(device_handle, command_pool, nullptr);
                command_pool = VK_NULL_HANDLE;
            }

            if (device_handle != VK_NULL_HANDLE){
                vkDestroyDevice(device_handle, nullptr);
                device_handle = VK_NULL_HANDLE;
            }

            if (instance != VK_NULL_HANDLE){
                vkDestroyInstance(instance, nullptr);
                instance = VK_NULL_HANDLE;
            }
        };

    private:
        version<u32> api_version;

        // App context
        VkInstance instance{};
        
        // Device
        std::vector<VkPhysicalDevice> devices;
        VkPhysicalDevice device{};
        VkDevice device_handle{};

        // Queues
        u32 queue_family = 0;
        VkQueue queue_handle{};

        // Command pool
        VkCommandPool command_pool{};

        auto is_valid_workgroup_size(vec3<u32> work_group_size) -> bool {
            if (work_group_size.x == 0 || work_group_size.y == 0 || work_group_size.z == 0){
                return false;
            }

            constexpr auto workgroup_size_max = std::numeric_limits<u32>::max();
            if (work_group_size.x > workgroup_size_max || work_group_size.y > workgroup_size_max || work_group_size.z > workgroup_size_max){
                return false;
            }
            return true;
        };

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

            VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            cpci.queueFamilyIndex = queue_family;
            cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

            if (vkCreateCommandPool(device_handle, &cpci, nullptr, &command_pool) != VK_SUCCESS){
                vkDestroyDevice(device_handle, nullptr);
                device_handle = VK_NULL_HANDLE;
                return false;
            }

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

        auto find_shaderrc_vulkan_shader_version(version<u32> ver) -> std::expected<shaderc_env_version, device_error>{
            if (ver.major == 1 && ver.minor == 0) return shaderc_env_version_vulkan_1_0;
            if (ver.major == 1 && ver.minor == 1) return shaderc_env_version_vulkan_1_1;
            if (ver.major == 1 && ver.minor == 2) return shaderc_env_version_vulkan_1_2;
            if (ver.major == 1 && ver.minor == 3) return shaderc_env_version_vulkan_1_3;
            if (ver.major == 1 && ver.minor == 4) return shaderc_env_version_vulkan_1_4;
            return std::unexpected{device_error::shader_version_or_type_not_supported};
        }

        auto compile_glsl_to_spv(kernel_config& krnl_opts) -> std::expected<std::vector<u32>, device_error>{
            
            // Check if compute context version is compatible with shaderrc version
            shaderc::Compiler comp; 
            shaderc::CompileOptions opts; 
            auto shaderrc_version = find_shaderrc_vulkan_shader_version(krnl_opts.type_version);
            if(!shaderrc_version.has_value()) return std::unexpected{shaderrc_version.error()};

            opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderrc_version.value());
            
            // Load glsl shader into raw data
            std::ifstream kernel_file(krnl_opts.path);
            str kernel_raw((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
            kernel_file.close();

            // Compile shader into SpvCompilationResult vector
            auto shader_bin_obj = comp.CompileGlslToSpv(kernel_raw, shaderc_compute_shader, krnl_opts.name.c_str());
            if (shader_bin_obj.GetCompilationStatus() != shaderc_compilation_status_success) {
                return std::unexpected{device_error::could_not_compile_shader};
            }

            // Convert SpvCompilationResult vector into raw u32 vector and store binary in seperate file for later
            std::vector<u32> shader_bin(shader_bin_obj.cbegin(), shader_bin_obj.cend());
            std::ofstream outFile(krnl_opts.path_bin, std::ios::binary);
            outFile.write(reinterpret_cast<const char*>(shader_bin.data()),
                        shader_bin.size() * sizeof(u32));
            outFile.close();

            // Return raw u32 vector representing spv binary as words.
            return shader_bin;
        }

        auto register_spv_to_pipeline(
            kernel_config& krnl_opts,
            std::initializer_list<device_buffer<device_driver::vulkan_native>> buffers,
            std::vector<u32>& spv_binary,
            kernel<device_driver::vulkan_native>& krnl,
            vec3<u32> work_group_size
        ) -> std::expected<void, device_error> {
            // Configure descriptors for each needed buffer for kernel
            std::vector<VkDescriptorSetLayoutBinding> dslb;
            dslb.resize(buffers.size());

            for (i32 i=0; i<dslb.size(); ++i){ 
                dslb[i].binding=i; 
                dslb[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; 
                dslb[i].descriptorCount=1; 
                dslb[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; 
            }

            // Configure descriptor layout
            VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; 
            dlci.bindingCount=dslb.size(); 
            dlci.pBindings=dslb.data();
            
            if (vkCreateDescriptorSetLayout(device_handle, &dlci, nullptr, &krnl.descriptor_layout) != VK_SUCCESS ){
                return std::unexpected{device_error::could_not_update_descriptors};
            }

            // Configure how many bytes to allocate for kernel parameters
            VkPushConstantRange pcr{}; 
            pcr.offset=0; 
            pcr.size=krnl_opts.param_size_bytes; 
            pcr.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;

            // Configure pipeline with buffer and kernel params info
            VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; 
            plci.setLayoutCount=1; 
            plci.pSetLayouts=&krnl.descriptor_layout; 
            plci.pushConstantRangeCount=1; 
            plci.pPushConstantRanges=&pcr;
            
            if(vkCreatePipelineLayout(device_handle, &plci, nullptr, &krnl.pipeline_layout) != VK_SUCCESS ){
                vkDestroyDescriptorSetLayout(device_handle, krnl.descriptor_layout, nullptr);
                krnl.descriptor_layout = VK_NULL_HANDLE;
                return std::unexpected{device_error::could_not_update_pipeline};
            }

            // Configure shader module with spv binary info
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; 
            smci.codeSize=spv_binary.size() * sizeof(u32); 
            smci.pCode=spv_binary.data();

            VkShaderModule sm; 
            if(vkCreateShaderModule(device_handle, &smci, nullptr, &sm) != VK_SUCCESS){
                vkDestroyPipelineLayout(device_handle, krnl.pipeline_layout, nullptr);
                krnl.pipeline_layout = VK_NULL_HANDLE;
                vkDestroyDescriptorSetLayout(device_handle, krnl.descriptor_layout, nullptr);
                krnl.descriptor_layout = VK_NULL_HANDLE;
                return std::unexpected{device_error::could_not_update_kernel_module};
            };

            std::array<u32, 3> work_group_size_values{
                work_group_size.x, work_group_size.y, work_group_size.z
            };

            std::array<VkSpecializationMapEntry, 3> specialization_entries{};
            for (uint32_t idx = 0; idx < specialization_entries.size(); ++idx){
                specialization_entries[idx].constantID = idx;
                specialization_entries[idx].offset = idx * sizeof(uint32_t);
                specialization_entries[idx].size = sizeof(uint32_t);
            }

            VkSpecializationInfo specialization_info{};
            specialization_info.mapEntryCount = static_cast<uint32_t>(specialization_entries.size());
            specialization_info.pMapEntries = specialization_entries.data();
            specialization_info.dataSize = work_group_size_values.size() * sizeof(u32);
            specialization_info.pData = work_group_size_values.data();

            // Stage shader module in pipeline
            VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; 
            ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; 
            ss.module=sm; 
            ss.pName="main"; // entry point name in shader code
            ss.pSpecializationInfo = &specialization_info;
            
            // Create compute pipeline
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; 
            cpci.stage=ss; 
            cpci.layout=krnl.pipeline_layout;
            
            if (vkCreateComputePipelines(device_handle, VK_NULL_HANDLE, 1, &cpci, nullptr, &krnl.pipeline) != VK_SUCCESS){
                vkDestroyShaderModule(device_handle, sm, nullptr);
                vkDestroyPipelineLayout(device_handle, krnl.pipeline_layout, nullptr);
                krnl.pipeline_layout = VK_NULL_HANDLE;
                vkDestroyDescriptorSetLayout(device_handle, krnl.descriptor_layout, nullptr);
                krnl.descriptor_layout = VK_NULL_HANDLE;
                return std::unexpected{device_error::could_not_create_pipeline};
            }
            vkDestroyShaderModule(device_handle, sm, nullptr);

            // Confgure descriptor pool to send buffers
            VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<u32>(buffers.size())};
            VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; 
            dpci.poolSizeCount=1; 
            dpci.pPoolSizes=&dps; 
            dpci.maxSets=1;
            
            if (vkCreateDescriptorPool(device_handle, &dpci, nullptr, &krnl.descriptor_pool) != VK_SUCCESS){
                vkDestroyPipeline(device_handle, krnl.pipeline, nullptr);
                krnl.pipeline = VK_NULL_HANDLE;
                vkDestroyPipelineLayout(device_handle, krnl.pipeline_layout, nullptr);
                krnl.pipeline_layout = VK_NULL_HANDLE;
                vkDestroyDescriptorSetLayout(device_handle, krnl.descriptor_layout, nullptr);
                krnl.descriptor_layout = VK_NULL_HANDLE;
                return std::unexpected{device_error::could_not_update_descriptors};
            }

            VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cbai.commandPool = command_pool;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;

            if (vkAllocateCommandBuffers(device_handle, &cbai, &krnl.command_buffer) != VK_SUCCESS){
                vkDestroyDescriptorPool(device_handle, krnl.descriptor_pool, nullptr);
                krnl.descriptor_pool = VK_NULL_HANDLE;
                vkDestroyPipeline(device_handle, krnl.pipeline, nullptr);
                krnl.pipeline = VK_NULL_HANDLE;
                vkDestroyPipelineLayout(device_handle, krnl.pipeline_layout, nullptr);
                krnl.pipeline_layout = VK_NULL_HANDLE;
                vkDestroyDescriptorSetLayout(device_handle, krnl.descriptor_layout, nullptr);
                krnl.descriptor_layout = VK_NULL_HANDLE;
                return std::unexpected{device_error::could_not_register_kernel};
            }

            return {};


        }

        auto update_descriptor_sets(
            kernel<device_driver::vulkan_native>& task,
            std::initializer_list<device_buffer<device_driver::vulkan_native>> buffs
        ) -> bool{
            if (task.descriptor_pool == VK_NULL_HANDLE || task.descriptor_layout == VK_NULL_HANDLE){
                return false;
            }

            if (vkResetDescriptorPool(device_handle, task.descriptor_pool, 0) != VK_SUCCESS){
                return false;
            }

            // Configure new descriptor layout
            VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; 
            dsai.descriptorPool=task.descriptor_pool; 
            dsai.descriptorSetCount=1; 
            dsai.pSetLayouts=&task.descriptor_layout;

            if(vkAllocateDescriptorSets(device_handle, &dsai, &task.descriptor) != VK_SUCCESS){
                return false;
            }

            // Assign buffers to descriptor layout
            std::vector<VkDescriptorBufferInfo> dbis;
            dbis.resize(buffs.size());

            for(u32 i=0; i<buffs.size(); ++i){
                auto buff = *std::next(buffs.begin(), i);

                dbis[i] = VkDescriptorBufferInfo{
                    buff.buff_handle,
                    0,
                    buff.size_bytes
                };
            }

            // Create write descriptors for transfer of buffers to sm
            std::vector<VkWriteDescriptorSet> write_descriptors;
            write_descriptors.resize(buffs.size());

            for (i32 i=0; i<buffs.size(); ++i){ 
                write_descriptors[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; 
                write_descriptors[i].dstSet=task.descriptor; 
                write_descriptors[i].dstBinding=i; 
                write_descriptors[i].descriptorCount=1; 
                write_descriptors[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; 
                write_descriptors[i].pBufferInfo=&dbis[i]; 
            }

            vkUpdateDescriptorSets(device_handle, buffs.size(), write_descriptors.data(), 0, nullptr);

            return true;
        }

        template<class_type KernelParams>
        auto dispatch_kernel_to_command_buffer(
            kernel<device_driver::vulkan_native>& task, 
            vec3<u32> workgroup_size, 
            KernelParams kernel_params
        ) -> bool {
            // Configure command buffer info
            if (task.command_buffer == VK_NULL_HANDLE){
                return false;
            }

            vkResetCommandBuffer(task.command_buffer, 0);
            
            // Start command buffer
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; 
            if(vkBeginCommandBuffer(task.command_buffer, &cbbi) != VK_SUCCESS ){
                return false;
            }

            // Bind updated pipeline and descripor sets
            vkCmdBindPipeline(task.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.pipeline);
            vkCmdBindDescriptorSets(task.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.pipeline_layout, 0, 1, &task.descriptor, 0, nullptr);

            // Push kernel params for launch 
            vkCmdPushConstants(task.command_buffer, task.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelParams), &kernel_params);

            // Set workgroup size
            vkCmdDispatch(task.command_buffer, workgroup_size.x, workgroup_size.y, workgroup_size.z);
            
            // End command buffer
            if(vkEndCommandBuffer(task.command_buffer) != VK_SUCCESS){
                return false;
            }

            // Submit command buffer to queue
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; 
            si.commandBufferCount=1; 
            si.pCommandBuffers=&task.command_buffer;
            
            // Create fence to indicate compute has finshed
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; 
            if (task.lock != VK_NULL_HANDLE){
                vkDestroyFence(device_handle, task.lock, nullptr);
                task.lock = VK_NULL_HANDLE;
            }
            if(vkCreateFence(device_handle, &fci, nullptr, &task.lock) != VK_SUCCESS){
                return false;
            }
            
            if(vkQueueSubmit(queue_handle, 1, &si, task.lock) != VK_SUCCESS){
                return false;
            }

            return true;
        }

    }; // vulkan_native_driver


    template<> struct device_driver_impl<device_driver::vulkan_native>{
        using type = vulkan_native_driver;
    }; // device_driver_impl<device_driver::vulkan_native>
    
}; // nameapce tether_io

#endif // TARGET_VULKAN_NATIVE
