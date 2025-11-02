#pragma once

#include <functional>
#include <expected>

#include "types.hpp"

#ifdef TARGET_VULKAN_NATIVE

#include "context/vulkan_native.hpp"

#endif // TARGET_VULKAN_NATIVE

namespace tether_io {

template<device_driver D>    
struct compute_context{
    using driver_type = typename device_driver_impl<D>::type;

    template<typename... Args>
    auto init(Args&&... opts) -> std::expected<void, device_error>{
        auto result = driver.init(opts...);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return {};
    };

    auto init(std::function<bool()> custom_init) -> std::expected<void, device_error> { 
        if (!custom_init()) return std::unexpected{device_error::init_failed};
    };

    auto set_device(device_select preferred_type) -> std::expected<void, device_error>{
        auto result = driver.set_device(preferred_type);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return {};
    };

    auto set_device(usize device_number) -> std::expected<void, device_error>{
        // auto result = driver.set_device(device_number);
        // if (!result.has_value()) return std::unexpected{ result.error() };
        return {};
    };

    template<typename... Args>
    auto allocate(
        usize size_bytes, 
        alloc_method method = alloc_method::default, 
        Args&&... opts
    ) -> std::expected<device_buffer<D>, device_error> {
        auto result = driver.allocate(size_bytes, method, opts...);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return result.value();
    };

    template<typename T, typename... Args>
    auto upload(
        device_buffer<D>& dest, 
        std::span<T> src,
        upload_method method = upload_method::sync, 
        Args&&... opts
    ) -> std::expected<void, device_error> {
        auto result = driver.upload(dest, src, method, opts...);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return {};
    };

    template<typename T, typename... Args>
    auto download(
        std::span<T> dest, 
        device_buffer<D>& src,
        download_method method = download_method::sync, 
        Args&&... opts
    ) -> std::expected<void, device_error> {
        auto result = driver.download(dest, src, method, opts...);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return {};
    };

    template<typename... Args>
    auto register_kernel(
        kernel_config& krnl_opts, 
        vec3<u32> workgroup_size,
        std::initializer_list<device_buffer<D>> buffers, 
        Args&&... opts
    ) -> std::expected<kernel<D>, device_error> {
        auto result = driver.register_kernel(krnl_opts, workgroup_size, buffers, opts...);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return result.value();
    };

    template<typename... Args>
    auto launch_kernel(
        kernel<D>& task,
        vec3<u32> workgroup_size,
        std::initializer_list<device_buffer<device_driver::vulkan_native>> buffers, 
        launch_method method = launch_method::sync, 
        Args&&... opts
    ) -> std::expected<void, device_error> {
        auto result = driver.launch_kernel(task, workgroup_size, buffers, method, opts...);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return {};
    };

    template<typename... Args>
    auto wait_for_kernel(
        kernel<D>& task, 
        usize time_out, 
        Args&&... opts
    ) -> std::expected<void, device_error> {
        auto result = driver.wait_for_kernel(task, time_out, opts...);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return {};
    }

    template<typename... Args>
    auto wait_for_last_kernel(
        usize time_out, 
        Args&&... opts
    ) -> std::expected<void, device_error> {
        auto result = driver.wait_for_last_kernel(time_out, opts...);
        if (!result.has_value()) return std::unexpected{ result.error() };
        return {};
    }

    auto limits() -> std::expected<device_limits, device_error>{
        auto result = driver.limits();
        if (!result.has_value()) return std::unexpected{ result.error() };
        return result.value();
    }

    void destroy_kernel(kernel<D>& task){
        driver.destroy_kernel(task);
    }

    void exit(){
        driver.exit();
    }


private:
    driver_type driver;

};


} // tether_io
