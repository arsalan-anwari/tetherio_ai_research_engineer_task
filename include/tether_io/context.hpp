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

    auto init() -> std::expected<void, device_error>{};
    auto init(std::function<bool()> custom_init) -> std::expected<void, device_error> { 
        if (!custom_init()) return std::unexpected{device_error::init_failed};
    };

    auto set_device(device_select preferred_type) -> std::expected<void, device_error>{};
    auto set_device(usize device_number) -> std::expected<void, device_error>{};

    template<typename... Args>
    auto allocate(
        usize size_bytes, 
        alloc_method method = alloc_method::default, Args&&... opts
    ) -> std::expected<device_buffer<D>, device_error> {
        return device_buffer<D>{};
    };

    template<typename T, typename... Args>
    auto upload(
        device_buffer<D>& dest, std::span<T> src,
        upload_method method = upload_method::sync, Args&&... opts
    ) -> std::expected<void, device_error> {

    };

    template<typename T, typename... Args>
    auto download(
        std::span<T> dest, device_buffer<D>& src, usize offset,
        download_method method = download_method::sync, Args&&... opts
    ) -> std::expected<void, device_error> {

    };

    template<device_buffer<D>... Buffs, typename... Args>
    auto register_kernel(
        kernel_config krnl_opts, Buffs&&... buffers, Args&&... opts
    ) -> std::expected<kernel<D>, device_error> {};

    template<typename... Args>
    auto launch_kernel(
        kernel<D>& task, vec3<usize> workgroup_size, 
        launch_method method = launch_method::sync, Args&&... opts
    ) -> std::expected<void, device_error> {};



private:
    driver_type driver;

};


} // tether_io
