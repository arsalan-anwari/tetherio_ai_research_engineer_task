#pragma once

#include <vector>
#include <expected>
#include <concepts>
#include <span>

#include "../../types.hpp"
#include "../../context.hpp"

namespace tether_io{

// template<execution_method M, typename T>

// auto fill(
//     compute_context<device_driver::vulkan_native>& ctx,
//     application_config& config,
//     vec3<u32> work_group_size,
//     std::span<T>& out,
//     T fill_value
// ) -> std::expected<void, device_error>;

template<typename T>
auto fill_vulkan_native_sequenced(
    compute_context<device_driver::vulkan_native>& ctx,
    application_config& config,
    vec3<u32> work_group_size,
    device_buffer<device_driver::vulkan_native>& d_buff,
    T fill_value
) -> std::expected<void, device_error>{
    kernel_config kernel_opts = config.kernels["fill"];

    struct KernelParams { 
        T value; u32 count; 
    } kernel_params { fill_value, static_cast<u32>(d_buff.size_bytes / sizeof(T)) };

    auto kernel = ctx.register_kernel(kernel_opts, work_group_size, {d_buff});
    if (!kernel.has_value()){
        ctx.exit();
        return std::unexpected{kernel.error()};
    }

    auto res = ctx.launch_kernel(
        kernel.value(), 
        work_group_size, 
        {d_buff}, 
        launch_method::sync, 
        kernel_params
    );

    if (!res.has_value()){
        ctx.destroy_kernel(kernel.value());
        ctx.exit();
        return std::unexpected{res.error()};
    }

    return {};
}

template<typename T>
auto fill_vulkan_native_standalone(
    compute_context<device_driver::vulkan_native>& ctx,
    application_config& config,
    vec3<u32> work_group_size,
    std::span<T>& out,
    T fill_value
) -> std::expected<void, device_error>{
    kernel_config kernel_opts = config.kernels["fill"];

    struct KernelParams { 
        T value; u32 count; 
    } kernel_params { fill_value, static_cast<u32>(out.size()) };

    auto d_buff = ctx.allocate(out.size() * sizeof(T), alloc_method::base);

    auto kernel = ctx.register_kernel(kernel_opts, work_group_size, {d_buff});
    if (!kernel.has_value()){
        ctx.exit();
        return std::unexpected{kernel.error()};
    }

    auto res = ctx.launch_kernel(
        kernel.value(), 
        work_group_size, 
        {d_buff}, 
        launch_method::sync, 
        kernel_params
    );

    if (!res.has_value()){
        ctx.destroy_kernel(kernel.value());
        ctx.exit();
        return std::unexpected{res.error()};
    }

    res = ctx.wait_for_kernel(kernel.value(), 1'000'000'000ull);
    if (!res.has_value()){
        ctx.destroy_kernel(kernel.value());
        ctx.exit();
        return std::unexpected{res.error()};
    }

    res = ctx.download(std::span<T>{out}, d_buff, download_method::sync);
    if (!res.has_value()){
        ctx.destroy_kernel(kernel.value());
        ctx.exit();
        return std::unexpected{res.error()};
    }

    ctx.destroy_kernel(kernel.value());

    return {};
}

}