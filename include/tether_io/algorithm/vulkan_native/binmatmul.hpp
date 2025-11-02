#pragma once

#include <vector>
#include <expected>
#include <concepts>
#include <span>

#include "../../types.hpp"
#include "../../context.hpp"

namespace tether_io{

/*

layout(push_constant) uniform PushConsts {
    uint M;        // rows of A / C
    uint N;        // cols of B / C
    uint K_bits;   // common dimension in bits (not words)
    uint K_words;  // K_bits / 32 rounded up
} pc;
*/

auto binmatmul_vulkan_native_sequenced(
    compute_context<device_driver::vulkan_native>& ctx,
    application_config& config,
    vec3<u32> grid_size,
    vec3<u32> local_size,
    std::initializer_list<device_buffer<device_driver::vulkan_native>> d_buffers,
    u32 m, u32 n, u32 k_bits, u32 k_words
) -> std::expected<void, device_error>{
    kernel_config kernel_opts = config.kernels["binmatmul"];

    struct KernelParams { 
        u32 m; u32 n;
        u32 k_bits; u32 k_words; 
    } kernel_params { m, n, k_bits, k_words };

    auto kernel = ctx.register_kernel(kernel_opts, local_size, d_buffers);
    if (!kernel.has_value()){
        ctx.exit();
        return std::unexpected{kernel.error()};
    }

    auto res = ctx.launch_kernel(
        kernel.value(), 
        grid_size, 
        d_buffers, 
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

// template<typename T> 
// auto binmatmul_vulkan_native_standalone(
//     compute_context<device_driver::vulkan_native>& ctx,
//     application_config& config,
//     vec3<u32> work_group_size,
//     std::span<T>& out,
//     u32 m, u32 n, u32 k_bits
// ) -> std::expected<void, device_error>{
//     kernel_config kernel_opts = config.kernels["binmatmul"];

//     const uint32_t K_words = (K_bits + 31u)/32u;

//     struct KernelParams { 
//         u32 m; u32 n;
//         u32 k_bits; u32 K_words; 
//     } kernel_params { M, N, K_bits, K_words };

//     auto d_buff = ctx.allocate(out.size() * sizeof(T), alloc_method::base);

//     auto kernel = ctx.register_kernel(kernel_opts, work_group_size, {d_buff});
//     if (!kernel.has_value()){
//         ctx.exit();
//         return std::unexpected{kernel.error()};
//     }

//     auto res = ctx.launch_kernel(
//         kernel.value(), 
//         work_group_size, 
//         {d_buff}, 
//         launch_method::sync, 
//         kernel_params
//     );

//     if (!res.has_value()){
//         ctx.destroy_kernel(kernel.value());
//         ctx.exit();
//         return std::unexpected{res.error()};
//     }

//     res = ctx.wait_for_kernel(kernel.value(), 1'000'000'000ull);
//     if (!res.has_value()){
//         ctx.destroy_kernel(kernel.value());
//         ctx.exit();
//         return std::unexpected{res.error()};
//     }

//     res = ctx.download(std::span<T>{out}, d_buff, download_method::sync);
//     if (!res.has_value()){
//         ctx.destroy_kernel(kernel.value());
//         ctx.exit();
//         return std::unexpected{res.error()};
//     }

//     ctx.destroy_kernel(kernel.value());

//     return {};
// }

}
