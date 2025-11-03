#pragma once

#include <expected>
#include <vector>
#include <filesystem>

#include "types.hpp"
#include "context.hpp"
#include "algorithm.hpp"

namespace tether_io {

template<sandbox_algorithm A, device_driver D>
struct sandbox;


template<> struct sandbox<sandbox_algorithm::binmatmul, device_driver::vulkan_native> {

    auto run(
        data_domain domain,
        u32 M, 
        u32 N,
        u32 K_bits
    ) -> std::expected<sandbox_results<sandbox_algorithm::binmatmul>, device_error> {

        // Load config
        auto cfg = parse_application_settings(rsc / "settings.json");
        if(!cfg.has_value()) { 
            return std::unexpected{ device_error::init_failed }; 
        }
        config = cfg.value();

        u32 K_words = (K_bits + 31u) / 32u;
        A.resize(M * K_bits);
        B.resize(K_bits * N);

        // Fill matrices A and B with random data
        auto A_res = host_kernel_launcher.random_mat_binary_f32_1d(
            domain, 
            M, 
            K_bits, 
            7937929
        );

        if(!A_res.has_value()) { 
            return std::unexpected{A_res.error()}; 
        }

        A = A_res.value();

        auto B_res = host_kernel_launcher.random_mat_binary_f32_1d(
            domain, 
            K_bits, 
            N, 
            732973980
        );

        if(!B_res.has_value()) { 
            return std::unexpected{B_res.error()}; 
        }

        B = B_res.value();

        auto A_bits_res = host_kernel_launcher.f32_mat_to_packed_u32(
            matrix_order::row_major, 
            A, 
            M, 
            K_bits
        );

        if(!A_bits_res.has_value()) { 
            return std::unexpected{A_bits_res.error()}; 
        }

        A_bits = A_bits_res.value();

        auto B_bits_res = host_kernel_launcher.f32_mat_to_packed_u32(
            matrix_order::col_major, 
            B, 
            N, 
            K_bits
        );

        if(!B_bits_res.has_value()) { 
            return std::unexpected{B_bits_res.error()}; 
        }

        B_bits = B_bits_res.value();

        auto C_host_res = host_kernel_launcher.binmatmul(
            A_bits, 
            B_bits, 
            M, 
            N, 
            K_bits
        );

        if(!C_host_res.has_value()) { 
            return std::unexpected{C_host_res.error()}; 
        }

        C_host = C_host_res.value();
        C_device.resize(C_host.size());

        result = ctx.init(version<u32>{0, 1, 1, 0}, gen_app_name(domain, M, K_bits, N));
        if(!result.has_value()) { 
            ctx.exit();
            return std::unexpected{result.error()}; 
        }
        
        result = ctx.set_device(device_select::first_compute_capable);
        if(!result.has_value()) { 
            ctx.exit();
            return std::unexpected{result.error()}; 
        }

        auto d_buff_A_res = ctx.allocate(A_bits.size() * sizeof(u32), alloc_method::base);
        if(!d_buff_A_res.has_value()) { 
            ctx.exit();
            return std::unexpected{d_buff_A_res.error()}; 
        }
        auto d_buff_A = d_buff_A_res.value();

        auto d_buff_B_res = ctx.allocate(A_bits.size() * sizeof(u32), alloc_method::base);
        if(!d_buff_B_res.has_value()) {
            ctx.exit(); 
            return std::unexpected{d_buff_B_res.error()}; 
        }
        auto d_buff_B = d_buff_B_res.value();
        
        auto d_buff_C_res = ctx.allocate(static_cast<usize>(M * N * sizeof(u32)), alloc_method::base);
        if(!d_buff_C_res.has_value()) { 
            ctx.exit();
            return std::unexpected{d_buff_C_res.error()}; 
        }
        auto d_buff_C = d_buff_C_res.value();

        result = ctx.upload(d_buff_A, std::span<u32>{A_bits}, upload_method::sync);
        if(!result.has_value()) { 
            ctx.exit();
            return std::unexpected{result.error()}; 
        }

        result = ctx.upload(d_buff_B, std::span<u32>{B_bits}, upload_method::sync);
        if(!result.has_value()) { 
            ctx.exit();
            return std::unexpected{result.error()}; 
        }

        algorithm<
            device_driver::vulkan_native, 
            execution_method::sequenced
        > device_kernel_launcher(ctx, config);

        auto device_limits_res = ctx.limits();
        if (!device_limits_res.has_value()){
            ctx.exit();
            return std::unexpected{device_limits_res.error()}; 
        }
        const auto device_limits = device_limits_res.value();

        u32 local_x = choose_tile(N, 16u, device_limits.max_compute_work_group_size.x);
        u32 local_y = choose_tile(M, 16u, device_limits.max_compute_work_group_size.y);
        vec3<u32> local_size{local_x, local_y, 1u};

        u32 groups_x = ceil_div(N, local_x);
        u32 groups_y = ceil_div(M, local_y);
        vec3<u32> grid_size{groups_x, groups_y, 1u};

        result = device_kernel_launcher.binmatmul(
            grid_size,
            local_size,
            {d_buff_A, d_buff_B, d_buff_C},
            M, N, K_bits, K_words
        );
        if(!result.has_value()) { 
            ctx.exit();
            return std::unexpected{result.error()}; 
        }

        result = ctx.wait_for_last_kernel(1'000'000'000ull);

        result = ctx.download(std::span<i32>{C_device}, d_buff_C, download_method::sync);
        if (!result.has_value()){
            ctx.exit();
            return std::unexpected{result.error()}; 
        }

        ctx.exit();

        i32 max_abs_err = 0; 
        usize mismatches = 0;
        for (usize i=0; i<C_device.size(); ++i){ 
            i32 e = std::abs(C_device[i] - C_host[i]); 
            if (e > max_abs_err) max_abs_err=e; 
            if (e != 0) ++mismatches; 
        }

        return sandbox_results<sandbox_algorithm::binmatmul>{max_abs_err, mismatches, C_host.size()};

    };

private:
    application_config config;
    std::expected<void, device_error> result;
    std::filesystem::path rsc = RESOURCE_DIR;
    compute_context<device_driver::vulkan_native> ctx;
    algorithm<device_driver::cpu_native, execution_method::standalone> host_kernel_launcher;
    
    std::vector<f32> A;
    std::vector<f32> B;
    std::vector<u32> A_bits;
    std::vector<u32> B_bits;
    std::vector<i32> C_host;
    std::vector<i32> C_device;

    auto choose_tile(
        u32 dim,
        u32 preferred,
        u32 max_local
    ) -> u32 {
        u32 capped = std::min(preferred, max_local);
        if (dim >= capped) return capped;
        // fall back to the largest power-of-two ≤ dim
        if (dim >= 8) return 8u;
        if (dim >= 4) return 4u;
        if (dim >= 2) return 2u;
        return 1u;
    };

    auto ceil_div(u32 value, u32 tile) -> u32 {
        return (value + tile - 1u) / tile;
    };

    auto gen_app_name(
        data_domain domain,
        u32 M, 
        u32 K_bits, 
        u32 N
    ) -> str {
        return 
            to_string(sandbox_algorithm::binmatmul) + "_" +
            to_string(domain) + "_" +
            std::to_string(M) + "x" + 
            std::to_string(N) + "[" +
            std::to_string(K_bits) + "bit]";

    };



};


} // tether_io