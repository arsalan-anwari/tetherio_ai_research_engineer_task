#pragma once

#include <concepts>

#include "types.hpp"
#include "context.hpp"
#include "config.hpp"

#ifdef TARGET_VULKAN_NATIVE

#include "algorithm/vulkan_native/fill.hpp"
#include "algorithm/vulkan_native/multiply.hpp"
#include "algorithm/vulkan_native/binmatmul.hpp"


#endif // TARGET_VULKAN_NATIVE

#include "algorithm/cpu_native/data_formatting.hpp"
#include "algorithm/cpu_native/binmatmul.hpp"

namespace tether_io{

template<device_driver D, execution_method M>
struct algorithm;

template<device_driver D> 
struct algorithm<D, execution_method::standalone>{
    compute_context<D>& ctx;
    application_config& config;

    algorithm(compute_context<D>& c, application_config& cfg)
        : ctx(c), config(cfg) {}

    template<typename T, typename... Args>
    requires std::integral<T> || std::floating_point<T>
    auto fill(
        vec3<u32> work_group_size,
        std::span<T>& out,
        T fill_value,
        Args&&... opts
    ) -> std::expected<void, device_error>{
        std::expected<void, device_error> res;

        if constexpr(D == device_driver::vulkan_native){
            res = fill_vulkan_native_standalone(ctx, config, work_group_size, out, fill_value, opts...);
        }

        if (!res.has_value()) return std::unexpected{ res.error() };
        return{};
    };

    template<typename T, typename... Args>
    requires std::integral<T> || std::floating_point<T>
    auto multiply(
        vec3<u32> work_group_size,
        std::span<T>& out,
        T mull_factor,
        Args&&... opts
    ) -> std::expected<void, device_error>{
        std::expected<void, device_error> res;

        if constexpr(D == device_driver::vulkan_native){
            res = multiply_vulkan_native_standalone(ctx, config, work_group_size, out, mull_factor, opts...);
        }

        if (!res.has_value()) return std::unexpected{ res.error() };
        return{};
    };

    // auto binmatmul(
    //     compute_context<device_driver::vulkan_native>& ctx,
    //     application_config& config,
    //     vec3<u32> work_group_size,
    //     std::span<T>& out,
    //     u32 m, u32 n, u32 k_bits
    // ) -> std::expected<void, device_error>{
    //     std::expected<void, device_error> res;
        
    //     if constexpr(D == device_driver::vulkan_native){
    //         res = binmatmul_vulkan_native_standalone(ctx, config, work_group_size, out, m, n, k_bits, opts...);
    //     }

    //     if (!res.has_value()) return std::unexpected{ res.error() };
    //     return{};
    // }

};

template<device_driver D> 
struct algorithm<D, execution_method::sequenced> {
    compute_context<D>& ctx;
    application_config& config;

    algorithm(compute_context<D>& c, application_config& cfg)
        : ctx(c), config(cfg) {}

    template<typename T, typename... Args>
    requires std::integral<T> || std::floating_point<T>
    auto fill(
        vec3<u32> work_group_size,
        device_buffer<D>& d_buff,
        T fill_value,
        Args&&... opts
    ) -> std::expected<void, device_error>{
        std::expected<void, device_error> res;

        if constexpr(D == device_driver::vulkan_native){
            res = fill_vulkan_native_sequenced(ctx, config, work_group_size, d_buff, fill_value, opts...);
        }

        if (!res.has_value()) return std::unexpected{ res.error() };
        return{};
    };

    template<typename T, typename... Args>
    requires std::integral<T> || std::floating_point<T>
    auto multiply(
        vec3<u32> work_group_size,
        device_buffer<D>& d_buff,
        T mull_factor,
        Args&&... opts
    ) -> std::expected<void, device_error>{
        std::expected<void, device_error> res;
        if constexpr(D == device_driver::vulkan_native){
            res = multiply_vulkan_native_sequenced(ctx, config, work_group_size, d_buff, mull_factor, opts...);
        }

        if (!res.has_value()) return std::unexpected{ res.error() };
        return{};
    };

    template<typename... Args>
    auto binmatmul(
        vec3<u32> grid_size,
        vec3<u32> local_size,
        std::initializer_list<device_buffer<D>> d_buffers,
        u32 m, u32 n, u32 k_bits, u32 k_words,
        Args&&... opts
    ) -> std::expected<void, device_error>{
        std::expected<void, device_error> res;

        if constexpr(D == device_driver::vulkan_native){
            res = binmatmul_vulkan_native_sequenced(ctx, config, grid_size, local_size, d_buffers, m, n, k_bits, k_words, opts...);
        }

        if (!res.has_value()) return std::unexpected{ res.error() };
        return{};
    }

};

template <> struct algorithm<device_driver::cpu_native, execution_method::standalone>{

    auto f32_mat_to_packed_u32(
        matrix_order order,
        std::span<f32> in,
        u32 matrix_side,
        u32 k_bits
    ) -> std::expected<std::vector<u32>, device_error>{
        std::expected<std::vector<u32>, device_error> res;

        if (order == matrix_order::row_major){
            res = f32_mat_to_packed_u32_row_major_cpu_native_standalone(
                in, matrix_side, k_bits
            );   
        }else{
            res = f32_mat_to_packed_u32_col_major_cpu_native_standalone(
                in, matrix_side, k_bits
            ); 
        }

        if (!res.has_value()) return std::unexpected{ res.error() };
        return res.value();
    }


    auto binmatmul(
        std::span<const u32> a_bits,
        std::span<const u32> b_bits,
        u32 m, u32 n, u32 k_bits
    ) -> std::expected<std::vector<i32>, device_error>{
        std::expected<std::vector<i32>, device_error> res;

        res = binmatmul_cpu_native_standalone(a_bits, b_bits, m, n, k_bits);

        if (!res.has_value()) return std::unexpected{ res.error() };
        return res.value();
    }

    auto random_mat_binary_f32_1d(
        data_domain data_range,
        u32 rows, 
        u32 cols, 
        u32 seed
    ) -> std::expected<std::vector<f32>, device_error>{
        std::expected<std::vector<f32>, device_error> res;

        switch(data_range){

            case data_domain::pm_one: 
                 res = random_mat_binary_f32_1d_pm_one_dist_cpu_native_standalone(rows, cols, seed);
                 break;
            case data_domain::zero_one:
                res = random_mat_binary_f32_1d_zero_one_dist_cpu_native_standalone(rows, cols, seed);
                break;
            case data_domain::full_range:
                res = random_mat_binary_f32_1d_full_range_dist_cpu_native_standalone(rows, cols, seed);
                break;
            default:
                return std::unexpected{device_error::not_available};

        }


        if (!res.has_value()) return std::unexpected{ res.error() };
        return res.value();
    }

};

} // namespace tether_io



