#pragma once

#include "types.hpp"

#ifdef TARGET_VULKAN_NATIVE

#include <concepts>

#include "context.hpp"
#include "config.hpp"
#include "algorithm/vulkan_native/fill.hpp"
#include "algorithm/vulkan_native/multiply.hpp"

#endif // TARGET_VULKAN_NATIVE

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
            res = fill_standalone_vulkan_native(ctx, config, work_group_size, out, fill_value, opts...);
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
            res = multiply_standalone_vulkan_native(ctx, config, work_group_size, out, mull_factor, opts...);
        }

        if (!res.has_value()) return std::unexpected{ res.error() };
        return{};
    };

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
            res = fill_sequenced_vulkan_native(ctx, config, work_group_size, d_buff, fill_value, opts...);
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
            res = multiply_sequenced_vulkan_native(ctx, config, work_group_size, d_buff, mull_factor, opts...);
        }

        if (!res.has_value()) return std::unexpected{ res.error() };
        return{};
    };

};

} // namespace tether_io



// template<device_driver D, typename T, typename... Args>
// auto fill(
//     compute_context<D>& ctx,
//     device_buffer<D>& d_buff,
//     application_config& config,
//     vec3<u32> work_group_size,
//     std::span<T>& out,
//     T fill_value
// ) -> std::expected<void, device_error>{}
