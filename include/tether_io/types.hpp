#pragma once

#include <cstdint>
#include <string>
#include <concepts>
#include <filesystem>
#include <unordered_map>

namespace tether_io {

// Base Types
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

using isize = ptrdiff_t;
using usize = size_t;

using cstr = std::string_view;
using str = std::string;

template<std::integral T>
struct vec2 { T x; T y; };

template<std::integral T>
struct vec3 { T x; T y; T z; };

template<std::integral T>
struct vec4 { T x; T y; T z; T w; };

template<std::integral T>
struct api_version { T version; T major; T minor; T patch; };

// Device Context Types
enum class device_driver : u8 { vulkan_native, ggml_vulkan, cuda_native, opencl_native };
enum class device_select : u8 { first_available, first_compute_capable, discrete, integrated };

template<device_driver D>
struct device_driver_impl;

// Buffer Types
template<device_driver D>
struct device_buffer;

enum class alloc_method { base, custom };
enum class upload_method { sync, async };
enum class download_method { sync, async };

enum class precision : u8 { binary_1bit = 1 /* 1-bit weights/acts */ };
enum class data_domain   : u8 { pm_one, zero_one };  // ±1 or {0,1}
enum class matrix_order   : u8 { row_major, col_major };

// Kernel types
enum class kernel_type : u8 { vulkan_compute_shader /* future: CUDA, Metal */ };
enum class kernel_format : u8 { glsl, spirv, hlsl };
enum class launch_method : u8 { sync, async, interrupt };

struct kernel_config {
    kernel_type type { kernel_type::vulkan_compute_shader };
    kernel_format format { kernel_format::glsl };
    std::filesystem::path path;
    std::filesystem::path path_bin;
};

template<device_driver D>
struct kernel;

// Configuration setting for whole application
struct application_config {
    std::filesystem::path resource_dir;
    std::filesystem::path kernel_dir;
    kernel_format kernel_bin_format { kernel_format::spirv };
    std::unordered_map<str, kernel_config> kernels;
};

// Error types
enum class json_error : u8 {
    invalid_json_format, key_not_found, invalid_value_type
};

enum class file_error : u8 {
    file_not_found, could_not_parse_file
};

enum class device_error : u8 {
    init_failed,
    could_not_create_instance,
    no_available_devices, 
    could_not_create_selected_device,
    not_available, 
    unexpected_crash, 
    alloc_failed, 
    could_not_create_buffer,
    upload_failed, 
    download_failed,
    launch_failed
};



} // namespace tether_io