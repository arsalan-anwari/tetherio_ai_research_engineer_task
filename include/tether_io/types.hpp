#pragma once

#include <cstdint>
#include <string>
#include <concepts>
#include <filesystem>
#include <unordered_map>
#include <iostream>

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
struct version { 
    T variant; T major; T minor; T patch; 
    auto operator<=>(const version&) const = default;
};

// concepts

template<typename T>
concept class_type = std::is_class_v<T>;

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
    str name;
    bool recompile;
    kernel_type type;
    kernel_format format;
    version<u32> type_version;
    usize param_size_bytes; 
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
    launch_failed,
    could_not_compile_shader,
    shader_version_or_type_not_supported,
    could_not_update_descriptors,
    could_not_update_pipeline,
    could_not_update_kernel_module,
    could_not_create_pipeline,
    could_not_register_kernel,
    could_not_dispatch_kernel_to_command_buffer,
    kernel_timout_reached,
};

std::ostream& operator<<(std::ostream& os, const json_error& error) {
    switch (error) {
        case json_error::invalid_json_format :
            os << "Json file has invalid format";
            break;
        case json_error::key_not_found :
            os << "Could not find keys needed in json file";
            break;
        case json_error::invalid_value_type :
            os << "Key value is incorrect type or value";
            break; 
        default:
            os << "Unkown error with parsing json file";
            break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const device_error& error) {
    switch (error) {
        case device_error::init_failed :
            os << "Could not initialize compute context";
            break;
        case device_error::could_not_create_instance :
            os << "Could not create instance";
            break;
        case device_error::no_available_devices :
            os << "No devices available";
            break;
        case device_error::could_not_create_selected_device :
            os << "Could find requested device type";
            break;
        case device_error::not_available :
            os << "Device or feature is not available";
            break;
        case device_error::unexpected_crash :
            os << "Device or feature crashed without message";
            break;
        case device_error::alloc_failed :
            os << "Could not allocate memory on device";
            break;
        case device_error::could_not_create_buffer :
            os << "Could not create memory buffer on device";
            break;
        case device_error::upload_failed :
            os << "Could not upload data to device buffer";
            break;
        case device_error::download_failed :
            os << "Could not download data from device buffer";
            break;
        case device_error::launch_failed :
            os << "Could not launch kernel on device";
            break;
        case device_error::could_not_compile_shader :
            os << "Could not compile shader into requested format";
            break;
        case device_error::shader_version_or_type_not_supported :
            os << "Version or type of shader not supported with selected compute context";
            break;
        case device_error::could_not_update_descriptors :
            os << "Could not update descriptor layout with new kernel config";
            break;
        case device_error::could_not_update_pipeline :
            os << "Could not update piepline with new kernel config";
            break;
        case device_error::could_not_update_kernel_module :
            os << "Could not update kernel module with new kernel config";
            break;
        case device_error::could_not_create_pipeline :
            os << "Could not create pipeline for new kernel config";
            break; 
        case device_error::could_not_register_kernel :
            os << "Could not register or shedule kernel with new kernel config";
            break; 
        case device_error::could_not_dispatch_kernel_to_command_buffer :
            os << "Could not dispatch the kernel to the command buffer";
            break; 
        case device_error::kernel_timout_reached : 
            os << "Timeout reached, kernel is not responding complete status";
            break; 
        default:
            os << "Unkown error with device";
            break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const version<u32>& ver){
    os << "(" 
        << ver.variant << ", " 
        << ver.major << ", "
        << ver.minor << ", "
        << ver.patch << ""
        << ")";

    return os;
};

std::ostream& operator<<(std::ostream& os, const kernel_config& cfg){
    os  << "kernel_config[" << cfg.name << "]"  << std::endl
        << "\t - recompile = " << cfg.recompile << std::endl
        << "\t - type = " << (cfg.type == kernel_type::vulkan_compute_shader ? "vulkan_compute_shader" : "unknown") << std::endl
        << "\t - format = " << (cfg.format == kernel_format::glsl ? "glsl" : "unknown") << std::endl
        << "\t - type_version = " << cfg.type_version << std::endl
        << "\t - path = " << cfg.path << std::endl
        << "\t - path_bin = " << cfg.path_bin << std::endl;
    
        return os;
}



} // namespace tether_io