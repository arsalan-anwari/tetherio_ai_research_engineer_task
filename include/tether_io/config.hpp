#pragma once

#include <expected>
#include <unordered_set>
#include <fstream>

#include <nlohmann/json.hpp>

#include "types.hpp"

namespace tether_io{
    using json = nlohmann::json;
}

namespace {

static inline auto read_json_file(const std::filesystem::path& p, tether_io::json& out) -> std::expected<void, tether_io::file_error> {
    std::ifstream ifs(p);
    
    if (!ifs) return std::unexpected{ tether_io::file_error::file_not_found };

    try {
        ifs >> out;
        return {};
    } catch (...) {
        return std::unexpected{ tether_io::file_error::could_not_parse_file };
    }
}

static inline auto file_contains_keys(const tether_io::json& in, const std::unordered_set<tether_io::cstr>& keys) -> bool {
    for(const auto& key : keys){
        if (!in.contains(key)) return false;
    }
    return true;
}

}


namespace tether_io{

inline void from_json(const json& j, vec3<u32>& v) {
    v.x = j.at(0).get<tether_io::u32>();
    v.y = j.at(1).get<tether_io::u32>();
    v.z = j.at(2).get<tether_io::u32>();
}

inline void from_json(const json& j, version<u32>& v) {
    v.variant = j.at(0).get<u32>();
    v.major = j.at(1).get<u32>();
    v.minor = j.at(2).get<u32>();
    v.patch = j.at(3).get<u32>();
}

auto kernel_type_from_str(str value) -> std::expected<kernel_type, json_error>{
    if (value == "vulkan_compute_shader") return kernel_type::vulkan_compute_shader;
    return std::unexpected{ json_error::invalid_value_type };
};

auto kernel_format_from_str(str value) -> std::expected<kernel_format, json_error>{
    if (value == "glsl") return kernel_format::glsl;
    if (value == "spirv") return kernel_format::spirv;
    if (value == "hlsl") return kernel_format::hlsl;
    return std::unexpected{ json_error::invalid_value_type };
};

auto kernel_bin_format_from_kernel_type(kernel_type value) -> kernel_format {
    switch(value){
        case kernel_type::vulkan_compute_shader: return kernel_format::spirv;
        default: kernel_format::spirv;
    }
    return kernel_format::spirv;
};

auto dir_name_from_kernel_type(kernel_type value) -> str {
    switch(value){
        case kernel_type::vulkan_compute_shader : return "vk";
        default: return "";
    }
};

auto file_type_from_kernel_format(kernel_format value) -> str {
    switch(value){
        case kernel_format::glsl : return ".glsl";
        case kernel_format::spirv : return ".spv";
        case kernel_format::hlsl : return ".hlsl";
        default: return "";
    }
};

inline auto parse_application_settings(std::filesystem::path path) -> std::expected<application_config, json_error>{
    
    json app_settings;
    if( !read_json_file(path, app_settings).has_value() || !app_settings.is_object() ){
        return std::unexpected{ json_error::invalid_json_format };
    }

    if (!file_contains_keys(app_settings, {"kernel_type", "kernel_format_out"})){
        return std::unexpected{ json_error::key_not_found };
    }

    application_config cfg;
    cfg.resource_dir = std::filesystem::path(RESOURCE_DIR); 
    
    auto comp_type = kernel_type_from_str(app_settings["kernel_type"].get<str>());
    if (!comp_type.has_value()) return std::unexpected{comp_type.error()};

    cfg.kernel_dir = cfg.resource_dir / str("kernels") / dir_name_from_kernel_type(comp_type.value());
    cfg.kernel_bin_format = kernel_bin_format_from_kernel_type(comp_type.value());

    // Now parse all available kernels based on application settings
    json kernel_settings;
    if( !read_json_file(cfg.kernel_dir / "index.json", kernel_settings).has_value() || !kernel_settings.is_object() ){
        return std::unexpected{ json_error::invalid_json_format };
    }

    if (!file_contains_keys(kernel_settings, {"compute"})){
        return std::unexpected{ json_error::key_not_found };
    }

    for (const auto& entry : kernel_settings["compute"]) {
        try {
            if (!file_contains_keys(entry, {"recompile", "version", "param_size_bytes", "name", "format", "file"})){
                return std::unexpected{ json_error::key_not_found };
            }

            kernel_config krnl;
            krnl.name = entry["name"].get<str>();
            krnl.recompile = entry["recompile"].get<bool>();
            krnl.type = comp_type.value();

            auto format = kernel_format_from_str(entry["format"].get<str>());
            if(!format.has_value()) return std::unexpected{format.error()};
            krnl.format = format.value();

            krnl.type_version = entry["version"].get<version<u32>>();
            krnl.param_size_bytes = entry["param_size_bytes"].get<usize>();

            krnl.path = cfg.kernel_dir / entry["file"].get<str>();
            str bin_file_name = entry["name"].get<str>() + file_type_from_kernel_format(cfg.kernel_bin_format);
            krnl.path_bin = cfg.kernel_dir / "bin" / bin_file_name;

            cfg.kernels[entry["name"].get<str>()] = krnl;
            
        } catch (const std::exception& e) {
            return std::unexpected { json_error::invalid_json_format };
        }
    }

    return cfg;

}



}