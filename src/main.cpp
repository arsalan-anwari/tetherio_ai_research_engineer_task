#include <iostream>
#include <array>
#include <chrono>
#include <thread>

#include <tether_io/config.hpp>
#include <tether_io/context.hpp>

// struct kernel_config {
//     str name;
//     bool recompile;
//     kernel_type type;
//     kernel_format format;
//     version<u32> type_version;
//     usize param_size_bytes; 
//     std::filesystem::path path;
//     std::filesystem::path path_bin;
// };


int main() {
    using namespace tether_io;

    std::filesystem::path rsc = RESOURCE_DIR;
    auto config = parse_application_settings(rsc / "settings.json");
    if(!config.has_value()) { 
        std::cout << config.error() << std::endl; 
        return -1; 
    }
    
    kernel_config kernel_fill_opts = config.value().kernels["fill"];
    kernel_config kernel_mul_opts = config.value().kernels["multiply"];

    std::array<float, 100> h_buff_1 = {0};
    std::array<float, 100> g_buff_1 = {0};
    struct KernelFillParams { float value; u32 count; } kernel_fill_params { 53.455f, 100 };
    struct KernelMullParams { float factor; u32 count; } kernel_mull_params { 2.0f, 100 };
  
    compute_context<device_driver::vulkan_native> ctx;
    auto res = ctx.init(version<u32>{0, 1, 1, 0}, "HelloWorld");
    res = ctx.set_device(device_select::first_compute_capable);

    auto d_buff_1 = ctx.allocate(h_buff_1.size() * sizeof(float), alloc_method::base);
    res = ctx.upload(d_buff_1.value(), std::span<float>{h_buff_1}, upload_method::sync);

    // Register kernels ahead of time so they can be queued back-to-back.
    auto kernel_fill = ctx.register_kernel(kernel_fill_opts, {d_buff_1.value()});
    if (!kernel_fill.has_value()){
        std::cout << kernel_fill.error() << std::endl;
        ctx.exit({d_buff_1.value()});
        return -1;
    }

    auto kernel_mull = ctx.register_kernel(kernel_mul_opts, {d_buff_1.value()});
    if (!kernel_mull.has_value()){
        std::cout << kernel_mull.error() << std::endl;
        ctx.destroy_kernel(kernel_fill.value());
        ctx.exit({d_buff_1.value()});
        return -1;
    }

    // Launch fill first, followed immediately by multiply while data stays on device.
    res = ctx.launch_kernel(kernel_fill.value(), vec3<usize>{64, 1, 1}, {d_buff_1.value()}, launch_method::sync, kernel_fill_params);
    if (!res.has_value()){
        std::cout << res.error() << std::endl;
        ctx.destroy_kernel(kernel_mull.value());
        ctx.destroy_kernel(kernel_fill.value());
        ctx.exit({d_buff_1.value()});
        return -1;
    }

    res = ctx.launch_kernel(kernel_mull.value(), vec3<usize>{64, 1, 1}, {d_buff_1.value()}, launch_method::sync, kernel_mull_params);
    if (!res.has_value()){
        std::cout << res.error() << std::endl;
        ctx.destroy_kernel(kernel_mull.value());
        ctx.destroy_kernel(kernel_fill.value());
        ctx.exit({d_buff_1.value()});
        return -1;
    }

    // Wait once on the last kernel; the queue guarantees the fill submission has finished as well.
    res = ctx.wait_for_kernel(kernel_mull.value(), 1'000'000'000ull);
    if (!res.has_value()){
        std::cout << res.error() << std::endl;
        ctx.destroy_kernel(kernel_mull.value());
        ctx.destroy_kernel(kernel_fill.value());
        ctx.exit({d_buff_1.value()});
        return -1;
    }

    // Download results back to host
    res = ctx.download(std::span<float>{g_buff_1}, d_buff_1.value(), download_method::sync);
    if (!res.has_value()){
        std::cout << res.error() << std::endl;
        ctx.destroy_kernel(kernel_mull.value());
        ctx.destroy_kernel(kernel_fill.value());
        ctx.exit({d_buff_1.value()});
        return -1;
    }
    
    // Close compute context
    ctx.destroy_kernel(kernel_mull.value());
    ctx.destroy_kernel(kernel_fill.value());
    ctx.exit({d_buff_1.value()});

    // Print out results
    std::cout << "g_buff_1[100] = { " << std::endl;
    int i = 0;
    for(const float& e: g_buff_1){
        std::cout << e << " ";
        if (i >= 20){
            std::cout << std::endl;
            i = 0;
        }
        i++;
    }

    std::cout << std::endl << "}" << std::endl;

    return 0;

}
