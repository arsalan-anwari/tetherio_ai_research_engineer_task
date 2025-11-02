#include <iostream>
#include <array>
#include <chrono>
#include <thread>

#include <tether_io/config.hpp>
#include <tether_io/context.hpp>
#include <tether_io/algorithm.hpp>

int main() {
    using namespace tether_io;

    std::filesystem::path rsc = RESOURCE_DIR;
    auto config = parse_application_settings(rsc / "settings.json");
    if(!config.has_value()) { 
        std::cout << config.error() << std::endl; 
        return -1; 
    }

    std::array<float, 100> g_buff_raw = {0};
    auto g_buff = std::span<float>{g_buff_raw};
      
    compute_context<device_driver::vulkan_native> ctx;
    auto res = ctx.init(version<u32>{0, 1, 1, 0}, "HelloWorld");
    res = ctx.set_device(device_select::first_compute_capable);

    auto d_buff = ctx.allocate(g_buff.size() * sizeof(float), alloc_method::base);

    algorithm<device_driver::vulkan_native, execution_method::sequenced> kernel_launcher(ctx, config.value());

    res = kernel_launcher.fill(vec3<u32>{64, 1, 1}, d_buff.value(), 128.0f);
    res = kernel_launcher.multiply(vec3<u32>{64, 1, 1}, d_buff.value(), 2.0f);

    ctx.wait_for_last_kernel(1'000'000'000ull);

    res = ctx.download(g_buff, d_buff.value(), download_method::sync);
    if (!res.has_value()){
        ctx.exit();
        std::cout << res.error() << std::endl;
        return -1;
    }

    ctx.exit();

    // Print out results
    std::cout << "g_buff[100] = { " << std::endl;
    int i = 0;
    for(const float& e: g_buff){
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
