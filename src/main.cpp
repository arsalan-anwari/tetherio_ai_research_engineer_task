

#include <iostream>
#include <array>
#include <chrono>
#include <thread>

// #include <tether_io/config.hpp>
#include <tether_io/context.hpp>

int main() {
    using namespace tether_io;

    // std::filesystem::path rsc = RESOURCE_DIR;
    // auto config = parse_application_settings(rsc / "settings.json");

    // if (!config.has_value()) {
    //     // print the error
    //     const auto& err = config.error();
    //     std::cerr << "Failed to load config: " << err << "\n"; // if you add an overload
    //     // or, temporarily:
    //     // std::cerr << "Failed to load config\n";
    //     return 1;
    // }

    // // print the value
    // std::cout << *config << '\n';          // or: std::cout << config.value() << '\n';
    device_buffer<device_driver::vulkan_native> tmp{};

    std::array<float, 100> h_buff;
    std::array<float, 100> g_buff = {0};
    h_buff.fill(456.f);

    compute_context<device_driver::vulkan_native> ctx;
    auto res = ctx.init(api_version<u32>{0, 1, 1, 0}, "HelloWorld");
    if(!res.has_value()){
        std::cout << "Could not Init" << std::endl;
        ctx.kill(tmp);
        return -1;
    }

    auto res2 = ctx.set_device(device_select::first_compute_capable);
    if(!res2.has_value()){
        std::cout << "Could not set device" << std::endl;
        ctx.kill(tmp);
        return -1;
    }

    auto d_buff = ctx.allocate(h_buff.size() * sizeof(float), alloc_method::base);
    if(!d_buff.has_value()){
        std::cout << "Could not allocate" << std::endl;
        ctx.kill(d_buff.value());
        return -1;
    }
    
    auto res3 = ctx.upload(d_buff.value(), std::span<float>{h_buff}, upload_method::sync);
    if(!res3.has_value()){
        std::cout << "Could not upload" << std::endl;
        ctx.kill(d_buff.value());
        return -1;
    }

    // simulate compute
    std::this_thread::sleep_for (std::chrono::seconds(5));
    // ctx.signal(d_buff.value());
    
    auto res4 = ctx.download(std::span<float>{g_buff}, d_buff.value(), download_method::sync);
    if(!res4.has_value()){
        std::cout << "Could not download" << std::endl;
        ctx.kill(d_buff.value());
        return -1;
    }

    // ctx.kill(tmp);
    ctx.kill(d_buff.value());

    std::cout << "g_buff[100] = { " << std::endl;
    int i = 0;
    for(const float& e: g_buff){
        std::cout << e << " ";
        if (i == 20){
            std::cout << std::endl;
            i = 0;
        }
        i++;
    }

    std::cout << std::endl << "}" << std::endl;

    return 0;

}

