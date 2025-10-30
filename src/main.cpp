

#include <iostream>

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

    compute_context<device_driver::vulkan_native> ctx;
    ctx.test();

}

