#include <iostream>
#include <array>
#include <chrono>
#include <thread>

#include <tether_io/config.hpp>
#include <tether_io/context.hpp>
#include <tether_io/algorithm.hpp>

int main() {
    using namespace tether_io;

// Load config
    std::filesystem::path rsc = RESOURCE_DIR;
    auto config = parse_application_settings(rsc / "settings.json");
    if(!config.has_value()) { 
        std::cout << config.error() << std::endl; 
        return -1; 
    }


// Prepare host side executor
    algorithm<device_driver::cpu_native, execution_method::standalone> host_kernel_launcher;

// Prepare host and device side data
    // Simulation constants
    const u32 M = 256, K_bits = 64, N = 256; // K_bits is the shared dimension (# of features)
    const u32 K_words = (K_bits + 31u)/32u;

    // Matrices A and B
    std::vector<f32> A(M * K_bits);
    std::vector<f32> B(K_bits * N);

    // Fill matrices A and B with random data
    A = host_kernel_launcher.random_mat_binary_f32_1d(data_domain::pm_one, M, K_bits, 123).value();
    B = host_kernel_launcher.random_mat_binary_f32_1d(data_domain::pm_one, K_bits, N, 321).value();

    // Pack matrices as u32 words of bits, each bit of each u32 is one float state (±1)
    std::vector<u32> A_bits = host_kernel_launcher.f32_mat_to_packed_u32(matrix_order::row_major, A, M, K_bits).value();
    std::vector<u32> B_bits = host_kernel_launcher.f32_mat_to_packed_u32(matrix_order::col_major, B, N, K_bits).value();
    
    // Calculate host version of GEMM operation for validation
    std::vector<i32> C_host = host_kernel_launcher.binmatmul(A_bits, B_bits, M, N, K_bits).value();

    // Create result buffer for device
    std::vector<i32> C_device;
    C_device.resize(C_host.size());

// Prepare device side context
    compute_context<device_driver::vulkan_native> ctx;
    auto res = ctx.init(version<u32>{0, 1, 1, 0}, "SingleBitMull_Demo");
    res = ctx.set_device(device_select::first_compute_capable);

// Allocate buffers on device
    auto d_buff_A = ctx.allocate(A_bits.size() * sizeof(u32), alloc_method::base).value();
    auto d_buff_B = ctx.allocate(A_bits.size() * sizeof(u32), alloc_method::base).value();
    auto d_buff_C = ctx.allocate(static_cast<usize>(M * N * sizeof(u32)), alloc_method::base).value();

    res = ctx.upload(d_buff_A, std::span<u32>{A_bits}, upload_method::sync);
    res = ctx.upload(d_buff_B, std::span<u32>{B_bits}, upload_method::sync);

// Prepare device side executor    
    algorithm<device_driver::vulkan_native, execution_method::sequenced> device_kernel_launcher(ctx, config.value());

    auto choose_tile = [](
        u32 dim,
        u32 preferred,
        u32 max_local
    ) {
        u32 capped = std::min(preferred, max_local);
        if (dim >= capped) return capped;
        // fall back to the largest power-of-two ≤ dim
        if (dim >= 8) return 8u;
        if (dim >= 4) return 4u;
        if (dim >= 2) return 2u;
        return 1u;
    };

    auto ceil_div = [](u32 value, u32 tile) {
        return (value + tile - 1u) / tile;
    };

    auto device_limits_res = ctx.limits();
    if (!device_limits_res.has_value()){
        ctx.exit();
        std::cout << device_limits_res.error() << std::endl;
        return -1;
    }
    const auto device_limits = device_limits_res.value();

    uint32_t local_x = choose_tile(N, 16u, device_limits.max_compute_work_group_size.x);
    uint32_t local_y = choose_tile(M, 16u, device_limits.max_compute_work_group_size.y);
    vec3<u32> local_size{local_x, local_y, 1u};

    uint32_t groups_x = ceil_div(N, local_x);
    uint32_t groups_y = ceil_div(M, local_y);
    vec3<u32> grid_size{groups_x, groups_y, 1u};

    res = device_kernel_launcher.binmatmul(
        grid_size,
        local_size,
        {d_buff_A, d_buff_B, d_buff_C},
        M, N, K_bits, K_words
    );

// Wait for device kernel to finish    
    res = ctx.wait_for_last_kernel(1'000'000'000ull);

// Download result back to host buffer
    res = ctx.download(std::span<i32>{C_device}, d_buff_C, download_method::sync);
    if (!res.has_value()){
        ctx.exit();
        std::cout << res.error() << std::endl;
        return -1;
    }

// Close device context
    ctx.exit();

// Print out results
    
    i32 max_abs_err = 0; 
    usize mismatches = 0;
    for (usize i=0; i<C_device.size(); ++i){ 
        i32 e = std::abs(C_device[i] - C_host[i]); 
        if (e > max_abs_err) max_abs_err=e; 
        if (e != 0) ++mismatches; 
    }

    // Print a tiny view
    auto print_matrix = [&](const std::vector<i32>& C){
        u32 M_capped = std::min(16u, M);
        u32 N_capped = std::min(16u, N);

        for (u32 r=0; r<M_capped; ++r){
            for (u32 c=0; c<N_capped; ++c){ 
                std::cout << C[usize(r) * N_capped + c] << ( c+1 == N_capped ? '\n' : '\t'); 
            }
        }
    };

    std::cout << "CPU reference [:16][:16]" << std::endl; print_matrix(C_host);
    std::cout << "GPU reference [:16][:16]" << std::endl; print_matrix(C_device);
    std::cout << "Max abs error: " << max_abs_err << ", mismatches: " << mismatches << " / " << C_host.size() << "\n";

    bool ok = (mismatches == 0);
    std::cout << (ok ? "SUCCESS: GPU matches CPU (1-bit GEMM)" : "FAIL: mismatch detected") << "\n";

    return 0;

}
