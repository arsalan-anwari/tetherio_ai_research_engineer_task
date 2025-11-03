#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

#include <tether_io/sanbox.hpp>

using namespace tether_io;

namespace {

auto make_case_label(data_domain domain, u32 M, u32 N, u32 K_bits) -> std::string {
    return to_string(domain) + "_" +
           std::to_string(M) + "x" + std::to_string(N) + "_" +
           std::to_string(K_bits) + "bit";
}

auto execute_case(data_domain domain, u32 M, u32 N, u32 K_bits) -> bool {
    const std::string case_label = make_case_label(domain, M, N, K_bits);

    sandbox<sandbox_algorithm::binmatmul, device_driver::vulkan_native> bench;
    auto result = bench.run(domain, M, N, K_bits);
    if (!result.has_value()) {
        std::cerr << "[binmatmul] " << case_label
                  << " failed: " << result.error() << "\n";
        return false;
    }

    const auto metrics = result.value();
    const auto expected_total = static_cast<usize>(M) * static_cast<usize>(N);
    if (metrics.total_size != expected_total) {
        std::cerr << "[binmatmul] " << case_label
                  << " unexpected total_size=" << metrics.total_size
                  << " (expected " << expected_total << ")\n";
        return false;
    }

    if (metrics.mismatches != 0 || metrics.max_abs_err != 0) {
        std::cerr << "[binmatmul] " << case_label
                  << " mismatches=" << metrics.mismatches
                  << " max_abs_err=" << metrics.max_abs_err << "\n";
        return false;
    }

    std::cout << "[binmatmul] " << case_label
              << " ok (M=" << M
              << ", N=" << N
              << ", K_bits=" << K_bits
              << ", total=" << metrics.total_size
              << ")" << std::endl;

    return true;
}

} // namespace

auto main() -> int {
    constexpr std::array data_domains{
        data_domain::full_range,
        data_domain::pm_one,
        data_domain::zero_one,
        data_domain::trinary};

    constexpr std::array<u32, 4> k_bit_values{16u, 32u, 48u, 64u};

    bool all_passed = true;
    usize total_cases = 0;

    for (auto domain : data_domains) {
        bool domain_passed = true;
        usize domain_cases = 0;

        for (u32 M = 8u; M <= 256u; M += 8u) {
            const u32 N = M;

            for (auto K_bits : k_bit_values) {
                domain_cases++;
                total_cases++;

                const bool ok = execute_case(domain, M, N, K_bits);
                domain_passed = ok && domain_passed;
                all_passed = ok && all_passed;
            }
        }

        if (domain_passed) {
            std::cout << "[binmatmul] domain=" << to_string(domain)
                      << " all cases passed (" << domain_cases << ")\n";
        } else {
            std::cerr << "[binmatmul] domain=" << to_string(domain)
                      << " detected failures (" << domain_cases << " total cases)\n";
        }
    }

    if (all_passed) {
        std::cout << "[binmatmul] completed " << total_cases << " combinations without error\n";
    } else {
        std::cerr << "[binmatmul] sandbox regression detected across "
                  << total_cases << " combinations\n";
    }

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
