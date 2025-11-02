#pragma once

#include <span>
#include <expected>

#include "../../types.hpp"

namespace tether_io{

auto binmatmul_cpu_native_standalone(
    std::span<const u32> a_bits,
    std::span<const u32> b_bits,
    u32 m, u32 n, u32 k_bits
) -> std::expected<std::vector<i32>, device_error> {
    const u32 k_words = (k_bits + 31u) / 32u;

    const usize a_needed = static_cast<usize>(m) * k_words; // A: [m x k_words]
    const usize b_needed = static_cast<usize>(n) * k_words; // B: [n x k_words], each original column becomes a row

    if (a_bits.size() != a_needed || b_bits.size() != b_needed) {
        return std::unexpected(device_error::launch_failed);
    }

    std::vector<i32> c;
    c.assign(static_cast<usize>(m) * n, 0);

    const u32 rem        = (k_bits & 31u);
    const u32 tail_mask  = (rem == 0u) ? 0xFFFFFFFFu : ((1u << rem) - 1u);

    for (u32 r = 0; r < m; ++r) {
        const usize a_row = static_cast<usize>(r) * k_words;

        for (u32 col = 0; col < n; ++col) {
            const usize b_row = static_cast<usize>(col) * k_words;

            u32 matches = 0u;
            for (u32 kw = 0; kw < k_words; ++kw) {
                u32 x = ~(a_bits[a_row + kw] ^ b_bits[b_row + kw]); // XNOR
                if (kw + 1u == k_words) x &= tail_mask;             // mask final partial word
                matches += std::popcount(x);
            }

            // Convert XNOR-popcount to {-1,+1} dot: 2*matches - k_bits
            c[static_cast<usize>(r) * n + col] =
                static_cast<i32>(matches) * 2 - static_cast<i32>(k_bits);
        }
    }

    return c;
}

} // tether_io
