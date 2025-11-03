#pragma once

#include <expected>
#include <span>
#include <random>

#include "../../types.hpp"

namespace tether_io{

// A is row-major [matrix_side x k_bits] with values in { -1, +1 } (or any float; >=0 -> bit 1)
// Output: row-major bit-pack along K => [matrix_side x k_words]
auto f32_mat_to_packed_u32_row_major_cpu_native_standalone(
    std::span<f32> in,
    u32 matrix_side,
    u32 k_bits
) -> std::expected<std::vector<u32>, device_error> {
    const u32 k_words = (k_bits + 31u) / 32u;

    // Expect exactly matrix_side * k_bits input scalars
    if(in.size() != static_cast<usize>(matrix_side) * static_cast<usize>(k_bits)){
        return std::unexpected{ device_error::launch_failed };
    }

    std::vector<u32> out;
    out.assign(static_cast<usize>(matrix_side) * k_words, 0u);

    for (u32 r = 0; r < matrix_side; ++r) {
        const usize row_off_in  = static_cast<usize>(r) * k_bits;
        const usize row_off_out = static_cast<usize>(r) * k_words;

        for (u32 k = 0; k < k_bits; ++k) {
            const u32 bit = (in[row_off_in + k] >= 0.0f) ? 1u : 0u; // +1 -> 1, -1 -> 0
            const u32 kw  = k >> 5;          // which u32 word
            const u32 off = k & 31u;         // bit position within the word
            out[row_off_out + kw] |= (bit << off);
        }
    }
    return out;
}


// B is row-major [k_bits x matrix_side] with values in { -1, +1 } (>=0 -> bit 1)
// We pack "columns as matrix_side": each original column becomes one packed row
// Output: [matrix_side x k_words]
auto f32_mat_to_packed_u32_col_major_cpu_native_standalone(
    std::span<f32> in,
    u32 matrix_side,
    u32 k_bits
) -> std::expected<std::vector<u32>, device_error> {
    const u32 k_words = (k_bits + 31u) / 32u;
    // Expect exactly k_bits * matrix_side input scalars
    if(in.size() != static_cast<usize>(k_bits) * static_cast<usize>(matrix_side)){
        return std::unexpected{ device_error::launch_failed };
    }

    std::vector<u32> out;
    out.assign(static_cast<usize>(matrix_side) * k_words, 0u);

    for (u32 c = 0; c < matrix_side; ++c) {
        const usize row_off_out = static_cast<usize>(c) * k_words;

        for (u32 k = 0; k < k_bits; ++k) {
            // Access B[k, c] where B is row-major [k_bits x matrix_side]
            const f32 v = in[static_cast<usize>(k) * matrix_side + c];
            const u32 bit = (v >= 0.0f) ? 1u : 0u;
            const u32 kw  = k >> 5;
            const u32 off = k & 31u;
            out[row_off_out + kw] |= (bit << off);
        }
    }
    return out;
}

// Create a random matrix with binary distribution as floating point representation (-1.f, 1.0f)
auto random_mat_binary_f32_1d_pm_one_dist_cpu_native_standalone(
    u32 rows, u32 cols, u32 seed
) -> std::expected<std::vector<f32>, device_error> {
    if (rows == 0 || cols == 0){
        return std::unexpected{ device_error::launch_failed };
    }

    std::vector<f32> out;

    std::mt19937 rng(seed); 
    std::uniform_int_distribution<i32> d(0,1);
    
    out.resize(static_cast<usize>(rows) * cols);

    for (usize i = 0; i < out.size(); ++i){
        out[i] = (d(rng) == 1 ? 1.0f : -1.0f);
    }

    return out;
}

auto random_mat_binary_f32_1d_zero_one_dist_cpu_native_standalone(
    u32 rows, u32 cols, u32 seed
) -> std::expected<std::vector<f32>, device_error> {
    if (rows == 0 || cols == 0){
        return std::unexpected{ device_error::launch_failed };
    }

    std::vector<f32> out;

    f32 lower_bound = 0.0f;
    f32 upper_bound = 1.0f;

    // Create random number generator
    std::random_device rd;  // Seed source (hardware)
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<f32> dist(lower_bound, upper_bound);

    out.resize(static_cast<usize>(rows) * cols);

    for (usize i = 0; i < out.size(); ++i){
        out[i] =  dist(gen);
    }


    return out;
}

auto random_mat_binary_f32_1d_full_range_dist_cpu_native_standalone(
    u32 rows, u32 cols, u32 seed
) -> std::expected<std::vector<f32>, device_error> {
    if (rows == 0 || cols == 0){
        return std::unexpected{ device_error::launch_failed };
    }

    std::vector<f32> out;

    f32 lower_bound = -1e6f;
    f32 upper_bound = 1e6f;

    // Create random number generator
    std::random_device rd;  // Seed source (hardware)
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<f32> dist(lower_bound, upper_bound);

    out.resize(static_cast<usize>(rows) * cols);

    for (usize i = 0; i < out.size(); ++i){
        out[i] =  dist(gen);
    }


    return out;
}



} // tether_io



