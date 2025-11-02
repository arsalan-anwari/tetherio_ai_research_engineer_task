#pragma once

#ifdef ENABLE_LLAMA_CPP

#include <algorithm>
#include <expected>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include <llama.h>
#include <ggml-backend-reg.h>

#include <tether_io/config.hpp>
#include <tether_io/context.hpp>
#include <tether_io/algorithm.hpp>
#include <tether_io/types.hpp>

namespace tether_io::integration{

struct llama_vulkan_binmm_adapter{
    explicit llama_vulkan_binmm_adapter(application_config cfg)
        : config_(std::move(cfg)) {}

    inline auto init() -> std::expected<void, device_error> {
        auto res = ctx_.init(version<u32>{0, 1, 3, 0}, "llama_vulkan_binmm");
        if (!res.has_value()) return std::unexpected{ res.error() };
        res = ctx_.set_device(device_select::first_compute_capable);
        if (!res.has_value()) return std::unexpected{ res.error() };
        device_kernel_ = std::make_unique<
            algorithm<device_driver::vulkan_native, execution_method::sequenced>>(ctx_, config_);
        return {};
    }

    inline auto attach(ggml_backend_dev_t device) -> std::expected<void, device_error> {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        using fn_register = void (*)(
            ggml_backend_reg_t,
            enum ggml_op,
            ggml_backend_reg_can_t,
            ggml_backend_reg_compute_t,
            void *
        );
        
        auto* proc = reinterpret_cast<fn_register>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_node_handler")
        );

        if (!proc) return std::unexpected{ device_error::not_available };

        static backend_state state{};
        state.adapter = this;

        proc(reg, GGML_OP_MUL_MAT,
             [](ggml_tensor* node, void* user) -> bool {
                 return static_cast<backend_state*>(user)->adapter->can_handle(node);
             },
             [](ggml_tensor* node, void* user) -> enum ggml_status {
                 return static_cast<backend_state*>(user)->adapter->run_node(node);
             },
             &state);
        return {};
    }

    inline auto can_handle(const ggml_tensor* node) const -> bool {
        if (!node || node->op != GGML_OP_MUL_MAT) return false;
        const ggml_tensor* A = node->src[0];
        const ggml_tensor* B = node->src[1];
        return A && B && A->type == GGML_TYPE_F32 && B->type == GGML_TYPE_F32;
    }

    inline auto run_node(ggml_tensor* node) -> enum ggml_status {
        if (!can_handle(node)) return GGML_STATUS_FAILED;

        ggml_tensor* dst = node;
        ggml_tensor* A   = node->src[0];
        ggml_tensor* B   = node->src[1];

        const u32 n       = static_cast<u32>(dst->ne[0]);
        const u32 m       = static_cast<u32>(ggml_nrows(dst));
        const u32 k_bits  = static_cast<u32>(A->ne[0]);

        auto cap = ensure_capacity(m, n, k_bits);
        if (!cap.has_value()) return GGML_STATUS_FAILED;

        auto a_span = std::span<const f32>(static_cast<const f32*>(A->data), usize(m) * k_bits);
        auto b_span = std::span<const f32>(static_cast<const f32*>(B->data), usize(n) * k_bits);

        auto pack_a = cpu_tools_.f32_mat_to_packed_u32(matrix_order::row_major, a_span, m, k_bits);
        if (!pack_a.has_value()) return GGML_STATUS_FAILED;
        auto pack_b = cpu_tools_.f32_mat_to_packed_u32(matrix_order::col_major, b_span, n, k_bits);
        if (!pack_b.has_value()) return GGML_STATUS_FAILED;

        act_bits_ = std::move(pack_a.value());
        wt_bits_  = std::move(pack_b.value());

        if (!ctx_.upload(d_act_, std::span<const u32>(act_bits_)).has_value()) 
            return GGML_STATUS_FAILED;

        if (!ctx_.upload(d_wt_, std::span<const u32>(wt_bits_)).has_value())
            return GGML_STATUS_FAILED;

        auto limits = ctx_.limits();
        if (!limits.has_value()) return GGML_STATUS_FAILED;
        const auto max_local = limits.value().max_compute_work_group_size;

        auto choose_tile = [](u32 dim, u32 preferred, u32 max_dim) {
            u32 capped = std::min(preferred, max_dim);
            if (dim >= capped) return capped;
            if (dim >= 8) return 8u;
            if (dim >= 4) return 4u;
            if (dim >= 2) return 2u;
            return 1u;
        };
        auto ceil_div = [](u32 value, u32 tile) {
            return (value + tile - 1u) / tile;
        };

        const u32 local_x = choose_tile(n, 16u, max_local.x);
        const u32 local_y = choose_tile(m, 16u, max_local.y);
        const vec3<u32> local_size{local_x, local_y, 1u};
        const vec3<u32> grid_size{ceil_div(n, local_x), ceil_div(m, local_y), 1u};
        const u32 k_words = (k_bits + 31u) / 32u;

        auto res = device_kernel_->binmatmul(
            grid_size,
            local_size,
            {d_act_, d_wt_, d_out_},
            m, n, k_bits, k_words);
        if (!res.has_value()) return GGML_STATUS_FAILED;

        res = ctx_.wait_for_last_kernel(1'000'000'000ull);
        if (!res.has_value()) return GGML_STATUS_FAILED;

        auto out_view = std::span<i32>(out_accum_);
        auto download = ctx_.download(out_view, d_out_);
        if (!download.has_value()) return GGML_STATUS_FAILED;

        float* dst_data = static_cast<float*>(dst->data);
        std::transform(
            out_view.begin(), 
            out_view.begin() + usize(m) * n, 
            dst_data,
            [](i32 v) { return static_cast<float>(v); }
        );

        return GGML_STATUS_SUCCESS;
    }

private:
    struct backend_state {
        llama_vulkan_binmm_adapter* adapter{};
    };

    inline auto ensure_capacity(u32 m, u32 n, u32 k_bits) -> std::expected<void, device_error> {
        const u32 k_words = (k_bits + 31u) / 32u;
        const usize act_size = usize(m) * k_words;
        const usize wt_size  = usize(n) * k_words;
        const usize out_size = usize(m) * n;

        if (m == cached_m_ && n == cached_n_ && k_bits == cached_k_bits_) return {};

        if (act_bits_.size() < act_size) act_bits_.resize(act_size);
        if (wt_bits_.size()  < wt_size)  wt_bits_.resize(wt_size);
        if (out_accum_.size() < out_size) out_accum_.resize(out_size);

        if (!d_act_.buff_handle) {
            auto buf = ctx_.allocate(act_size * sizeof(u32));
            if (!buf.has_value()) return std::unexpected{ buf.error() };
            d_act_ = buf.value();
        }
        if (!d_wt_.buff_handle) {
            auto buf = ctx_.allocate(wt_size * sizeof(u32));
            if (!buf.has_value()) return std::unexpected{ buf.error() };
            d_wt_ = buf.value();
        }
        if (!d_out_.buff_handle) {
            auto buf = ctx_.allocate(out_size * sizeof(i32));
            if (!buf.has_value()) return std::unexpected{ buf.error() };
            d_out_ = buf.value();
        }

        cached_m_ = m;
        cached_n_ = n;
        cached_k_bits_ = k_bits;
        return {};
    }

    application_config config_;
    compute_context<device_driver::vulkan_native> ctx_;
    std::unique_ptr<algorithm<device_driver::vulkan_native, execution_method::sequenced>> device_kernel_;
    algorithm<device_driver::cpu_native, execution_method::standalone> cpu_tools_;

    device_buffer<device_driver::vulkan_native> d_act_{};
    device_buffer<device_driver::vulkan_native> d_wt_{};
    device_buffer<device_driver::vulkan_native> d_out_{};

    std::vector<u32> act_bits_;
    std::vector<u32> wt_bits_;
    std::vector<i32> out_accum_;

    u32 cached_m_{0};
    u32 cached_n_{0};
    u32 cached_k_bits_{0};
};

inline auto register_llama_vulkan_binmm_backend(llama_vulkan_binmm_adapter& adapter) -> ggml_backend_reg_t {
    ggml_backend_reg_t reg = ggml_backend_reg_init("vulkan-binmm");
    ggml_backend_reg_set_name(reg, "vulkan-binmm");
    ggml_backend_dev_t dev = ggml_backend_reg_new_device(reg, "vulkan-binmm-device");
    adapter.attach(dev);
    ggml_backend_device_register(dev);
    return reg;
}

inline auto run_llama_with_binmm(
    const char* model_path,
    const char* prompt,
    application_config cfg
) -> int {
    llama_backend_init();

    llama_vulkan_binmm_adapter adapter(std::move(cfg));
    if (!adapter.init().has_value()) return 1;

    auto reg = register_llama_vulkan_binmm_backend(adapter);

    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = true;

    llama_model* model = llama_load_model_from_file(model_path, model_params);
    if (!model) return 1;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) return 1;

    llama_batch batch = llama_batch_get_one(prompt, 0, LLAMA_FTYPE_ALL_F32);
    int decode_ok = llama_decode(ctx, batch);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_unload(reg);
    llama_backend_free();

    return decode_ok;
}

} // namespace tether_io::integration

#endif // ENABLE_LLAMA_CPP
