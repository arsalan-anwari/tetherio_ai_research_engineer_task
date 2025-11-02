#include <filesystem>
#include <iostream>
#include <thread>

#include <llama.h>

#include <tether_io/config.hpp>
#include <tether_io/integration/llama_vulkan_binmatmul.hpp>

auto main() -> int {
    llama_backend_init();

    std::filesystem::path resource = RESOURCE_DIR;
    auto cfg = tether_io::parse_application_settings(resource / "settings.json");
    if (!cfg.has_value()) {
        std::cerr << cfg.error() << "\n";
        return 1;
    }

    tether_io::integration::llama_vulkan_binmm_adapter adapter(cfg.value());
    if (!adapter.init().has_value()) {
        std::cerr << "could not init Vulkan binary matmul adapter\n";
        return 1;
    }

    // Register backend before loading the model so ggml can pick it
    auto reg = tether_io::integration::register_llama_vulkan_binmm_backend(adapter);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // let our adapter handle matmul
    model_params.use_extra_bufts = true; // enable weight repacking

    auto model_file = resource / "models" / "tiny-llama.gguf";

    auto model = llama_load_model_from_file(model_file.generic_string().c_str(), model_params);
    if (!model) {
        std::cerr << "failed to load model\n";
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_batch = 1;

    auto ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "failed to create context\n";
        return 1;
    }

    const char* prompt = "Binary matmul Vulkan integration test.";
    llama_batch batch = llama_batch_get_one(prompt, 0, LLAMA_FTYPE_MOSTLY_Q4_0);

    if (llama_decode(ctx, batch)) {
        std::cerr << "llama_decode failed\n";
        return 1;
    }

    std::cout << "llama.cpp ran with Vulkan 1-bit matmul backend\n";
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_unload(reg);
    llama_backend_free();

    return 0;
}
