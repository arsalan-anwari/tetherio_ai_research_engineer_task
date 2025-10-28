#include <llama.h>
#include <vulkan/vulkan.h>
#include <iostream>
#include <string>

static void log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    // Pipe all llama.cpp/ggml logs to stdout; you can filter if you like
    std::cout << text;
}

int main() {
    // optional: print header version you already had
    std::cout << "Vulkan header version: "
              << VK_API_VERSION_MAJOR(VK_HEADER_VERSION_COMPLETE) << "."
              << VK_API_VERSION_MINOR(VK_HEADER_VERSION_COMPLETE) << "."
              << VK_API_VERSION_PATCH(VK_HEADER_VERSION_COMPLETE) << "\n";

    // 1) install logger first (so Vulkan discovery messages are captured)
    llama_log_set(log_cb, /* user_data = */ nullptr);

    // 2) init backends
    llama_backend_init();

    // 3) load any small GGUF just to trigger backend selection
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 1;                  // offload ≥1 layer -> uses GPU backend if available
    // (Optionally pin a device via env: GGML_VULKAN_DEVICE=0/1/2 ...)

    const char* model_path = "C:\\Users\\arsalan.anwari_nscal\\Workspaces\\tether_io\\tetherio_ai_research_engineer_task\\res\\models\\tiny-llama.gguf"; // put any valid small model here
    llama_model * model = llama_load_model_from_file(model_path, mp);
    if (!model) {
        std::cerr << "Failed to load model\n";
        llama_backend_free();
        return 1;
    }

    // If Vulkan is enabled and chosen, you’ll see ggml_vulkan / "loaded Vulkan backend" lines above.

    llama_free_model(model);
    llama_backend_free();
}
