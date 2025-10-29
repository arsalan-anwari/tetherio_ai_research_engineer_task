// Minimal 1‑bit (binary) matrix multiplication on Vulkan + CPU reference.
// Single-file demo with runtime shader compilation via shaderc.

// What this does:
//  - Packs two matrices A (MxK), B (KxN) into 1‑bit along K using 32-bit words
//  - Vulkan compute shader performs C = A x B using XNOR + bitCount per (i,j)
//    dot = 2*matches - K  in the ±1 domain
//  - Compares GPU result to CPU reference and prints a small matrix
//
// Notes:
//  - This is a tiny, happy-path demo with minimal error checking.
//  - For clarity, both activations and weights are treated as 1-bit (±1).
//  - For integration with llama.cpp, you would adapt the weight layout and
//    dispatch this kernel inside ggml-vulkan when encountering a Q1 format.

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <random>
#include <limits>
#include <bit>  // for std::popcount on MSVC / C++20

// ======== Simple helpers ========

static const char* kShaderGLSL = R"(// 1-bit GEMM: C = A x B, both A and B are bit-packed along K into 32-bit words
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Buffers
layout(set = 0, binding = 0) readonly buffer A_buf { uint A_bits[]; };
layout(set = 0, binding = 1) readonly buffer B_buf { uint B_bits[]; };
layout(set = 0, binding = 2) writeonly buffer C_buf { int C_out[]; };

// Push constants for dimensions
layout(push_constant) uniform PushConsts {
    uint M;        // rows of A / C
    uint N;        // cols of B / C
    uint K_bits;   // common dimension in bits (not words)
    uint K_words;  // K_bits / 32 rounded up
} pc;

// Index helpers: A row-major packed [M x K_words], B column-major packed [N x K_words]
uint idxA(uint row, uint kw) { return row * pc.K_words + kw; }
uint idxB(uint col, uint kw) { return col * pc.K_words + kw; }
uint idxC(uint row, uint col) { return row * pc.N + col; }

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    if (row >= pc.M || col >= pc.N) return;

    uint matches = 0u;
    for (uint kw = 0u; kw < pc.K_words; ++kw) {
        uint a = A_bits[idxA(row, kw)];
        uint b = B_bits[idxB(col, kw)];
        uint xnor = ~(a ^ b);
        // Handle tail bits if K_bits is not a multiple of 32
        if (kw == pc.K_words - 1u) {
            uint valid = (pc.K_bits & 31u) == 0u ? 0xFFFFFFFFu : ((1u << (pc.K_bits & 31u)) - 1u);
            xnor &= valid;
        }
        matches += bitCount(xnor);
    }
    int dot = int(matches) * 2 - int(pc.K_bits);
    C_out[idxC(row, col)] = dot;
}
)";

struct VK {
    VkInstance instance{};
    VkPhysicalDevice pdev{};
    uint32_t queueFamily = 0;
    VkDevice dev{};
    VkQueue queue{};
    VkDescriptorSetLayout dsetLayout{};
    VkPipelineLayout pipeLayout{};
    VkPipeline pipeline{};
    VkDescriptorPool dpool{};
    VkCommandPool cpool{};
};

struct Buf { VkBuffer buf{}; VkDeviceMemory mem{}; size_t sz{}; };

uint32_t findMemoryType(VkPhysicalDevice pdev, uint32_t typeBits, VkMemoryPropertyFlags req) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & req) == req) return i;
    }
    throw std::runtime_error("No suitable memory type");
}

void createBuffer(VK& vk, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, Buf& out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk.dev, &bi, nullptr, &out.buf) != VK_SUCCESS) throw std::runtime_error("vkCreateBuffer failed");
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(vk.dev, out.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemoryType(vk.pdev, mr.memoryTypeBits, props);
    if (vkAllocateMemory(vk.dev, &ai, nullptr, &out.mem) != VK_SUCCESS) throw std::runtime_error("vkAllocateMemory failed");
    vkBindBufferMemory(vk.dev, out.buf, out.mem, 0);
    out.sz = size;
}

std::vector<uint32_t> compileGLSLtoSPV(const std::string& src, shaderc_shader_kind kind) {
    shaderc::Compiler comp; shaderc::CompileOptions opts; opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    auto res = comp.CompileGlslToSpv(src, kind, "binmatmul.comp");
    if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::string("shaderc error: ") + res.GetErrorMessage());
    }
    return {res.cbegin(), res.cend()};
}

void createCompute(VK& vk, const std::vector<uint32_t>& spv) {
    // Descriptor set: 3 storage buffers
    VkDescriptorSetLayoutBinding b[3]{};
    for (int i=0;i<3;++i){ b[i].binding=i; b[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount=1; b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dlci.bindingCount=3; dlci.pBindings=b;
    vkCreateDescriptorSetLayout(vk.dev, &dlci, nullptr, &vk.dsetLayout);

    VkPushConstantRange pcr{}; pcr.offset=0; pcr.size=sizeof(uint32_t)*4; pcr.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount=1; plci.pSetLayouts=&vk.dsetLayout; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    vkCreatePipelineLayout(vk.dev, &plci, nullptr, &vk.pipeLayout);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=spv.size()*sizeof(uint32_t); smci.pCode=spv.data();
    VkShaderModule sm; vkCreateShaderModule(vk.dev, &smci, nullptr, &sm);

    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sm; ss.pName="main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage=ss; cpci.layout=vk.pipeLayout;
    if (vkCreateComputePipelines(vk.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &vk.pipeline) != VK_SUCCESS) throw std::runtime_error("create pipeline failed");
    vkDestroyShaderModule(vk.dev, sm, nullptr);

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps; dpci.maxSets=1;
    vkCreateDescriptorPool(vk.dev, &dpci, nullptr, &vk.dpool);

    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci2.queueFamilyIndex=vk.queueFamily; cpci2.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(vk.dev, &cpci2, nullptr, &vk.cpool);
}

// Pack a float matrix (values assumed ±1) into bit-packed uint32 words along K
void packBitsRowMajorA(const std::vector<float>& A, uint32_t M, uint32_t K_bits, std::vector<uint32_t>& A_bits) {
    const uint32_t K_words = (K_bits + 31u) / 32u;
    A_bits.assign(size_t(M) * K_words, 0u);
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t k = 0; k < K_bits; ++k) {
            uint32_t bit = (A[size_t(r) * K_bits + k] >= 0.0f) ? 1u : 0u; // +1 -> 1, -1 -> 0
            uint32_t kw = k >> 5; uint32_t off = k & 31u;
            A_bits[size_t(r) * K_words + kw] |= (bit << off);
        }
    }
}

// Pack B as column-major bit rows so that each column of original B (KxN) becomes a row in packed space: [N x K_words]
void packBitsColMajorB(const std::vector<float>& B, uint32_t K_bits, uint32_t N, std::vector<uint32_t>& B_bits) {
    const uint32_t K_words = (K_bits + 31u) / 32u;
    B_bits.assign(size_t(N) * K_words, 0u);
    for (uint32_t c = 0; c < N; ++c) {
        for (uint32_t k = 0; k < K_bits; ++k) {
            uint32_t bit = (B[size_t(k) * N + c] >= 0.0f) ? 1u : 0u; // B is row-major [K x N]
            uint32_t kw = k >> 5; uint32_t off = k & 31u;
            B_bits[size_t(c) * K_words + kw] |= (bit << off);
        }
    }
}

// CPU reference: dot in ±1 domain using XNOR+popcount on packed
void cpuBinaryGEMM(const std::vector<uint32_t>& A_bits, const std::vector<uint32_t>& B_bits, uint32_t M, uint32_t N, uint32_t K_bits, std::vector<int>& C) {
    const uint32_t K_words = (K_bits + 31u) / 32u;
    C.assign(size_t(M) * N, 0);
    uint32_t tailMask = (K_bits & 31u) == 0u ? 0xFFFFFFFFu : ((1u << (K_bits & 31u)) - 1u);
    for (uint32_t r=0;r<M;++r){
        for (uint32_t c=0;c<N;++c){
            uint32_t matches = 0u;
            for (uint32_t kw=0; kw<K_words; ++kw){
                uint32_t a = A_bits[size_t(r) * K_words + kw];
                uint32_t b = B_bits[size_t(c) * K_words + kw];
                uint32_t x = ~(a ^ b);
                if (kw == K_words-1u) x &= tailMask;
                matches += std::popcount(x);
            }
            C[size_t(r)*N+c] = int(matches) * 2 - int(K_bits);
        }
    }
}

// Create a tiny random ±1 matrix
void makeRandomPM1(std::vector<float>& Mtx, uint32_t rows, uint32_t cols, uint32_t seed){
    std::mt19937 rng(seed); std::uniform_int_distribution<int> d(0,1);
    Mtx.resize(size_t(rows)*cols);
    for (auto &v : Mtx) v = d(rng) ? 1.0f : -1.0f;
}

int main(){
    // Problem size: small demo. You can tweak these.
    const uint32_t M = 8, K_bits = 64, N = 8; // K_bits is the shared dimension (# of features)
    const uint32_t K_words = (K_bits + 31u)/32u;

    // Generate test data (±1)
    std::vector<float> A(M * K_bits), B(K_bits * N);
    makeRandomPM1(A, M, K_bits, 123);
    makeRandomPM1(B, K_bits, N, 321);

    // Pack into bits
    std::vector<uint32_t> A_bits, B_bits;
    packBitsRowMajorA(A, M, K_bits, A_bits);
    packBitsColMajorB(B, K_bits, N, B_bits);

    // CPU reference
    std::vector<int> C_ref; cpuBinaryGEMM(A_bits, B_bits, M, N, K_bits, C_ref);

    // ===== Vulkan init (instance, device, queue) =====
    VK vk{};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1; app.pApplicationName = "binmatmul";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    if (vkCreateInstance(&ici, nullptr, &vk.instance) != VK_SUCCESS) { std::cerr << "vkCreateInstance failed\n"; return 1; }

    uint32_t pdCount=0; vkEnumeratePhysicalDevices(vk.instance, &pdCount, nullptr); if (!pdCount){ std::cerr<<"No physical devices\n"; return 1; }
    std::vector<VkPhysicalDevice> pds(pdCount); vkEnumeratePhysicalDevices(vk.instance, &pdCount, pds.data());

    // Choose first device with a compute queue
    for (auto pd : pds){
        uint32_t qfCount=0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfp(qfCount); vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfp.data());
        for (uint32_t i=0;i<qfCount;++i){ if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT){ vk.pdev=pd; vk.queueFamily=i; break; } }
        if (vk.pdev) break;
    }
    if (!vk.pdev){ std::cerr << "No compute-capable device found\n"; return 1; }

    float prio=1.0f; VkDeviceQueueCreateInfo dq{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; dq.queueFamilyIndex=vk.queueFamily; dq.queueCount=1; dq.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&dq;
    if (vkCreateDevice(vk.pdev, &dci, nullptr, &vk.dev)!=VK_SUCCESS){ std::cerr<<"vkCreateDevice failed\n"; return 1; }
    vkGetDeviceQueue(vk.dev, vk.queueFamily, 0, &vk.queue);

    // Compile shader
    auto spv = compileGLSLtoSPV(kShaderGLSL, shaderc_compute_shader);
    createCompute(vk, spv);

    // Buffers: A_bits, B_bits, C_out
    Buf bufA, bufB, bufC;
    VkDeviceSize szA = VkDeviceSize(A_bits.size()*sizeof(uint32_t));
    VkDeviceSize szB = VkDeviceSize(B_bits.size()*sizeof(uint32_t));
    VkDeviceSize szC = VkDeviceSize(size_t(M)*N*sizeof(int));
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    createBuffer(vk, szA, usage, props, bufA);
    createBuffer(vk, szB, usage, props, bufB);
    createBuffer(vk, szC, usage, props, bufC);

    // Upload A, B; zero C
    void* map=nullptr;
    vkMapMemory(vk.dev, bufA.mem, 0, szA, 0, &map); std::memcpy(map, A_bits.data(), (size_t)szA); vkUnmapMemory(vk.dev, bufA.mem);
    vkMapMemory(vk.dev, bufB.mem, 0, szB, 0, &map); std::memcpy(map, B_bits.data(), (size_t)szB); vkUnmapMemory(vk.dev, bufB.mem);
    vkMapMemory(vk.dev, bufC.mem, 0, szC, 0, &map); std::memset(map, 0, (size_t)szC); vkUnmapMemory(vk.dev, bufC.mem);

    // Descriptor set allocate & update
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool=vk.dpool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&vk.dsetLayout;
    VkDescriptorSet dset; vkAllocateDescriptorSets(vk.dev, &dsai, &dset);

    VkDescriptorBufferInfo dbiA{bufA.buf, 0, szA};
    VkDescriptorBufferInfo dbiB{bufB.buf, 0, szB};
    VkDescriptorBufferInfo dbiC{bufC.buf, 0, szC};
    VkWriteDescriptorSet writes[3]{};
    for (int i=0;i<3;++i){ writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet=dset; writes[i].dstBinding=i; writes[i].descriptorCount=1; writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; }
    writes[0].pBufferInfo=&dbiA; writes[1].pBufferInfo=&dbiB; writes[2].pBufferInfo=&dbiC;
    vkUpdateDescriptorSets(vk.dev, 3, writes, 0, nullptr);

    // Command buffer and dispatch
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=vk.cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(vk.dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipeLayout, 0, 1, &dset, 0, nullptr);

    struct Push { uint32_t M,N,K_bits,K_words; } push{M,N,K_bits,K_words};
    vkCmdPushConstants(cmd, vk.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t gx = (N + 7u)/8u; uint32_t gy = (M + 7u)/8u;
    vkCmdDispatch(cmd, gx, gy, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(vk.dev, &fci, nullptr, &fence);
    vkQueueSubmit(vk.queue, 1, &si, fence);
    vkWaitForFences(vk.dev, 1, &fence, VK_TRUE, 1000000000ull);
    vkDestroyFence(vk.dev, fence, nullptr);

    // Read back C
    std::vector<int> C_gpu(size_t(M)*N);
    vkMapMemory(vk.dev, bufC.mem, 0, szC, 0, &map); std::memcpy(C_gpu.data(), map, (size_t)szC); vkUnmapMemory(vk.dev, bufC.mem);

    // Compare
    int max_abs_err = 0; size_t mismatches = 0;
    for (size_t i=0;i<C_gpu.size();++i){ int e = std::abs(C_gpu[i] - C_ref[i]); if (e>max_abs_err) max_abs_err=e; if (e!=0) ++mismatches; }

    // Print a tiny view
    auto printMat = [&](const std::vector<int>& C){
        for (uint32_t r=0;r<M;++r){
            for (uint32_t c=0;c<N;++c){ std::cout << C[size_t(r)*N+c] << (c+1==N?'\n':'\t'); }
        }
    };

    std::cout << "CPU reference C:" << std::endl; printMat(C_ref);
    std::cout << "GPU Vulkan C:" << std::endl; printMat(C_gpu);
    std::cout << "Max abs error: " << max_abs_err << ", mismatches: " << mismatches << " / " << C_gpu.size() << "\n";

    bool ok = (mismatches == 0);
    std::cout << (ok ? "SUCCESS: GPU matches CPU (1-bit GEMM)" : "FAIL: mismatch detected") << "\n";

    // Cleanup
    vkDeviceWaitIdle(vk.dev);
    vkDestroyDescriptorPool(vk.dev, vk.dpool, nullptr);
    vkDestroyPipeline(vk.dev, vk.pipeline, nullptr);
    vkDestroyPipelineLayout(vk.dev, vk.pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(vk.dev, vk.dsetLayout, nullptr);
    vkDestroyBuffer(vk.dev, bufA.buf, nullptr); vkFreeMemory(vk.dev, bufA.mem, nullptr);
    vkDestroyBuffer(vk.dev, bufB.buf, nullptr); vkFreeMemory(vk.dev, bufB.mem, nullptr);
    vkDestroyBuffer(vk.dev, bufC.buf, nullptr); vkFreeMemory(vk.dev, bufC.mem, nullptr);
    vkDestroyCommandPool(vk.dev, vk.cpool, nullptr);
    vkDestroyDevice(vk.dev, nullptr);
    vkDestroyInstance(vk.instance, nullptr);

    // ===== Bonus stub: llama.cpp integration sketch =====
    // In ggml-vulkan, add a new op/quant type (e.g., GGML_TYPE_Q1X) storing bit-packed rows with K_words stride.
    // Dispatch this pipeline when weights are Q1X and inputs are binarized or pre-thresholded. The buffer bindings
    // align with A_bits (activations), B_bits (weights), C_out (int dot). Then apply per-row scales/bias on GPU or CPU.

    return ok ? 0 : 2;
}
