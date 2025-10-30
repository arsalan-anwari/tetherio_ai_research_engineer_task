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