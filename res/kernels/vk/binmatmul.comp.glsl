#version 450

layout(constant_id = 0) const uint LOCAL_SIZE_X = 8;
layout(constant_id = 1) const uint LOCAL_SIZE_Y = 8;
layout(constant_id = 2) const uint LOCAL_SIZE_Z = 1;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly buffer A_buf { uint A_bits[]; };
layout(set = 0, binding = 1) readonly buffer B_buf { uint B_bits[]; };
layout(set = 0, binding = 2) writeonly buffer C_buf { int C_out[]; };

layout(push_constant) uniform PushConsts {
    uint M;
    uint N;
    uint K_bits;
    uint K_words;
} pc;

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row >= pc.M || col >= pc.N)
        return;

    // Early out for degenerate case
    if (pc.K_words == 0u || pc.K_bits == 0u) {
        C_out[row * pc.N + col] = 0;
        return;
    }

    uint baseA = row * pc.K_words;
    uint baseB = col * pc.K_words;
    uint cIndex = row * pc.N + col;

    uint lastKw   = pc.K_words - 1u;
    uint tailBits = pc.K_bits & 31u;
    uint tailMask = (tailBits == 0u)
        ? 0xFFFFFFFFu
        : ((1u << tailBits) - 1u);

    uint matches = 0u;

    // Main loop over all full words except the last one
    for (uint kw = 0u; kw < lastKw; ++kw) {
        uint a = A_bits[baseA + kw];
        uint b = B_bits[baseB + kw];
        uint xnor = ~(a ^ b);
        matches += bitCount(xnor);
    }

    // Last word, with tail mask (or full mask if no tail)
    uint aLast = A_bits[baseA + lastKw];
    uint bLast = B_bits[baseB + lastKw];
    uint xnorLast = ~(aLast ^ bLast);
    xnorLast &= tailMask;
    matches += bitCount(xnorLast);

    int dot = int(matches) * 2 - int(pc.K_bits);
    C_out[cIndex] = dot;
}
