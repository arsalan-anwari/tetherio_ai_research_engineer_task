#version 450
// 1D work; 64 threads per workgroup ⇒ ceil(100/64) = 2 groups
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// SSBO: data to write (set=0, binding=0)
layout(set = 0, binding = 0) buffer DataBuf {
    float data[];
};

// Push constants: the value to write, and (recommended) count for bounds checks
layout(push_constant) uniform PC {
    float value;
    uint  count;   // number of elements to fill (set to 100)
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.count) {
        data[i] = pc.value;
    }
}
