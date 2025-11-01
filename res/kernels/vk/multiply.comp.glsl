#version 450

layout(constant_id = 0) const uint LOCAL_SIZE_X = 64;
layout(constant_id = 1) const uint LOCAL_SIZE_Y = 1;
layout(constant_id = 2) const uint LOCAL_SIZE_Z = 1;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer DataBuf {
    float data[];
};

layout(push_constant) uniform PC {
    float factor;
    uint  count;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.count) {
        data[i] *= pc.factor;
    }
}
