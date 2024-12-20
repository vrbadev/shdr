#ifdef CPU
#include "shdr.h"
#else
#version 320 es
#define MAIN_SHADER_FUNCTION void main()

layout(rgba8ui, binding = 0) readonly uniform highp uimage2D input_image2d;
layout(rgba8ui, binding = 1) writeonly uniform highp uimage2D output_image2d;
layout(std430, binding = 2) buffer kernel_ssbo { uint kernel_ssbo_data[]; };
#endif

MAIN_SHADER_FUNCTION
{
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size_in = imageSize(input_image2d);
    ivec2 size_out = imageSize(output_image2d);
    float kernel_size = sqrt(float(kernel_ssbo_data.length()));
    float res = 0.0f;
    int kernel_span, i, x, y;

    if (kernel_size != floor(kernel_size) || mod(kernel_size, 2.0f) != 1.0f) {
        return;
    }

    if (size_in.x != size_out.x || size_in.y != size_out.y) {
        return;
    }

    kernel_span = (int(kernel_size) - 1) / 2;
    if (pos.x >= kernel_span && pos.x < (size_in.x - kernel_span) && pos.y >= kernel_span && pos.y < (size_in.y - kernel_span)) {
        i = 0;
        for (y = -kernel_span; y <= kernel_span; y++) {
            for (x = -kernel_span; x <= kernel_span; x++) {
                res += float(imageLoad(input_image2d, pos + ivec2(x, y)).r) * kernel_ssbo_data[i];
                i++;
            }
        }
    }

    imageStore(output_image2d, pos, uvec4(uint(res), uint(res), uint(res), 1.0f));
}