// A 3x3 image filter with explicit edge clamping, identical on CPU and GPU.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in; // Cover a 8 by 8 group of output coordinates.
layout(r32f, binding = 0) readonly uniform highp image2D input_image; // Read the r32f image at this image binding.
layout(r32f, binding = 1) writeonly uniform highp image2D output_image; // Write the r32f image at this image binding.

void main() { // Average the clamped three-by-three neighborhood of one image pixel.
    ivec2 position = ivec2(gl_GlobalInvocationID.xy); // Address the pixel assigned to this two-dimensional invocation.
    ivec2 dimensions = imageSize(input_image); // Read the bound image extent before accessing its pixels.
    if (position.x >= dimensions.x || position.y >= dimensions.y) { // Guard invocations outside the image in the partial final tile.
        return; // Stop this invocation before performing another resource access.
    } // End this block.
    float result = 0.0; // Start the nine-sample sum at zero.
    for (int y = -1; y <= 1; ++y) { // Visit the three neighboring rows.
        for (int x = -1; x <= 1; ++x) { // Visit the three neighboring columns.
            ivec2 sample_position = ivec2(clamp(position.x + x, 0, dimensions.x - 1), // Clamp the horizontal neighbor to the nearest valid image edge.
                                          clamp(position.y + y, 0, dimensions.y - 1)); // Clamp the vertical neighbor to the nearest valid image edge.
            result += imageLoad(input_image, sample_position).x; // Accumulate the red component of this clamped scalar image sample.
        } // End this block.
    } // End this block.
    imageStore(output_image, position, vec4(result / 9.0, 0.0, 0.0, 1.0)); // Divide the sum by nine and store the averaged scalar pixel.
} // End this block.
