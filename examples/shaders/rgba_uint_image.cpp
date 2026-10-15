// Apply unsigned integer image arithmetic directly to stored channel values.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in; // Cover a 4 by 4 group of output coordinates.
layout(rgba32ui, binding = 0) readonly uniform highp uimage2D input_image; // Read the rgba32ui image at this image binding.
layout(rgba32ui, binding = 1) writeonly uniform highp uimage2D output_image; // Write the rgba32ui image at this image binding.

void main() { // Add deterministic per-channel offsets to one RGBA image pixel.
    ivec2 position = ivec2(gl_GlobalInvocationID.xy); // Address the pixel assigned to this two-dimensional invocation.
    ivec2 dimensions = imageSize(input_image); // Read the bound image extent before accessing its pixels.
    if (position.x >= dimensions.x || position.y >= dimensions.y) // Guard invocations outside the image in the partial final tile.
        return; // Stop this invocation before performing another resource access.
    imageStore(output_image, position, imageLoad(input_image, position) + uvec4(1u, 2u, 3u, 4u)); // Store the transformed RGBA pixel in the separate output image.
} // End this block.
