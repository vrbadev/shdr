// GLES requires separate readonly/writeonly declarations for RGBA images.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in; // Cover a 4 by 4 group of output coordinates.
layout(rgba32f, binding = 0) readonly uniform highp image2D input_image; // Read the rgba32f image at this image binding.
layout(rgba32f, binding = 1) writeonly uniform highp image2D output_image; // Write the rgba32f image at this image binding.

void main() { // Add deterministic per-channel offsets to one RGBA image pixel.
    ivec2 position = ivec2(gl_GlobalInvocationID.xy); // Address the pixel assigned to this two-dimensional invocation.
    ivec2 dimensions = imageSize(input_image); // Read the bound image extent before accessing its pixels.
    if (position.x >= dimensions.x || position.y >= dimensions.y) // Guard invocations outside the image in the partial final tile.
        return; // Stop this invocation before performing another resource access.
    imageStore( // Store the transformed RGBA pixel in the separate output image.
        output_image, position, imageLoad(input_image, position) + vec4(0.25, 0.5, 0.75, 1.0)); // Load all four input components and add the specified per-channel offsets.
} // End this block.
