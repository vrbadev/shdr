// End an image pipeline with an SSBO, ready for another stage or host readback.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in; // Cover a 8 by 8 group of output coordinates.
layout(r32f, binding = 0) readonly uniform highp image2D input_image; // Read the r32f image at this image binding.
layout(std430, binding = 0) writeonly buffer Storage_output_values { // Write this stage's output elements through buffer binding 0.
    float output_values[]; // Store this stage's output elements.
}; // Finish this declaration.

void main() { // Copy one image pixel into its row-major buffer element.
    ivec2 position = ivec2(gl_GlobalInvocationID.xy); // Address the pixel assigned to this two-dimensional invocation.
    ivec2 dimensions = imageSize(input_image); // Read the bound image extent before accessing its pixels.
    if (position.x < dimensions.x && position.y < dimensions.y) { // Guard invocations outside the image in the partial final tile.
        uint i = uint(position.y * dimensions.x + position.x); // Flatten the pixel coordinate into the output buffer index.
        output_values[i] = imageLoad(input_image, position).x; // Copy the r32f red component into the corresponding buffer element.
    } // End this block.
} // End this block.
