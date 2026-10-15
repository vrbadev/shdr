// Demonstrate the same resource moving from an SSBO into an r32f image.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in; // Cover a 8 by 8 group of output coordinates.
layout(std430, binding = 0) readonly buffer Storage_input_values { // Read the preceding stage's input elements through buffer binding 0.
    float input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(r32f, binding = 0) writeonly uniform highp image2D output_image; // Write the r32f image at this image binding.

void main() { // Copy one row-major buffer element into its corresponding image pixel.
    ivec2 position = ivec2(gl_GlobalInvocationID.xy); // Address the pixel assigned to this two-dimensional invocation.
    ivec2 dimensions = imageSize(output_image); // Read the bound image extent before accessing its pixels.
    if (position.x < dimensions.x && position.y < dimensions.y) { // Guard invocations outside the image in the partial final tile.
        uint i = uint(position.y * dimensions.x + position.x); // Flatten the pixel coordinate into the input buffer index.
        imageStore(output_image, position, vec4(input_values[i], 0.0, 0.0, 1.0)); // Copy the scalar value into the red component of the r32f image.
    } // End this block.
} // End this block.
