// Two-channel 2D convolution, x/y can also hold a complex sample's real/imaginary
// parts. Scalar kernel coefficients are supplied at runtime, in row order.
// This is true convolution: coefficient offsets are reversed around the center.
// Out-of-image samples use clamp-to-edge. Input/output buffers must be distinct.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in; // Cover a 8 by 8 group of output coordinates.
layout(std430, binding = 0) readonly buffer Input { // Read the preceding stage's input elements through buffer binding 0.
    vec2 input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Output { // Write this stage's output elements through buffer binding 1.
    vec2 output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
layout(std430, binding = 2) readonly buffer Coefficients { // Read runtime filter coefficients in x-major order through buffer binding 2.
    float coefficients[]; // Store runtime filter coefficients in x-major order.
}; // Finish this declaration.
uniform uvec2 dimensions; // Set the image or volume extent in x, y and optional z order.
uniform uvec2 kernel_size; // Set the number of coefficients along each convolution axis.

void main() { // Convolve both channels at one output pixel with the dynamic coefficient grid.
    uvec2 p = gl_GlobalInvocationID.xy; // Select the output pixel or voxel assigned to this invocation.
    if (any(greaterThanEqual(p, dimensions))) // Guard every coordinate against the output image or volume extent.
        return; // Stop this invocation before performing another resource access.
    ivec2 center = ivec2(kernel_size) / 2; // Center odd or even coefficient grids using integer division.
    vec2 total = vec2(0.0); // Start the independent two-channel accumulation at zero.
    for (uint ky = 0u; ky < kernel_size.y; ++ky) { // Visit each row of the coefficient grid.
        for (uint kx = 0u; kx < kernel_size.x; ++kx) { // Visit each column of the coefficient grid.
            ivec2 q = clamp(ivec2(p) + center - ivec2(kx, ky), ivec2(0), ivec2(dimensions) - 1); // Reverse the coefficient offset for true convolution and clamp the sample coordinate.
            uint sample_index = uint(q.x) + dimensions.x * uint(q.y); // Flatten the clamped input coordinate with x varying fastest.
            total += input_values[sample_index] * coefficients[kx + kernel_size.x * ky]; // Multiply both input channels by this scalar coefficient and accumulate.
        } // End this block.
    } // End this block.
    output_values[p.x + dimensions.x * p.y] = total; // Write the two-channel result at this output coordinate.
} // End this block.
