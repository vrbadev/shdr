// Scalar-volume 3D convolution with dynamic coefficients and clamp-to-edge.
// Both samples and coefficients are flattened with x varying fastest, then y,
// then z. Axis-sized kernels also build separable X/Y/Z convolution pipelines.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in; // Cover a 4 by 4 by 4 group of output coordinates.
layout(std430, binding = 0) readonly buffer Input { // Read the preceding stage's input elements through buffer binding 0.
    float input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Output { // Write this stage's output elements through buffer binding 1.
    float output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
layout(std430, binding = 2) readonly buffer Coefficients { // Read runtime filter coefficients in x-major order through buffer binding 2.
    float coefficients[]; // Store runtime filter coefficients in x-major order.
}; // Finish this declaration.
uniform uvec3 dimensions; // Set the image or volume extent in x, y and optional z order.
uniform uvec3 kernel_size; // Set the number of coefficients along each convolution axis.

void main() { // Convolve one output voxel with the dynamic three-dimensional coefficient grid.
    uvec3 p = gl_GlobalInvocationID.xyz; // Select the output pixel or voxel assigned to this invocation.
    if (any(greaterThanEqual(p, dimensions))) // Guard every coordinate against the output image or volume extent.
        return; // Stop this invocation before performing another resource access.
    ivec3 center = ivec3(kernel_size) / 2; // Center the coefficient volume on the output voxel.
    float total = 0.0; // Start the scalar voxel accumulation at zero.
    for (uint kz = 0u; kz < kernel_size.z; ++kz) { // Visit each depth slice of the coefficient volume.
        for (uint ky = 0u; ky < kernel_size.y; ++ky) { // Visit each coefficient row in this depth slice.
            for (uint kx = 0u; kx < kernel_size.x; ++kx) { // Visit each coefficient column in this row.
                ivec3 q = // Prepare the input coordinate for the reversed convolution offset.
                    clamp(ivec3(p) + center - ivec3(kx, ky, kz), ivec3(0), ivec3(dimensions) - 1); // Reverse the coefficient offset and clamp it to the volume boundary.
                uint sample_index = // Prepare the flattened input voxel address.
                    uint(q.x) + dimensions.x * (uint(q.y) + dimensions.y * uint(q.z)); // Flatten x, then y, then z for the clamped input voxel.
                uint coefficient_index = kx + kernel_size.x * (ky + kernel_size.y * kz); // Flatten the matching coefficient coordinate in the same axis order.
                total += input_values[sample_index] * coefficients[coefficient_index]; // Accumulate this input voxel multiplied by its dynamic coefficient.
            } // End this block.
        } // End this block.
    } // End this block.
    output_values[p.x + dimensions.x * (p.y + dimensions.y * p.z)] = total; // Publish the convolution result at the original output voxel.
} // End this block.
