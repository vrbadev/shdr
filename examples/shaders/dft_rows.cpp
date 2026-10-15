// Educational direct DFT along each row of a row-major complex vec2 buffer.
// direction=-1 gives forward DFT, +1 with normalization=1/width gives inverse.
// Direct summation costs O(width^2 * height). Input/output must be distinct.
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
uniform uvec2 dimensions; // Set the image or volume extent in x, y and optional z order.
uniform float direction; // Choose a negative Fourier phase for forward transforms or a positive phase for inverse transforms.
uniform float normalization; // Scale the transform result, usually by the reciprocal axis length for an inverse transform.

void main() { // Compute one complex frequency bin along this row.
    uvec2 p = gl_GlobalInvocationID.xy; // Select the output pixel or voxel assigned to this invocation.
    if (any(greaterThanEqual(p, dimensions))) // Guard every coordinate against the output image or volume extent.
        return; // Stop this invocation before performing another resource access.
    const float tau = 6.28318530717958647692; // Use a complete turn in radians for the complex Fourier phase.
    vec2 total = vec2(0.0); // Accumulate the real and imaginary components independently.
    for (uint n = 0u; n < dimensions.x; ++n) { // Visit every sample along the transformed axis.
        float angle = direction * tau * float(p.x) * float(n) / float(dimensions.x); // Form this frequency bin's signed phase at the current sample.
        vec2 phase = vec2(cos(angle), sin(angle)); // Represent the unit complex phase as cosine and sine.
        vec2 value = input_values[n + dimensions.x * p.y]; // Read the original complex sample along this axis.
        total += vec2(value.x * phase.x - value.y * phase.y, value.x * phase.y + value.y * phase.x); // Multiply the sample by the unit complex phase and accumulate.
    } // End this block.
    output_values[p.x + dimensions.x * p.y] = total * normalization; // Apply the configured forward or inverse normalization to the completed bin.
} // End this block.
