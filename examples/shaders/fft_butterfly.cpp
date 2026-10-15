// Apply one radix-two FFT stage with precomputed complex phases.
#ifdef __cplusplus // Include the runtime only in the generated C++ build.
#include "shdr.h" // Use portable complex values stored as vec2.
#endif // Keep the shader source valid for a native GLSL compiler.
layout(local_size_x = 64) in; // Each invocation writes one butterfly result.
layout(std430, binding = 0) readonly buffer Input { // Read the previous stage.
    vec2 input_values[]; // Read the previous stage.
}; // Read the previous stage.
layout(std430, binding = 1) writeonly buffer Output { // Write the next stage.
    vec2 output_values[]; // Write the next stage.
}; // Write the next stage.
layout(std430, binding = 2) readonly buffer Phases { // Share one phase table between channels.
    vec2 twiddles[]; // Share one phase table between channels.
}; // Share one phase table between channels.
uniform uint block_length; // Number of complex values per transform.
uniform uint channels; // Number of independent transforms.
uniform uint span; // Number of values covered by this stage's butterfly groups.
uniform float direction; // Use +1 for forward phases and -1 for their conjugates.
uniform float normalization; // Apply 1/N only on the final inverse stage.
void main() { // Each stage writes independent positions in the next working buffer.
    uint index = gl_GlobalInvocationID.x; // Select one output value.
    if (index >= block_length * channels) // Ignore padding in the dispatch.
        return; // Ignore padding in the dispatch.
    uint position = index % block_length; // Locate the value within its channel.
    uint half_span = span / 2u; // Separate the two inputs of a butterfly.
    uint offset = position % half_span; // Select the phase within this butterfly group.
    uint base = // Find the group's first input.
        index - position + (position / span) * span + offset; // Find the group's first input.
    vec2 first = input_values[base]; // Read the even half of the group.
    vec2 second = input_values[base + half_span]; // Read the odd half of the group.
    vec2 phase = twiddles[offset * (block_length / span)]; // Reuse phases from the full-block table.
    phase.y *= direction; // Conjugation changes a forward FFT into an inverse FFT.
    vec2 rotated = vec2(second.x * phase.x - second.y * phase.y, // Multiply two complex values.
                        second.x * phase.y + second.y * phase.x); // Multiply two complex values.
    vec2 value = position % span < half_span // Choose the sum or difference output.
                     ? first + rotated // Choose the sum or difference output.
                     : first - rotated; // Choose the sum or difference output.
    output_values[index] = value * normalization; // Scale the final inverse result when requested.
} // A dispatch boundary makes this stage visible to the next stage.
