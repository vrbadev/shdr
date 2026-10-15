// Arrange complex values in the order needed by radix-two butterflies.
#ifdef __cplusplus // Include the runtime only in the generated C++ build.
#include "shdr.h" // Use the same resource types as the other examples.
#endif // Finish header inclusion for the generated C++ build.
layout(local_size_x = 64) in; // Each invocation arranges one complex value.
layout(std430, binding = 0) readonly buffer Input { // Read channel-major input.
    vec2 input_values[]; // Read channel-major input.
}; // Read channel-major input.
layout(std430, binding = 1) writeonly buffer Output { // Write to separate working storage.
    vec2 output_values[]; // Write to separate working storage.
}; // Write to separate working storage.
uniform uint block_length; // Number of complex values in one channel.
uniform uint channels; // Number of independent transforms.
uniform uint bits; // Number of binary digits in a channel index.
void main() { // Run the same index permutation on the CPU or GPU.
    uint index = gl_GlobalInvocationID.x; // Identify one output value across all channels.
    if (index >= block_length * channels) // Ignore extra invocations in the last group.
        return; // Ignore extra invocations in the last group.
    uint position = index % block_length; // Find the position within this channel.
    uint reversed = 0u; // A single-sample transform has only position zero.
    if (bits != 0u) // Keep only the reversed index bits.
        reversed = bitfieldReverse(position) >> (32u - bits); // Keep only the reversed index bits.
    output_values[index - position + reversed] = // Permute the positions within this signal channel.
        input_values[index]; // Permute the positions within this signal channel.
} // All later stages read this completed permutation.
