// Apply a rectangular, Hann, Hamming, or Blackman window before a block FFT.
#ifdef __cplusplus // Include the runtime only in the generated C++ build.
#include "shdr.h" // Use portable complex storage for real or complex input.
#endif // Finish the C++ runtime include guard.
layout(local_size_x = 64) in; // Each invocation windows one input value.
layout(std430, binding = 0) readonly buffer Input { // Read the original block.
    vec2 input_values[]; // Read the original block.
}; // Read the original block.
layout(std430, binding = 1) writeonly buffer Output { // Store the windowed complex output block.
    vec2 output_values[]; // Store the windowed complex output block.
}; // Store the windowed complex output block.
uniform uint block_length; // Number of values in each windowed block.
uniform uint channels; // Number of independent blocks.
uniform uint mode; // Select rectangular 0, Hann 1, Hamming 2, or Blackman 3.
void main() { // Both real and imaginary components get the same gain.
    uint index = gl_GlobalInvocationID.x; // Locate one value across all channels.
    if (index >= block_length * channels) // Ignore extra invocations.
        return; // Ignore extra invocations.
    float gain = 1.0; // Rectangular and single-sample windows use unity gain.
    if (block_length > 1u && mode != 0u) { // Symmetric windows include both endpoints.
        float phase = 6.28318530717958647692 * float(index % block_length) / // Sweep one full period over the block.
                      float(block_length - 1u); // Sweep one full period over the block.
        if (mode == 1u) // Hann endpoints reach zero.
            gain = 0.5 - 0.5 * cos(phase); // Hann endpoints reach zero.
        if (mode == 2u) // Hamming retains a small endpoint weight.
            gain = 0.54 - 0.46 * cos(phase); // Hamming retains a small endpoint weight.
        if (mode == 3u) // Blackman uses a second cosine to reduce sidelobes.
            gain = 0.42 - 0.5 * cos(phase) + // Blackman uses a second cosine to reduce sidelobes.
                   0.08 * cos(2.0 * phase); // Blackman uses a second cosine to reduce sidelobes.
    } // Window gains define the amplitudes represented by subsequent FFTs.
    output_values[index] = input_values[index] * gain; // Store the weighted complex value.
} // Link this output to the first FFT stage when spectral analysis needs a window.
