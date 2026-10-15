// Causal FIR over independent channel-major samples, with zero-prefix batch history.
#ifdef __cplusplus // Enable this block only for the indicated compilation environment.
#include "shdr.h" // Use the same header for CPU and GPU resource wiring.
#endif // Finish the conditional compilation block.
layout(local_size_x = 64) in; // Process up to 64 independent invocations in each work group.
layout(std430, binding = 0) readonly buffer Inputs { // Read channel-major input samples through this storage binding.
    float input_values[]; // Store input samples in channel-major order.
}; // Finish this shader storage declaration.
layout(std430, binding = 1) writeonly buffer Outputs { // Write one filtered sample to the matching output position.
    float output_values[]; // Store filtered samples in matching channel-major order.
}; // Finish this shader storage declaration.
layout(std430, binding = 2) readonly buffer Coefficients { // Read the shared filter coefficients from this storage binding.
    float coefficients[]; // Store the finite filter coefficients.
}; // Finish this shader storage declaration.
uniform uint count; // Receive the sample count for each channel in this block.
uniform uint channels; // Receive the number of independent signal channels.
uniform uint taps; // Receive the causal FIR coefficient count.
void main() { // Filter the single causal output sample assigned to this invocation.
    uint index = gl_GlobalInvocationID.x; // Find the sample handled by this invocation.
    if (index >= count * channels) // Ignore unused invocations outside the sample buffer.
        return; // Finish invocations beyond the configured signal range.
    uint position = index % count; // Find the sample position within its channel.
    float value = 0.0; // Start the scalar accumulator at zero.
    for (uint tap = 0u; tap < taps && tap <= position; ++tap) // Use only causal taps that lie inside this batch.
        value += coefficients[tap] * input_values[index - tap]; // Accumulate the next available causal FIR tap.
    output_values[index] = value; // Store the result at the matching sample position.
} // Finish the main operation.
