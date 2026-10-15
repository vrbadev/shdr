// Transposed direct-form II biquad, one sequential channel per invocation.
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
layout(std430, binding = 3) buffer State { // Read and update one recurrence pair per channel.
    vec2 state[]; // Keep z1 and z2 together for each channel.
}; // Finish this shader storage declaration.
uniform uint count; // Receive the sample count for each channel in this block.
uniform uint channels; // Receive the number of independent signal channels.
void main() { // Filter this channel's temporal block using its retained state.
    uint channel = gl_GlobalInvocationID.x; // Give this invocation one independent channel.
    if (channel >= channels) // Ignore unused invocations outside the channel count.
        return; // Finish invocations beyond the configured signal range.
    vec2 previous = state[channel]; // Load both recurrence values from the previous block.
    float z1 = previous.x, z2 = previous.y; // Keep the channel recurrence values in private registers.
    float b0 = coefficients[0], b1 = coefficients[1], b2 = coefficients[2]; // Load the three normalized feed-forward coefficients.
    float a1 = coefficients[3], a2 = coefficients[4]; // Load the two normalized feedback coefficients.
    for (uint position = 0u; position < count; ++position) { // Process the channel samples in temporal order.
        uint index = channel * count + position; // Find the sample handled by this invocation.
        float x = input_values[index], y = b0 * x + z1; // Read the current sample and compute its filtered output.
        z1 = b1 * x - a1 * y + z2; // Advance the first transposed recurrence value.
        z2 = b2 * x - a2 * y; // Advance the second transposed recurrence value.
        output_values[index] = y; // Store the result at the matching sample position.
    } // Finish this channel, sample or feature pass.
    state[channel] = vec2(z1, z2); // Keep both recurrence values for the next block.
} // Finish the main operation.
