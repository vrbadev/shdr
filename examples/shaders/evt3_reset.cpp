// Reset state inside a pipeline when independent batches or reproducible timings are needed.
#ifdef __cplusplus // Include the runtime only when compiling the CPU adapter.
#include "shdr.h" // Share the portable resource and math declarations.
#endif // Finish the C++ declaration include guard.
layout(local_size_x = 1) in; // Use one invocation for this ordered stream-state operation.
layout(std430, binding = 0) writeonly buffer State { // Access decoder state retained across input chunks.
    uvec4 stream_state[]; // Retain pixel, vector and timestamp state between input chunks.
}; // Finish this shader storage resource declaration.
void main() { // Initialize decoder state before processing an independent event stream.
    stream_state[0] = uvec4(0u); // Clear Y, camera, vector X, vector polarity and high timestamp bits.
    stream_state[1] = uvec4(0u); // Clear low time, rollover count and the synchronization flag.
} // Finish this decoder or fitting branch.
