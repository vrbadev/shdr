// Resolve inter-block dependencies using small summaries, retaining stream state.
#ifdef __cplusplus // Include the runtime only when compiling the CPU adapter.
#include "shdr.h" // Share the portable resource and math declarations.
#endif // Finish the C++ declaration include guard.
layout(local_size_x = 1) in; // Use one invocation for this ordered stream-state operation.
layout(std430, binding = 0) readonly buffer Summaries { // Exchange compact block summaries between decoder stages.
    uvec4 summaries[]; // Store three state and event-count records per input block.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 1) writeonly buffer Seeds { // Exchange the decoder state preceding each block.
    uvec4 seeds[]; // Store the two decoder-state records preceding each block.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 2) writeonly buffer Sizes { // Write event counts used to allocate exclusive output intervals.
    uint sizes[]; // Store the synchronized event count of each input block.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 3) buffer State { // Access decoder state retained across input chunks.
    uvec4 stream_state[]; // Retain pixel, vector and timestamp state between input chunks.
}; // Finish this shader storage resource declaration.
uniform uint blocks; // Receive the number of summarized input blocks.

void main() { // Resolve block seeds in order and advance persistent decoder state.
    // a=(Y+camera, vector X, vector polarity, high time), z=(low time, wraps, synchronized, reserved).
    uvec4 a = stream_state[0], z = stream_state[1]; // Load pixel and high-time state together with low time, wraps and synchronization.
    for (uint b = 0u; b < blocks; ++b) { // Carry each small summary into the following block's initial state.
        seeds[2u * b] = a; // Save the pixel and high-time state preceding this block.
        seeds[2u * b + 1u] = z; // Save low time, rollover count and synchronization preceding this block.
        uvec4 last = summaries[3u * b], meta = summaries[3u * b + 1u], // Read the final state fields and replacement metadata for this block.
              count = summaries[3u * b + 2u]; // Read the total and newly synchronized event counts.
        sizes[b] = z.z != 0u ? count.x : count.y; // Include all events only when the stream was already synchronized.
        if ((meta.w & 1u) != 0u) // Replace Y and camera state only if the block supplied them.
            a.x = last.x; // Carry the final Y-coordinate and camera bit to later blocks.
        if ((meta.w & 2u) != 0u) { // Replace vector state when this block supplied a new vector base.
            a.y = last.y; // Carry the final vector X-coordinate to later blocks.
            a.z = last.z; // Carry the final vector polarity to later blocks.
        } else // Preserve the old vector base when this block only advanced mask positions.
            a.y = (a.y + last.y) & 2047u; // Apply the block's net vector-base advance with eleven-bit wrapping.
        if ((meta.w & 4u) != 0u) { // Update high time and synchronization when this block supplied high-time words.
            uint cross = (z.z != 0u && a.w > meta.z && a.w - meta.z >= 4085u) ? 1u : 0u; // Detect a timestamp rollover between the preceding state and this block's first high time.
            z.y += meta.y + cross; // Accumulate internal and cross-block 24-bit timestamp rollovers.
            a.w = last.w; // Carry the last high timestamp payload forward.
            z.z = 1u; // Mark the stream as synchronized after receiving high time.
        } // Finish this decoder or fitting branch.
        if ((meta.w & 8u) != 0u) // Replace low time only when this block changed timestamp state.
            z.x = meta.x; // Carry the final low timestamp payload forward.
    } // Finish this decoder or fitting branch.
    stream_state[0] = a; // Save pixel, vector and high-time state for the next input chunk.
    stream_state[1] = z; // Save low time, rollover count and synchronization for the next chunk.
} // Finish this decoder or fitting branch.
