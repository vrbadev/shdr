// Decode blocks concurrently into disjoint compact output intervals, in input order.
#ifdef __cplusplus // Include the runtime only when compiling the CPU adapter.
#include "shdr.h" // Share the portable resource and math declarations.
#endif // Finish the C++ declaration include guard.
layout(local_size_x = 64) in; // Assign one independent block or event to each invocation.
layout(std430, binding = 0) readonly buffer Input { // Read the packed sensor stream.
    uint packed_words[]; // Read two little-endian 16-bit sensor words from each uint.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 1) readonly buffer Seeds { // Exchange the decoder state preceding each block.
    uvec4 seeds[]; // Store the two decoder-state records preceding each block.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 2) readonly buffer Offsets { // Read each block's exclusive output interval and the final total.
    uint offsets[]; // Read exclusive event offsets and the final total event count.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 3) writeonly buffer Events { // Access decoded records in their original event order.
    uvec4 events[]; // Use decoded pixel or trigger records with paired timestamp words.
}; // Finish this shader storage resource declaration.
uniform uint word_count; // Receive the number of valid 16-bit input words.
uniform uint block_size; // Receive the maximum input words handled by one block.

void main() { // Decode one seeded block into its exclusive ordered output interval.
    uint b = gl_GlobalInvocationID.x, begin = b * block_size; // Choose this block and its first packed input word.
    if (begin >= word_count) // Ignore invocations beyond the last input block.
        return; // Stop dispatch padding at the input-block boundary.
    uint end = min(begin + block_size, word_count), out_index = offsets[b]; // Bound this block and start writing at its exclusive event offset.
    uvec4 a = seeds[2u * b], z = seeds[2u * b + 1u]; // Load the decoder state established immediately before this block.
    for (uint i = begin; i < end; ++i) { // Interpret this block's words in their original stream order.
        uint word = (packed_words[i / 2u] >> ((i % 2u) * 16u)) & 65535u; // Extract the selected 16-bit word from its packed uint.
        uint type = word >> 12u, value = word & 4095u; // Separate the four-bit word type from its twelve-bit payload.
        if (type == 0u) // Handle a Y-coordinate and camera word.
            a.x = value; // Replace the Y-coordinate and camera state.
        else if (type == 3u) { // Handle a vector base and polarity word.
            a.y = value & 2047u; // Replace the eleven-bit vector X-coordinate.
            a.z = value >> 11u; // Replace the vector polarity bit.
        } else if (type == 6u) // Handle a low timestamp word.
            z.x = value; // Replace the low twelve timestamp bits.
        else if (type == 8u) { // Handle high timestamp bits and synchronization.
            if (z.z != 0u && a.w > value && a.w - value >= 4085u) // Detect a tolerated high-time rollover after synchronization.
                ++z.y; // Extend the timestamp by another 24-bit rollover.
            a.w = value; // Replace the high twelve timestamp bits.
            z.x = 0u; // Reset low time when a new high-time word arrives.
            z.z = 1u; // Allow subsequent events to carry synchronized timestamps.
        } else if (type == 2u || type == 4u || type == 5u || type == 10u) { // Decode absolute events, vector events or external triggers.
            uint time_low = (z.y << 24u) | (a.w << 12u) | z.x, time_high = z.y >> 8u; // Reconstruct the extended timestamp as low and high uint32 words.
            if (type == 2u && z.z != 0u) { // Emit a synchronized absolute pixel event.
                uint yp = (a.x & 2047u) | ((value >> 11u) << 11u) | ((a.x >> 11u) << 12u); // Pack eleven-bit Y, event polarity and camera into the coordinate metadata.
                events[out_index++] = uvec4(value & 2047u, yp, time_low, time_high); // Write the absolute X-coordinate, pixel metadata and complete timestamp.
            } else if (type == 10u && z.z != 0u) { // Emit a synchronized external trigger record.
                events[out_index++] = // Write a trigger record into this block's exclusive output interval.
                    uvec4(value >> 8u, 2147483648u | (value & 1u), time_low, time_high); // Store trigger identifier, trigger marker, edge polarity and timestamp.
            } else if (type == 4u || type == 5u) { // Expand a vector mask into individual pixel events.
                uint bits = type == 4u ? value : value & 255u; // Keep the twelve-bit or eight-bit event mask for this word type.
                if (z.z != 0u) { // Emit vector events only after timestamp synchronization.
                    uint yp = (a.x & 2047u) | (a.z << 11u) | ((a.x >> 11u) << 12u); // Pack Y, vector polarity and camera for all events in this mask.
                    // Visit only set bits, dense and sparse vector masks share the same path.
                    while (bits != 0u) { // Emit one event for each remaining set mask bit.
                        uint bit = uint(findLSB(bits)); // Select the lowest remaining event position to preserve vector order.
                        bits &= bits - 1u; // Clear the emitted mask bit before selecting the next set bit.
                        events[out_index++] = uvec4((a.y + bit) & 2047u, yp, time_low, time_high); // Write this vector pixel with eleven-bit X wrapping and complete time.
                    } // Finish this decoder or fitting branch.
                } // Finish this decoder or fitting branch.
                a.y = (a.y + (type == 4u ? 12u : 8u)) & 2047u; // Advance the vector base even when unsynchronized events were suppressed.
            } // Finish this decoder or fitting branch.
        } // Finish this decoder or fitting branch.
    } // Finish this decoder or fitting branch.
} // Finish this decoder or fitting branch.
