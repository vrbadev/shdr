// Summarize independent EVT3 blocks. Two 16-bit sensor words share each uint.
#ifdef __cplusplus // Include the runtime only when compiling the CPU adapter.
#include "shdr.h" // Share the portable resource and math declarations.
#endif // Finish the C++ declaration include guard.
layout(local_size_x = 64) in; // Assign one independent block or event to each invocation.
layout(std430, binding = 0) readonly buffer Input { // Read the packed sensor stream.
    uint packed_words[]; // Read two little-endian 16-bit sensor words from each uint.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 1) writeonly buffer Summaries { // Exchange compact block summaries between decoder stages.
    uvec4 summaries[]; // Store three state and event-count records per input block.
}; // Finish this shader storage resource declaration.
uniform uint word_count; // Receive the number of valid 16-bit input words.
uniform uint block_size; // Receive the maximum input words handled by one block.

void main() { // Summarize this block using local state changes and event counts.
    uint b = gl_GlobalInvocationID.x; // Select the input block summarized by this invocation.
    uint begin = b * block_size; // Locate the first 16-bit word of this block.
    if (begin >= word_count) // Ignore invocations beyond the last input block.
        return; // Stop dispatch padding at the input-word boundary.
    uint end = min(begin + block_size, word_count); // Clamp the final block to the valid input word count.
    // Flags mark state fields replaced by this block, X otherwise adds to its seed.
    uint flags = 0u, y = 0u, x = 0u, p = 0u, hi = 0u, lo = 0u; // Track replaced state fields and the block's final pixel and time state.
    uint first_hi = 0u, wraps = 0u, total = 0u, after_sync = 0u; // Track the first high-time word, internal wraps and both event counts.
    for (uint i = begin; i < end; ++i) { // Interpret this block's words in their original stream order.
        uint word = (packed_words[i / 2u] >> ((i % 2u) * 16u)) & 65535u; // Extract the selected 16-bit word from its packed uint.
        uint type = word >> 12u, value = word & 4095u; // Separate the four-bit word type from its twelve-bit payload.
        uint emitted = 0u; // Initialize this word's emitted-event count to zero.
        if (type == 0u) { // Handle a Y-coordinate and camera-state word.
            y = value; // Remember the final Y-coordinate and camera bit.
            flags |= 1u; // Mark Y and camera state as replaced by this block.
        } else if (type == 3u) { // Handle a vector base coordinate and polarity word.
            x = value & 2047u; // Replace the vector base with the eleven-bit X coordinate.
            p = value >> 11u; // Remember the vector polarity bit.
            flags |= 2u; // Mark the new vector X and polarity as absolute state replacements.
        } else if (type == 4u || type == 5u) { // Handle a twelve-bit or eight-bit vector event mask.
            uint bits = type == 4u ? value : value & 255u; // Keep only the mask bits belonging to this vector word type.
            emitted = uint(bitCount(bits)); // Count one event for every set mask bit.
            x = (x + (type == 4u ? 12u : 8u)) & 2047u; // Advance the vector base by this mask's width with eleven-bit wrapping.
        } else if (type == 2u || type == 10u) // Count an absolute pixel event or external trigger word.
            emitted = 1u; // Count the single record represented by this word.
        else if (type == 6u) { // Handle a low twelve-bit timestamp word.
            lo = value; // Remember the final low timestamp bits.
            flags |= 8u; // Mark the low timestamp state as replaced.
        } else if (type == 8u) { // Handle high timestamp bits and synchronization.
            if ((flags & 4u) == 0u) // Record the first high-time word for cross-block wrap detection.
                first_hi = value; // Save the first high-time payload seen in this block.
            else if (hi > value && hi - value >= 4085u) // Recognize a high-time wrap using the EVT3 tolerated rollover range.
                ++wraps; // Count another 24-bit timestamp rollover inside this block.
            hi = value; // Remember the most recent high timestamp bits.
            lo = 0u; // Reset low timestamp bits whenever a high-time word arrives.
            flags |= 12u; // Mark both high and low timestamp fields as replaced.
        } // Finish this decoder or fitting branch.
        total += emitted; // Count all event-bearing words in this block.
        if ((flags & 4u) != 0u) // Use the locally synchronized count for an initially unsynchronized stream.
            after_sync += emitted; // Count events appearing after this block's first high-time word.
    } // Finish this decoder or fitting branch.
    summaries[3u * b] = uvec4(y, x, p, hi); // Publish final pixel, vector-polarity and high-time fields.
    summaries[3u * b + 1u] = uvec4(lo, wraps, first_hi, flags); // Publish final low time, internal wraps, first high time and replacement flags.
    summaries[3u * b + 2u] = uvec4(total, after_sync, 0u, 0u); // Publish total and newly synchronized event counts.
} // Finish this decoder or fitting branch.
