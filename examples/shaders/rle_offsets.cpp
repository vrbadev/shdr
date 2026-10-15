// Serial prefix stage for a small tile table, independent encoders use these offsets.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; // Group 1 independent invocations along x.
layout(std430, binding = 0) readonly buffer Sizes { // Read the size of each independently processed tile through buffer binding 0.
    uint sizes[]; // Store the size of each independently processed tile.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Offsets { // Write exclusive tile offsets followed by the complete payload size through buffer binding 1.
    uint offsets[]; // Store exclusive tile offsets followed by the complete payload size.
}; // Finish this declaration.
uniform uint blocks; // Set the number of tile sizes to include in the prefix calculation.
void main() { // Compute exclusive cumulative offsets for the input tile sizes.
    uint sum = 0u; // Start the cumulative size at zero.
    offsets[0] = 0u; // Place the first tile at the beginning of the destination.
    for (uint b = 0u; b < blocks; ++b) { // Visit every tile size in original order.
        sum += sizes[b]; // Add this tile's size in the caller's selected units.
        offsets[b + 1u] = sum; // Write the next tile's exclusive start and the final total after the last tile.
    } // End this block.
} // End this block.
