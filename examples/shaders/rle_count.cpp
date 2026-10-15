// Count compact (value,run length) pairs independently within each fixed-size tile.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Values { // Read the original unsigned values to encode through buffer binding 0.
    uint values[]; // Store the original unsigned values to encode.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Sizes { // Write the size of each independently processed tile through buffer binding 1.
    uint sizes[]; // Store the size of each independently processed tile.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.
uniform uint block_size; // Limit each independent encoder or decoder to this many original values.
void main() { // Count the encoded payload size of one independent input tile.
    uint block = gl_GlobalInvocationID.x; // Assign one independent input tile to this invocation.
    uint begin = block * block_size; // Locate the first original value in this tile.
    if (begin >= count) // Ignore workgroup padding after the final tile.
        return; // Stop this invocation before performing another resource access.
    uint end = min(begin + block_size, count); // Stop the last tile at the actual input length.
    uint runs = 1u; // Count the first value as the start of one run.
    for (uint i = begin + 1u; i < end; ++i) // Inspect the remaining values within this tile.
        if (values[i] != values[i - 1u]) // Start another run whenever the original value changes.
            ++runs; // Count this transition as one additional encoded pair.
    sizes[block] = runs * 2u; // Reserve two payload words for each value and run-length pair.
} // End this block.
