// Reconstruct each tile from its compact range, preserving every input sample bit.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Encoded { // Read alternating original values and unsigned run lengths through buffer binding 0.
    uint encoded[]; // Store alternating original values and unsigned run lengths.
}; // Finish this declaration.
layout(std430, binding = 1) readonly buffer Offsets { // Read exclusive tile offsets followed by the complete payload size through buffer binding 1.
    uint offsets[]; // Store exclusive tile offsets followed by the complete payload size.
}; // Finish this declaration.
layout(std430, binding = 2) writeonly buffer Restored { // Write the losslessly reconstructed unsigned values through buffer binding 2.
    uint restored[]; // Store the losslessly reconstructed unsigned values.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.
uniform uint block_size; // Limit each independent encoder or decoder to this many original values.
void main() { // Reconstruct one original tile from its encoded value and run-length pairs.
    uint block = gl_GlobalInvocationID.x; // Assign one independent output tile to this invocation.
    uint begin = block * block_size; // Locate this tile's first reconstructed value.
    if (begin >= count) // Ignore workgroup padding after the final tile.
        return; // Stop this invocation before performing another resource access.
    uint write_position = begin; // Write restored values into their original tile positions.
    for (uint i = offsets[block]; i < offsets[block + 1u]; i += 2u) { // Visit each encoded value and run-length pair in this tile's prefix range.
        uint value = encoded[i]; // Read the exact unsigned value recorded by the encoder.
        uint run = encoded[i + 1u]; // Read the number of repetitions of that value.
        for (uint j = 0u; j < run && write_position < count; ++j) // Expand the run while respecting the actual original input length.
            restored[write_position++] = value; // Restore one original value and advance to the next destination.
    } // End this block.
} // End this block.
