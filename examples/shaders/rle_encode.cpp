// Lossless tiled RLE: disjoint tile ranges produce one compact global word stream.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Values { // Read the original unsigned values to encode through buffer binding 0.
    uint values[]; // Store the original unsigned values to encode.
}; // Finish this declaration.
layout(std430, binding = 1) readonly buffer Offsets { // Read exclusive tile offsets followed by the complete payload size through buffer binding 1.
    uint offsets[]; // Store exclusive tile offsets followed by the complete payload size.
}; // Finish this declaration.
layout(std430, binding = 2) writeonly buffer Encoded { // Write alternating original values and unsigned run lengths through buffer binding 2.
    uint encoded[]; // Store alternating original values and unsigned run lengths.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.
uniform uint block_size; // Limit each independent encoder or decoder to this many original values.
void main() { // Write the lossless value and run-length pairs for one tile.
    uint block = gl_GlobalInvocationID.x; // Assign one independent input tile to this invocation.
    uint begin = block * block_size; // Locate the first original value in this tile.
    if (begin >= count) // Ignore workgroup padding after the final tile.
        return; // Stop this invocation before performing another resource access.
    uint end = min(begin + block_size, count); // Stop the last tile at the actual input length.
    uint write_position = offsets[block]; // Use the prefix stage's disjoint destination range for this tile.
    uint value = values[begin]; // Begin the first run with the tile's first original value.
    uint run = 1u; // Count the first sample in the current run.
    for (uint i = begin + 1u; i < end; ++i) { // Scan the remaining original values in this tile.
        if (values[i] == value) // Extend the current run when its value repeats.
            ++run; // Include this repeated value in the current run.
        else { // Emit the completed run and start a run for the newly encountered value.
            encoded[write_position++] = value; // Write the original value of the completed run.
            encoded[write_position++] = run; // Write the number of repetitions and advance to the next pair.
            value = values[i]; // Start the next run with this newly encountered value.
            run = 1u; // Count the first sample in the new run.
        } // End this block.
    } // End this block.
    encoded[write_position++] = value; // Write the original value of the completed run.
    encoded[write_position] = run; // Finish the final run in this tile with its repetition count.
} // End this block.
