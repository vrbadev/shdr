#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

// Keep positive response peaks, deterministic index ordering breaks equal-value ties.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Inputs { // Read the preceding stage's input elements through buffer binding 0.
    float input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Outputs { // Write this stage's output elements through buffer binding 1.
    uint output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
uniform uint width; // Set the row width of the flattened image buffer.
uniform uint height; // Set the number of rows in the image buffer.
uniform float threshold; // Set the smallest response that can be accepted as a corner.

void main() { // Select this pixel only when it is a deterministic positive local maximum.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i >= width * height) // Guard the flattened image against padding invocations.
        return; // Stop this invocation before performing another resource access.
    output_values[i] = 0u; // Clear the output flag before checking this candidate.
    float value = input_values[i]; // Read the candidate corner response.
    if (value <= threshold) // Reject weak responses before inspecting the neighborhood.
        return; // Stop this invocation before performing another resource access.
    int x = int(i % width), y = int(i / width); // Recover the column and row from the flattened pixel index.
    for (int dy = -1; dy <= 1; ++dy) { // Visit the three neighboring rows.
        for (int dx = -1; dx <= 1; ++dx) { // Visit the three neighboring columns.
            int nx = clamp(x + dx, 0, int(width) - 1); // Clamp this neighbor column to the valid image.
            int ny = clamp(y + dy, 0, int(height) - 1); // Clamp this neighbor row to the valid image.
            uint neighbor_index = uint(ny) * width + uint(nx); // Flatten the neighbor coordinate for a deterministic tie break.
            float neighbor = input_values[neighbor_index]; // Read the neighboring corner response.
            if (neighbor > value || (neighbor == value && neighbor_index < i)) // Reject stronger neighbors and choose the smallest index for equal responses.
                return; // Stop this invocation before performing another resource access.
        } // End this block.
    } // End this block.
    output_values[i] = 1u; // Mark this surviving positive local maximum.
} // End this block.
