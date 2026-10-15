// A small deterministic reduction: one invocation reads the preceding stage.
// For large buffers, split the reduction into several independently run groups.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; // Group 1 independent invocations along x.
layout(std430, binding = 0) readonly buffer Storage_input_values { // Read the preceding stage's input elements through buffer binding 0.
    float input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Storage_output_values { // Write this stage's output elements through buffer binding 1.
    float output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.

void main() { // Reduce the entire input into one deterministic sum.
    float result = 0.0; // Start the ordered accumulation at zero.
    for (uint i = 0u; i < count; ++i) { // Visit every input in the same order on both backends.
        result += input_values[i]; // Add this input element to the running total.
    } // End this block.
    output_values[0] = result; // Publish the one-element result for the following stage or host.
} // End this block.
