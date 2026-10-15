// A second stage can use a different workgroup size and share buffer objects.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in; // Group 32 independent invocations along x.
layout(std430, binding = 0) readonly buffer Storage_input_values { // Read the preceding stage's input elements through buffer binding 0.
    float input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Storage_output_values { // Write this stage's output elements through buffer binding 1.
    float output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.
uniform float lower; // Set the smallest permitted output value.
uniform float upper; // Set the largest permitted output value.

void main() { // Clip one input element to the configured numeric interval.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i < count) { // Ignore padding invocations in the final workgroup.
        output_values[i] = clamp(input_values[i], lower, upper); // Keep the value inside the inclusive lower and upper bounds.
    } // End this block.
} // End this block.
