// Apply a scale and offset to a buffer. The build embeds this native GLSL and generates its C++11 CPU adapter.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Storage_input_values { // Read the preceding stage's input elements through buffer binding 0.
    float input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Storage_output_values { // Write this stage's output elements through buffer binding 1.
    float output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.
uniform float scale; // Multiply each input value by this scalar gain.
uniform float offset; // Add this bias after scaling the input.

void main() { // Apply the configured scale and offset to one input element.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i < count) { // Ignore padding invocations in the final workgroup.
        output_values[i] = input_values[i] * scale + offset; // Apply the gain and then add the bias, with one independent output per invocation.
    } // End this block.
} // End this block.
