#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

// Encode the symmetric gradient outer product as (dx*dx,dx*dy,dy*dy,0).
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Inputs { // Read the preceding stage's input elements through buffer binding 0.
    vec2 input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Outputs { // Write this stage's output elements through buffer binding 1.
    vec4 output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.

void main() { // Form one symmetric structure tensor from the two image derivatives.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i >= count) // Stop padding invocations before reading or writing an element.
        return; // Stop this invocation before performing another resource access.
    vec2 g = input_values[i]; // Read the horizontal and vertical image derivatives.
    output_values[i] = vec4(g.x * g.x, g.x * g.y, g.y * g.y, 0.0); // Pack dx squared, dx times dy and dy squared into the symmetric tensor.
} // End this block.
