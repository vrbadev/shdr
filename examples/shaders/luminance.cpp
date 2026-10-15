// Linear RGB to Rec.709 luminance. Transfer-function decoding belongs upstream.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Colors { // Read normalized linear RGBA colors through buffer binding 0.
    vec4 colors[]; // Store normalized linear RGBA colors.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Gray { // Write one Rec.709 luminance value per pixel through buffer binding 1.
    float gray[]; // Store one Rec.709 luminance value per pixel.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.
void main() { // Convert one linear RGB color to Rec.709 luminance.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i < count) // Ignore padding invocations in the final workgroup.
        gray[i] = dot(colors[i].rgb, vec3(0.2126, 0.7152, 0.0722)); // Weight linear red, green and blue using the Rec.709 luminance coefficients.
} // End this block.
