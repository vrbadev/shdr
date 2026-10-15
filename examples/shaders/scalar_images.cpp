// Three image bindings fit even the minimum GLES 3.1 compute image limit.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in; // Cover a 4 by 4 group of output coordinates.
layout(r32f, binding = 0) uniform highp image2D float_scalar; // Read and update the r32f image at this image binding.
layout(r32i, binding = 1) uniform highp iimage2D int_scalar; // Read and update the r32i image at this image binding.
layout(r32ui, binding = 2) uniform highp uimage2D uint_scalar; // Read and update the r32ui image at this image binding.

void main() { // Add deterministic offsets to scalar pixels of all three numeric types.
    ivec2 position = ivec2(gl_GlobalInvocationID.xy); // Address the pixel assigned to this two-dimensional invocation.
    ivec2 dimensions = imageSize(float_scalar); // Read the bound image extent before accessing its pixels.
    if (position.x >= dimensions.x || position.y >= dimensions.y) // Guard invocations outside the image in the partial final tile.
        return; // Stop this invocation before performing another resource access.
    imageStore(float_scalar, position, imageLoad(float_scalar, position) + vec4(0.5)); // Load the scalar pixel, add this type's test offset and store the same numeric type.
    imageStore(int_scalar, position, imageLoad(int_scalar, position) + ivec4(-2)); // Load the scalar pixel, add this type's test offset and store the same numeric type.
    imageStore(uint_scalar, position, imageLoad(uint_scalar, position) + uvec4(3u)); // Load the scalar pixel, add this type's test offset and store the same numeric type.
} // End this block.
