#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

// Normalized Sobel gradients, stored as (dx,dy) in one std430 vec2 element.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Inputs { // Read the preceding stage's input elements through buffer binding 0.
    float input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Outputs { // Write this stage's output elements through buffer binding 1.
    vec2 output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
uniform uint width; // Set the row width of the flattened image buffer.
uniform uint height; // Set the number of rows in the image buffer.

// Neighborhood reads address the immutable blurred input, with clamped borders.
float sample_value(int x, int y) { // Read a grayscale neighbor with the shared clamp-to-edge border rule.
    x = clamp(x, 0, int(width) - 1); // Clamp the neighboring column to the closest valid image pixel.
    y = clamp(y, 0, int(height) - 1); // Clamp the neighboring row to the closest valid image pixel.
    return input_values[uint(y) * width + uint(x)]; // Read this clamped coordinate from the row-major input buffer.
} // End this block.

void main() { // Compute the horizontal and vertical Sobel derivatives at one pixel.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i >= width * height) // Guard the flattened image against padding invocations.
        return; // Stop this invocation before performing another resource access.
    int x = int(i % width), y = int(i / width); // Recover the column and row from the flattened pixel index.
    float dx = // Begin the horizontal derivative with the right-minus-left Sobel stencil.
        (sample_value(x + 1, y - 1) + 2.0 * sample_value(x + 1, y) + sample_value(x + 1, y + 1) - // Weight the right neighboring column by one, two and one.
         sample_value(x - 1, y - 1) - 2.0 * sample_value(x - 1, y) - sample_value(x - 1, y + 1)) / // Subtract the matching weighted left neighboring column.
        8.0; // Normalize the stencil by its total absolute weight.
    float dy = // Begin the vertical derivative with the bottom-minus-top Sobel stencil.
        (sample_value(x - 1, y + 1) + 2.0 * sample_value(x, y + 1) + sample_value(x + 1, y + 1) - // Weight the bottom neighboring row by one, two and one.
         sample_value(x - 1, y - 1) - 2.0 * sample_value(x, y - 1) - sample_value(x + 1, y - 1)) / // Subtract the matching weighted top neighboring row.
        8.0; // Normalize the stencil by its total absolute weight.
    output_values[i] = vec2(dx, dy); // Store the horizontal and vertical derivatives together for the tensor stage.
} // End this block.
