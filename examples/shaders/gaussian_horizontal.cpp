#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

// First pass of a separable Gaussian approximation: [1,4,6,4,1] / 16.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Inputs { // Read the preceding stage's input elements through buffer binding 0.
    float input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Outputs { // Write this stage's output elements through buffer binding 1.
    float output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
uniform uint width; // Set the row width of the flattened image buffer.
uniform uint height; // Set the number of rows in the image buffer.

// Clamp-to-edge makes the border rule explicit on both backends.
float sample_value(int x, int y) { // Read a grayscale neighbor with the shared clamp-to-edge border rule.
    x = clamp(x, 0, int(width) - 1); // Clamp the neighboring column to the closest valid image pixel.
    y = clamp(y, 0, int(height) - 1); // Clamp the neighboring row to the closest valid image pixel.
    return input_values[uint(y) * width + uint(x)]; // Read this clamped coordinate from the row-major input buffer.
} // End this block.

void main() { // Apply the five-tap Gaussian approximation along one image row.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i >= width * height) // Guard the flattened image against padding invocations.
        return; // Stop this invocation before performing another resource access.
    int x = int(i % width), y = int(i / width); // Recover the column and row from the flattened pixel index.
    output_values[i] = // Write this pixel's horizontal Gaussian approximation.
        (sample_value(x - 2, y) + 4.0 * sample_value(x - 1, y) + 6.0 * sample_value(x, y) + // Weight the first three neighbors by one, four and six.
         4.0 * sample_value(x + 1, y) + sample_value(x + 2, y)) / // Weight the final two neighbors by four and one.
        16.0; // Normalize by the sum of the five integer weights.
} // End this block.
