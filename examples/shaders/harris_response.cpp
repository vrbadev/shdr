#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

// Smooth the structure tensor with a 3x3 [1,2,1] outer product, then evaluate Harris.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Inputs { // Read the preceding stage's input elements through buffer binding 0.
    vec4 input_values[]; // Store the preceding stage's input elements.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Outputs { // Write this stage's output elements through buffer binding 1.
    float output_values[]; // Store this stage's output elements.
}; // Finish this declaration.
uniform uint width; // Set the row width of the flattened image buffer.
uniform uint height; // Set the number of rows in the image buffer.
uniform float harris_k; // Control the trace penalty in the Harris corner response.

// Clamp the tensor neighborhood independently of the output response storage.
vec4 sample_tensor(int x, int y) { // Read a tensor neighbor with the shared clamp-to-edge border rule.
    x = clamp(x, 0, int(width) - 1); // Clamp the neighboring column to the closest valid image pixel.
    y = clamp(y, 0, int(height) - 1); // Clamp the neighboring row to the closest valid image pixel.
    return input_values[uint(y) * width + uint(x)]; // Read this clamped coordinate from the row-major input buffer.
} // End this block.

void main() { // Smooth one structure tensor and compute its Harris corner response.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i >= width * height) // Guard the flattened image against padding invocations.
        return; // Stop this invocation before performing another resource access.
    int x = int(i % width), y = int(i / width); // Recover the column and row from the flattened pixel index.
    vec4 tensor = vec4(0.0); // Start the weighted neighborhood tensor at zero.
    for (int dy = -1; dy <= 1; ++dy) { // Visit each neighboring tensor row.
        for (int dx = -1; dx <= 1; ++dx) { // Visit each neighboring tensor column.
            float weight_x = dx == 0 ? 2.0 : 1.0; // Give the center column twice the weight of either side column.
            float weight_y = dy == 0 ? 2.0 : 1.0; // Give the center row twice the weight of either side row.
            tensor += sample_tensor(x + dx, y + dy) * (weight_x * weight_y / 16.0); // Accumulate the normalized separable three-by-three tensor smoothing weight.
        } // End this block.
    } // End this block.
    float determinant_value = tensor.x * tensor.z - tensor.y * tensor.y; // Measure the product of the tensor eigenvalues through its determinant.
    float trace_value = tensor.x + tensor.z; // Measure the sum of the tensor eigenvalues through its trace.
    output_values[i] = determinant_value - harris_k * trace_value * trace_value; // Penalize a large trace relative to the determinant to distinguish corners from edges.
} // End this block.
