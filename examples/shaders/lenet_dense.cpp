// Apply a learned dense layer with an optional ReLU activation.
#ifdef __cplusplus // Include the runtime only when building the CPU adapter.
#include "shdr.h" // Share resource and math types with the application.
#endif // Finish header inclusion for the generated C++ build.
layout(local_size_x = 64) in; // Give each invocation one dense output feature.
layout(std430, binding = 0) readonly buffer Inputs { // Read the complete preceding feature vector.
    float input_values[]; // Store one contiguous feature vector per image.
}; // Finish the input resource declaration.
layout(std430, binding = 1) readonly buffer Parameters { // Read flattened model weights and biases.
    float parameters[]; // Match the offline model parameter order.
}; // Finish the parameter resource declaration.
layout(std430, // Use tightly packed shader storage for the following port.
       binding = 2) writeonly buffer Outputs { // Write this layer into distinct feature storage.
    float output_values[]; // Keep one contiguous output vector per image.
}; // Finish the output resource declaration.
uniform uint batch_count; // Receive the number of independent images.
uniform uint input_features; // Receive the preceding feature-vector length.
uniform uint output_features; // Receive the learned output-vector length.
uniform uint weight_offset; // Receive the first dense weight in the flat model.
uniform uint bias_offset; // Receive the first dense bias in the flat model.
uniform uint activation; // Select zero for logits and one for ReLU features.
void main() { // Compute the dense output assigned to this invocation.
    uint index = gl_GlobalInvocationID.x; // Locate this flattened batch and output feature.
    if (index >= batch_count * output_features) // Ignore excess invocations beyond the dense output.
        return; // Ignore unused invocations after the output vector.
    uint row = index / output_features, // Decode the image index and its output feature index.
         column = index % output_features; // Recover the image and output feature indices.
    float value = parameters[bias_offset + column]; // Start with this learned output bias.
    for (uint source = 0u; source < input_features; // Visit every component of this image's input feature vector.
         ++source) { // Visit every feature contributing to this output.
        value += // Accumulate one learned projection term.
            parameters[weight_offset + column * input_features + source] * // Multiply one row-major learned weight by its source feature.
            input_values[row * input_features + source]; // Accumulate one learned projection term.
    } // Finish the dot product for this output feature.
    output_values[index] = // Store the completed feature with the chosen activation.
        activation != 0u ? max(value, 0.0) : value; // Apply ReLU only for the hidden dense layers.
} // Finish this dense-layer invocation.
