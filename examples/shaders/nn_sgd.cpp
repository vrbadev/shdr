// Batch-reduce independent parameter gradients and perform an in-place SGD update.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 64) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Gradients { // Read sample-major derivatives for every trainable parameter through buffer binding 0.
    float gradients[]; // Store sample-major derivatives for every trainable parameter.
}; // Finish this declaration.
layout(std430, binding = 1) buffer Parameters { // Read and update the complete flat learned parameter vector through buffer binding 1.
    float parameters[]; // Store the complete flat learned parameter vector.
}; // Finish this declaration.
uniform uint batch_count; // Set the number of independent samples in this model batch.
uniform uint parameter_count; // Set the number of trainable values in each sample gradient record.
uniform float learning_rate; // Scale the averaged loss derivative before updating each parameter.
uniform float weight_decay; // Add an L2 parameter penalty to the averaged loss derivative.
void main() { // Average the batch gradients and update one trainable parameter.
    uint parameter = gl_GlobalInvocationID.x; // Assign one distinct trainable parameter to this invocation.
    if (parameter >= parameter_count) // Ignore padding outside the complete model vector.
        return; // Stop this invocation before performing another resource access.
    float gradient = 0.0; // Start the batch gradient reduction at zero.
    for (uint row = 0u; row < batch_count; ++row) // Read this parameter's derivative from every sample.
        gradient += gradients[row * parameter_count + parameter]; // Accumulate the sample-major derivative at this parameter index.
    parameters[parameter] -= // Update this parameter in place after the gradient stage has completed.
        learning_rate * (gradient / float(batch_count) + weight_decay * parameters[parameter]); // Update this parameter in place after the gradient stage has completed.
} // End this block.
