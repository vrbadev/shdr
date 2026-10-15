// A fully connected 2 -> ReLU(8) -> sigmoid(1) classifier, one invocation per row.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 64) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Inputs { // Read the model's sample-major input features through buffer binding 0.
    float inputs[]; // Store the model's sample-major input features.
}; // Finish this declaration.
layout(std430, binding = 1) readonly buffer Parameters { // Read the complete flat learned parameter vector through buffer binding 1.
    float parameters[]; // Store the complete flat learned parameter vector.
}; // Finish this declaration.
layout(std430, binding = 2) writeonly buffer Predictions { // Write one predicted class probability per sample through buffer binding 2.
    float predictions[]; // Store one predicted class probability per sample.
}; // Finish this declaration.
uniform uint batch_count; // Set the number of independent samples in this model batch.
float logistic(float value) { // Evaluate the sigmoid with a sign-selected, numerically stable exponential.
    if (value >= 0.0) // Use the negative exponent when the input logit is nonnegative.
        return 1.0 / (1.0 + exp(-value)); // Convert a nonnegative logit into its sigmoid probability.
    float e = exp(value); // Use a nonpositive exponent for the negative logit branch.
    return e / (1.0 + e); // Convert a negative logit into its sigmoid probability.
} // End this block.
void main() { // Evaluate the fully connected binary classifier for one input row.
    uint row = gl_GlobalInvocationID.x; // Select the independent input sample assigned to this invocation.
    if (row >= batch_count) // Ignore invocations outside the sample batch.
        return; // Stop this invocation before performing another resource access.
    float x = inputs[2u * row], y = inputs[2u * row + 1u], logit = parameters[32]; // Read the two input features and initialize the output with its bias.
    for (uint neuron = 0u; neuron < 8u; ++neuron) { // Evaluate each of the eight hidden neurons.
        float activation = max(0.0, // Apply ReLU to this hidden neuron's weighted input and bias.
                               parameters[2u * neuron] * x + parameters[2u * neuron + 1u] * y + // Multiply the two input features by this neuron's learned weights.
                                   parameters[16u + neuron]); // Include this hidden neuron's bias before the ReLU.
        logit += activation * parameters[24u + neuron]; // Add this hidden activation through its learned output weight.
    } // End this block.
    predictions[row] = logistic(logit); // Convert the completed output logit into a binary class probability.
} // End this block.
