// Analytic binary cross-entropy gradients for every row and every ANN parameter.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 64) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Inputs { // Read the model's sample-major input features through buffer binding 0.
    float inputs[]; // Store the model's sample-major input features.
}; // Finish this declaration.
layout(std430, binding = 1) readonly buffer Targets { // Read one binary training label per sample through buffer binding 1.
    float targets[]; // Store one binary training label per sample.
}; // Finish this declaration.
layout(std430, binding = 2) readonly buffer Parameters { // Read the complete flat learned parameter vector through buffer binding 2.
    float parameters[]; // Store the complete flat learned parameter vector.
}; // Finish this declaration.
layout(std430, binding = 3) writeonly buffer Gradients { // Write sample-major derivatives for every trainable parameter through buffer binding 3.
    float gradients[]; // Store sample-major derivatives for every trainable parameter.
}; // Finish this declaration.
uniform uint batch_count; // Set the number of independent samples in this model batch.
float logistic(float value) { // Evaluate the sigmoid with a sign-selected, numerically stable exponential.
    if (value >= 0.0) // Use the negative exponent when the input logit is nonnegative.
        return 1.0 / (1.0 + exp(-value)); // Convert a nonnegative logit into its sigmoid probability.
    float e = exp(value); // Use a nonpositive exponent for the negative logit branch.
    return e / (1.0 + e); // Convert a negative logit into its sigmoid probability.
} // End this block.
void main() { // Differentiate one sample loss with respect to one ANN parameter.
    uint index = gl_GlobalInvocationID.x, row = index / 33u, parameter = index % 33u; // Select one sample-major derivative among all 33 ANN parameters.
    if (row >= batch_count) // Ignore invocations outside the sample batch.
        return; // Stop this invocation before performing another resource access.
    float x = inputs[2u * row], y = inputs[2u * row + 1u], logit = parameters[32]; // Read the two input features and initialize the output with its bias.
    float activations[8], preactivations[8]; // Retain hidden preactivations and ReLU outputs for the chain rule.
    for (uint neuron = 0u; neuron < 8u; ++neuron) { // Evaluate each of the eight hidden neurons.
        preactivations[neuron] = parameters[2u * neuron] * x + parameters[2u * neuron + 1u] * y + // Multiply the two input features by this neuron's learned weights.
                                 parameters[16u + neuron]; // Include this hidden neuron's bias before the ReLU.
        activations[neuron] = max(0.0, preactivations[neuron]); // Form the hidden neuron's weighted input before its nonlinearity.
        logit += activations[neuron] * parameters[24u + neuron]; // Keep positive preactivations and set the negative branch to zero.
    } // End this block.
    float delta = logistic(logit) - targets[row], gradient = delta; // Use prediction minus target as the cross-entropy derivative with respect to the logit.
    if (parameter < 16u) { // Differentiate one of the sixteen hidden input weights.
        uint neuron = parameter / 2u; // Find the hidden neuron owning this input weight.
        gradient = preactivations[neuron] > 0.0 // Form the hidden neuron's weighted input before its nonlinearity.
                       ? delta * parameters[24u + neuron] * (parameter % 2u == 0u ? x : y) // Multiply the output derivative by its output weight and the selected input feature.
                       : 0.0; // Use zero derivative for the inactive ReLU branch.
    } else if (parameter < 24u) { // Differentiate one of the eight hidden biases.
        uint neuron = parameter - 16u; // Find the hidden neuron owning this bias.
        gradient = preactivations[neuron] > 0.0 ? delta * parameters[24u + neuron] : 0.0; // Form the hidden neuron's weighted input before its nonlinearity.
    } else if (parameter < 32u) // Differentiate one of the eight output weights.
        gradient = delta * activations[parameter - 24u]; // Multiply the output logit derivative by this hidden activation.
    gradients[index] = gradient; // Store this sample's derivative for the following batch SGD stage.
} // End this block.
