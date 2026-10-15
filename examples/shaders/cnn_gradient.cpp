// Train convolution coefficients, biases, and the classifier head with analytic gradients.
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
void main() { // Differentiate one image loss with respect to one CNN parameter.
    uint index = gl_GlobalInvocationID.x, row = index / 23u, parameter = index % 23u; // Select one sample-major derivative among all 23 CNN parameters.
    if (row >= batch_count) // Ignore invocations outside the sample batch.
        return; // Stop this invocation before performing another resource access.
    float logit = parameters[22], derivative = parameter == 22u ? 1.0 : 0.0; // Initialize the output bias and its derivative for this parameter.
    for (uint channel = 0u; channel < 2u; ++channel) { // Evaluate each of the two learned convolution filters.
        float pooled = 0.0, pooled_derivative = 0.0; // Start the mean feature and its derivative at zero.
        for (uint y = 0u; y < 6u; ++y) // Visit the six valid output rows of the three-by-three filter.
            for (uint x = 0u; x < 6u; ++x) { // Visit the six valid output columns of this filter.
                float activation = parameters[18u + channel]; // Start this convolution window with the filter's learned bias.
                for (uint ky = 0u; ky < 3u; ++ky) // Visit each row of the three-by-three learned kernel.
                    for (uint kx = 0u; kx < 3u; ++kx) // Visit each column of the three-by-three learned kernel.
                        activation += parameters[9u * channel + 3u * ky + kx] * // Multiply this kernel coefficient by its matching input image sample.
                                      inputs[64u * row + 8u * (y + ky) + x + kx]; // Read the sample from this image's flattened eight-by-eight plane.
                if (activation > 0.0) { // Pass feature derivatives only through the active ReLU branch.
                    pooled += activation / 36.0; // Accumulate the active convolution feature in the global mean.
                    if (parameter < 18u && parameter / 9u == channel) { // Differentiate a coefficient only when it belongs to this filter.
                        uint coefficient = parameter % 9u; // Locate the selected three-by-three coefficient within its filter.
                        pooled_derivative += // Accumulate this kernel coefficient's contribution to the pooled derivative.
                            inputs[64u * row + 8u * (y + coefficient / 3u) + x + coefficient % 3u] / // Read the sample from this image's flattened eight-by-eight plane.
                            36.0; // Normalize the derivative by the valid six-by-six output area.
                    } else if (parameter == 18u + channel) // Handle the derivative of this filter's additive bias.
                        pooled_derivative += 1.0 / 36.0; // Normalize the derivative by the valid six-by-six output area.
                } // End this block.
            } // End this block.
        logit += parameters[20u + channel] * pooled; // Connect this mean convolution feature through its learned head weight.
        derivative += parameters[20u + channel] * pooled_derivative; // Propagate the pooled feature derivative through the head weight.
        if (parameter == 20u + channel) // Handle the derivative of the head weight for this filter.
            derivative += pooled; // The head weight derivative is this filter's pooled feature.
    } // End this block.
    gradients[index] = (logistic(logit) - targets[row]) * derivative; // Multiply the output-loss derivative by the full parameter chain-rule derivative.
} // End this block.
