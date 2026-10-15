// 8x8 grayscale -> two valid 3x3 filters -> ReLU -> global mean -> sigmoid classifier.
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
void main() { // Evaluate the two-filter convolutional classifier for one image.
    uint row = gl_GlobalInvocationID.x; // Select the independent input sample assigned to this invocation.
    if (row >= batch_count) // Ignore invocations outside the sample batch.
        return; // Stop this invocation before performing another resource access.
    float logit = parameters[22]; // Initialize the binary output logit from its learned bias.
    for (uint channel = 0u; channel < 2u; ++channel) { // Evaluate each of the two learned convolution filters.
        float pooled = 0.0; // Start the mean feature and its derivative at zero.
        for (uint y = 0u; y < 6u; ++y) // Visit the six valid output rows of the three-by-three filter.
            for (uint x = 0u; x < 6u; ++x) { // Visit the six valid output columns of this filter.
                float activation = parameters[18u + channel]; // Start this convolution window with the filter's learned bias.
                for (uint ky = 0u; ky < 3u; ++ky) // Visit each row of the three-by-three learned kernel.
                    for (uint kx = 0u; kx < 3u; ++kx) // Visit each column of the three-by-three learned kernel.
                        activation += parameters[9u * channel + 3u * ky + kx] * // Multiply this kernel coefficient by its matching input image sample.
                                      inputs[64u * row + 8u * (y + ky) + x + kx]; // Read the sample from this image's flattened eight-by-eight plane.
                pooled += max(0.0, activation) / 36.0; // Apply ReLU and add one thirty-sixth of this valid convolution feature.
            } // End this block.
        logit += parameters[20u + channel] * pooled; // Connect this mean convolution feature through its learned head weight.
    } // End this block.
    predictions[row] = logistic(logit); // Convert the two-filter output logit into its binary class probability.
} // End this block.
