// Compute mean binary cross-entropy for a small validation batch.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 1) in; // Group 1 independent invocations along x.
layout(std430, binding = 0) readonly buffer Predictions { // Read one predicted class probability per sample through buffer binding 0.
    float predictions[]; // Store one predicted class probability per sample.
}; // Finish this declaration.
layout(std430, binding = 1) readonly buffer Targets { // Read one binary training label per sample through buffer binding 1.
    float targets[]; // Store one binary training label per sample.
}; // Finish this declaration.
layout(std430, binding = 2) writeonly buffer Loss { // Write the single mean loss value through buffer binding 2.
    float loss[]; // Store the single mean loss value.
}; // Finish this declaration.
uniform uint batch_count; // Set the number of independent samples in this model batch.
void main() { // Reduce the batch predictions into their mean binary cross entropy.
    float total = 0.0; // Start the total negative log likelihood at zero.
    for (uint row = 0u; row < batch_count; ++row) { // Include every predicted probability and binary label in the batch.
        float p = clamp(predictions[row], 0.000001, 0.999999); // Keep probabilities away from zero and one before taking logarithms.
        total -= targets[row] * log(p) + (1.0 - targets[row]) * log(1.0 - p); // Accumulate binary cross entropy for the label and its complement.
    } // End this block.
    loss[0] = total / float(batch_count); // Publish the mean sample loss as one scalar result.
} // End this block.
