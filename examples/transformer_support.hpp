/** @file transformer_support.hpp
 * @brief Host wiring for a four-token transformer with analytic SGD training.
 * The same native shader functions perform CPU and GPU inference and evaluate
 * every parameter derivative. Gradient dispatches read the current parameters.
 * Subsequent SGD dispatches apply their averaged updates.
 */
#ifndef SHDR_EXAMPLES_TRANSFORMER_SUPPORT_HPP // Include this example helper only once.
#define SHDR_EXAMPLES_TRANSFORMER_SUPPORT_HPP // Remember that this example helper is available.
#include "shdr.h" // Use the same header for CPU and GPU resource wiring.
#include <cmath> // Use ordinary scalar math in host designers and oracles.
#include <random> // Create repeatable random fixtures and initial parameters.

namespace shdr_examples { // Keep these example helpers in their own namespace.

/** @brief A tiny single-head sequence classifier with all 134 parameters learned.
 * Architecture: four 4-component tokens, learned positions, scaled attention,
 * output residual, RMSNorm, a 4-wide ReLU feed-forward residual, mean pooling,
 * and a two-class softmax classifier. This compact implementation demonstrates
 * training mechanics and explicit CPU/GPU resource wiring.
 */
class transformer_model { // Bundle all transformer resources and linked stages.
    static unsigned checked_batch(unsigned count) { // Validate the batch before allocating model resources.
        if (!count || count > 65536u) // Handle this validation or boundary case explicitly.
            throw shdr::error("transformer batch count must be between 1 and 65536"); // Report the invalid design or failed check clearly.
        return count; // Return the validated sequence count.
    } // Finish the checked_batch operation.
    unsigned batch_count_; // Remember how many token sequences share this model.
    std::unique_ptr<shdr::kernel> infer_, gradient_, update_; // Own the three compiled transformer stages.
    shdr::pipeline inference_, training_; // Keep inference and training as separate linked pipelines.

  public: // Expose the resources and processing helpers.
    enum { token_count = 4, width = 4, parameter_count = 134 }; // Fix the four-token, four-feature model and its parameter count.
    shdr::buffer<float> inputs, parameters, gradients; // Keep input sequences, trainable values and per-sample gradients.
    shdr::buffer<unsigned> labels; // Keep each sequence class in a portable unsigned buffer.
    shdr::buffer<shdr::glsl::vec4> predictions; // Keep probabilities, loss and predicted class for each sequence.

    /** @brief Allocate and link inference, analytic gradients and averaged SGD.
     * @param batch_count Positive number of four-token training sequences.
     * @param learning_rate SGD step size applied to the batch-mean gradient.
     */
    explicit transformer_model(unsigned batch_count = 8, float learning_rate = 0.15f) // Connect the transformer inference and training stages.
        : batch_count_(checked_batch(batch_count)), // Validate the batch size before allocating model resources.
          infer_(shdr::make_kernel("transformer_infer_kernel")), // Create the next compiled inference, derivative or update stage.
          gradient_(shdr::make_kernel("transformer_gradient_kernel")), // Create the next compiled inference, derivative or update stage.
          update_(shdr::make_kernel("nn_sgd_kernel")), inputs(batch_count_ * token_count * width), // Create the next compiled inference, derivative or update stage.
          parameters(parameter_count), gradients(batch_count_ * parameter_count), // Allocate every learned transformer value.
          labels(batch_count_), predictions(batch_count_) { // Allocate one target class for each sequence.
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) // Reject a nonfinite or out-of-range value.
            throw shdr::error("transformer learning rate must be positive and finite"); // Report the invalid design or failed check clearly.
        infer_->bind("inputs", inputs) // Connect the next model stage to its shared resources.
            .bind("parameters", parameters) // Connect the next named resource to the same shader instance.
            .bind("labels", labels) // Connect the next named resource to the same shader instance.
            .bind("predictions", predictions) // Connect the next named resource to the same shader instance.
            .set<unsigned>("batch_count", batch_count_); // Set the shader dimensions or scalar processing parameter.
        gradient_->bind("inputs", inputs) // Connect the next model stage to its shared resources.
            .bind("parameters", parameters) // Connect the next named resource to the same shader instance.
            .bind("labels", labels) // Connect the next named resource to the same shader instance.
            .bind("gradients", gradients) // Connect the next named resource to the same shader instance.
            .set<unsigned>("batch_count", batch_count_); // Set the shader dimensions or scalar processing parameter.
        update_->bind("parameters", parameters) // Connect the next model stage to its shared resources.
            .bind("gradients", gradients) // Connect the next named resource to the same shader instance.
            .set<unsigned>("batch_count", batch_count_) // Set the shader dimensions or scalar processing parameter.
            .set<unsigned>("parameter_count", parameter_count) // Set the shader dimensions or scalar processing parameter.
            .set<float>("learning_rate", learning_rate) // Set the shader dimensions or scalar processing parameter.
            .set<float>("weight_decay", 0.0f); // Set the shader dimensions or scalar processing parameter.
        inference_ = shdr::dispatch(*infer_, shdr::extent((batch_count_ + 63u) / 64u)); // Dispatch one inference invocation per sequence.
        training_ = shdr::dispatch(*gradient_, // Dispatch one analytic derivative per sequence and parameter.
                                   shdr::extent((batch_count_ * parameter_count + 63u) / 64u)) >> // Cover every sequence and parameter in the derivative dispatch.
                    shdr::dispatch(*update_, shdr::extent((parameter_count + 63u) / 64u)); // Reduce batch gradients and update each parameter exactly once.
        reset_inputs(); // Create the initial repeatable training dataset.
        reset_parameters(); // Create the initial repeatable trainable parameters.
    } // Finish the transformer_model operation.
    /** @brief Recreate a balanced, repeatable two-class sequence dataset.
     * The class controls a feature pattern shared across tokens.  small seeded
     * noise and token variations prevent identical sequences within each class.
     */
    void reset_inputs(unsigned seed = 7127u) { // Build the repeatable two-class token dataset.
        std::mt19937 generator(seed); // Use a fixed seed so every backend receives identical data.
        std::uniform_real_distribution<float> noise(-0.12f, 0.12f); // Choose a bounded range for reproducible noise or weights.
        const float pattern[4] = {1.0f, -0.7f, 0.4f, -0.2f}; // Define the feature pattern distinguishing the two classes.
        float* values = inputs.data(); // Acquire the host storage that this initialization will overwrite.
        unsigned* targets = labels.data(); // Acquire the host storage that this initialization will overwrite.
        for (unsigned row = 0; row < batch_count_; ++row) { // Visit each token sequence in the batch.
            targets[row] = row % 2u; // Alternate the two classes for a balanced dataset.
            const float sign = targets[row] ? 1.0f : -1.0f; // Choose the class-dependent sign of the feature pattern.
            for (unsigned token = 0; token < token_count; ++token) // Visit each of the four tokens.
                for (unsigned c = 0; c < width; ++c) // Visit every value needed by this calculation.
                    values[row * 16u + token * 4u + c] = // Write this feature into its sequence, token and component slot.
                        sign * pattern[c] + noise(generator) + float(token) * 0.02f; // Combine the class pattern with small noise and token variation.
        } // Finish this channel, sample or feature pass.
    } // Finish the reset_inputs operation.
    /** @brief Restore deterministic trainable parameters, including RMSNorm scales. */
    void reset_parameters(unsigned seed = 59315u) { // Restore repeatable trainable model parameters.
        std::mt19937 generator(seed); // Use a fixed seed so every backend receives identical data.
        std::uniform_real_distribution<float> distribution(-0.15f, 0.15f); // Choose a bounded range for reproducible noise or weights.
        float* values = parameters.data(); // Acquire the host storage that this initialization will overwrite.
        for (unsigned i = 0; i < parameter_count; ++i) // Visit every trainable parameter.
            values[i] = distribution(generator); // Initialize this trainable value from the fixed random seed.
        for (unsigned i = 0; i < 16u; ++i) // Visit every value needed by this calculation.
            values[i] *= 0.1f; // Keep positional embeddings small at initialization.
        for (unsigned i = 80u; i < 84u; ++i) // Visit every value needed by this calculation.
            values[i] = 1.0f; // Initialize RMS normalization scales to unity.
        for (unsigned i = 100u; i < 104u; ++i) // Visit every value needed by this calculation.
            values[i] = 0.5f; // Keep the initial feed-forward hidden units active.
        for (unsigned i = 120u; i < 124u; ++i) // Visit every value needed by this calculation.
            values[i] = 0.0f; // Start the output feed-forward bias at zero.
    } // Finish the reset_parameters operation.
    /** @brief Return the sample count shared by inference and gradient dispatches. */
    unsigned batch_count() const { // Return the number of sequences in this model batch.
        return batch_count_; // Remember how many token sequences share this model.
    } // Finish the batch_count operation.
    /** @brief Return a two-stage analytic gradient followed by averaged SGD update. */
    const shdr::pipeline& training_pipeline() const { // Expose analytic derivatives followed by averaged SGD.
        return training_; // Expose analytic gradients followed by averaged SGD.
    } // Finish the training_pipeline operation.
    /** @brief Return the classifier inference dispatch. */
    const shdr::pipeline& prediction_pipeline() const { // Expose the classifier inference dispatch.
        return inference_; // Expose the classifier inference dispatch.
    } // Finish the prediction_pipeline operation.
    /** @brief Return the dispatch that computes analytic per-sample derivatives. */
    shdr::pipeline gradient_pipeline() const { // Build the analytic derivative dispatch.
        return shdr::pipeline( // Expose the per-sample analytic gradient stage.
            shdr::dispatch(*gradient_, shdr::extent((batch_count_ * parameter_count + 63u) / 64u))); // Cover the sample-major gradient resource with 64-wide groups.
    } // Finish the gradient_pipeline operation.
    /** @brief Read batch-mean cross entropy after inference has produced predictions. */
    float loss() const { // Read the mean cross entropy for the current predictions.
        const shdr::glsl::vec4* values = predictions.data(); // Read the resource after any required GPU result synchronization.
        double sum = 0.0; // Start the scalar accumulator at zero.
        for (unsigned row = 0; row < batch_count_; ++row) // Visit each token sequence in the batch.
            sum += values[row].z; // Add this sequence loss to the batch total.
        return static_cast<float>(sum / batch_count_); // Read the mean cross entropy for the current predictions.
    } // Finish the loss operation.
    /** @brief Read the number of correctly classified training sequences. */
    unsigned correct() const { // Count correctly classified sequences.
        const shdr::glsl::vec4* values = predictions.data(); // Read the resource after any required GPU result synchronization.
        const unsigned* targets = labels.data(); // Read the resource after any required GPU result synchronization.
        unsigned count = 0; // Begin counting correct sequence classifications.
        for (unsigned row = 0; row < batch_count_; ++row) // Visit each token sequence in the batch.
            if (static_cast<unsigned>(values[row].w) == targets[row]) // Handle this validation or boundary case explicitly.
                ++count; // Count this correctly classified sequence.
        return count; // Return the validated sequence count.
    } // Finish the correct operation.
}; // Finish the transformer_model resource helper.
} // namespace shdr_examples
#endif // Finish the conditional compilation block.
