/** @file neural_support.hpp
 * @brief Small fully trainable ANN/CNN pipelines built from ordinary shdr resources.
 * Model parameters, batches, gradients and predictions are ordinary shdr buffers.
 * Training computes independent sample gradients and reduces them in a separate SGD dispatch.
 */
#ifndef SHDR_EXAMPLES_NEURAL_SUPPORT_HPP // Open this include guard or optional feature branch.
#define SHDR_EXAMPLES_NEURAL_SUPPORT_HPP // Define SHDR_EXAMPLES_NEURAL_SUPPORT_HPP for the declarations that follow.
#include "shdr.h" // Include shdr.h declarations.
#include <algorithm> // Include algorithm declarations.
#include <cmath> // Include cmath declarations.
#include <random> // Include random declarations.
#include <vector> // Include vector declarations.

namespace shdr_examples { // Group these helpers within their namespace.

/** @brief Binary ANN or CNN with separate training, inference, and evaluation pipelines.
 * ANN: two input features, eight ReLU neurons, sigmoid output, 33 parameters.
 * CNN: 8x8 grayscale, two valid 3x3 ReLU filters, global pooling, sigmoid, 23 parameters.
 * All parameters train. A new instance reproduces the same seeded data and initial weights.
 */
class binary_network { // Declare binary network.
    bool cnn_; // Declare cnn for the following calculation.
    unsigned batch_, parameter_count_; // Declare batch for the following calculation.
    std::vector<float> initial_; // Declare initial for the following calculation.
    std::unique_ptr<shdr::kernel> infer_, gradient_, sgd_, loss_shader_; // Declare infer for the following calculation.
    shdr::pipeline train_, predict_, evaluate_, differentiate_; // Declare train for the following calculation.
    static unsigned batch(unsigned n) { // Implement batch for the operations below.
        if (!n || n > 4096u) // Check this condition before entering the following branch.
            throw shdr::error("neural example batch size must be between 1 and 4096"); // Report neural example batch size must be between 1 and 4096.
        return n; // Return n.
    } // End this block.

  public: // Set public access for the following class members.
    shdr::buffer<float> inputs, targets, parameters, gradients, predictions, loss; // Declare inputs for the following calculation.

    /** @brief Build a deterministic XOR-feature or image-orientation training problem. */
    explicit binary_network(bool cnn = false, unsigned batch_count = 32) // Set cnn from this expression.
        : cnn_(cnn), batch_(batch(batch_count)), parameter_count_(cnn ? 23u : 33u), // Declare cnn with the parameters continued below.
          initial_(parameter_count_), // Apply initial to these arguments.
          infer_(shdr::make_kernel(cnn ? "cnn_infer_kernel" : "ann_infer_kernel")), // Create the linked shader instance.
          gradient_(shdr::make_kernel(cnn ? "cnn_gradient_kernel" : "ann_gradient_kernel")), // Create the linked shader instance.
          sgd_(shdr::make_kernel("nn_sgd_kernel")), // Create the linked nn sgd kernel instance.
          loss_shader_(shdr::make_kernel("nn_binary_loss_kernel")), // Create the linked nn binary loss kernel instance.
          inputs(batch_ * (cnn ? 64u : 2u)), targets(batch_), parameters(parameter_count_), // Apply inputs to these arguments.
          gradients(batch_ * parameter_count_), predictions(batch_), loss(1) { // Apply gradients to these arguments.
        std::mt19937 random(12345u); // Construct random with these initial values.
        std::uniform_real_distribution<float> noise(-0.12f, 0.12f), weights(-0.15f, 0.15f); // Construct noise with these initial values.
        for (float& value : initial_) // Visit each value required by this loop.
            value = weights(random); // Set value from this expression.
        if (cnn_) { // Check this condition before entering the following branch.
            for (unsigned filter = 0; filter < 2; ++filter) // Iterate over filter for this calculation.
                for (unsigned y = 0; y < 3; ++y) // Iterate over y for this calculation.
                    for (unsigned x = 0; x < 3; ++x) // Iterate over x for this calculation.
                        initial_[9 * filter + 3 * y + x] = // Set initial [9 * filter + 3 * y + x] from this expression.
                            ((filter == 0 ? y : x) == 1 ? 0.4f : -0.2f) + weights(random) * 0.1f; // Apply weights to these arguments.
            initial_[18] = initial_[19] = 0.04f; // Set initial [18] from this expression.
            for (unsigned sample = 0; sample < batch_; ++sample) { // Iterate over sample for this calculation.
                const bool horizontal = (sample & 1u) != 0; // Set horizontal from this expression.
                const unsigned position = 2u + unsigned(random()) % 3u; // Set position from this expression.
                targets.data()[sample] = horizontal ? 1.0f : 0.0f; // Apply data to these arguments.
                for (unsigned y = 0; y < 8; ++y) // Iterate over y for this calculation.
                    for (unsigned x = 0; x < 8; ++x) { // Iterate over x for this calculation.
                        const unsigned coordinate = horizontal ? y : x; // Set coordinate from this expression.
                        const float signal = // Set signal from this expression.
                            (coordinate == position || coordinate == position + 1) ? 0.9f : 0.0f; // Continue the expression using coordinate, position, coordinate.
                        inputs.data()[64 * sample + 8 * y + x] = signal + noise(random) * 0.4f; // Apply data to these arguments.
                    } // End this block.
            } // End this block.
        } else { // Initialize the eight-neuron ANN and its two-feature training inputs.
            const float directions[8] = {1, 1, 1, -1, -1, 1, -1, -1}; // Set directions[8] from this expression.
            std::copy(directions, directions + 8, initial_.begin()); // Apply copy to these arguments.
            for (unsigned neuron = 0; neuron < 8; ++neuron) // Iterate over neuron for this calculation.
                initial_[16 + neuron] = -0.15f; // Set initial [16 + neuron] from this expression.
            for (unsigned sample = 0; sample < batch_; ++sample) { // Iterate over sample for this calculation.
                const bool positive_x = (sample & 1u) != 0, positive_y = (sample & 2u) != 0; // Set positive x from this expression.
                inputs.data()[2 * sample] = (positive_x ? 1.0f : -1.0f) + noise(random); // Apply data to these arguments.
                inputs.data()[2 * sample + 1] = (positive_y ? 1.0f : -1.0f) + noise(random); // Apply data to these arguments.
                targets.data()[sample] = positive_x != positive_y ? 1.0f : 0.0f; // Apply data to these arguments.
            } // End this block.
        } // End this block.
        reset_parameters(); // Apply reset parameters to these arguments.
        infer_->bind("inputs", inputs) // Connect this shader port to its application resource.
            .bind("parameters", parameters) // Connect this shader port to its application resource.
            .bind("predictions", predictions) // Connect this shader port to its application resource.
            .set("batch_count", batch_); // Apply set to these arguments.
        gradient_->bind("inputs", inputs) // Connect this shader port to its application resource.
            .bind("targets", targets) // Connect this shader port to its application resource.
            .bind("parameters", parameters) // Connect this shader port to its application resource.
            .bind("gradients", gradients) // Connect this shader port to its application resource.
            .set("batch_count", batch_); // Apply set to these arguments.
        sgd_->bind("parameters", parameters) // Connect this shader port to its application resource.
            .bind("gradients", gradients) // Connect this shader port to its application resource.
            .set("batch_count", batch_) // Apply set to these arguments.
            .set("parameter_count", parameter_count_) // Apply set to these arguments.
            .set("learning_rate", cnn ? 1.5f : 0.4f) // Apply set to these arguments.
            .set("weight_decay", 0.0f); // Apply set to these arguments.
        loss_shader_->bind("predictions", predictions) // Connect this shader port to its application resource.
            .bind("targets", targets) // Connect this shader port to its application resource.
            .bind("loss", loss) // Connect this shader port to its application resource.
            .set("batch_count", batch_); // Apply set to these arguments.
        differentiate_ = // Set differentiate from this expression.
            shdr::dispatch(*gradient_, shdr::extent((batch_ * parameter_count_ + 63) / 64)); // Add a shader dispatch with the required workgroup counts.
        train_ = differentiate_; // Set train from this expression.
        train_ += shdr::dispatch(*sgd_, shdr::extent((parameter_count_ + 63) / 64)); // Add a shader dispatch with the required workgroup counts.
        predict_ = shdr::dispatch(*infer_, shdr::extent((batch_ + 63) / 64)); // Add a shader dispatch with the required workgroup counts.
        evaluate_ = predict_; // Set evaluate from this expression.
        evaluate_ += shdr::dispatch(*loss_shader_, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
    } // End this block.
    /** @brief Restore deterministic initial weights, outside a timed or queued processing loop. */
    void reset_parameters() { // Implement reset parameters for the operations below.
        std::copy(initial_.begin(), initial_.end(), parameters.data()); // Apply copy to these arguments.
    } // End this block.
    /** @brief Compute all independent per-sample analytic parameter gradients. */
    const shdr::pipeline& gradient_pipeline() const { // Implement gradient pipeline for the operations below.
        return differentiate_; // Return differentiate.
    } // End this block.
    /** @brief One complete mini-batch update: gradients followed by mean-gradient SGD. */
    const shdr::pipeline& training_pipeline() const { // Implement training pipeline for the operations below.
        return train_; // Return train.
    } // End this block.
    /** @brief Predict probabilities for the currently bound batch. */
    const shdr::pipeline& prediction_pipeline() const { // Implement prediction pipeline for the operations below.
        return predict_; // Return predict.
    } // End this block.
    /** @brief Predict, then compute a mean binary cross-entropy diagnostic. */
    const shdr::pipeline& evaluation_pipeline() const { // Implement evaluation pipeline for the operations below.
        return evaluate_; // Return evaluate.
    } // End this block.
    /** @brief Number of independent samples in the batch. */
    unsigned batch_count() const { // Implement batch count for the operations below.
        return batch_; // Return batch.
    } // End this block.
    /** @brief Number of trainable weights and biases. */
    unsigned parameter_count() const { // Implement parameter count for the operations below.
        return parameter_count_; // Return parameter count.
    } // End this block.
    /** @brief Whether this instance uses the convolutional image classifier. */
    bool convolutional() const { // Implement convolutional for the operations below.
        return cnn_; // Return cnn.
    } // End this block.
}; // Finish this declaration.

/** @brief Independent double-precision prediction oracle for ANN/CNN fixtures. */
inline double binary_reference_probability(const binary_network& network, // Declare binary reference probability with the parameters continued below.
                                           const std::vector<double>& weights, // Declare weights for the following calculation.
                                           unsigned sample) { // Declare sample for the following calculation.
    const float* input = network.inputs.data(); // Set input from this expression.
    double logit = weights.back(); // Use this floating-point value or resource in the following calculation.
    if (network.convolutional()) { // Check this condition before entering the following branch.
        for (unsigned filter = 0; filter < 2; ++filter) { // Iterate over filter for this calculation.
            double pooled = 0; // Use this floating-point value or resource in the following calculation.
            for (unsigned y = 0; y < 6; ++y) // Iterate over y for this calculation.
                for (unsigned x = 0; x < 6; ++x) { // Iterate over x for this calculation.
                    double value = weights[18 + filter]; // Use this floating-point value or resource in the following calculation.
                    for (unsigned ky = 0; ky < 3; ++ky) // Iterate over ky for this calculation.
                        for (unsigned kx = 0; kx < 3; ++kx) // Iterate over kx for this calculation.
                            value += weights[9 * filter + 3 * ky + kx] * // Update value with this contribution.
                                     input[64 * sample + 8 * (y + ky) + x + kx]; // Continue the expression using input, sample, y.
                    pooled += std::max(0.0, value) / 36.0; // Update pooled with this contribution.
                } // End this block.
            logit += weights[20 + filter] * pooled; // Update logit with this contribution.
        } // End this block.
    } else // Evaluate all eight ANN hidden neurons for the independent prediction.
        for (unsigned neuron = 0; neuron < 8; ++neuron) { // Iterate over neuron for this calculation.
            const double activation = std::max(0.0, // Set activation from this expression.
                                               weights[2 * neuron] * input[2 * sample] + // Continue the expression using weights, neuron, input.
                                                   weights[2 * neuron + 1] * input[2 * sample + 1] + // Continue the expression using weights, neuron, input.
                                                   weights[16 + neuron]); // Continue the expression using weights, neuron.
            logit += weights[24 + neuron] * activation; // Update logit with this contribution.
        } // End this block.
    if (logit >= 0) // Check this condition before entering the following branch.
        return 1.0 / (1.0 + std::exp(-logit)); // Return the unit-valued numeric result.
    const double e = std::exp(logit); // Set e from this expression.
    return e / (1.0 + e); // Return e / (1.0 + e).
} // End this block.

/** @brief Check mean analytic gradients against central differences of a double oracle. */
inline void check_binary_gradients(binary_network& network) { // Implement check binary gradients for the operations below.
    network.gradient_pipeline().run(); // Execute the configured pipeline on the requested backend.
    const shdr::buffer<float>& parameters = network.parameters; // Use a constant buffer view so GPU writes synchronize before host reads.
    std::vector<double> weights(parameters.data(), parameters.data() + parameters.size()); // Construct weights with these initial values.
    const float* gradient = static_cast<const shdr::buffer<float>&>(network.gradients).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
    const float* target = static_cast<const shdr::buffer<float>&>(network.targets).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
    const auto mean_loss = [&]() { // Set mean loss from this expression.
        double sum = 0; // Use this floating-point value or resource in the following calculation.
        for (unsigned sample = 0; sample < network.batch_count(); ++sample) { // Iterate over sample for this calculation.
            const double p = binary_reference_probability(network, weights, sample); // Set p from this expression.
            sum -= target[sample] * std::log(p) + (1.0 - target[sample]) * std::log(1.0 - p); // Update sum with this contribution.
        } // End this block.
        return sum / network.batch_count(); // Return sum / network.batch count().
    }; // Finish this declaration.
    for (unsigned parameter = 0; parameter < network.parameter_count(); ++parameter) { // Iterate over parameter for this calculation.
        const double original = weights[parameter], epsilon = 0.0001; // Set original from this expression.
        weights[parameter] = original + epsilon; // Set weights[parameter] from this expression.
        const double plus = mean_loss(); // Set plus from this expression.
        weights[parameter] = original - epsilon; // Set weights[parameter] from this expression.
        const double minus = mean_loss(); // Set minus from this expression.
        weights[parameter] = original; // Set weights[parameter] from this expression.
        const double numerical = (plus - minus) / (2 * epsilon); // Set numerical from this expression.
        double analytic = 0; // Use this floating-point value or resource in the following calculation.
        for (unsigned sample = 0; sample < network.batch_count(); ++sample) // Iterate over sample for this calculation.
            analytic += gradient[sample * network.parameter_count() + parameter] / // Update analytic with this contribution.
                        double(network.batch_count()); // Use this floating-point value or resource in the following calculation.
        if (!std::isfinite(analytic) || // Check this condition before entering the following branch.
            std::fabs(analytic - numerical) > 0.002 + 0.02 * std::fabs(numerical)) // Apply fabs to these arguments.
            throw shdr::error("neural analytic gradient differs from finite-difference oracle"); // Report neural analytic gradient differs from finite-difference oracle.
    } // End this block.
} // End this block.
} // namespace shdr_examples
#endif // Finish this conditional compilation branch.
