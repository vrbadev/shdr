/** @file transformer_reference.hpp
 * @brief Double-precision, value-only oracle for the transformer example.
 * This host implementation uses ordinary arrays and matrix products in
 * double precision. Its finite differences validate the training derivatives.
 * Optimization uses the analytic gradients produced by the native shader.
 */
#ifndef SHDR_EXAMPLES_TRANSFORMER_REFERENCE_HPP // Include this example helper only once.
#define SHDR_EXAMPLES_TRANSFORMER_REFERENCE_HPP // Remember that this example helper is available.
#include <algorithm> // Use standard copying, limits and zero filling.
#include <array> // Use fixed-size host coefficient and prediction arrays.
#include <cmath> // Use ordinary scalar math in host designers and oracles.

namespace shdr_examples { // Keep these example helpers in their own namespace.
namespace detail { // Keep these example helpers in their own namespace.

inline void // Declare the reusable host calculation or check.
transformer_projection(const double source[16], const double* weights, double result[16]) { // Apply a four-by-four projection to every token.
    for (unsigned token = 0; token < 4; ++token) // Visit each of the four tokens.
        for (unsigned column = 0; column < 4; ++column) { // Visit each of the four feature coordinates.
            double value = 0.0; // Start this sample or projected-feature accumulator at zero.
            for (unsigned input = 0; input < 4; ++input) // Visit every value needed by this calculation.
                value += source[4 * token + input] * weights[4 * input + column]; // Accumulate one entry of the learned matrix product.
            result[4 * token + column] = value; // Store this projected token feature.
        } // Finish this channel, sample or feature pass.
} // Finish the transformer_projection operation.

inline std::array<double, 4> // Declare the reusable host calculation or check.
transformer_reference(const float* input, const double* parameters, unsigned label) { // Compute the transformer independently in double precision.
    double tokens[16], queries[16], keys[16], values[16], context[16], residual[16], features[16]; // Keep each intermediate reference tensor as ordinary double arrays.
    for (unsigned i = 0; i < 16; ++i) // Visit every value needed by this calculation.
        tokens[i] = input[i] + parameters[i]; // Add the learned positional value to this input feature.
    transformer_projection(tokens, parameters + 16, queries); // Apply a four-by-four projection to every token.
    transformer_projection(tokens, parameters + 32, keys); // Apply a four-by-four projection to every token.
    transformer_projection(tokens, parameters + 48, values); // Apply a four-by-four projection to every token.
    for (unsigned token = 0; token < 4; ++token) { // Visit each of the four tokens.
        double scores[4] = {}, maximum = -1e300, total = 0.0; // Start the attention scores and softmax denominator at zero.
        for (unsigned source = 0; source < 4; ++source) { // Combine the contributing source features or samples.
            for (unsigned column = 0; column < 4; ++column) // Visit each of the four feature coordinates.
                scores[source] += queries[4 * token + column] * keys[4 * source + column] / 2.0; // Accumulate the scaled query-key attention score.
            maximum = std::max(maximum, scores[source]); // Keep the largest score for a stable softmax shift.
        } // Finish this channel, sample or feature pass.
        for (unsigned source = 0; source < 4; ++source) { // Combine the contributing source features or samples.
            scores[source] = std::exp(scores[source] - maximum); // Exponentiate the shifted attention score.
            total += scores[source]; // Accumulate the attention normalization denominator.
        } // Finish this channel, sample or feature pass.
        for (unsigned column = 0; column < 4; ++column) { // Visit each of the four feature coordinates.
            context[4 * token + column] = 0.0; // Start the attended feature at zero.
            for (unsigned source = 0; source < 4; ++source) // Combine the contributing source features or samples.
                context[4 * token + column] += scores[source] * values[4 * source + column] / total; // Weight this value feature by its attention probability.
        } // Finish this channel, sample or feature pass.
    } // Finish this channel, sample or feature pass.
    transformer_projection(context, parameters + 64, residual); // Apply a four-by-four projection to every token.
    for (unsigned i = 0; i < 16; ++i) // Visit every value needed by this calculation.
        residual[i] += tokens[i]; // Add the original token features as the attention residual.
    for (unsigned token = 0; token < 4; ++token) { // Visit each of the four tokens.
        double square = 1e-5, normalized[4], hidden[4]; // Add a small RMS stabilizer before accumulating squared features.
        for (unsigned column = 0; column < 4; ++column) // Visit each of the four feature coordinates.
            square += residual[4 * token + column] * residual[4 * token + column] / 4.0; // Accumulate the mean squared token magnitude.
        for (unsigned column = 0; column < 4; ++column) // Visit each of the four feature coordinates.
            normalized[column] = // Prepare the scaled RMS-normalized token feature.
                residual[4 * token + column] * parameters[80 + column] / std::sqrt(square); // Divide by the RMS magnitude and apply the learned scale.
        for (unsigned column = 0; column < 4; ++column) { // Visit each of the four feature coordinates.
            double value = parameters[100 + column]; // Start the first feed-forward projection with its bias.
            for (unsigned source = 0; source < 4; ++source) // Combine the contributing source features or samples.
                value += normalized[source] * parameters[84 + 4 * source + column]; // Accumulate the first feed-forward projection.
            hidden[column] = std::max(0.0, value); // Apply ReLU to the hidden feed-forward value.
        } // Finish this channel, sample or feature pass.
        for (unsigned column = 0; column < 4; ++column) { // Visit each of the four feature coordinates.
            double value = normalized[column] + parameters[120 + column]; // Start the second projection with its normalized residual and bias.
            for (unsigned source = 0; source < 4; ++source) // Combine the contributing source features or samples.
                value += hidden[source] * parameters[104 + 4 * source + column]; // Accumulate the second feed-forward projection.
            features[4 * token + column] = value; // Store the feed-forward residual feature.
        } // Finish this channel, sample or feature pass.
    } // Finish this channel, sample or feature pass.
    double logits[2] = {parameters[132], parameters[133]}; // Initialize the two class logits with their learned biases.
    for (unsigned column = 0; column < 4; ++column) { // Visit each of the four feature coordinates.
        double mean = 0.0; // Start the pooled feature accumulator at zero.
        for (unsigned token = 0; token < 4; ++token) // Visit each of the four tokens.
            mean += features[4 * token + column] / 4.0; // Average this feature over the four tokens.
        for (unsigned output = 0; output < 2; ++output) // Visit every value needed by this calculation.
            logits[output] += mean * parameters[124 + 2 * column + output]; // Add the pooled feature contribution to this class logit.
    } // Finish this channel, sample or feature pass.
    const double maximum = std::max(logits[0], logits[1]); // Use the maximum logit to stabilize the exponentials.
    const double first = std::exp(logits[0] - maximum), second = std::exp(logits[1] - maximum), // Compute the two stabilized class exponentials and their sum.
                 total = first + second; // Sum the stable class exponentials.
    return std::array<double, 4>{{first / total, // Return both probabilities, loss and predicted class.
                                  second / total, // Return the second class probability.
                                  std::log(total) + maximum - logits[label], // Return stable cross entropy for the known target class.
                                  logits[1] > logits[0] ? 1.0 : 0.0}}; // Return the class with the larger logit.
} // Finish the transformer_reference operation.
} // namespace detail
} // namespace shdr_examples
#endif // Finish the conditional compilation block.
