/** @file lenet_checks.hpp
 * @brief Independent double-precision checks for the complete LeNet-style model.
 * The reference uses ordinary host arrays and explicit layer calculations.
 * It checks every intermediate feature and final logit, including edge padding,
 * channel ordering, flattening, pooling and the two independent batch images.
 */
#ifndef SHDR_EXAMPLES_LENET_CHECKS_HPP // Include this independent model check only once.
#define SHDR_EXAMPLES_LENET_CHECKS_HPP // Remember that the model check is available.
#include "runner.hpp" // Select CPU or a supported graphics execution profile.
#include "lenet_support.hpp" // Use the complete seven-stage native classifier.
#include <random> // Build repeatable independent model and input fixtures.
namespace shdr_examples { // Keep the model checks with the reusable examples.
namespace lenet_detail { // Keep reference calculations separate from deployed shader code.
struct reference_result { // Retain every independently calculated model layer.
    std::vector<double> conv1, pool1, conv2, pool2, hidden1, hidden2, // Keep spatial features, dense features and final scores in double precision.
        logits; // Keep spatial features, dense features and final scores in double precision.
}; // Finish the reference feature record.
inline std::vector<double> // Compute a padded learned convolution in ordinary double precision.
convolution(const std::vector<double>& input, // Compute a padded learned convolution in ordinary double precision.
            const std::vector<double>& weights, // Read the flattened model in double precision.
            unsigned batch, // Process this many independent images.
            unsigned width, // Specify the width of each input feature plane.
            unsigned sources, // Read this many source channels or vector features.
            unsigned outputs, // Produce this many channels or vector features.
            unsigned padding, // Specify the number of zero pixels around the input.
            unsigned weight_offset, // Locate the first weight of this layer.
            unsigned bias_offset) { // Compute a padded learned convolution in ordinary double precision.
    const unsigned side = // Find the output side for a five-by-five valid kernel.
        width + 2u * padding - 4u; // Find the output side for a five-by-five valid kernel.
    std::vector<double> result(batch * outputs * side * // Allocate each complete output channel plane.
                               side); // Allocate each complete output channel plane.
    for (unsigned image = 0; image < batch; // Keep independent batch images in their original order.
         ++image) { // Keep independent batch images in their original order.
        for (unsigned target = 0; target < outputs; ++target) { // Visit each learned output filter.
            for (unsigned y = 0; y < side; ++y) { // Visit each output row.
                for (unsigned x = 0; x < side; ++x) { // Visit each output column.
                    double value = weights[bias_offset + // Start with the output filter's learned bias.
                                           target]; // Start with the output filter's learned bias.
                    for (unsigned source = 0; source < sources; // Sum all source feature-plane contributions.
                         ++source) { // Sum all source feature-plane contributions.
                        for (unsigned ky = 0; ky < 5u; ++ky) { // Visit the five kernel rows.
                            for (unsigned kx = 0; kx < 5u; ++kx) { // Visit the five kernel columns.
                                const int sx = // Find the corresponding unpadded input column.
                                    static_cast<int>(x + kx) - // Find the corresponding unpadded input column.
                                    static_cast<int>( // Find the corresponding unpadded input column.
                                        padding); // Find the corresponding unpadded input column.
                                const int sy = // Find the corresponding unpadded input row.
                                    static_cast<int>(y + ky) - // Find the corresponding unpadded input row.
                                    static_cast<int>( // Find the corresponding unpadded input row.
                                        padding); // Find the corresponding unpadded input row.
                                if (sx < 0 || sy < 0 || sx >= static_cast<int>(width) || // Treat all pixels outside the image as zero.
                                    sy >= static_cast<int>(width)) // Treat all pixels outside the image as zero.
                                    continue; // Treat all pixels outside the image as zero.
                                const unsigned position = // Select the matching NCHW feature value.
                                    (image * sources + source) * width * width + // Select the matching NCHW feature value.
                                    unsigned(sy) * width + // Select the matching NCHW feature value.
                                    unsigned(sx); // Select the matching NCHW feature value.
                                const unsigned coefficient = // Select the independently indexed learned coefficient.
                                    weight_offset + ((target * sources + source) * 5u + ky) * 5u + // Select the independently indexed learned coefficient.
                                    kx; // Select the independently indexed learned coefficient.
                                value += // Accumulate this contribution in double precision.
                                    weights[coefficient] * // Accumulate this contribution in double precision.
                                    input // Accumulate this contribution in double precision.
                                        [position]; // Accumulate this contribution in double precision.
                            } // Finish this kernel row.
                        } // Finish this source kernel.
                    } // Finish the sum over source channels.
                    result[(image * outputs + target) * side * side + y * side + x] = // Apply ReLU after the complete convolution sum.
                        std::max(0.0, value); // Apply ReLU after the complete convolution sum.
                } // Finish this output row.
            } // Finish this output channel.
        } // Finish this image's learned feature planes.
    } // Finish all independent batch images.
    return result; // Return the independently calculated convolution layer.
} // Finish the convolution reference.
inline std::vector<double> // Downsample channel planes with an explicit double average.
pooling(const std::vector<double>& input, // Downsample channel planes with an explicit double average.
        unsigned batch, // Process this many independent images.
        unsigned channels, // Pool each feature channel independently.
        unsigned width) { // Downsample channel planes with an explicit double average.
    const unsigned side = width / 2u; // Halve the even spatial dimension.
    std::vector<double> result(batch * channels * side * // Allocate each complete downsampled plane.
                               side); // Allocate each complete downsampled plane.
    for (unsigned plane = 0; plane < batch * channels; // Keep batch and channel planes independent.
         ++plane) { // Keep batch and channel planes independent.
        for (unsigned y = 0; y < side; ++y) { // Visit each pooled output row.
            for (unsigned x = 0; x < side; ++x) { // Visit each pooled output column.
                double sum = 0.0; // Begin the four-pixel double accumulator at zero.
                for (unsigned dy = 0; dy < 2u; ++dy) { // Visit the upper and lower source rows.
                    for (unsigned dx = 0; dx < 2u; // Visit the left and right source columns.
                         ++dx) { // Visit the left and right source columns.
                        sum += input[plane * width * width + (2u * y + dy) * width + 2u * x + // Add this source pixel to the explicit double sum.
                                     dx]; // Add this source pixel to the explicit double sum.
                    } // Finish this source square row.
                } // Finish this complete two-by-two source square.
                result[plane * side * side + y * side + x] = // Store the average in the same plane ordering.
                    sum / 4.0; // Store the average in the same plane ordering.
            } // Finish this pooled output row.
        } // Finish this pooled output plane.
    } // Finish the independent image and channel planes.
    return result; // Return the independently downsampled feature layer.
} // Finish the average-pooling reference.
inline std::vector<double> // Compute an ordinary double matrix-vector product for every image.
dense(const std::vector<double>& input, // Compute an ordinary double matrix-vector product for every image.
      const std::vector<double>& weights, // Read the flattened model in double precision.
      unsigned batch, // Process this many independent images.
      unsigned sources, // Read this many source channels or vector features.
      unsigned outputs, // Produce this many channels or vector features.
      unsigned weight_offset, // Locate the first weight of this layer.
      unsigned bias_offset, // Locate the first bias of this layer.
      bool relu) { // Compute an ordinary double matrix-vector product for every image.
    std::vector<double> result(batch * // Allocate a contiguous output vector for each image.
                               outputs); // Allocate a contiguous output vector for each image.
    for (unsigned image = 0; image < batch; // Process the independent flattened input vectors.
         ++image) { // Process the independent flattened input vectors.
        for (unsigned target = 0; target < outputs; // Visit each learned output coordinate.
             ++target) { // Visit each learned output coordinate.
            double sum = // Start the dot product with the learned bias.
                weights[bias_offset + target]; // Start the dot product with the learned bias.
            for (unsigned source = 0; source < sources; // Visit every contributing input feature.
                 ++source) { // Visit every contributing input feature.
                sum += // Accumulate the row-major learned projection.
                    weights[weight_offset + target * sources + source] * // Accumulate the row-major learned projection.
                    input[image * sources + source]; // Accumulate the row-major learned projection.
            } // Finish the input-feature sum.
            result[image * outputs + target] = // Keep logits linear and hidden features nonnegative.
                relu ? std::max(0.0, sum) // Keep logits linear and hidden features nonnegative.
                     : sum; // Keep logits linear and hidden features nonnegative.
        } // Finish the image's output vector.
    } // Finish all independent image vectors.
    return result; // Return the independently calculated dense layer.
} // Finish the dense reference.
inline void // Compare every native feature with its independent double result.
compare(const shdr::buffer<float>& actual, // Compare every native feature with its independent double result.
        const std::vector<double>& expected, // Use the independently computed double-precision features.
        const char* layer) { // Compare every native feature with its independent double result.
    if (actual.size() != expected.size()) // Require matching layer dimensions before comparing values.
        throw shdr::error( // Reject an inconsistent model shape before reading output.
            "LeNet reference layer size mismatch"); // Reject an inconsistent model shape before reading output.
    const float* values = // Obtain the completed native output values.
        actual.data(); // Synchronize the layer only for this untimed correctness check.
    for (unsigned i = 0; i < expected.size(); // Verify every image, channel and feature coordinate.
         ++i) { // Verify every image, channel and feature coordinate.
        const float reference = static_cast<float>( // Convert the double oracle value for a float comparison.
            expected[i]); // Convert the double oracle value for a float comparison.
        require_close( // Check this value against the independent layer oracle.
            values[i], // Use the native CPU or GPU feature value.
            reference, // Use the oracle feature rounded to storage precision.
            layer, // Identify the layer if this comparison fails.
            0.00002f + // Allow a small absolute error around zero.
                0.00003f * // Scale the tolerance with the expected feature magnitude.
                    std::fabs( // Measure the absolute expected magnitude.
                        reference)); // Allow float accumulation error while retaining a strict layer check.
    } // Finish all feature comparisons.
} // Finish the model-layer comparison.
} // namespace lenet_detail
/** @brief Calculate every model layer independently using a double host reference. */
inline lenet_detail::reference_result lenet_reference( // Accept the same complete input and parameter layout as the native classifier.
    const std::vector<float>& inputs, // Accept the complete batch of normalized input images.
    const std::vector<float>& parameters, // Accept the documented flattened model record.
    unsigned // Accept the number of independent images in this reference batch.
        batch) { // Accept the same complete input and parameter layout as the native classifier.
    if (!batch || batch > 512u || inputs.size() != batch * 784u || parameters.size() != 61706u) // Require the exact batch, image and model geometry.
        throw shdr::error( // Require complete model and batch records.
            "invalid LeNet reference geometry"); // Require complete model and batch records.
    const std::vector<double> image(inputs.begin(), inputs.end()), // Convert each input value to the oracle's double precision.
        weights(parameters.begin(), // Convert each stored parameter to double precision.
                parameters.end()); // Convert the float parameters to the independent double reference.
    lenet_detail::reference_result result; // Keep every independently computed intermediate layer.
    result.conv1 = // Compute the independent first convolution features.
        lenet_detail::convolution(image, // Use explicit double-precision convolution loops.
                                  weights, // Read the independent double-precision model record.
                                  batch, // Process every image in this oracle call.
                                  28u, // Use the 28-pixel input dimension.
                                  1u, // Read one grayscale source plane.
                                  6u, // Use the six first-block feature planes.
                                  2u, // Pad each edge with two implicit zero pixels.
                                  0u, // Start the first convolution weight record.
                                  150u); // Reproduce padded first convolution in double precision.
    result.pool1 = // Compute the independent first pooled features.
        lenet_detail::pooling(result.conv1, batch, 6u, 28u); // Reproduce first average pooling.
    result.conv2 = lenet_detail::convolution( // Compute the full six-to-sixteen second convolution.
        result.pool1, // Read the first independent pooled feature record.
        weights, // Read the independent double-precision model record.
        batch, // Process every image in this oracle call.
        14u, // Use the 14-pixel first pooled dimension.
        6u, // Use the six first-block feature planes.
        16u, // Use the sixteen second-block feature planes.
        0u, // Apply the second convolution over valid five-by-five input windows.
        156u, // Start the second convolution weight record.
        2556u); // Reproduce full C3 connectivity across six source planes.
    result.pool2 = lenet_detail::pooling( // Compute the independent second average pooling.
        result.conv2, batch, 16u, 10u); // Reproduce second average pooling in NCHW order.
    result.hidden1 = lenet_detail::dense(result.pool2, // Project the independent pooled features to 120 values.
                                         weights, // Read the independent double-precision model record.
                                         batch, // Process every image in this oracle call.
                                         400u, // Read all sixteen five-by-five pooled planes.
                                         120u, // Use the 120-feature first dense representation.
                                         2572u, // Start the 400-to-120 dense weight record.
                                         50572u, // Start the 120 first dense biases.
                                         true); // Reproduce the flattened full C5 projection.
    result.hidden2 = lenet_detail::dense(result.hidden1, // Project the independent hidden features to 84 values.
                                         weights, // Read the independent double-precision model record.
                                         batch, // Process every image in this oracle call.
                                         120u, // Use the 120-feature first dense representation.
                                         84u, // Use the 84-feature second dense representation.
                                         50692u, // Start the 120-to-84 dense weight record.
                                         60772u, // Start the 84 second dense biases.
                                         true); // Reproduce the F6 feature projection.
    result.logits = lenet_detail::dense(result.hidden2, // Compute the ten independent linear class logits.
                                        weights, // Read the independent double-precision model record.
                                        batch, // Process every image in this oracle call.
                                        84u, // Use the 84-feature second dense representation.
                                        10u, // Use the ten-value dimension required by this layer.
                                        60856u, // Start the 84-to-10 classifier weight record.
                                        61696u, // Start the ten final classifier biases.
                                        false); // Reproduce the learned linear ten-class output.
    return result; // Return all reference layers for detailed validation.
} // Finish complete independent model inference.
/** @brief Verify all seven native layers on two distinct seeded and impulse images. */
inline void check_lenet_classifier( // Verify the complete model on the chosen execution backend.
    runner& run) { // Check geometry and every model layer against independent calculations.
    lenet_classifier model(2); // Allocate two independent complete classifiers in one batch.
    std::mt19937 generator(1978u); // Reproduce the same synthetic model on every backend.
    std::uniform_real_distribution<float> weights(-0.08f, 0.08f), // Keep model values bounded and input pixels in the documented range.
        pixels(0.0f, 1.0f); // Keep model values bounded and input pixels in the documented range.
    std::vector<float> parameters( // Allocate one complete deterministic test model.
        model.parameter_count); // Hold every learned value in its published layout.
    for (float& value : parameters) // Initialize every learned model value.
        value = weights( // Draw a reproducible small signed parameter.
            generator); // Give all convolution and dense coefficients nontrivial seeded values.
    for (unsigned i = 0; i < 6u; ++i) // Initialize each first convolution bias.
        parameters[150u + i] = 0.01f + 0.003f * i; // Keep distinct first-layer biases visible.
    for (unsigned i = 0; i < 16u; ++i) // Initialize each second convolution bias.
        parameters[2556u + i] = // Keep second-block activations available for later checks.
            0.02f + 0.005f * (i % 3u); // Keep second-layer channels distinguishable.
    for (unsigned i = 0; i < 120u; ++i) // Initialize every first dense bias.
        parameters[50572u + i] = // Keep first dense activations available for later checks.
            0.02f + 0.002f * (i % 7u); // Exercise nonzero hidden features after flattening.
    for (unsigned i = 0; i < 84u; ++i) // Initialize every second dense bias.
        parameters[60772u + i] = // Keep second dense activations available for later checks.
            0.01f + 0.003f * (i % 5u); // Exercise the learned 84-feature projection.
    for (unsigned i = 0; i < 10u; ++i) // Initialize all final digit-score biases.
        parameters[61696u + i] = // Give each class a distinct deterministic bias.
            -0.02f + 0.004f * i; // Keep the final linear class scores distinct.
    std::vector<float> inputs(2u * 784u, // Allocate two initially zero test images.
                              0.0f); // Keep the second image zero except for its explicit impulses.
    for (unsigned i = 0; i < 784u; ++i) // Fill the first image with reproducible grayscale data.
        inputs[i] = pixels(generator); // Use a seeded grayscale image in the first batch slot.
    inputs[784u] = 1.0f; // Put an impulse at the second image's upper-left edge.
    inputs[784u + 14u * 28u + 12u] = 0.75f; // Put a distinct interior impulse in the second image.
    inputs[2u * 784u - 1u] = 0.4f; // Put a third impulse at the lower-right edge.
    model.load_parameters( // Load the independent deterministic model into native storage.
        parameters); // Copy the complete synthetic model into shared CPU and GPU storage.
    std::copy( // Copy both test images into the model input resource.
        inputs.begin(), // Start at the first normalized test image.
        inputs.end(), // Finish at the end of the second test image.
        model.inputs.data()); // Copy both independent input images in their original order.
    const lenet_detail::reference_result expected = // Compute every layer with a separate double-precision oracle.
        lenet_reference(inputs, // Run the independent complete classifier calculation.
                        parameters, // Use the same stored model values for the oracle.
                        2u); // Compute the full independent double oracle before native execution.
    run.run( // Execute all seven linked native model stages.
        model // Use the classifier's complete inference pipeline.
            .inference_pipeline()); // Execute the same seven linked native stages on the selected backend.
    lenet_detail::compare( // Compare every feature of the next native layer.
        model.conv1, // Read the first native convolution feature record.
        expected.conv1, // Use the independent first convolution features.
        "LeNet padded first convolution"); // Check every padded border and first-layer feature.
    lenet_detail::compare( // Compare every feature of the next native layer.
        model.pool1, // Read the first native pooled feature record.
        expected.pool1, // Use the independent first pooling features.
        "LeNet first average pooling"); // Check the first downsampled feature planes.
    lenet_detail::compare( // Compare every feature of the next native layer.
        model.conv2, // Read the second native convolution feature record.
        expected.conv2, // Use the independent second convolution features.
        "LeNet full C3 convolution"); // Check every learned cross-channel contribution.
    lenet_detail::compare( // Compare every feature of the next native layer.
        model.pool2, // Read the second native pooled feature record.
        expected.pool2, // Use the independent second pooling features.
        "LeNet second average pooling"); // Check the NCHW features feeding the dense classifier.
    lenet_detail::compare( // Compare every feature of the next native layer.
        model.hidden1, // Read the native 120-feature hidden representation.
        expected.hidden1, // Use the independent 120-feature hidden representation.
        "LeNet full C5 projection"); // Check flattening and the 120-feature projection.
    lenet_detail::compare( // Compare every feature of the next native layer.
        model.hidden2, expected.hidden2, "LeNet F6 projection"); // Check the 84-feature ReLU stage.
    lenet_detail::compare(model.logits, // Compare all native linear digit scores.
                          expected.logits, // Use the independently computed linear digit scores.
                          "LeNet ten-class logits"); // Check every final linear class score.
    std::cout // Print a concise confirmation of complete model verification.
        << "LeNet-style classifier: all seven layers and 61706 parameters verified against a double oracle\n"; // Report the complete independent inference check.
} // Finish the native classifier validation.
} // namespace shdr_examples
#endif // Finish the model-check include guard.
