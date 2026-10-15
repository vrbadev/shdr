/** @file transformer.cpp
 * @brief Train and validate a tiny transformer on the CPU or selected GL backend.
 * Analytic forward derivatives train every parameter. Finite differences are
 * used only to validate the gradient implementation before optimization.
 */
#include "runner.hpp" // Select CPU execution or the requested graphics profile.
#include "transformer_support.hpp" // Use the reusable tiny transformer model.
#include "transformer_reference.hpp" // Use an independent double-precision prediction oracle.
#include <algorithm> // Use standard copying, limits and zero filling.
#include <cmath> // Use ordinary scalar math in host designers and oracles.
#include <iostream> // Report verified results and useful failures.
#include <vector> // Use owned host arrays for coefficients and reference results.

namespace { // Keep these example helpers in their own namespace.

void check_gradients(shdr_examples::transformer_model& model, shdr_examples::runner& run) { // Compare analytic derivatives with the independent double oracle.
    run.run(model.gradient_pipeline()); // Execute the linked stages on the selected backend.
    const float* derivative = static_cast<const shdr::buffer<float>&>(model.gradients).data(); // Read the resource after any required GPU result synchronization.
    std::vector<float> analytic(model.parameter_count, 0.0f); // Keep the averaged analytic derivative for every parameter.
    for (unsigned parameter = 0; parameter < model.parameter_count; ++parameter) // Visit every trainable parameter.
        for (unsigned row = 0; row < model.batch_count(); ++row) // Visit each token sequence in the batch.
            analytic[parameter] += // Average this analytic derivative over the batch.
                derivative[row * model.parameter_count + parameter] / model.batch_count(); // Read the sample-major gradient for this parameter.
    const float* original = static_cast<const shdr::buffer<float>&>(model.parameters).data(); // Read the resource after any required GPU result synchronization.
    std::vector<double> parameters(original, original + model.parameter_count); // Copy model parameters into the independent double reference.
    const float* inputs = static_cast<const shdr::buffer<float>&>(model.inputs).data(); // Read the resource after any required GPU result synchronization.
    const unsigned* labels = static_cast<const shdr::buffer<unsigned>&>(model.labels).data(); // Read the resource after any required GPU result synchronization.
    run.run(model.prediction_pipeline()); // Execute the linked stages on the selected backend.
    const shdr::glsl::vec4* predictions = // Read the resource after any required GPU result synchronization.
        static_cast<const shdr::buffer<shdr::glsl::vec4>&>(model.predictions).data(); // Read probabilities, losses and classes for the double-oracle check.
    for (unsigned row = 0; row < model.batch_count(); ++row) { // Visit each token sequence in the batch.
        const std::array<double, 4> expected = shdr_examples::detail::transformer_reference( // Compute the independent double probabilities and loss for this sequence.
            inputs + 16u * row, parameters.data(), labels[row]); // Use this sequence and its label in the double reference.
        for (unsigned c = 0; c < 4; ++c) // Visit every value needed by this calculation.
            shdr_examples::require_close(predictions[row][c], // Compare this result with its independent expected value.
                                         static_cast<float>(expected[c]), // Convert the computed host value to the shader float type.
                                         "transformer double prediction oracle", // Identify this check clearly in any failure message.
                                         0.00002f); // Apply the declared float error allowance.
    } // Finish this channel, sample or feature pass.
    const double step = 0.00001; // Use a small perturbation only for derivative validation.
    for (unsigned parameter = 0; parameter < model.parameter_count; ++parameter) { // Visit every trainable parameter.
        const double value = parameters[parameter]; // Save this double parameter before perturbing it.
        double above = 0.0, below = 0.0; // Prepare the two perturbed-reference loss accumulators.
        parameters[parameter] = value + step; // Perturb this double reference parameter upward.
        for (unsigned row = 0; row < model.batch_count(); ++row) // Visit each token sequence in the batch.
            above += shdr_examples::detail::transformer_reference( // Accumulate the independent perturbed-reference loss.
                inputs + 16u * row, parameters.data(), labels[row])[2]; // Use this sequence and its label in the double reference.
        parameters[parameter] = value - step; // Perturb this double reference parameter downward.
        for (unsigned row = 0; row < model.batch_count(); ++row) // Visit each token sequence in the batch.
            below += shdr_examples::detail::transformer_reference( // Accumulate the independent perturbed-reference loss.
                inputs + 16u * row, parameters.data(), labels[row])[2]; // Use this sequence and its label in the double reference.
        parameters[parameter] = value; // Restore the double reference parameter after its check.
        const float numerical = // Compute the central-difference mean-loss derivative.
            static_cast<float>((above - below) / (2.0 * step * model.batch_count())); // Convert the computed host value to the shader float type.
        const float tolerance = // Allow float rounding while keeping the derivative check strict.
            0.00002f + 0.001f * std::max(std::fabs(numerical), std::fabs(analytic[parameter])); // Combine a small absolute error allowance with a relative one.
        shdr_examples::require_close(analytic[parameter], // Compare this result with its independent expected value.
                                     numerical, // Supply the independently computed expected value.
                                     "transformer analytic gradient " + std::to_string(parameter), // Identify this check clearly in any failure message.
                                     tolerance); // Apply the declared float error allowance.
    } // Finish this channel, sample or feature pass.
} // Finish the check_gradients operation.

void check_trained_cpu_parity(shdr_examples::transformer_model& gpu) { // Compare every trained CPU and GPU parameter and prediction.
    shdr_examples::transformer_model cpu(gpu.batch_count()); // Build an independent CPU model with the same seed.
    for (unsigned step = 0; step < 32u; ++step) // Visit every value needed by this calculation.
        cpu.training_pipeline().run(); // Apply the same analytic update independently on the CPU.
    cpu.prediction_pipeline().run(); // Compute the independently trained CPU predictions.
    const float* expected = static_cast<const shdr::buffer<float>&>(cpu.parameters).data(); // Read the resource after any required GPU result synchronization.
    const float* actual = static_cast<const shdr::buffer<float>&>(gpu.parameters).data(); // Read the resource after any required GPU result synchronization.
    for (unsigned i = 0; i < cpu.parameter_count; ++i) // Visit every trainable parameter.
        shdr_examples::require_close( // Compare this result with its independent expected value.
            actual[i], expected[i], "trained transformer CPU/GPU parameter", 0.0003f); // Supply the actual and expected result values for comparison.
    expected = static_cast<const shdr::buffer<float>&>(cpu.gradients).data(); // Read the corresponding CPU and GPU result resource.
    actual = static_cast<const shdr::buffer<float>&>(gpu.gradients).data(); // Read the corresponding CPU and GPU result resource.
    for (unsigned i = 0; i < cpu.gradients.size(); ++i) // Visit every value needed by this calculation.
        shdr_examples::require_close( // Compare this result with its independent expected value.
            actual[i], expected[i], "trained transformer CPU/GPU gradient", 0.0001f); // Supply the actual and expected result values for comparison.
    const shdr::glsl::vec4* expected_predictions = // Read the resource after any required GPU result synchronization.
        static_cast<const shdr::buffer<shdr::glsl::vec4>&>(cpu.predictions).data(); // Read the independently trained CPU prediction records.
    const shdr::glsl::vec4* actual_predictions = // Read the resource after any required GPU result synchronization.
        static_cast<const shdr::buffer<shdr::glsl::vec4>&>(gpu.predictions).data(); // Read the trained GPU prediction records.
    for (unsigned row = 0; row < cpu.batch_count(); ++row) // Visit each token sequence in the batch.
        for (unsigned c = 0; c < 4; ++c) // Visit every value needed by this calculation.
            shdr_examples::require_close(actual_predictions[row][c], // Compare this result with its independent expected value.
                                         expected_predictions[row][c], // Supply the independently computed expected value.
                                         "trained transformer CPU/GPU prediction", // Identify this check clearly in any failure message.
                                         0.0001f); // Apply the declared float error allowance.
} // Finish the check_trained_cpu_parity operation.

void train(shdr_examples::runner& run) { // Verify the complete transformer training example.
    shdr_examples::transformer_model model(8); // Allocate the seeded eight-sequence transformer fixture.
    check_gradients(model, run); // Run the corresponding independent correctness check.
    model.reset_parameters(); // Restore the seed before measuring complete training convergence.
    run.run(model.prediction_pipeline()); // Execute the linked stages on the selected backend.
    const float initial = model.loss(); // Keep the loss before the first training update.
    const float* start = static_cast<const shdr::buffer<float>&>(model.parameters).data(); // Read the resource after any required GPU result synchronization.
    const std::vector<float> original(start, start + model.parameter_count); // Keep initial parameters for the learned-family update check.
    for (unsigned step = 0; step < 32u; ++step) // Visit every value needed by this calculation.
        run.run(model.training_pipeline()); // Execute the linked stages on the selected backend.
    run.run(model.prediction_pipeline()); // Execute the linked stages on the selected backend.
    const float final = model.loss(); // Read the loss after all analytic training steps.
    const unsigned correct = model.correct(); // Count the trained model predictions matching their labels.
    if (!std::isfinite(final) || final >= initial * 0.35f || correct != model.batch_count()) // Reject a nonfinite or out-of-range value.
        throw shdr::error("transformer training did not converge on the seeded dataset"); // Report the invalid design or failed check clearly.
    // Each learned family must participate in optimization, including Q and K.
    const unsigned boundaries[] = {0, 16, 32, 48, 64, 80, 84, 100, 104, 120, 124, 132, 134}; // Mark the learned parameter families so every family is checked.
    const float* trained = static_cast<const shdr::buffer<float>&>(model.parameters).data(); // Read the resource after any required GPU result synchronization.
    for (unsigned family = 0; family + 1u < sizeof(boundaries) / sizeof(boundaries[0]); ++family) { // Visit every value needed by this calculation.
        bool changed = false; // Start checking this learned parameter family.
        for (unsigned parameter = boundaries[family]; parameter < boundaries[family + 1u]; // Visit every trainable parameter.
             ++parameter) // Advance through every parameter in this learned family.
            changed = changed || trained[parameter] != original[parameter]; // Notice whether any parameter in this family was updated.
        if (!changed) // Require this learned parameter family to participate.
            throw shdr::error("a transformer parameter family did not train"); // Report the invalid design or failed check clearly.
    } // Finish this channel, sample or feature pass.
    if (run.gpu()) // Use the requested GPU path or explicit CPU transition.
        check_trained_cpu_parity(model); // Run the corresponding independent correctness check.
    std::cout << "Transformer: 134 analytic parameter gradients verified. Cross entropy " << initial // Print the verified result or the reason a check failed.
              << " -> " << final << ", classification " << correct << '/' << model.batch_count() // Complete the readable result message.
              << " after 32 SGD steps\n"; // Complete the readable result message.
} // Finish the train operation.
} // namespace

int main(int argc, char** argv) { // Run the selected CPU or GPU example.
    try { // Run the operation while preserving its failure diagnostic.
        shdr_examples::runner run(argc, argv); // Select CPU execution or the requested GPU profile.
        train(run); // Run the complete transformer gradient and training checks.
        return 0; // Report successful validation to the caller.
    } // Finish the checked operation.
#ifdef SHDR_HAVE_EGL // Enable this block only for the indicated compilation environment.
    catch (const shdr_examples::context_unavailable& e) { // Handle this expected backend or validation error.
        std::cerr << "SKIP: " << e.what() << '\n'; // Print the verified result or the reason a check failed.
        return 77; // Mark the unavailable graphics profile as skipped.
    } // Finish handling this reported failure.
#endif // Finish the conditional compilation block.
    catch (const std::exception& e) { // Handle this expected backend or validation error.
        std::cerr << e.what() << '\n'; // Print the verified result or the reason a check failed.
        return 1; // Report the validation failure to the caller.
    } // Finish handling this reported failure.
} // Finish the main operation.
