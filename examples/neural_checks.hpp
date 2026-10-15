/** @file neural_checks.hpp
 * @brief Training, inference and numerical validation shared by the ANN/CNN applications.
 */
#ifndef SHDR_EXAMPLES_NEURAL_CHECKS_HPP // Open this include guard or optional feature branch.
#define SHDR_EXAMPLES_NEURAL_CHECKS_HPP // Define SHDR_EXAMPLES_NEURAL_CHECKS_HPP for the declarations that follow.
#include "runner.hpp" // Include runner.hpp declarations.
#include "neural_support.hpp" // Include neural_support.hpp declarations.

namespace shdr_examples { // Group these helpers within their namespace.

/** @brief Run complete model checks on the selected backend, including every parameter. */
inline int binary_example_main(int argc, char** argv, bool cnn) { // Implement binary example main for the operations below.
    try { // Run the operation while retaining the error handling below.
        runner run(argc, argv); // Construct run with these initial values.
        binary_network model(cnn); // Construct model with these initial values.
        check_binary_gradients(model); // Apply check binary gradients to these arguments.
        const float* cpu_gradients = // Set cpu gradients from this expression.
            static_cast<const shdr::buffer<float>&>(model.gradients).data(); // Apply data to these arguments.
        const std::vector<float> expected_gradient(cpu_gradients, // Declare expected gradient with the parameters continued below.
                                                   cpu_gradients + model.gradients.size()); // Apply size to these arguments.
        if (run.gpu()) { // Check this condition before entering the following branch.
            run.run(model.gradient_pipeline()); // Execute the configured pipeline on the requested backend.
            const float* gpu_gradients = // Set gpu gradients from this expression.
                static_cast<const shdr::buffer<float>&>(model.gradients).data(); // Apply data to these arguments.
            for (std::size_t i = 0; i < expected_gradient.size(); ++i) // Iterate over i for this calculation.
                require_close( // Check the computed value against the independent reference.
                    gpu_gradients[i], expected_gradient[i], "neural CPU/GPU gradient", 0.0002f); // Continue the expression using gpu gradients, i, expected gradient.
        } // End this block.
        run.run(model.evaluation_pipeline()); // Execute the configured pipeline on the requested backend.
        const float initial_loss = static_cast<const shdr::buffer<float>&>(model.loss).data()[0]; // Use a constant buffer view so GPU writes synchronize before host reads.
        const unsigned steps = cnn ? 180u : 160u; // Set steps from this expression.
        for (unsigned step = 0; step < steps; ++step) // Iterate over step for this calculation.
            run.run(model.training_pipeline()); // Execute the configured pipeline on the requested backend.
        run.run(model.evaluation_pipeline()); // Execute the configured pipeline on the requested backend.
        const float final_loss = static_cast<const shdr::buffer<float>&>(model.loss).data()[0]; // Use a constant buffer view so GPU writes synchronize before host reads.
        const shdr::buffer<float>& parameter_buffer = model.parameters; // Use a constant buffer view so GPU writes synchronize before host reads.
        const std::vector<double> weights(parameter_buffer.data(), // Declare weights with the parameters continued below.
                                          parameter_buffer.data() + parameter_buffer.size()); // Apply data to these arguments.
        const float* probabilities = // Set probabilities from this expression.
            static_cast<const shdr::buffer<float>&>(model.predictions).data(); // Apply data to these arguments.
        const float* labels = static_cast<const shdr::buffer<float>&>(model.targets).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
        unsigned correct = 0; // Set correct from this expression.
        for (unsigned sample = 0; sample < model.batch_count(); ++sample) { // Iterate over sample for this calculation.
            require_close(probabilities[sample], // Check the computed value against the independent reference.
                          float(binary_reference_probability(model, weights, sample)), // Apply float to these arguments.
                          "neural inference oracle", // Provide this text argument for the operation above.
                          0.0002f); // Complete this expression or declaration.
            if ((probabilities[sample] >= 0.5f) == (labels[sample] >= 0.5f)) // Check this condition before entering the following branch.
                ++correct; // Continue the expression using correct.
        } // End this block.
        if (!(final_loss < initial_loss * 0.8f) || correct < model.batch_count() * 9u / 10u) // Check this condition before entering the following branch.
            throw shdr::error( // Report this failed operation.
                "neural training did not reduce loss and learn the seeded classification task"); // Provide this text argument for the operation above.
        if (run.gpu()) { // Check this condition before entering the following branch.
            binary_network cpu(cnn); // Construct cpu with these initial values.
            for (unsigned step = 0; step < steps; ++step) // Iterate over step for this calculation.
                cpu.training_pipeline().run(); // Execute the configured pipeline on the requested backend.
            cpu.prediction_pipeline().run(); // Execute the configured pipeline on the requested backend.
            const float* cpu_weights = // Set cpu weights from this expression.
                static_cast<const shdr::buffer<float>&>(cpu.parameters).data(); // Apply data to these arguments.
            const float* cpu_probabilities = // Set cpu probabilities from this expression.
                static_cast<const shdr::buffer<float>&>(cpu.predictions).data(); // Apply data to these arguments.
            for (unsigned parameter = 0; parameter < model.parameter_count(); ++parameter) // Iterate over parameter for this calculation.
                require_close(float(weights[parameter]), // Check the computed value against the independent reference.
                              cpu_weights[parameter], // Continue the expression using cpu weights, parameter.
                              "neural trained CPU/GPU weights", // Provide this text argument for the operation above.
                              0.002f); // Complete this expression or declaration.
            for (unsigned sample = 0; sample < model.batch_count(); ++sample) // Iterate over sample for this calculation.
                require_close(probabilities[sample], // Check the computed value against the independent reference.
                              cpu_probabilities[sample], // Continue the expression using cpu probabilities, sample.
                              "neural trained CPU/GPU predictions", // Provide this text argument for the operation above.
                              0.002f); // Complete this expression or declaration.
        } // End this block.
        std::cout << (cnn ? "CNN" : "ANN") << ": " << model.parameter_count() // Write this part of the example result or diagnostic.
                  << " parameters trained for " << steps << " steps, loss " << initial_loss // Write this part of the example result or diagnostic.
                  << " -> " << final_loss << ", accuracy " << correct << '/' << model.batch_count() // Write this part of the example result or diagnostic.
                  << "; analytic gradients and inference verified\n"; // Write this part of the example result or diagnostic.
        return 0; // Report successful completion.
    } // End this block.
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
    catch (const context_unavailable& e) { // Handle the indicated exception from the preceding operation.
        std::cerr << "SKIP: " << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 77; // Report an unavailable optional GPU backend.
    } // End this block.
#endif // Finish this conditional compilation branch.
    catch (const std::exception& e) { // Handle the indicated exception from the preceding operation.
        std::cerr << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
} // namespace shdr_examples
#endif // Finish this conditional compilation branch.
