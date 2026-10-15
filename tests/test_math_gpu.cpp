/** @file test_math_gpu.cpp
 * The native math shader is compiled through the generated GCC adapter and
 * dispatched on CPU and every enabled GLES/GL profile. Known references check
 * formulas independently. GPU runs additionally compare against CPU results.
 */
#include "examples/runner.hpp" // Include examples/runner.hpp declarations.
#include <cstdint> // Include cstdint declarations.
#include <vector> // Include vector declarations.

namespace { // Group these helpers within their namespace.
const float expected[] = { // Declare expected for the following calculation.
    -3, 6,    -3, 32, 5,  0.6f,  0.8f, 1,      -1, 0.5f, -0.5f, 0.75f, 0, 2,    2,  -2, -2,   -88, // Complete this expression or declaration.
    11, 16,   8,  18, 7,  22,    1,    0,      4,  3,    15,    16,    2, 1,    3,  3,  2.5f, 4, // Complete this expression or declaration.
    3,  0.5f, 0,  1,  90, 0.25f, -1,   -0.25f, 4,  4,    -2,    -1,    1, 0.5f, -1, 1,  2.5f, 9}; // Complete this expression or declaration.
const std::uint32_t expected_bits[] = {0x3f800000u, // Declare expected bits for the following calculation.
                                       0x80000000u, // Complete this expression or declaration.
                                       32u, // Complete this expression or declaration.
                                       3u, // Complete this expression or declaration.
                                       0u, // Complete this expression or declaration.
                                       0x80000000u, // Complete this expression or declaration.
                                       0xffffffffu, // Complete this expression or declaration.
                                       0xff03u, // Complete this expression or declaration.
                                       1u, // Complete this expression or declaration.
                                       1u, // Complete this expression or declaration.
                                       0xfffffffeu, // Complete this expression or declaration.
                                       1u, // Complete this expression or declaration.
                                       0xc0003c00u, // Complete this expression or declaration.
                                       0xffff0000u, // Complete this expression or declaration.
                                       0x00ff00ffu, // Complete this expression or declaration.
                                       0xffffffffu, // Complete this expression or declaration.
                                       1u}; // Complete this expression or declaration.
const std::size_t float_count = sizeof(expected) / sizeof(expected[0]); // Set float count from this expression.
const std::size_t bit_count = sizeof(expected_bits) / sizeof(expected_bits[0]); // Set bit count from this expression.

void check(const shdr::buffer<float>& results, // Declare check with the parameters continued below.
           const shdr::buffer<unsigned>& bits, // Declare bits for the following calculation.
           const std::string& label) { // Declare label for the following calculation.
    for (std::size_t i = 0; i < float_count; ++i) // Iterate over i for this calculation.
        shdr_examples::require_close( // Check the computed value against the independent reference.
            results.data()[i], expected[i], label + " float[" + std::to_string(i) + "]", 0.0002f); // Apply data to these arguments.
    for (std::size_t i = 0; i < bit_count; ++i) // Iterate over i for this calculation.
        if (bits.data()[i] != expected_bits[i]) // Check this condition before entering the following branch.
            throw std::runtime_error(label + " bits[" + std::to_string(i) + "]: expected " + // Report  bits[.
                                     std::to_string(expected_bits[i]) + ", received " + // Apply to string to these arguments.
                                     std::to_string(bits.data()[i])); // Apply to string to these arguments.
} // End this block.
} // namespace

int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner runner(argc, argv); // Construct runner with these initial values.
        shdr::buffer<float> results(float_count); // Construct results with these initial values.
        shdr::buffer<unsigned> bits(bit_count); // Construct bits with these initial values.
        std::unique_ptr<shdr::kernel> shader = shdr::make_kernel("math_kernel"); // Create the linked math kernel instance.
        shader->bind("results", results); // Connect this shader port to its application resource.
        shader->bind("bits", bits); // Connect this shader port to its application resource.
        shader->set<shdr::glsl::vec2>("controls", shdr::glsl::vec2(1, -1)); // Supply this shader uniform with its typed application value.
        shdr::pipeline pipeline; // Declare pipeline for the following calculation.
        pipeline += shdr::dispatch(*shader, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
        pipeline.run(); // Execute the configured pipeline on the requested backend.
        check(results, bits, "CPU math"); // Apply check to these arguments.
        const shdr::buffer<float>& host_results = results; // Use a constant buffer view so GPU writes synchronize before host reads.
        const shdr::buffer<unsigned>& host_bits = bits; // Use a constant buffer view so GPU writes synchronize before host reads.
        const std::vector<float> cpu(host_results.data(), host_results.data() + float_count); // Construct cpu with these initial values.
        const std::vector<unsigned> cpu_bits(host_bits.data(), host_bits.data() + bit_count); // Construct cpu bits with these initial values.
        if (runner.gpu()) { // Check this condition before entering the following branch.
            runner.run(pipeline); // Execute the configured pipeline on the requested backend.
            check(results, bits, "GPU math"); // Apply check to these arguments.
            for (std::size_t i = 0; i < float_count; ++i) // Iterate over i for this calculation.
                shdr_examples::require_close( // Check the computed value against the independent reference.
                    host_results.data()[i], cpu[i], "CPU/GPU math parity", 0.0002f); // Apply data to these arguments.
            for (std::size_t i = 0; i < bit_count; ++i) // Iterate over i for this calculation.
                if (host_bits.data()[i] != cpu_bits[i]) // Check this condition before entering the following branch.
                    throw std::runtime_error("CPU/GPU bit operation parity"); // Report CPU/GPU bit operation parity.
        } // End this block.
        std::cout << "Native math shader " << (runner.gpu() ? "CPU/GPU" : "CPU") // Write this part of the example result or diagnostic.
                  << " checks passed\n"; // Write this part of the example result or diagnostic.
        return 0; // Report successful completion.
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
    } catch (const shdr_examples::context_unavailable& e) { // Report the exception from the preceding operation.
        std::cerr << "SKIP: " << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 77; // Report an unavailable optional GPU backend.
#endif // Finish this conditional compilation branch.
    } catch (const std::exception& e) { // Report the exception from the preceding operation.
        std::cerr << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
