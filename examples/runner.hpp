#ifndef SHDR_EXAMPLES_RUNNER_HPP // Open this include guard or optional feature branch.
#define SHDR_EXAMPLES_RUNNER_HPP // Define SHDR_EXAMPLES_RUNNER_HPP for the declarations that follow.

#include "shdr.h" // Include shdr.h declarations.
#include <cmath> // Include cmath declarations.
#include <iostream> // Include iostream declarations.
#include <memory> // Include memory declarations.
#include <random> // Include random declarations.
#include <stdexcept> // Include stdexcept declarations.
#include <string> // Include string declarations.
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
#include "egl_context.hpp" // Include egl_context.hpp declarations.
#endif // Finish this conditional compilation branch.

namespace shdr_examples { // Group these helpers within their namespace.

/// Select CPU execution by default, or an explicitly requested GL profile.
class runner { // Own an optional GPU device and select the pipeline execution backend.
    std::unique_ptr<shdr::device> device_; // Keep the optional headless context alive until the example finishes.

  public: // Set public access for the following class members.
    runner(int argc, char** argv) { // Read the common optional GPU profile from this example's command line.
        if (argc == 1) // Select CPU execution for the default command line.
            return; // Finish initializing the CPU runner.
        if (argc != 3 || std::string(argv[1]) != "--gpu") // Accept exactly the documented --gpu PROFILE option.
            throw std::invalid_argument("usage: executable [--gpu es31|es32|gl43|gl45|gl46]"); // Report usage: executable [--gpu es31|es32|gl43|gl45|gl46].
        const std::string name(argv[2]); // Construct name with these initial values.
        shdr::profile profile; // Declare profile for the following calculation.
        if (name == "es31") // Check this condition before entering the following branch.
            profile = shdr::profile::es31; // Set profile from this expression.
        else if (name == "es32") // Select the GLES 3.2 device profile.
            profile = shdr::profile::es32; // Set profile from this expression.
        else if (name == "gl43") // Select the desktop GL 4.3 device profile.
            profile = shdr::profile::gl43; // Set profile from this expression.
        else if (name == "gl45") // Select the desktop GL 4.5 device profile.
            profile = shdr::profile::gl45; // Set profile from this expression.
        else if (name == "gl46") // Select the desktop GL 4.6 device profile.
            profile = shdr::profile::gl46; // Set profile from this expression.
        else // Reject profile names outside the supported list.
            throw std::invalid_argument("unknown GL profile: " + name); // Report unknown GL profile: .
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
        device_.reset(new shdr::headless_device(profile)); // Create the requested headless context through the public library backend.
#else // Select the alternative implementation for this build.
        (void)profile; // Consume the parsed profile value in the CPU build configuration.
        throw std::runtime_error( // Report this failed operation.
            "GPU examples were not enabled. Configure with EGL development files"); // Provide this text argument for the operation above.
#endif // Finish this conditional compilation branch.
    } // End this block.

    void run(const shdr::pipeline& pipeline) { // Execute any prepared pipeline using this example's selected backend.
        if (device_) // Use the GPU device only when the command line requested one.
            pipeline.run(*device_); // Run default stages on GPU while preserving explicitly selected CPU stages.
        else // Execute the default stages on the CPU backend.
            pipeline.run(); // Run the pipeline through the standard C++ implementation.
    } // End this block.

    bool gpu() const { // Expose the selected backend for result labels and optional GPU checks.
        return bool(device_); // Report whether this runner owns a headless GPU device.
    } // End this block.
    shdr::device* device() const { // Expose the optional device for interop, synchronization and timing.
        return device_.get(); // Return the retained device or null for ordinary CPU execution.
    } // End this block.
}; // Finish this declaration.

/// Fail examples and tests with a useful diagnostic when CPU/GPU results diverge.
inline void // Declare void for the following calculation.
require_close(float actual, float expected, const std::string& label, float tolerance = 0.0001f) { // Compare a computed scalar with its independently calculated reference.
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) { // Reject nonfinite results and differences above the chosen absolute tolerance.
        throw std::runtime_error(label + ": expected " + std::to_string(expected) + ", received " + // Identify the failed case and show both expected and received values.
                                 std::to_string(actual)); // Include the computed value in the comparison diagnostic.
    } // End this block.
} // End this block.

/// Use repeatable random inputs so each backend receives identical test data.
inline void randomize(shdr::buffer<float>& values, unsigned seed, float lower, float upper) { // Fill a host buffer with reproducible uniformly distributed test values.
    std::mt19937 generator(seed); // Initialize the deterministic random sequence from the caller's seed.
    std::uniform_real_distribution<float> distribution(lower, upper); // Select the caller's inclusive lower and upper distribution bounds.
    float* data = values.data(); // Mark the host resource modified before filling its elements.
    for (std::size_t i = 0; i < values.size(); ++i) // Initialize every element of the buffer exactly once.
        data[i] = distribution(generator); // Store the next random sample from the selected interval.
} // End this block.

} // namespace shdr_examples
#endif // Finish this conditional compilation branch.
