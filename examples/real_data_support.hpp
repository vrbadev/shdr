/** @file real_data_support.hpp
 * @brief Portable fixture readers and complete CPU/GPU output comparisons.
 * Dataset paths contain data only. Shader sources remain embedded in the executable.
 */
#ifndef SHDR_EXAMPLES_REAL_DATA_SUPPORT_HPP // Include fixture support only once.
#define SHDR_EXAMPLES_REAL_DATA_SUPPORT_HPP // Mark the support declarations as available.
#include "runner.hpp" // Select the same execution backend as the other examples.
#include <cstdint> // Assemble endian-defined dataset words explicitly.
#include <cstdlib> // Allow a caller to choose a dataset directory through the environment.
#include <cstring> // Copy float representations with std::memcpy.
#include <fstream> // Read ordinary binary datasets from disk.
#include <functional> // Restore mutable resources before comparing independent executions.
#include <limits> // Check that float fixtures have the documented representation.

namespace shdr_examples { // Keep fixture utilities outside the public library API.
namespace real_data { // Group the real-data example's input and comparison helpers.
inline std::string directory() { // Resolve the supplied fixture directory at runtime.
    const char* override_path = std::getenv("SHDR_DATA_DIR"); // Check an optional caller override.
    if (override_path && *override_path) { // Accept a nonempty explicit dataset location.
        return override_path; // Use only this directory for the real-data fixture reads.
    } // Ordinary builds use the CMake-provided source fixture directory.
#ifdef SHDR_EXAMPLE_DATA_DIR // Select the dataset directory recorded by the example build.
    return SHDR_EXAMPLE_DATA_DIR; // Return the ordinary example fixture path.
#else // Keep a useful relative fallback for manual builds.
    return "examples/data"; // Resolve fixtures relative to the caller's working directory.
#endif // Finish the build-path selection.
} // Use this selected directory to resolve prepared dataset filenames.
inline std::vector<unsigned char> // Read one bounded fixture file.
bytes(const std::string& name, // Read one bounded fixture file.
      const std::string& base = directory()) { // Read one bounded fixture file.
    const std::string path = base + "/" + name; // Join the fixed fixture filename to its directory.
    std::ifstream input(path.c_str(), // Open at the end to measure size.
                        std::ios::binary | std::ios::ate); // Open at the end to measure size.
    if (!input) { // Identify a missing prepared dataset immediately.
        throw shdr::error("cannot open dataset " + path + // Report the failed format, numerical or resource precondition.
                          ", run scripts/prepare_datasets.py first"); // Explain preparation.
    } // The dataset path is now open as an ordinary binary stream.
    const std::streamoff count = input.tellg(); // Measure the exact byte count before allocation.
    if (count <= 0 || // Reject empty or unexpectedly large example inputs.
        count > 64 * 1024 * 1024) { // Reject empty or unexpectedly large example inputs.
        throw shdr::error("invalid example dataset size: " + // Report the failed format, numerical or resource precondition.
                          name); // Keep fixture allocations bounded.
    } // All documented public fixtures fit within this simple limit.
    std::vector<unsigned char> result( // Allocate the measured payload.
        static_cast<std::size_t>(count)); // Allocate the measured payload.
    input.seekg(0); // Return to the first dataset byte.
    input.read(reinterpret_cast<char*>(result.data()), count); // Read the complete fixture payload.
    if (!input) { // Catch incomplete file reads before interpreting any value.
        throw shdr::error("dataset read was incomplete: " + // Report the failed format, numerical or resource precondition.
                          name); // Report truncation or an I/O failure.
    } // The caller receives a fully read binary fixture.
    return result; // Transfer the owned byte array to its decoder.
} // Preserve the file order of the raw byte values.
inline std::vector<float> // Decode little-endian IEEE float32 fixtures.
floats(const std::string& name) { // Decode little-endian IEEE float32 fixtures.
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, // Check representation.
                  "float32 fixtures require IEEE floats"); // Check representation.
    const std::vector<unsigned char> raw = bytes(name); // Read the complete named float fixture.
    if (raw.size() % 4u) { // Require four bytes for every declared float32 value.
        throw shdr::error("float dataset contains a partial value: " + // Report the failed format, numerical or resource precondition.
                          name); // Reject incomplete representations.
    } // Whole values can now be assembled independently of endian order.
    std::vector<float> result(raw.size() / 4u); // Allocate one native float per encoded value.
    for (std::size_t i = 0; i < result.size(); ++i) { // Decode every payload element.
        const std::uint32_t bits = // Assemble bytes in documented order.
            std::uint32_t(raw[4u * i]) | (std::uint32_t(raw[4u * i + 1u]) << 8u) | // Assemble bytes in documented order.
            (std::uint32_t(raw[4u * i + 2u]) << 16u) | // Assemble bytes in documented order.
            (std::uint32_t(raw[4u * i + 3u]) << 24u); // Assemble bytes in documented order.
        std::memcpy( // Copy the assembled float32 representation into its float value.
            &result[i], &bits, sizeof(float)); // Copy the assembled float32 representation into its float value.
        if (!std::isfinite(result[i])) { // Keep dataset corruption out of numerical comparisons.
            throw shdr::error("nonfinite float in dataset: " + // Report the failed format, numerical or resource precondition.
                              name); // Reject invalid signal or model values.
        } // Valid values retain their published float32 bit patterns.
    } // The full fixture is now usable by either backend.
    return result; // Supply the decoded scalar array.
} // The example supports both little-endian and big-endian hosts.
inline std::vector<std::uint16_t> // Decode EVT3 little-endian words.
words(const std::string& name) { // Decode EVT3 little-endian words.
    const std::vector<unsigned char> raw = // Read the public recording's prepared prefix.
        bytes(name); // Read the public recording's prepared prefix.
    if (raw.size() % 2u) { // Require complete sensor words.
        throw shdr::error("EVT3 dataset contains a partial word"); // Report a truncated fixture.
    } // Each pair of bytes forms one sensor word.
    std::vector<std::uint16_t> result(raw.size() / 2u); // Allocate the word-oriented decoder input.
    for (std::size_t i = 0; i < result.size(); ++i) { // Convert every complete word.
        result[i] = static_cast<std::uint16_t>( // Decode low byte first.
            unsigned(raw[2u * i]) | (unsigned(raw[2u * i + 1u]) << 8u)); // Decode low byte first.
    } // Assemble each little-endian word from its two stored bytes.
    return result; // Return the original sensor word sequence.
} // The RAW text header is removed only by the documented preparation script.
template <class T> // Apply the same helper to each compatible scalar or vector resource type.
inline shdr::buffer<T> resource(const std::vector<T>& values) { // Upload ordinary prepared arrays.
    shdr::buffer<T> result(values.size()); // Allocate a typed resource of the exact element count.
    std::copy(values.begin(), // Copy the preserved source values into their input resource.
              values.end(), // Copy the dataset into host resource storage.
              result.data()); // Copy the dataset into host resource storage.
    return result; // Share the resulting resource with pipeline stages.
} // GPU upload remains deferred until a device actually uses the resource.
inline double difference(float actual, // Measure a finite scalar output difference.
                         float expected) { // Measure a finite scalar output difference.
    if (!std::isfinite(actual) || // Reject numerical failures on either backend.
        !std::isfinite(expected)) { // Reject numerical failures on either backend.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "real-data pipeline produced a nonfinite output"); // Name an invalid processing result.
    } // Relative scaling makes the comparison meaningful for logits and signal amplitudes.
    return std::fabs(double(actual) - expected) / // Combine absolute and relative error.
           (1.0 + std::fabs(double(expected))); // Combine absolute and relative error.
} // A tolerance of zero requires bit-identical numerical values.
inline double difference(unsigned actual, unsigned expected) { // Compare discrete outputs exactly.
    return actual == expected ? 0.0 : 1.0; // Keep event words, masks and lengths lossless.
} // Require exact equality for integer outputs.
template <class T, std::size_t N> // Apply the same helper to each compatible scalar or vector resource type.
inline double difference(const glsl::vec<T, N>& actual, // Compare every vector component.
                         const glsl::vec<T, N>& expected) { // Compare every vector component.
    double maximum = 0.0; // Track the worst normalized component difference.
    for (std::size_t component = 0; component < N; // Include both coordinates and metadata components.
         ++component) { // Include both coordinates and metadata components.
        maximum = std::max(maximum, // Apply the matching scalar policy.
                           difference(actual[component], // Apply the matching scalar policy.
                                      expected[component])); // Apply the matching scalar policy.
    } // Vectors use their strictest component result.
    return maximum; // Return the complete element's comparison error.
} // Compare the typed vector components through the normalized scalar-error helper.
class output_check { // Provide a common interface for several typed stage outputs.
  public: // Expose only the capture and comparison operations needed by the evaluator.
    virtual ~output_check() { // Destroy a concrete checker through this base type safely.
    } // Destroy a concrete checker through this base type safely.
    virtual void capture() = 0; // Save completed CPU output as the backend comparison reference.
    virtual double verify() const = 0; // Compare the requested backend against that saved output.
}; // A pipeline can validate several differently typed resources together.
template <class T> // Apply the same helper to each compatible scalar or vector resource type.
class checked_output : public output_check { // Keep an output resource and its CPU snapshot.
    const shdr::buffer<T>& resource_; // Read completed results through a constant resource reference.
    std::vector<T> reference_; // Preserve a complete CPU result across GPU execution.
    double tolerance_; // Record the permitted normalized floating-point difference.
    std::size_t // Compare only the meaningful output prefix when storage has spare capacity.
        count_; // Compare only the meaningful output prefix when storage has spare capacity.
  public: // Let each test choose the resource, error limit and optional compact count.
    explicit checked_output(const shdr::buffer<T>& value, // Configure the output comparison.
                            double tolerance = 0.0001, // Configure the output comparison.
                            std::size_t count = 0) // Configure the output comparison.
        : resource_(value), tolerance_(tolerance), // Configure the output comparison.
          count_(count ? count : value.size()) { // Configure the output comparison.
        if (count_ > // Prevent a checker from reading beyond allocated resource storage.
            value.size()) { // Prevent a checker from reading beyond allocated resource storage.
            throw shdr::error( // Report the failed format, numerical or resource precondition.
                "real-data comparison count exceeds resource storage"); // Explain the invalid check.
        } // The meaningful output range is now bounded.
    } // The reference is populated only after CPU processing.
    void capture() override { // Save a complete finite CPU output reference.
        const T* values = resource_.data(); // Obtain completed host-readable resource values.
        reference_.assign(values, values + count_); // Keep all meaningful output elements.
        for (const T& value : // Validate CPU numerical results before using them as a reference.
             reference_) { // Validate CPU numerical results before using them as a reference.
            (void)difference(value, value); // Self-comparison detects nonfinite float components.
        } // Discrete values remain exact in the snapshot.
    } // Keep an independent CPU reference for later GPU comparisons.
    double verify() const override { // Compare every completed output value with its CPU reference.
        const T* values = // Read back GPU results only after the pipeline has been submitted.
            resource_.data(); // Read back GPU results only after the pipeline has been submitted.
        double maximum = 0.0; // Track a useful measured backend discrepancy.
        for (std::size_t i = 0; i < reference_.size(); ++i) { // Check the full meaningful result.
            maximum = std::max( // Compare the appropriate value type.
                maximum, // Compare the appropriate value type.
                difference(values[i], reference_[i])); // Compare the appropriate value type.
            if (maximum > // Fail as soon as a result exceeds its explicit error limit.
                tolerance_) { // Fail as soon as a result exceeds its explicit error limit.
                throw shdr::error("real-data CPU/GPU output differs at element " + // Report the failed format, numerical or resource precondition.
                                  std::to_string(i)); // Identify the bad location.
            } // Successful outputs stay within the documented comparison tolerance.
        } // All elements have now been compared.
        return maximum; // Report the worst measured normalized difference.
    } // Finish comparing completed results with the stored CPU reference.
}; // Own the CPU reference vector and borrow the pipeline resource.
inline void // Execute identical real inputs on both backends.
evaluate(runner& run, // Execute identical real inputs on both backends.
         const std::string& label, // Execute identical real inputs on both backends.
         const shdr::pipeline& pipeline, // Execute identical real inputs on both backends.
         std::initializer_list<output_check*> outputs, // Execute identical real inputs on both backends.
         const std::function<void()>& reset = // Execute identical real inputs on both backends.
             std::function<void()>()) { // Execute identical real inputs on both backends.
    if (reset) { // Restore trainable parameters or streaming state when a case needs it.
        reset(); // Establish the CPU run's documented starting state.
    } // Stateless cases use their existing immutable input resources.
    pipeline.run(); // Compute the complete CPU reference from real dataset values.
    for (output_check* output : outputs) { // Capture every selected intermediate and final result.
        output->capture(); // Preserve completed output before a device may overwrite it.
    } // The comparison snapshots now describe this exact pipeline invocation.
    if (run.gpu()) { // Run the requested profile only when the user selected a GPU backend.
        if (reset) { // Give GPU execution the same initial mutable resources as CPU execution.
            reset(); // Restore parameters or state before replaying the pipeline.
        } // Input datasets remain identical across both runs.
        run.run(pipeline); // Submit the linked stages through the selected library device.
    } // CPU-only runs still validate finite and exact output checks below.
    double maximum = 0.0; // Track the worst error across all selected stage resources.
    for (const output_check* output : outputs) { // Validate the full result for each typed output.
        maximum = // Include every recorded stage in the comparison.
            std::max(maximum, output->verify()); // Include every recorded stage in the comparison.
    } // Backend errors become ordinary test failures with useful element diagnostics.
    std::cout << label << ": verified, normalized CPU/GPU error " << maximum // Report the completed real-data processing check.
              << '\n'; // Report a measured functional check.
} // Timed randomized benchmarks remain separate from these correctness runs.
} // namespace real_data
} // namespace shdr_examples
#endif // Finish the fixture support include guard.
