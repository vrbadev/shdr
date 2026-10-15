#include "runner.hpp" // Include runner.hpp declarations.

/// Interleave GPU and CPU stages, shared buffers synchronize at backend changes.
int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Select CPU execution or the requested headless GPU profile.
        const unsigned count = 129; // Use a nonmultiple of the workgroup size to exercise the shader bounds guards.
        shdr::buffer<float> input(count), scaled(count), clipped(count), result(1); // Allocate distinct resources for the input, intermediate stages and final result.
        shdr_examples::randomize(input, 1103u, 0.0f, 4.0f); // Fill the input with reproducible random values.
        auto affine_owner = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
        shdr::kernel& affine = *affine_owner; // Retain the owned shader instance for every stage that references it.
        affine.bind("input_values", input); // Read the resource produced by the preceding stage or the original host input.
        affine.bind("output_values", scaled); // Write the shared resource consumed by the next stage or the final host check.
        affine.set<unsigned>("count", count); // Limit the shader to the actual number of valid buffer elements.
        affine.set<float>("scale", 4.0f); // Configure the first stage's multiplicative gain.
        affine.set<float>("offset", -1.0f); // Configure the bias added after the first stage scales each input.

        auto clip_owner = shdr::make_kernel("clamp_kernel"); // Create the linked clamp kernel instance.
        shdr::kernel& clip = *clip_owner; // Retain the owned shader instance for every stage that references it.
        clip.bind("input_values", scaled); // Read the resource produced by the preceding stage or the original host input.
        clip.bind("output_values", clipped); // Write the shared resource consumed by the next stage or the final host check.
        clip.set<unsigned>("count", count); // Limit the shader to the actual number of valid buffer elements.
        clip.set<float>("lower", 0.0f); // Set the lower clipping bound in the second stage.
        clip.set<float>("upper", 7.0f); // Set the upper clipping bound in the second stage.

        auto sum_owner = shdr::make_kernel("sum_kernel"); // Create the linked sum kernel instance.
        shdr::kernel& sum = *sum_owner; // Retain the owned shader instance for every stage that references it.
        sum.bind("input_values", clipped); // Read the resource produced by the preceding stage or the original host input.
        sum.bind("output_values", result); // Write the shared resource consumed by the next stage or the final host check.
        sum.set<unsigned>("count", count); // Limit the shader to the actual number of valid buffer elements.

        shdr::pipeline pipeline = // Set pipeline from this expression.
            shdr::dispatch(affine, shdr::extent((count + 63) / 64)) >> // Start the pipeline with the affine shader and enough 64-wide groups.
            shdr::dispatch(clip, shdr::extent((count + 31) / 32), shdr::backend::cpu) >> // Force clipping onto the CPU so the shared buffer crosses both backend boundaries.
            shdr::dispatch(sum, shdr::extent(1)); // Return to the selected backend for the final one-invocation reduction.
        run.run(pipeline); // Execute the configured pipeline on the requested backend.
        float expected = 0.0f; // Start the independent scalar reference at zero.
        const shdr::buffer<float>& read_input = input; // Use a constant buffer view so GPU writes synchronize before host reads.
        for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
            expected += std::min(7.0f, std::max(0.0f, read_input.data()[i] * 4.0f - 1.0f)); // Apply the same affine and clipping formula directly to the original input.
        const shdr::buffer<float>& read_result = result; // Use a constant buffer view so GPU writes synchronize before host reads.
        shdr_examples::require_close(read_result.data()[0], expected, "hybrid pipeline sum"); // Check the computed value against the independent reference.
        std::cout << (run.gpu() ? "GPU/CPU/GPU" : "CPU") << " pipeline: " << read_result.data()[0] // Write this part of the example result or diagnostic.
                  << "\n"; // Write this part of the example result or diagnostic.
        return 0; // Report successful completion.
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
    } catch (const shdr_examples::context_unavailable& error) { // Report the exception from the preceding operation.
        std::cerr << "SKIP: " << error.what() << "\n"; // Write this part of the example result or diagnostic.
        return 77; // Report an unavailable optional GPU backend.
#endif // Finish this conditional compilation branch.
    } catch (const std::exception& error) { // Report the exception from the preceding operation.
        std::cerr << error.what() << "\n"; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
