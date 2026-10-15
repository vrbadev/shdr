#include "runner.hpp" // Include runner.hpp declarations.

/// Build and verify three linked stages from the exact same C++/GLSL sources.
int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Select ordinary CPU execution or the requested headless GPU profile.
        const unsigned count = 257; // Uneven sizes exercise shader bounds guards.
        shdr::buffer<float> input(count), scaled(count), clipped(count), result(1); // Allocate one resource for each stage boundary and a one-element reduction result.
        shdr_examples::randomize(input, 1101u, -4.0f, 12.0f); // Fill the input with reproducible random values.

        auto affine_owner = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
        shdr::kernel& affine = *affine_owner; // Keep the affine instance alive while its pipeline stages refer to it.
        affine.bind("input_values", input); // Read the resource produced by the preceding stage or the original host input.
        affine.bind("output_values", scaled); // Write the resource consumed by the next stage or the final host check.
        affine.set<unsigned>("count", count); // Guard the partial final workgroup using the actual number of input values.
        affine.set<float>("scale", 2.0f); // Double each input element in the first stage.
        affine.set<float>("offset", 1.0f); // Add one after scaling each element.

        auto clip_owner = shdr::make_kernel("clamp_kernel"); // Create the linked clamp kernel instance.
        shdr::kernel& clip = *clip_owner; // Keep the clipping instance alive through pipeline execution.
        clip.bind("input_values", scaled); // Read the resource produced by the preceding stage or the original host input.
        clip.bind("output_values", clipped); // Write the resource consumed by the next stage or the final host check.
        clip.set<unsigned>("count", count); // Guard the partial final workgroup using the actual number of input values.
        clip.set<float>("lower", 0.0f); // Clip values below zero in the second stage.
        clip.set<float>("upper", 8.0f); // Clip values above eight in the second stage.

        auto sum_owner = shdr::make_kernel("sum_kernel"); // Create the linked sum kernel instance.
        shdr::kernel& sum = *sum_owner; // Keep the reduction instance alive through pipeline execution.
        sum.bind("input_values", clipped); // Read the resource produced by the preceding stage or the original host input.
        sum.bind("output_values", result); // Write the resource consumed by the next stage or the final host check.
        sum.set<unsigned>("count", count); // Guard the partial final workgroup using the actual number of input values.

        // Resources are linked by binding the same object to adjacent shader ports.
        shdr::pipeline pipeline = shdr::dispatch(affine, shdr::extent((count + 63) / 64)) >> // Start with enough 64-wide groups to cover all original inputs.
                                  shdr::dispatch(clip, shdr::extent((count + 31) / 32)) >> // Run the linked clip shader with its own 32-wide workgroup size.
                                  shdr::dispatch(sum, shdr::extent(1)); // Finish with one deterministic reduction invocation over the clipped resource.
        run.run(pipeline); // Execute the configured pipeline on the requested backend.

        float expected = 0.0f; // Start the independent host reference sum at zero.
        const shdr::buffer<float>& read_input = input; // Use a constant buffer view so GPU writes synchronize before host reads.
        for (unsigned i = 0; i < count; ++i) // Apply the same scale, offset and clipping formula to every original input.
            expected += std::min(8.0f, std::max(0.0f, read_input.data()[i] * 2.0f + 1.0f)); // Accumulate the reference from the original inputs and the stated scale, offset and clipping formula.
        const shdr::buffer<float>& read_result = result; // Use a constant buffer view so GPU writes synchronize before host reads.
        shdr_examples::require_close(read_result.data()[0], expected, "pipeline sum"); // Check the computed value against the independent reference.
        std::cout << (run.gpu() ? "GPU" : "CPU") << " buffer pipeline: " << read_result.data()[0] // Write this part of the example result or diagnostic.
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
