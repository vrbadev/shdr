/** @file real_data.cpp
 * @brief Evaluate the example algorithms with public images, signals, events and digits.
 * Every requested GPU profile receives the same inputs and initial mutable state as CPU.
 * Independent randomized references remain part of the individual example tests.
 */
#include "real_data_support.hpp" // Read documented public fixtures and compare complete outputs.
#include "fft_support.hpp" // Build native radix-two FFT and frequency-filter pipelines.
#include "signal_filters.hpp" // Build causal FIR and stateful biquad pipelines.
#include "evt3_support.hpp" // Decode the original public EVT3 recording prefix.
#include "evt3_flow_checks.hpp" // Compare event flow with an independent double-precision fit.
#include "neural_support.hpp" // Train and infer the small ANN and CNN examples.
#include "transformer_support.hpp" // Train and infer the small attention model.
#include "lenet_support.hpp" // Run the documented modern LeNet digit classifier.
#include "lenet_checks.hpp" // Validate every named-model stage with an independent double reference.
#ifdef SHDR_PLUGIN_PATH // Enable the platform-specific dynamic-library example when it is built.
#include <dlfcn.h> // Load the documented shader plugin through the ordinary POSIX interface.
#endif // Compile the plugin case on platforms with the dynamic-loader API.
#include <chrono> // Bound asynchronous validation waits by ordinary completion objects.
#include <future> // Observe queued CPU work through its completion future.
#include <numeric> // Summarize classification accuracy and signal energy.
#include <thread> // Yield while an asynchronous device submission is incomplete.

namespace { // Keep the real-data evaluator's case wiring private to this executable.
using shdr_examples::runner; // Reuse the other examples' backend selection syntax.
namespace data = shdr_examples::real_data; // Keep fixture and output-check names short.
using glsl::uvec4; // Store exact decoded event records.
using glsl::vec2; // Store complex samples and image derivatives in native shader layout.
using glsl::vec4; // Store RGBA pixels, predictions and optical-flow results.
const unsigned width = 64u, height = 64u, // Match the prepared NASA image size.
               count = width * height; // Match the prepared NASA image size.

std::vector<float> image() { // Convert the public RGB image into a scalar luminance fixture.
    const auto rgb = data::bytes("earth_rgb.bin"); // Read the documented 64 by 64 RGB8 image.
    if (rgb.size() != count * 3u) { // Require the preparation script's exact image shape.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "NASA image fixture must contain 64 by 64 RGB8 pixels"); // Reject a mismatched fixture.
    } // Every pixel has three complete published color components.
    std::vector<float> result(count); // Allocate one normalized luminance value per pixel.
    for (unsigned i = 0; i < count; ++i) { // Visit every real photograph sample.
        result[i] = // Apply the documented RGB luminance weights.
            (0.2126f * rgb[3u * i] + 0.7152f * rgb[3u * i + 1u] + 0.0722f * rgb[3u * i + 2u]) / // Apply the documented RGB luminance weights.
            255.0f; // Apply the documented RGB luminance weights.
    } // The resulting scalar input can feed all image and resource examples.
    return result; // Return a reproducible real-data image to each case.
} // Load the prepared RGB bytes directly for native image processing.
void basic(runner& run, // Exercise linked resources and asynchronous execution with image samples.
           const std::vector<float>& // Exercise linked resources and asynchronous execution with image samples.
               source) { // Exercise linked resources and asynchronous execution with image samples.
    auto input = data::resource(source); // Share the same original real pixels with all stages.
    shdr::buffer<float> scaled(count), clipped(count), // Allocate intermediate and reduction results.
        total(1); // Allocate intermediate and reduction results.
    auto affine = shdr::make_kernel("affine_kernel"); // Create the scale-and-offset shader.
    auto clamp = shdr::make_kernel("clamp_kernel"); // Create the range-limiting shader.
    auto sum = shdr::make_kernel("sum_kernel"); // Create the deterministic reduction shader.
    affine->bind("input_values", input) // Connect the input_values shader port to input storage.
        .bind("output_values", scaled) // Connect the output_values shader port to scaled storage.
        .set("count", count) // Supply the count processing parameter.
        .set("scale", 1.7f) // Supply the scale processing parameter.
        .set("offset", -0.2f); // Connect the first transform.
    clamp->bind("input_values", scaled) // Connect the input_values shader port to scaled storage.
        .bind("output_values", clipped) // Connect the output_values shader port to clipped storage.
        .set("count", count) // Supply the count processing parameter.
        .set("lower", 0.0f) // Supply the lower processing parameter.
        .set("upper", 1.0f); // Connect range limiting.
    sum->bind("input_values", clipped) // Connect the input_values shader port to clipped storage.
        .bind("output_values", total) // Connect the output_values shader port to total storage.
        .set("count", count); // Reduce the transformed pixels.
    const auto stages = // Link three ordinary resource stages.
        shdr::dispatch(*affine, shdr::extent((count + 63u) / 64u)) >> // Dispatch the affine stage with its configured geometry.
        shdr::dispatch(*clamp, shdr::extent((count + 31u) / 32u)) >> // Dispatch the clamp stage with its configured geometry.
        shdr::dispatch(*sum, shdr::extent(1)); // Link three ordinary resource stages.
    data::checked_output<float> scaled_check(scaled), clipped_check(clipped), // Keep the complete stage output for the backend comparison.
        total_check(total); // Check intermediate values as well as the final sum.
    data::evaluate( // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
        run, // Select the same execution backend for this complete case.
        "NASA linked affine/clamp/reduction", // Use the exact diagnostic or comparison label for this case.
        stages, // Run the complete processing sequence in dependency order.
        {&scaled_check, &clipped_check, &total_check}); // Compare complete CPU and GPU outputs.
    double expected = 0.0; // Accumulate an independent host reduction.
    for (unsigned i = 0; i < count; // Check each transform against ordinary scalar arithmetic.
         ++i) { // Check each transform against ordinary scalar arithmetic.
        const float value = // Apply the documented transformation independently.
            std::max(0.0f, // Apply the documented transformation independently.
                     std::min(1.0f, // Apply the documented transformation independently.
                              source[i] * 1.7f - // Apply the documented transformation independently.
                                  0.2f)); // Apply the documented transformation independently.
        shdr_examples::require_close(static_cast<const shdr::buffer<float>&>(clipped).data()[i], // Validate the actual transformed pixel.
                                     value, // Validate the actual transformed pixel.
                                     "real affine/clamp"); // Validate the actual transformed pixel.
        expected += value; // Sum with a wider independent accumulator.
    } // Complete transformed data has an independent reference.
    shdr_examples::require_close(static_cast<const shdr::buffer<float>&>(total).data()[0], // Allow long float summation error.
                                 float(expected), // Allow long float summation error.
                                 "real reduction", // Use the exact diagnostic or comparison label for this case.
                                 0.02f); // Allow long float summation error.
    const auto hybrid = // Force one stage onto CPU in a mixed pipeline.
        shdr::dispatch(*affine, shdr::extent((count + 63u) / 64u)) >> // Dispatch the affine stage with its configured geometry.
        shdr::dispatch(*clamp, shdr::extent((count + 31u) / 32u), shdr::backend::cpu) >> // Dispatch the clamp stage with its configured geometry.
        shdr::dispatch(*sum, shdr::extent(1)); // Force one stage onto CPU in a mixed pipeline.
    data::evaluate( // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
        run, // Select the same execution backend for this complete case.
        "NASA mixed CPU/GPU resources", // Use the exact diagnostic or comparison label for this case.
        hybrid, // Validate real-data ownership changes between backends.
        {&clipped_check, &total_check}); // Validate real-data ownership changes between backends.
    if (run.gpu()) { // Exercise the library's headless asynchronous GPU path.
        stages.prepare(*run.device()); // Compile programs and allocate resources before polling.
        auto completion = // Queue the GPU pipeline and retain its completion handle.
            stages.submit(*run.device()); // Queue the GPU pipeline and retain its completion handle.
        while (!completion.ready()) { // Poll the completion fence from the context thread.
            std::this_thread:: // Allow other application work while processing remains in flight.
                yield(); // Allow other application work while processing remains in flight.
        } // All dependent dispatches have now completed.
    } else { // Exercise the bounded asynchronous CPU path with the same real input.
        shdr::executor executor(2, 2); // Create a small worker pool and bounded queue.
        executor.submit(stages).get(); // Observe the submitted linked pipeline's completion.
    } // Either backend uses the same captured pipeline resources.
    (void)total_check // Compare the asynchronous result with the earlier completed reference.
        .verify(); // Compare the asynchronous result with the earlier completed reference.
    std::cout // Report the completed real-data processing check.
        << "NASA asynchronous resource submission: verified\n"; // Report the real-data completion check.
#ifdef SHDR_PLUGIN_PATH // Exercise the built dynamic shader with the same real image values.
    void* library = // Load only the example plugin supplied by this build.
        dlopen(SHDR_PLUGIN_PATH, // Load only the example plugin supplied by this build.
               RTLD_NOW | RTLD_LOCAL); // Load only the example plugin supplied by this build.
    if (!library) { // Report a genuine plugin-loading failure.
        throw shdr::error(dlerror()); // Preserve the loader's useful diagnostic.
    } // The library stays open until its resources and pipeline are destroyed.
    typedef shdr::kernel* (*create_function)(); // Describe the documented exported kernel factory.
    typedef void (*destroy_function)( // Describe the matching plugin-owned destructor.
        shdr::kernel*); // Describe the matching plugin-owned destructor.
    const auto create = reinterpret_cast<create_function>( // Resolve the example's factory entry point.
        dlsym(library, "shdr_create_kernel")); // Resolve the example's factory entry point.
    const auto destroy = reinterpret_cast<destroy_function>( // Resolve the matching destruction entry point.
        dlsym(library, "shdr_destroy_kernel")); // Resolve the matching destruction entry point.
    if (!create || !destroy) { // Require both shader creation and destruction exports from the plugin.
        dlclose(library); // Release the unused library handle before reporting failure.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "real-data shader plugin has an incomplete factory interface"); // Explain the missing contract.
    } // Both exported functions are now available while the library is retained.
    shdr::kernel* dynamic = // Construct the same dual-compiled affine shader inside the library.
        create(); // Construct the same dual-compiled affine shader inside the library.
    if (!dynamic) { // Reject an unusable factory result.
        dlclose(library); // Release the plugin handle after an empty factory result.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "real-data shader plugin returned no kernel"); // Report the failed factory.
    } // The dynamic kernel can share the ordinary real-data input resources.
    try { // Keep destruction and unloading correct when validation fails.
        dynamic->bind("input_values", input) // Connect the input_values shader port to input storage.
            .bind("output_values", scaled) // Connect the output_values shader port to scaled storage.
            .set("count", count) // Supply the count processing parameter.
            .set("scale", 1.7f) // Supply the scale processing parameter.
            .set("offset", // Bind real pixels through the plugin's generic resource interface.
                 -0.2f); // Bind real pixels through the plugin's generic resource interface.
        const auto dynamic_stages = // Link the loaded stage with built-in stages.
            shdr::dispatch(*dynamic, shdr::extent((count + 63u) / 64u)) >> // Dispatch the dynamic stage with its configured geometry.
            shdr::dispatch(*clamp, shdr::extent((count + 31u) / 32u)) >> // Dispatch the clamp stage with its configured geometry.
            shdr::dispatch(*sum, shdr::extent(1)); // Link the loaded stage with built-in stages.
        data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                       "NASA dynamically loaded shader pipeline", // Use the exact diagnostic or comparison label for this case.
                       dynamic_stages, // Compare all complete real-data plugin outputs.
                       {&scaled_check, // Compare all complete real-data plugin outputs.
                        &clipped_check, // Compare all complete real-data plugin outputs.
                        &total_check}); // Compare all complete real-data plugin outputs.
    } catch (...) { // Clean up plugin ownership before propagating an actual validation failure.
        destroy(dynamic); // Destroy the kernel with the library's own matching function.
        dlclose(library); // Unload only after the pipeline and kernel have been released.
        throw; // Preserve the original useful validation diagnostic.
    } // The temporary linked pipeline has now left its lifetime scope.
    destroy(dynamic); // Release the plugin-owned shader object after all pipeline use.
    dlclose(library); // Release the library only after all live plugin objects are destroyed.
#endif // Finish the optional real-data dynamic-library evaluation.
} // Generic pipeline, hybrid and async examples now have real-data functional coverage.
void images(runner& run, // Move real pixels through image storage and an image filter.
            const std::vector<float>& // Move real pixels through image storage and an image filter.
                source) { // Move real pixels through image storage and an image filter.
    auto input = data::resource(source); // Allocate the original real scalar pixel resource.
    shdr::buffer<float> output(count); // Receive the filtered image as an ordinary buffer.
    shdr::image2d<float, 1> original(width, height), // Allocate portable r32f image resources.
        filtered(width, height); // Allocate portable r32f image resources.
    auto upload = shdr::make_kernel("buffer_to_image_kernel"); // Create the buffer-to-image stage.
    auto blur = shdr::make_kernel("blur_kernel"); // Create the explicit clamp-to-edge box filter.
    auto download = // Create the image-to-buffer stage.
        shdr::make_kernel("image_to_buffer_kernel"); // Create the image-to-buffer stage.
    upload->bind("input_values", input) // Connect the input_values shader port to input storage.
        .bind("output_image", original); // Connect original scalar pixels to image storage.
    blur->bind("input_image", original) // Connect the input_image shader port to original storage.
        .bind("output_image", filtered); // Connect the filtered image resource.
    download->bind("input_image", filtered) // Connect the input_image shader port to filtered storage.
        .bind("output_values", output); // Connect image results to a typed output buffer.
    const shdr::extent groups(width / 8u, height / 8u); // Cover every prepared image pixel.
    const auto stages = shdr::dispatch(*upload, groups) >> shdr::dispatch(*blur, groups) >> // Dispatch the upload stage with its configured geometry.
                        shdr::dispatch(*download, groups); // Link the image roundtrip and filter.
    data::checked_output<float> check(output); // Record the complete output image.
    data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                   "NASA buffer/image/box-filter pipeline", // Use the exact diagnostic or comparison label for this case.
                   stages, // Run the complete processing sequence in dependency order.
                   {&check}); // Compare full CPU and GPU image processing.
    for (unsigned y = 0; y < height; ++y) { // Independently validate each real image row.
        for (unsigned x = 0; x < width; ++x) { // Independently validate each output pixel.
            double sum = 0.0; // Use a double host accumulator for the reference neighborhood.
            for (int dy = -1; dy <= 1; ++dy) { // Visit the three reference neighborhood rows.
                for (int dx = -1; dx <= 1; // Visit the three reference neighborhood columns.
                     ++dx) { // Visit the three reference neighborhood columns.
                    const unsigned xx = unsigned( // Clamp the reference X coordinate.
                        std::max(0, // Clamp the reference X coordinate.
                                 std::min(int(width) - 1, // Clamp the reference X coordinate.
                                          int(x) + dx))); // Clamp the reference X coordinate.
                    const unsigned yy = unsigned( // Clamp the reference Y coordinate.
                        std::max(0, // Clamp the reference Y coordinate.
                                 std::min(int(height) - 1, // Clamp the reference Y coordinate.
                                          int(y) + dy))); // Clamp the reference Y coordinate.
                    sum += source[yy * width + xx]; // Add the original real image sample.
                } // Every horizontal neighbor has been included.
            } // Every neighborhood row has been included.
            shdr_examples::require_close( // Compare the independent local average.
                static_cast<const shdr::buffer<float>&>(output).data()[y * width + x], // Compare the independent local average.
                float(sum / 9.0), // Compare the independent local average.
                "real image box filter"); // Compare the independent local average.
        } // This complete row matches the documented image boundary policy.
    } // All real image pixels have independent filter references.
} // Complete the typed image-resource and independent box-filter comparisons.
void vision(runner& run, // Run the complete corner-detection chain on the public image.
            const std::vector<float>& // Run the complete corner-detection chain on the public image.
                source) { // Run the complete corner-detection chain on the public image.
    auto input = // Share original real pixels with the first smoothing stage.
        data::resource(source); // Share original real pixels with the first smoothing stage.
    shdr::buffer<float> horizontal(count), blurred(count), // Allocate Gaussian and Harris scalar outputs.
        response(count); // Allocate Gaussian and Harris scalar outputs.
    shdr::buffer<vec2> gradients(count); // Allocate the two Sobel components for each pixel.
    shdr::buffer<vec4> tensors(count); // Allocate the structure tensor components for each pixel.
    shdr::buffer<unsigned> corners(count); // Allocate the exact nonmaximum-suppression mask.
    auto gx = shdr::make_kernel( // Create the first separable Gaussian stage.
        "gaussian_horizontal_kernel"); // Create the first separable Gaussian stage.
    auto gy = shdr::make_kernel( // Create the second separable Gaussian stage.
        "gaussian_vertical_kernel"); // Create the second separable Gaussian stage.
    auto sobel = shdr::make_kernel("sobel_kernel"); // Create the image derivative stage.
    auto tensor = shdr::make_kernel("harris_tensor_kernel"); // Create the per-pixel tensor stage.
    auto harris = // Create the smoothed Harris response stage.
        shdr::make_kernel("harris_response_kernel"); // Create the smoothed Harris response stage.
    auto nms = shdr::make_kernel("nonmax_kernel"); // Create the deterministic peak-selection stage.
    gx->bind("input_values", input) // Connect the input_values shader port to input storage.
        .bind("output_values", horizontal); // Connect the horizontal Gaussian pass.
    gy->bind("input_values", horizontal) // Connect the input_values shader port to horizontal storage.
        .bind("output_values", blurred); // Connect the vertical Gaussian pass.
    sobel->bind("input_values", blurred) // Connect the input_values shader port to blurred storage.
        .bind("output_values", gradients); // Derive gradients from the blurred real image.
    tensor->bind("input_values", gradients) // Connect the input_values shader port to gradients storage.
        .bind("output_values", tensors) // Connect the output_values shader port to tensors storage.
        .set("count", count); // Form tensors from both derivative components.
    harris->bind("input_values", tensors) // Connect the input_values shader port to tensors storage.
        .bind("output_values", response) // Connect the output_values shader port to response storage.
        .set("harris_k", 0.04f); // Evaluate the standard Harris measure.
    nms->bind("input_values", response) // Connect the input_values shader port to response storage.
        .bind("output_values", corners) // Connect the output_values shader port to corners storage.
        .set("threshold", 0.000001f); // Select positive local peaks with a fixed threshold.
    for (shdr::kernel* shader : {gx.get(), // Set dimensions for every pixel-indexed stage.
                                 gy.get(), // Set dimensions for every pixel-indexed stage.
                                 sobel.get(), // Set dimensions for every pixel-indexed stage.
                                 harris.get(), // Set dimensions for every pixel-indexed stage.
                                 nms.get()}) { // Set dimensions for every pixel-indexed stage.
        shader->set("width", width) // Supply the width processing parameter.
            .set("height", height); // Use the same real image geometry throughout.
    } // The tensor stage needs only its element count.
    const shdr::extent groups((count + 63u) / 64u); // Cover every flattened real image pixel.
    const auto stages = shdr::dispatch(*gx, groups) >> shdr::dispatch(*gy, groups) >> // Dispatch the gx stage with its configured geometry.
                        shdr::dispatch(*sobel, groups) >> shdr::dispatch(*tensor, groups) >> // Dispatch the sobel stage with its configured geometry.
                        shdr::dispatch(*harris, groups) >> // Dispatch the harris stage with its configured geometry.
                        shdr::dispatch(*nms, groups); // Link the full vision pipeline.
    data::checked_output<float> blur_check(blurred), // Keep the complete stage output for the backend comparison.
        response_check(response); // Compare Gaussian and Harris scalar outputs.
    data::checked_output<vec2> gradient_check(gradients); // Compare both Sobel components.
    data::checked_output<vec4> tensor_check(tensors); // Compare every structure tensor component.
    data::checked_output<unsigned> corner_check(corners, // Keep the complete stage output for the backend comparison.
                                                0.0); // Require identical selected corner masks.
    data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                   "NASA Gaussian/Sobel/Harris/nonmaximum suppression", // Use the exact diagnostic or comparison label for this case.
                   stages, // Run the complete processing sequence in dependency order.
                   {&blur_check, // Validate every meaningful stage output.
                    &gradient_check, // Validate every meaningful stage output.
                    &tensor_check, // Validate every meaningful stage output.
                    &response_check, // Validate every meaningful stage output.
                    &corner_check}); // Validate every meaningful stage output.
    const unsigned* mask = // Read the completed peak mask.
        static_cast<const shdr::buffer<unsigned>&>(corners).data(); // Read the completed peak mask.
    const unsigned peaks = // Count the selected real image corners.
        std::accumulate(mask, mask + count, 0u); // Count the selected real image corners.
    if (!peaks || // Require meaningful real detections while preserving a valid mask count.
        peaks > count) { // Require meaningful real detections while preserving a valid mask count.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "real image corner detector produced no positive peaks"); // Report a failed useful real-data response.
    } // A public image has measurable local structure at this threshold.
    std::cout << "NASA selected Harris peaks: " << peaks // Report the completed real-data processing check.
              << '\n'; // Record the actual real-data corner count.
} // The guide distinguishes backend parity from annotated corner ground truth.
void spectral( // Filter and transform real image samples through native spectral stages.
    runner& run, // Filter and transform real image samples through native spectral stages.
    const std::vector<float>& // Filter and transform real image samples through native spectral stages.
        source) { // Filter and transform real image samples through native spectral stages.
    const glsl::uvec2 size(16u, 16u); // Keep the direct two-dimensional reference transform small.
    const unsigned samples = size.x * size.y; // Count the cropped real image samples.
    std::vector<vec2> original(samples); // Store real image values as complex samples.
    for (unsigned y = 0; y < size.y; // Read the selected image region from the prepared public RGB data.
         ++y) { // Read the selected image region from the prepared public RGB data.
        for (unsigned x = 0; x < size.x; ++x) { // Copy each pixel of the selected crop.
            original[y * size.x + x] = vec2(source[(y + 24u) * width + x + 24u], // Select the same central crop on every host.
                                            0.0f); // Select the same central crop on every host.
        } // Each row preserves its original image order.
    } // The crop supplies a real signal for convolution and Fourier processing.
    auto input = data::resource(original); // Bind the original image crop as a complex resource.
    shdr::buffer<vec2> convolved(samples), row_spectrum(samples), spectrum(samples), // Allocate complete spectral intermediates.
        inverse_rows(samples), restored(samples); // Allocate complete spectral intermediates.
    auto weights = data::resource( // Define a normalized cross-shaped convolution kernel.
        std::vector<float>{0.0f, // Define a normalized cross-shaped convolution kernel.
                           0.125f, // Define a normalized cross-shaped convolution kernel.
                           0.0f, // Define a normalized cross-shaped convolution kernel.
                           0.125f, // Define a normalized cross-shaped convolution kernel.
                           0.5f, // Define a normalized cross-shaped convolution kernel.
                           0.125f, // Define a normalized cross-shaped convolution kernel.
                           0.0f, // Define a normalized cross-shaped convolution kernel.
                           0.125f, // Define a normalized cross-shaped convolution kernel.
                           0.0f}); // Define a normalized cross-shaped convolution kernel.
    auto convolution = shdr::make_kernel( // Create the true two-dimensional convolution stage.
        "convolution2d_kernel"); // Create the true two-dimensional convolution stage.
    convolution->bind("input_values", input) // Connect the input_values shader port to input storage.
        .bind("output_values", convolved) // Connect the output_values shader port to convolved storage.
        .bind("coefficients", weights) // Connect the coefficients shader port to weights storage.
        .set("dimensions", size) // Supply the dimensions processing parameter.
        .set("kernel_size", glsl::uvec2(3u, 3u)); // Link pixels and dynamic coefficients.
    auto rows = shdr::make_kernel("dft_rows_kernel"); // Create the forward row transform.
    auto columns = shdr::make_kernel("dft_columns_kernel"); // Create the forward column transform.
    auto inverse_row = // Create the normalized inverse row transform.
        shdr::make_kernel("dft_rows_kernel"); // Create the normalized inverse row transform.
    auto inverse_column = // Create the normalized inverse column transform.
        shdr::make_kernel("dft_columns_kernel"); // Create the normalized inverse column transform.
    rows->bind("input_values", convolved) // Connect the input_values shader port to convolved storage.
        .bind("output_values", row_spectrum) // Connect the output_values shader port to row_spectrum storage.
        .set("dimensions", size) // Supply the dimensions processing parameter.
        .set("direction", -1.0f) // Supply the direction processing parameter.
        .set("normalization", 1.0f); // Transform real filtered image rows.
    columns->bind("input_values", row_spectrum) // Connect the input_values shader port to row_spectrum storage.
        .bind("output_values", spectrum) // Connect the output_values shader port to spectrum storage.
        .set("dimensions", size) // Supply the dimensions processing parameter.
        .set("direction", -1.0f) // Supply the direction processing parameter.
        .set("normalization", 1.0f); // Complete the full two-dimensional spectrum.
    inverse_row->bind("input_values", spectrum) // Connect the input_values shader port to spectrum storage.
        .bind("output_values", inverse_rows) // Connect the output_values shader port to inverse_rows storage.
        .set("dimensions", size) // Supply the dimensions processing parameter.
        .set("direction", 1.0f) // Supply the direction processing parameter.
        .set("normalization", 1.0f / size.x); // Reconstruct row-domain values.
    inverse_column->bind("input_values", inverse_rows) // Connect the input_values shader port to inverse_rows storage.
        .bind("output_values", restored) // Connect the output_values shader port to restored storage.
        .set("dimensions", size) // Supply the dimensions processing parameter.
        .set("direction", 1.0f) // Supply the direction processing parameter.
        .set("normalization", 1.0f / size.y); // Reconstruct the filtered real image.
    const shdr::extent groups(2u, 2u); // Cover the sixteen by sixteen output geometry.
    const auto stages = // Keep transform dependencies on the selected backend.
        shdr::dispatch(*convolution, groups) >> shdr::dispatch(*rows, groups) >> // Dispatch the convolution stage with its configured geometry.
        shdr::dispatch(*columns, groups) >> shdr::dispatch(*inverse_row, groups) >> // Dispatch the columns stage with its configured geometry.
        shdr::dispatch(*inverse_column, // Dispatch the inverse_column stage with its configured geometry.
                       groups); // Keep transform dependencies on the selected backend.
    data::checked_output<vec2> convolution_check(convolved), spectrum_check(spectrum, 0.002), // Keep the complete stage output for the backend comparison.
        restored_check(restored); // Record filtered pixels, frequencies and reconstructed samples.
    data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                   "NASA 2D convolution/DFT/inverse", // Use the exact diagnostic or comparison label for this case.
                   stages, // Run the complete processing sequence in dependency order.
                   {&convolution_check, // Compare every meaningful spectral result.
                    &spectrum_check, // Compare every meaningful spectral result.
                    &restored_check}); // Compare every meaningful spectral result.
    const vec2* expected = // Read the filtered real image before its spectral roundtrip.
        static_cast<const shdr::buffer<vec2>&>(convolved) // Read the filtered real image before its spectral roundtrip.
            .data(); // Read the filtered real image before its spectral roundtrip.
    const vec2* actual = static_cast<const shdr::buffer<vec2>&>(restored) // Read the reconstructed image after both inverse stages.
                             .data(); // Read the reconstructed image after both inverse stages.
    for (unsigned i = 0; i < samples; // Check an independent roundtrip invariant on every pixel.
         ++i) { // Check an independent roundtrip invariant on every pixel.
        shdr_examples::require_close(actual[i].x, // Verify the real image amplitude.
                                     expected[i].x, // Verify the real image amplitude.
                                     "real 2D inverse reconstruction", // Use the exact diagnostic or comparison label for this case.
                                     0.0001f); // Verify the real image amplitude.
        shdr_examples::require_close( // Verify that the reconstructed real image has a zero imaginary component.
            actual[i].y, // Verify that the reconstructed real image has a zero imaginary component.
            0.0f, // Verify that the reconstructed real image has a zero imaginary component.
            "real 2D inverse imaginary component", // Use the exact diagnostic or comparison label for this case.
            0.0001f); // Verify that the reconstructed real image has a zero imaginary component.
    } // The forward and inverse transforms agree on the real image fixture.
    const auto digits = // Read real digit images for a genuine image-stack volume.
        data::bytes("mnist_test.bin"); // Read real digit images for a genuine image-stack volume.
    const glsl::uvec3 volume_size( // Interpret eight digit images as separate depth planes.
        28u, 28u, 8u); // Interpret eight digit images as separate depth planes.
    std::vector<float> volume_values( // Allocate the volume formed by stacking real digit images along its depth axis.
        volume_size.x * volume_size.y * // Allocate the volume formed by stacking real digit images along its depth axis.
        volume_size.z); // Allocate the volume formed by stacking real digit images along its depth axis.
    for (unsigned i = 0; i < volume_values.size(); // Normalize the original dataset byte values.
         ++i) { // Normalize the original dataset byte values.
        volume_values[i] = // Preserve the first eight published digit images in order.
            digits.at(i) / 255.0f; // Preserve the first eight published digit images in order.
    } // Use each original digit image as one depth slice of the convolution input.
    auto volume = // Share the real volume with native convolution stages.
        data::resource(volume_values); // Share the real volume with native convolution stages.
    shdr::buffer<float> output(volume_values.size()); // Allocate the filtered real image stack.
    auto coefficients = data::resource( // Define a normalized three-dimensional box kernel.
        std::vector<float>(27u, 1.0f / 27.0f)); // Define a normalized three-dimensional box kernel.
    auto filter = shdr::make_kernel( // Create the dynamic three-dimensional convolution stage.
        "convolution3d_kernel"); // Create the dynamic three-dimensional convolution stage.
    filter->bind("input_values", volume) // Connect the input_values shader port to volume storage.
        .bind("output_values", output) // Connect the output_values shader port to output storage.
        .bind("coefficients", coefficients) // Connect the coefficients shader port to coefficients storage.
        .set("dimensions", volume_size) // Supply the dimensions processing parameter.
        .set("kernel_size", // Connect real voxels and kernel coefficients.
             glsl::uvec3(3u, 3u, 3u)); // Connect real voxels and kernel coefficients.
    data::checked_output<float> volume_check(output); // Compare every real-volume output sample.
    data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                   "MNIST image-stack 3D convolution", // Use the exact diagnostic or comparison label for this case.
                   shdr::pipeline(shdr::dispatch(*filter, shdr::extent(7u, 7u, 2u))), // Dispatch the filter stage with its configured geometry.
                   {&volume_check}); // Validate full CPU and GPU volume processing.
} // Randomized tests separately validate arbitrary 2D and 3D convolution coefficients.
unsigned bayer_channel(unsigned x, // Resolve the color sampled at a Bayer coordinate.
                       unsigned y, // Resolve the color sampled at a Bayer coordinate.
                       unsigned pattern) { // Resolve the color sampled at a Bayer coordinate.
    const unsigned phase = (y % 2u) * 2u + x % 2u; // Identify one of the four pixel phases.
    const unsigned red = pattern == 0u   ? 0u // Follow the documented RGGB/BGGR/GRBG/GBRG order.
                         : pattern == 1u ? 3u // Follow the documented RGGB/BGGR/GRBG/GBRG order.
                         : pattern == 2u ? 1u // Follow the documented RGGB/BGGR/GRBG/GBRG order.
                                         : 2u; // Follow the documented RGGB/BGGR/GRBG/GBRG order.
    return phase == red        ? 0u // Return red, blue or green for this phase.
           : phase == 3u - red ? 2u // Return red, blue or green for this phase.
                               : 1u; // Return red, blue or green for this phase.
} // This host sampler remains separate from the demosaicing shader's interpolation.
std::vector<unsigned> pack_bayer( // Encode real-derived Bayer samples into each supported transport representation.
    const std::vector<unsigned>& samples, // Encode real-derived Bayer samples into each supported transport representation.
    unsigned // Encode real-derived Bayer samples into each supported transport representation.
        packing) { // Encode real-derived Bayer samples into each supported transport representation.
    std::vector<unsigned> bytes; // Accumulate the exact little-endian packed transport bytes.
    for (std::size_t i = 0; i < samples.size();) { // Include every Bayer sensor-site value.
        if (packing == 0u) { // Store one unsigned byte per Bayer8 sample.
            bytes.push_back(samples[i++]); // Copy the original eight-bit value.
        } else if (packing == 1u) { // Store standard little-endian Bayer16 samples.
            const unsigned value = samples[i++]; // Read one complete sixteen-bit sensor value.
            bytes.push_back(value & 255u); // Emit the low transport byte.
            bytes.push_back(value >> 8u); // Emit the high transport byte.
        } else if (packing == 2u) { // Store standard four-sample RAW10 groups.
            unsigned low = 0u; // Collect the two low bits of every sensor sample.
            for (unsigned j = 0; j < 4u; ++j) { // Encode the group's four adjacent sensor values.
                bytes.push_back(samples[i + j] >> 2u); // Emit eight high bits from this sample.
                low |= (samples[i + j] & 3u) // Collect its two low bits in their standard slot.
                       << (2u * j); // Collect its two low bits in their standard slot.
            } // The group now has four high bytes and one low-bit byte.
            bytes.push_back(low); // Emit the combined low bits.
            i += 4u; // Continue with the next four real sensor-site values.
        } else { // Store standard two-sample RAW12 groups.
            bytes.push_back(samples[i] >> 4u); // Emit the first sample's high eight bits.
            bytes.push_back(samples[i + 1u] >> 4u); // Emit the second sample's high eight bits.
            bytes.push_back((samples[i] & 15u) | // Combine both low nibbles.
                            ((samples[i + 1u] & 15u) << 4u)); // Combine both low nibbles.
            i += 2u; // Continue with the next pair of sensor-site values.
        } // Every selected packing preserves the quantized real-derived sensor values.
    } // The complete byte transport is ready for word-oriented shader storage.
    std::vector<unsigned> words((bytes.size() + 3u) / 4u, // Allocate zero-padded uint storage words.
                                0u); // Allocate zero-padded uint storage words.
    for (unsigned i = 0; i < bytes.size(); ++i) { // Assemble all transport bytes explicitly.
        words[i / 4u] |= // Place each byte in its little-endian word slot.
            bytes[i] << (8u * (i % 4u)); // Place each byte in its little-endian word slot.
    } // Place each encoded byte at its explicit little-endian word position.
    return words; // Supply exactly the words consumed by the native unpack stage.
} // Derive each Bayer mosaic and its transport packing from the public RGB image.
void imaging( // Evaluate all Bayer formats and lossless compression with the public color image.
    runner& // Evaluate all Bayer formats and lossless compression with the public color image.
        run) { // Evaluate all Bayer formats and lossless compression with the public color image.
    const auto rgb = data::bytes("earth_rgb.bin"); // Read original real RGB8 fixture values.
    const unsigned depths[] = { // Match Bayer8, Bayer16, RAW10 and RAW12 packing modes.
        8u, 16u, 10u, 12u}; // Match Bayer8, Bayer16, RAW10 and RAW12 packing modes.
    for (unsigned packing = 0; packing < 4u; // Evaluate every supported sample transport layout.
         ++packing) { // Evaluate every supported sample transport layout.
        for (unsigned pattern = 0; pattern < 4u; // Evaluate every supported Bayer color phase.
             ++pattern) { // Evaluate every supported Bayer color phase.
            const unsigned maximum = // Calculate this transport format's full-scale value.
                (1u << depths[packing]) - 1u; // Calculate this transport format's full-scale value.
            std::vector<unsigned> mosaic(count); // Keep one sampled channel per real image pixel.
            for (unsigned y = 0; y < height; // Visit every image row in the derived Bayer mosaic.
                 ++y) { // Visit every image row in the derived Bayer mosaic.
                for (unsigned x = 0; x < width; // Select the channel measured at each sensor site.
                     ++x) { // Select the channel measured at each sensor site.
                    const unsigned i = y * width + x; // Locate the real RGB pixel.
                    mosaic[i] = // Quantize the sampled real color into the selected sensor depth.
                        (unsigned(rgb[3u * i + bayer_channel(x, y, pattern)]) * maximum + 127u) / // Quantize the sampled real color into the selected sensor depth.
                        255u; // Quantize the sampled real color into the selected sensor depth.
                } // All sensor sites in this row have real image-derived samples.
            } // The guide identifies this exact synthetic sampling of a real photograph.
            auto packed = data::resource( // Bind the original packed sensor transport words.
                pack_bayer(mosaic, packing)); // Bind the original packed sensor transport words.
            shdr::buffer<unsigned> decoded(count); // Allocate exact unpacked sensor samples.
            shdr::buffer<vec4> colors(count); // Allocate interpolated real image colors.
            shdr::buffer<float> gray(count); // Allocate the final luminance image.
            auto unpack = // Create the transport decoding stage.
                shdr::make_kernel("bayer_unpack_kernel"); // Create the transport decoding stage.
            auto debayer = // Create the native color interpolation stage.
                shdr::make_kernel("debayer_kernel"); // Create the native color interpolation stage.
            auto luma = // Create the luminance conversion stage.
                shdr::make_kernel("luminance_kernel"); // Create the luminance conversion stage.
            unpack->bind("packed_words", packed) // Connect the packed_words shader port to packed storage.
                .bind("samples", decoded) // Connect the samples shader port to decoded storage.
                .set("count", count) // Supply the count processing parameter.
                .set("packing", packing) // Supply the packing processing parameter.
                .set("bit_depth", depths[packing]); // Decode the selected real-derived transport.
            debayer->bind("samples", decoded) // Connect the samples shader port to decoded storage.
                .bind("colors", colors) // Connect the colors shader port to colors storage.
                .set("width", width) // Supply the width processing parameter.
                .set("height", height) // Supply the height processing parameter.
                .set("pattern", pattern) // Supply the pattern processing parameter.
                .set("bit_depth", depths[packing]); // Interpolate the selected Bayer phase.
            luma->bind("colors", colors) // Connect the colors shader port to colors storage.
                .bind("gray", gray) // Connect the gray shader port to gray storage.
                .set("count", count); // Produce normalized luminance from interpolated colors.
            const auto stages = // Link the complete image transport pipeline.
                shdr::dispatch(*unpack, shdr::extent(count / 64u)) >> // Dispatch the unpack stage with its configured geometry.
                shdr::dispatch(*debayer, shdr::extent(width / 8u, height / 8u)) >> // Dispatch the debayer stage with its configured geometry.
                shdr::dispatch( // Dispatch this configured shader with the following work group geometry.
                    *luma,
                    shdr::extent(count / 64u)); // Link the complete image transport pipeline.
            data::checked_output<unsigned> decoded_check( // Keep the complete stage output for the backend comparison.
                decoded, 0.0); // Require exact unpacked sensor words.
            data::checked_output<vec4> color_check( // Keep the complete stage output for the backend comparison.
                colors); // Compare all interpolated RGBA components.
            data::checked_output<float> gray_check(gray); // Compare final luminance values.
            data::evaluate( // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                run, // Select the same execution backend for this complete case.
                "NASA derived Bayer packing " + std::to_string(packing) + " phase " + // Use the exact diagnostic or comparison label for this case.
                    std::to_string(pattern), // Validate all sixteen real-image transport combinations.
                stages, // Run the complete processing sequence in dependency order.
                {&decoded_check, // Validate all sixteen real-image transport combinations.
                 &color_check, // Validate all sixteen real-image transport combinations.
                 &gray_check}); // Validate all sixteen real-image transport combinations.
            const unsigned* actual = static_cast<const shdr::buffer<unsigned>&>(decoded) // Read the exact unpacked samples.
                                         .data(); // Read the exact unpacked samples.
            if (!std::equal( // Compare with independently packed original sensor-site values.
                    mosaic.begin(), // Compare with independently packed original sensor-site values.
                    mosaic.end(), // Compare with independently packed original sensor-site values.
                    actual)) { // Compare with independently packed original sensor-site values.
                throw shdr::error( // Report the failed format, numerical or resource precondition.
                    "real image Bayer transport was not lossless"); // Report an incorrect sample representation.
            } // Packing and unpacking preserved every derived sensor value exactly.
        } // All four color phases have been processed for this format.
    } // Every Bayer depth and phase receives real-image functional coverage.
    std::vector<unsigned> pixels( // Compress original public RGB bytes as portable uint samples.
        rgb.begin(), rgb.end()); // Compress original public RGB bytes as portable uint samples.
    const unsigned samples = static_cast<unsigned>(pixels.size()), block_size = 64u, // Define fixed-size independent compression tiles.
                   blocks = (samples + block_size - 1u) / // Define fixed-size independent compression tiles.
                            block_size; // Define fixed-size independent compression tiles.
    auto input = data::resource(pixels); // Bind original real image color bytes.
    shdr::buffer<unsigned> sizes(blocks), offsets(blocks + 1u), encoded(samples * 2u), // Allocate bounded lossless codec resources.
        restored(samples); // Allocate bounded lossless codec resources.
    auto counter = // Count run pairs in independent input tiles.
        shdr::make_kernel("rle_count_kernel"); // Count run pairs in independent input tiles.
    auto prefix = // Place each tile's compact encoded interval.
        shdr::make_kernel("rle_offsets_kernel"); // Place each tile's compact encoded interval.
    auto encoder = shdr::make_kernel("rle_encode_kernel"); // Emit the lossless value/count pairs.
    auto decoder = shdr::make_kernel( // Reconstruct original image samples from those pairs.
        "rle_decode_kernel"); // Reconstruct original image samples from those pairs.
    counter->bind("values", input) // Connect the values shader port to input storage.
        .bind("sizes", sizes) // Connect the sizes shader port to sizes storage.
        .set("count", samples) // Supply the count processing parameter.
        .set("block_size", block_size); // Count runs using real image bytes.
    prefix->bind("sizes", sizes) // Connect the sizes shader port to sizes storage.
        .bind("offsets", offsets) // Connect the offsets shader port to offsets storage.
        .set("blocks", blocks); // Assign deterministic encoded offsets.
    encoder->bind("values", input) // Connect the values shader port to input storage.
        .bind("offsets", offsets) // Connect the offsets shader port to offsets storage.
        .bind("encoded", encoded) // Connect the encoded shader port to encoded storage.
        .set("count", samples) // Supply the count processing parameter.
        .set("block_size", block_size); // Encode the original data into disjoint intervals.
    decoder->bind("encoded", encoded) // Connect the encoded shader port to encoded storage.
        .bind("offsets", offsets) // Connect the offsets shader port to offsets storage.
        .bind("restored", restored) // Connect the restored shader port to restored storage.
        .set("count", samples) // Supply the count processing parameter.
        .set("block_size", block_size); // Decode the exact image byte sequence.
    const shdr::extent groups((blocks + 63u) / 64u); // Cover every independent compression tile.
    const auto stages = // Link count, prefix, encode and decode stages.
        shdr::dispatch(*counter, groups) >> shdr::dispatch(*prefix, shdr::extent(1)) >> // Dispatch the counter stage with its configured geometry.
        shdr::dispatch(*encoder, groups) >> // Dispatch the encoder stage with its configured geometry.
        shdr::dispatch(*decoder, groups); // Link count, prefix, encode and decode stages.
    data::checked_output<unsigned> sizes_check(sizes, 0.0), offsets_check(offsets, 0.0), // Keep the complete stage output for the backend comparison.
        restored_check(restored, // Compare compact layout and every reconstructed byte exactly.
                       0.0); // Compare compact layout and every reconstructed byte exactly.
    data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                   "NASA tiled lossless RLE image roundtrip", // Use the exact diagnostic or comparison label for this case.
                   stages, // Run the complete processing sequence in dependency order.
                   {&sizes_check, // Require identical CPU and GPU lossless processing.
                    &offsets_check, // Require identical CPU and GPU lossless processing.
                    &restored_check}); // Require identical CPU and GPU lossless processing.
    if (!std::equal( // Compare decoded values with the original real photograph bytes.
            pixels.begin(), // Compare decoded values with the original real photograph bytes.
            pixels.end(), // Compare decoded values with the original real photograph bytes.
            static_cast<const shdr::buffer<unsigned>&>(restored) // Compare decoded values with the original real photograph bytes.
                .data())) { // Compare decoded values with the original real photograph bytes.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "public image lossless compression roundtrip differs"); // Reject data loss even if both backends agree.
    } // RLE is correct whether this real image compresses well or expands.
    std::cout << "NASA RLE encoded words: " // Report the completed real-data processing check.
              << static_cast<const shdr::buffer<unsigned>&>(offsets).data()[blocks] << " from " // Report actual real-image compression behavior.
              << samples << " input words\n"; // Report actual real-image compression behavior.
} // Report the exact RLE round trip and encoded size for the original image bytes.
void signals( // Analyze the two recorded ECG channels with transform and filter comparisons.
    runner& // Analyze the two recorded ECG channels with transform and filter comparisons.
        run) { // Analyze the two recorded ECG channels with transform and filter comparisons.
    const auto samples = // Read the documented PhysioNet record 100 waveform prefix.
        data::floats("signal_ecg.bin"); // Read the documented PhysioNet record 100 waveform prefix.
    const unsigned length = 2048u, // Match the channel-major fixture shape and radix-two FFT length.
                   channels = 2u; // Match the channel-major fixture shape and radix-two FFT length.
    if (samples.size() != length * channels) { // Require two complete waveform channels.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "ECG fixture must contain two channels of 2048 float32 samples"); // Explain a preparation mismatch.
    } // The waveform's physical sample rate is 360 hertz.
    std::vector<vec2> complex_samples( // Store original real samples in portable complex FFT layout.
        samples.size()); // Store original real samples in portable complex FFT layout.
    for (unsigned i = 0; i < samples.size(); ++i) { // Copy every published waveform value.
        complex_samples[i] = vec2( // Preserve real amplitude and set the imaginary component to zero.
            samples[i], 0.0f); // Preserve real amplitude and set the imaginary component to zero.
    } // Both channels remain independent throughout signal processing.
    auto input = // Share the same real ECG input between spectral cases.
        data::resource(complex_samples); // Share the same real ECG input between spectral cases.
    shdr::buffer<vec2> spectrum(samples.size()), reconstructed(samples.size()), // Allocate reusable spectral outputs.
        output(samples.size()), windowed(samples.size()); // Allocate reusable spectral outputs.
    shdr_examples::fft1d forward( // Plan the unscaled batched forward FFT.
        input, spectrum, length, channels); // Plan the unscaled batched forward FFT.
    shdr_examples::fft1d inverse( // Plan the normalized inverse FFT.
        spectrum, reconstructed, length, channels, true); // Plan the normalized inverse FFT.
    shdr::pipeline transform; // Build a complete spectral roundtrip before execution.
    forward.append_to(transform); // Produce each channel's independent spectrum.
    inverse.append_to(transform); // Reconstruct each original real waveform from that spectrum.
    data::checked_output<vec2> spectrum_check(spectrum, 0.0005), // Keep the complete stage output for the backend comparison.
        reconstructed_check(reconstructed); // Compare every complex output value.
    data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                   "PhysioNet ECG 1D FFT/inverse", // Use the exact diagnostic or comparison label for this case.
                   transform, // Validate real spectral processing on both backends.
                   {&spectrum_check, // Validate real spectral processing on both backends.
                    &reconstructed_check}); // Validate real spectral processing on both backends.
    const vec2* restored = static_cast<const shdr::buffer<vec2>&>(reconstructed) // Read the completed inverse-transform waveform.
                               .data(); // Read the completed inverse-transform waveform.
    for (unsigned i = 0; i < samples.size(); // Check every original real waveform value independently.
         ++i) { // Check every original real waveform value independently.
        shdr_examples::require_close(restored[i].x, // Verify normalized amplitude recovery.
                                     samples[i], // Verify normalized amplitude recovery.
                                     "real ECG FFT reconstruction", // Use the exact diagnostic or comparison label for this case.
                                     0.0001f); // Verify normalized amplitude recovery.
        shdr_examples::require_close(restored[i].y, // Verify a real-valued roundtrip.
                                     0.0f, // Verify a real-valued roundtrip.
                                     "real ECG FFT imaginary residue", // Use the exact diagnostic or comparison label for this case.
                                     0.0001f); // Verify a real-valued roundtrip.
    } // The spectral representation preserves the original ECG input.
    for (unsigned mode = 0; mode < 4u; // Test every block-frequency filtering response on real waveform data.
         ++mode) { // Test every block-frequency filtering response on real waveform data.
        shdr_examples::fft_filter filter(input, // Use explicit physical frequency parameters.
                                         output, // Write results into the separate output resource.
                                         length, // Use the validated per-channel block length.
                                         channels, // Keep every channel independent.
                                         360.0f, // Use explicit physical frequency parameters.
                                         static_cast<shdr_examples::frequency_filter>(mode), // Use explicit physical frequency parameters.
                                         5.0f, // Use explicit physical frequency parameters.
                                         40.0f, // Use explicit physical frequency parameters.
                                         2.0f); // Use explicit physical frequency parameters.
        data::checked_output<vec2> output_check( // Keep the complete stage output for the backend comparison.
            output); // Compare complete reconstructed filtered waveforms.
        data::evaluate( // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
            run, // Select the same execution backend for this complete case.
            "PhysioNet ECG FFT filter " + std::to_string(mode), // Use the exact diagnostic or comparison label for this case.
            filter.processing_pipeline(), // Validate low, high, band and notch responses on both backends.
            {&output_check}); // Validate low, high, band and notch responses on both backends.
    } // Known-tone tests separately check the analytic frequency-response amplitudes.
    auto window = // Create the dynamic amplitude-window stage.
        shdr::make_kernel("signal_window_kernel"); // Create the dynamic amplitude-window stage.
    window->bind("input_values", input) // Connect the input_values shader port to input storage.
        .bind("output_values", windowed) // Connect the output_values shader port to windowed storage.
        .set("block_length", length) // Supply the per-channel block length.
        .set("channels", channels); // Link the original ECG channels to weighted samples.
    shdr_examples::fft1d weighted_fft( // Plan spectral analysis of the weighted real waveform.
        windowed, // Plan spectral analysis of the weighted real waveform.
        spectrum, // Read the completed complex frequency bins.
        length, // Use the validated per-channel block length.
        channels); // Plan spectral analysis of the weighted real waveform.
    shdr::pipeline weighted( // Begin with per-sample amplitude weighting.
        shdr::dispatch(*window, // Dispatch the window stage with its configured geometry.
                       shdr::extent((length * channels + 63u) / // Begin with per-sample amplitude weighting.
                                    64u))); // Begin with per-sample amplitude weighting.
    weighted_fft.append_to(weighted); // Keep window output on the same backend as the transform.
    for (unsigned mode = 0; mode < 4u; // Compare rectangular, Hann, Hamming and Blackman analysis windows.
         ++mode) { // Compare rectangular, Hann, Hamming and Blackman analysis windows.
        window->set("mode", mode); // Select the next documented window formula.
        data::checked_output<vec2> window_check(windowed), // Keep the complete stage output for the backend comparison.
            weighted_check(spectrum, 0.0005); // Compare both weighted input and its spectrum.
        data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                       "PhysioNet ECG spectral window " + std::to_string(mode), // Use the exact diagnostic or comparison label for this case.
                       weighted, // Validate the linked analysis stages.
                       {&window_check, &weighted_check}); // Validate the linked analysis stages.
    } // Each supported window now receives real waveform input.
    auto scalar_input = // Preserve original real samples for causal time-domain filtering.
        data::resource(samples); // Preserve original real samples for causal time-domain filtering.
    shdr::buffer<float> filtered(samples.size()); // Allocate channel-major real filter output.
    for (unsigned mode = 0; mode < 5u; // Exercise every public FIR design, including moving average.
         ++mode) { // Exercise every public FIR design, including moving average.
        const auto selected = static_cast<shdr_examples::filter_mode>( // Select the documented finite-impulse-response type.
            mode); // Select the documented finite-impulse-response type.
        const double lower = // Select the cutoff frequencies used by this waveform-processing demonstration.
            mode == 3u ? 48.0 // Select the cutoff frequencies used by this waveform-processing demonstration.
            : mode == 0u // Select the cutoff frequencies used by this waveform-processing demonstration.
                ? 40.0 // Select the cutoff frequencies used by this waveform-processing demonstration.
                : 5.0; // Select the cutoff frequencies used by this waveform-processing demonstration.
        const double upper = // Define the upper edge for band and notch designs.
            mode == 3u ? 52.0 : 40.0; // Define the upper edge for band and notch designs.
        shdr_examples::fir_pipeline filter(scalar_input, // Design and link a causal 65-tap filter.
                                           filtered, // Design and link a causal 65-tap filter.
                                           length, // Use the validated per-channel block length.
                                           channels, // Keep every channel independent.
                                           selected, // Design and link a causal 65-tap filter.
                                           360.0, // Design and link a causal 65-tap filter.
                                           lower, // Design and link a causal 65-tap filter.
                                           upper, // Design and link a causal 65-tap filter.
                                           65u); // Design and link a causal 65-tap filter.
        data::checked_output<float> check(filtered); // Compare every filtered real sample.
        data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                       "PhysioNet ECG FIR filter " + std::to_string(mode), // Use the exact diagnostic or comparison label for this case.
                       filter.processing_pipeline(), // Evaluate all finite-impulse-response modes.
                       {&check}); // Evaluate all finite-impulse-response modes.
    } // Independent impulse and tone references still test the coefficient mathematics.
    for (unsigned mode = 0; mode < 4u; // Exercise each recursive low, high, band and notch design.
         ++mode) { // Exercise each recursive low, high, band and notch design.
        const auto selected = static_cast<shdr_examples::filter_mode>( // Select the matching public biquad mode.
            mode); // Select the matching public biquad mode.
        shdr_examples::biquad_pipeline filter( // Set an explicit sampling rate, frequency and quality factor.
            scalar_input, // Set an explicit sampling rate, frequency and quality factor.
            filtered, // Set an explicit sampling rate, frequency and quality factor.
            length, // Use the validated per-channel block length.
            channels, // Keep every channel independent.
            selected, // Set an explicit sampling rate, frequency and quality factor.
            360.0, // Set an explicit sampling rate, frequency and quality factor.
            mode < 2u ? 40.0 : 50.0, // Set an explicit sampling rate, frequency and quality factor.
            0.7071067811865476); // Set an explicit sampling rate, frequency and quality factor.
        data::checked_output<float> check( // Keep the complete stage output for the backend comparison.
            filtered); // Compare the complete stateful waveform output.
        data::checked_output<vec2> state_check( // Keep the complete stage output for the backend comparison.
            filter.state); // Compare both delay values for every independent channel.
        data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                       "PhysioNet ECG biquad filter " + std::to_string(mode), // Use the exact diagnostic or comparison label for this case.
                       filter.processing_pipeline(), // Restore identical zero history before each backend run.
                       {&check, &state_check}, // Restore identical zero history before each backend run.
                       [&filter]() { // Restore identical zero history before each backend run.
                           filter.reset_state(); // Restore identical zero history before each backend run.
                       }); // Restore identical zero history before each backend run.
    } // Existing chunked tests separately validate state carried across successive waveform batches.
} // Complete the waveform reconstruction and filter-response comparisons.
std::vector<uvec4> decode_reference( // Interpret sensor words sequentially as an independent real-recording oracle.
    const std::vector<std::uint16_t>& // Interpret sensor words sequentially as an independent real-recording oracle.
        words) { // Interpret sensor words sequentially as an independent real-recording oracle.
    unsigned y = 0u, base = 0u, polarity = 0u, high = 0u, low = 0u, // Track the official scalar EVT3 stream state.
             wraps = 0u; // Track the official scalar EVT3 stream state.
    bool synchronized = false; // Suppress event emission until a TIME_HIGH word establishes time.
    std::vector<uvec4> events; // Preserve exact event order from the public sensor recording.
    const auto append = // Form timestamps independently of parallel block summaries.
        [&](unsigned x, // Form timestamps independently of parallel block summaries.
            unsigned packed_y) { // Form timestamps independently of parallel block summaries.
            const std::uint64_t time = std::uint64_t(wraps) * 16777216u + // Reconstruct extended time in wide host arithmetic.
                                       std::uint64_t(high) * 4096u + // Reconstruct extended time in wide host arithmetic.
                                       low; // Reconstruct extended time in wide host arithmetic.
            events.push_back( // Append one exact coordinate and timestamp record.
                uvec4(x, // Append one exact coordinate and timestamp record.
                      packed_y, // Append one exact coordinate and timestamp record.
                      unsigned(time), // Append one exact coordinate and timestamp record.
                      unsigned(time >> 32u))); // Append one exact coordinate and timestamp record.
        }; // All event types share the same reconstructed time base.
    for (std::uint16_t word : words) { // Read the original recording prefix in physical order.
        const unsigned type = word >> 12u, // Separate the protocol tag from its twelve-bit payload.
                       value = // Separate the protocol tag from its twelve-bit payload.
                           word & 4095u; // Separate the protocol tag from its twelve-bit payload.
        if (type == 0u) { // Update row and camera context.
            y = value; // Preserve the complete row context for later CD events.
        } else if (type == 3u) { // Update the vector's base X coordinate and polarity.
            base = value & 2047u; // Keep the eleven-bit pixel coordinate.
            polarity = value >> 11u; // Keep the vector's polarity bit.
        } else if (type == 6u) { // Update the low timestamp field.
            low = value; // Keep all twelve low timestamp bits.
        } else if (type == 8u) { // Establish or advance the high timestamp field.
            if (synchronized && high > value && // Detect a genuine high-time wrap using the protocol tolerance.
                high - value >= // Detect a genuine high-time wrap using the protocol tolerance.
                    4085u) { // Detect a genuine high-time wrap using the protocol tolerance.
                ++wraps; // Extend time beyond the twenty-four-bit hardware timestamp period.
            } // Retain the current wrap count for small backward high-word changes.
            high = value; // Store the current high timestamp field.
            low = 0u; // A TIME_HIGH word resets low time until the next TIME_LOW update.
            synchronized = true; // Permit event emission after a valid time base has arrived.
        } else if (type == 2u && synchronized) { // Emit a direct CD event with its own polarity.
            append(value & 2047u, // Preserve coordinate, camera and polarity.
                   (y & 2047u) | ((value >> 11u) << 11u) | // Preserve coordinate, camera and polarity.
                       ((y >> 11u) << 12u)); // Preserve coordinate, camera and polarity.
        } else if (type == 4u || type == 5u) { // Expand a twelve-bit or eight-bit event vector.
            const unsigned bits = // Respect the protocol's meaningful mask width.
                type == 4u ? 12u : 8u; // Respect the protocol's meaningful mask width.
            for (unsigned bit = 0; bit < bits; // Visit each potential CD event in the vector mask.
                 ++bit) { // Visit each potential CD event in the vector mask.
                if (synchronized && // Emit only set mask bits after time synchronization.
                    (value & (1u << bit))) { // Emit only set mask bits after time synchronization.
                    append((base + bit) & 2047u, // Restore every original vector event.
                           (y & 2047u) | (polarity << 11u) | // Restore every original vector event.
                               ((y >> 11u) << 12u)); // Restore every original vector event.
                } // Emit one event for each selected mask bit.
            } // Vector events remain in increasing X-bit order.
            base = (base + bits) & 2047u; // Advance the vector base by the complete encoded mask width.
        } else if (type == 10u && // Preserve external trigger records in their original order.
                   synchronized) { // Preserve external trigger records in their original order.
            append(value >> 8u, // Tag the trigger while retaining channel and edge.
                   0x80000000u | (value & 1u)); // Tag the trigger while retaining channel and edge.
        } // Finish handling this word according to its event type and synchronization state.
    } // All original sensor words have been interpreted once.
    return events; // Return the complete independent event reference for this prefix.
} // Interpret the EVT3 format sequentially as an independent decoder reference.
void events(runner& run) { // Decode and process an actual public Prophesee EVT3 recording prefix.
    const auto words = // Read the exact prepared sensor-word sequence.
        data::words("evt3_words.bin"); // Read the exact prepared sensor-word sequence.
    const auto expected = // Build an independent scalar reference for every emitted event.
        decode_reference(words); // Build an independent scalar reference for every emitted event.
    if (expected.empty()) { // Require real CD or trigger data after time synchronization.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "public EVT3 fixture contains no synchronized events"); // Reject an unusable prefix.
    } // The fixture now has a meaningful independent output reference.
    shdr_examples::evt3_decoder decoder(static_cast<unsigned>( // Allocate bounded reusable parallel decode resources.
        words.size())); // Allocate bounded reusable parallel decode resources.
    decoder.load(words); // Copy the unmodified sensor words into packed shader storage.
    shdr::buffer<vec4> flow( // Allocate only the actual meaningful decoded event count.
        expected.size()); // Allocate only the actual meaningful decoded event count.
    auto fit = shdr::make_kernel("evt3_flow_kernel"); // Create the normal-flow plane-fit stage.
    fit->bind("events", decoder.events) // Fit bounded recent same-polarity neighborhoods.
        .bind("offsets", decoder.offsets) // Fit bounded recent same-polarity neighborhoods.
        .bind("flow", flow) // Connect the flow shader port to flow storage.
        .set("blocks", decoder.blocks()) // Supply the blocks processing parameter.
        .set("radius", 5.0f) // Supply the radius processing parameter.
        .set("history", 64u) // Supply the history processing parameter.
        .set("time_window", 10000u); // Fit bounded recent same-polarity neighborhoods.
    shdr::pipeline stages = // Reset stream state before independently decoding this prefix.
        decoder.reset_pipeline(); // Reset stream state before independently decoding this prefix.
    stages += shdr::dispatch( // Dispatch this configured shader with the following work group geometry.
        *fit,
        shdr::extent((static_cast<unsigned>(expected.size()) + 63u) / // Feed the decoder's event buffer directly to the flow stage.
                     64u)); // Feed the decoder's event buffer directly to the flow stage.
    data::checked_output<uvec4> event_check( // Keep the complete stage output for the backend comparison.
        decoder.events, // Require exact ordered coordinates and timestamp words.
        0.0, // Require exact ordered coordinates and timestamp words.
        expected.size()); // Require exact ordered coordinates and timestamp words.
    data::checked_output<unsigned> offsets_check( // Keep the complete stage output for the backend comparison.
        decoder.offsets, 0.0); // Require identical compact event intervals and final count.
    data::checked_output<vec4> flow_check( // Keep the complete stage output for the backend comparison.
        flow, 0.02); // Allow float plane-fit conditioning while comparing complete flow outputs.
    data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                   "Prophesee real EVT3 decode/normal-flow pipeline", // Use the exact diagnostic or comparison label for this case.
                   stages, // Run the complete processing sequence in dependency order.
                   {&event_check, // Compare the full real decode and processing chain.
                    &offsets_check, // Compare the full real decode and processing chain.
                    &flow_check}); // Compare the full real decode and processing chain.
    if (decoder.event_count() != // Check parallel compact length against the independent sequential decoder.
        expected // Use the independently calculated expected values.
            .size()) { // Check parallel compact length against the independent sequential decoder.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "real EVT3 output count differs from scalar protocol reference"); // Reject a common output-length error.
    } // Every actual event has a matching independent reference record.
    const uvec4* actual = static_cast<const shdr::buffer<uvec4>&>(decoder.events) // Read completed decoded event records.
                              .data(); // Read completed decoded event records.
    if (!std::equal(expected.begin(), // Compare each coordinate, metadata word and extended timestamp.
                    expected.end(), // Compare each coordinate, metadata word and extended timestamp.
                    actual)) { // Compare each coordinate, metadata word and extended timestamp.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "real EVT3 ordered events differ from scalar protocol reference"); // Report any protocol interpretation error.
    } // Backend agreement also matches independent protocol decoding.
    const auto compact = shdr_examples::evt3_compact_cd_to64( // Exercise the CPU consumer of compact sixty-four-bit CD words.
        decoder.events, // Exercise the CPU consumer of compact sixty-four-bit CD words.
        decoder.event_count()); // Exercise the CPU consumer of compact sixty-four-bit CD words.
    const vec4* fitted = // Read completed normal-flow records.
        static_cast<const shdr::buffer<vec4>&>(flow).data(); // Read completed normal-flow records.
    unsigned valid = 0u; // Count neighborhoods that support a nonsingular plane fit.
    for (unsigned i = 0; i < expected.size(); // Inspect the full real-event processing output.
         ++i) { // Inspect the full real-event processing output.
        valid += fitted[i].w > 0.5f // Count only fits accepted by the native shader's validity rule.
                     ? 1u // Count only fits accepted by the native shader's validity rule.
                     : 0u; // Count only fits accepted by the native shader's validity rule.
        if (i % 97u == 0u && // Sample accepted real fits for an independent double-precision check.
            fitted[i].w > // Sample accepted real fits for an independent double-precision check.
                0.5f) { // Sample accepted real fits for an independent double-precision check.
            const vec4 oracle = shdr_examples::detail::evt3_reference_flow( // Solve the same documented neighborhood through independent elimination.
                expected, // Use the independently calculated expected values.
                i, // Solve the same documented neighborhood through independent elimination.
                5.0f, // Solve the same documented neighborhood through independent elimination.
                64u, // Use this fixed documented processing parameter.
                10000u); // Solve the same documented neighborhood through independent elimination.
            if (oracle.w > 0.5f && // Compare valid nonsingular fits with a conditioning-aware error limit.
                data::difference(fitted[i], oracle) > // Compare valid nonsingular fits with a conditioning-aware error limit.
                    0.05) { // Compare valid nonsingular fits with a conditioning-aware error limit.
                throw shdr::error( // Report the failed format, numerical or resource precondition.
                    "real event flow differs from independent plane-fit reference"); // Reject an incorrect real processing result.
            } // Validate this sampled accepted fit against the independent timestamp-plane oracle.
        } // Existing synthetic plane tests separately check exact known motion values.
    } // Every returned record has already received a finite-value and backend check.
    std::cout << "Prophesee recording: " << expected.size() << " decoded events, " << compact.size() // Report the completed real-data processing check.
              << " compact CD words, " << valid // Record useful real-data output counts.
              << " accepted normal-flow fits\n"; // Record useful real-data output counts.
} // The guide explains the aperture ambiguity and absence of annotated flow ground truth.
void neural(runner& run) { // Train the small neural examples on real MNIST zero-versus-one images.
    const auto images = // Read the documented balanced training subset.
        data::bytes("mnist_train.bin"); // Read the documented balanced training subset.
    const auto labels = data::bytes( // Read the corresponding original MNIST training labels.
        "mnist_train_labels.bin"); // Read the corresponding original MNIST training labels.
    if (images.size() != 128u * 784u || // Require the documented prepared subset shape.
        labels.size() != 128u) { // Require the documented prepared subset shape.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "binary MNIST training fixtures must contain 128 image-label pairs"); // Reject mismatched real training data.
    } // The guide distinguishes this small training demonstration from held-out LeNet evaluation.
    const unsigned batch = // Keep real analytic-gradient checks small enough for all supported devices.
        32u; // Keep real analytic-gradient checks small enough for all supported devices.
    std::vector<unsigned> // Preserve original fixture order while selecting a balanced batch.
        selected_indices; // Preserve original fixture order while selecting a balanced batch.
    unsigned selected_counts[2] = {0u, // Count selected real examples for each binary digit class.
                                   0u}; // Count selected real examples for each binary digit class.
    for (unsigned index = 0; index < labels.size(); // Scan the original-order training fixture labels.
         ++index) { // Scan the original-order training fixture labels.
        if (labels[index] > 1u) { // Require the documented zero-versus-one subset.
            throw shdr::error( // Report the failed format, numerical or resource precondition.
                "binary MNIST fixture contains another digit class"); // Reject an incompatible prepared subset.
        } // Original zero and one labels can now be balanced explicitly.
        if (selected_counts[labels[index]] < // Include only the first sixteen examples of each class.
            16u) { // Include only the first sixteen examples of each class.
            selected_indices.push_back(index); // Retain the corresponding original image position.
            ++selected_counts[labels[index]]; // Record this selected real training example.
        } // Extra examples remain available to other training demonstrations.
    } // Select the first sixteen images of each digit while preserving original dataset order.
    if (selected_indices.size() != // Require both classes to have enough original training examples.
        batch) { // Require both classes to have enough original training examples.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "binary MNIST fixture lacks sixteen samples of each class"); // Explain a deficient prepared dataset.
    } // Every neural case will use this exact balanced real mini-batch.
    for (bool convolutional : // Evaluate both the dense ANN and the compact CNN example.
         {false, true}) { // Evaluate both the dense ANN and the compact CNN example.
        shdr_examples::binary_network model( // Allocate the same trainable model used by the individual examples.
            convolutional, // Allocate the same trainable model used by the individual examples.
            batch); // Allocate the same trainable model used by the individual examples.
        float* input = // Replace only the synthetic constructor fixture with real MNIST features.
            model.inputs // Replace only the synthetic constructor fixture with real MNIST features.
                .data(); // Replace only the synthetic constructor fixture with real MNIST features.
        float* targets = // Replace class targets with the original published digit labels.
            model.targets.data(); // Replace class targets with the original published digit labels.
        for (unsigned sample = 0; sample < batch; ++sample) { // Select a balanced real mini-batch.
            const unsigned index = selected_indices // Select one of the first sixteen original examples from each class.
                [sample]; // Select one of the first sixteen original examples from each class.
            targets[sample] = float(labels[index]); // Copy the original zero-or-one labels.
            if (convolutional) { // Supply an eight-by-eight real digit image to the compact CNN.
                for (unsigned y = 0; y < 8u; // Sample each compact image row at fixed source positions.
                     ++y) { // Sample each compact image row at fixed source positions.
                    for (unsigned x = 0; x < 8u; // Sample each compact image column independently.
                         ++x) { // Sample each compact image column independently.
                        input[sample * 64u + y * 8u + x] = // Use deterministic center-nearest real image reduction.
                            images[index * 784u + ((2u * y + 1u) * 28u / 16u) * 28u + // Use deterministic center-nearest real image reduction.
                                   (2u * x + 1u) * 28u / 16u] / // Use deterministic center-nearest real image reduction.
                            255.0f; // Use deterministic center-nearest real image reduction.
                    } // The compact row contains real source intensities.
                } // The full compact CNN input preserves the documented preprocessing.
            } else { // Reduce real image geometry into the ANN's two input features.
                double mass = 0.0, horizontal_spread = 0.0, outer = 0.0, // Accumulate simple real-image stroke features.
                       center = 0.0; // Accumulate simple real-image stroke features.
                for (unsigned y = 0; y < 28u; ++y) { // Visit every original digit row.
                    for (unsigned x = 0; x < 28u; // Include every original real pixel in feature extraction.
                         ++x) { // Include every original real pixel in feature extraction.
                        const double value = images[index * 784u + y * 28u + x] / // Read normalized original ink intensity.
                                             255.0; // Read normalized original ink intensity.
                        mass += value; // Count total real-image stroke mass.
                        horizontal_spread += // Measure horizontal stroke spread about the image center.
                            value * (double(x) - 13.5) * // Measure horizontal stroke spread about the image center.
                            (double(x) - // Measure horizontal stroke spread about the image center.
                             13.5); // Measure horizontal stroke spread about the image center.
                        outer += x < 9u || x >= 19u // Measure ink in the two outer column bands.
                                     ? value // Measure ink in the two outer column bands.
                                     : 0.0; // Measure ink in the two outer column bands.
                        center += x >= 10u && x < 18u // Measure ink in the central column band.
                                      ? value // Measure ink in the central column band.
                                      : 0.0; // Measure ink in the central column band.
                    } // All pixels in this real digit row contribute to its features.
                } // Describe each recorded digit with its measured image features.
                input[2u * sample] = // Normalize horizontal spread to a small signed feature.
                    float(horizontal_spread / (mass + 1.0) / 30.0 - // Normalize horizontal spread to a small signed feature.
                          0.5); // Normalize horizontal spread to a small signed feature.
                input[2u * sample + 1u] = // Normalize the outer-versus-center stroke contrast.
                    float((outer - center) / // Normalize the outer-versus-center stroke contrast.
                          (mass + 1.0)); // Normalize the outer-versus-center stroke contrast.
            } // Both network variants receive deterministic documented real-data preprocessing.
        } // The same balanced mini-batch is now ready for both backends.
        model.evaluation_pipeline() // Measure initial real-data training loss before updating parameters.
            .run(); // Measure initial real-data training loss before updating parameters.
        const float initial_loss = // Preserve the starting mini-batch cross entropy.
            static_cast<const shdr::buffer<float>&>(model.loss) // Preserve the starting mini-batch cross entropy.
                .data()[0]; // Preserve the starting mini-batch cross entropy.
        shdr::pipeline stages; // Link several complete analytic-gradient and SGD updates.
        for (unsigned step = 0; step < 32u; // Demonstrate sustained learning on the same small real training batch.
             ++step) { // Demonstrate sustained learning on the same small real training batch.
            stages += // Update every trainable parameter with the native dual-compiled kernels.
                model // Update every trainable parameter with the native dual-compiled kernels.
                    .training_pipeline(); // Update every trainable parameter with the native dual-compiled kernels.
        } // Fit the selected training mini-batch and record its cross-entropy change.
        stages += model.evaluation_pipeline(); // Produce final predictions and mean cross entropy.
        data::checked_output<float> parameters(model.parameters, 0.001), // Keep the complete stage output for the backend comparison.
            gradients(model.gradients, 0.002), predictions(model.predictions, 0.001), // Compare complete training and inference resources.
            loss(model.loss, 0.001); // Compare complete training and inference resources.
        data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                       convolutional ? "MNIST compact CNN training/inference" // Restore identical initial weights before each backend's updates.
                                     : "MNIST two-feature ANN training/inference", // Restore identical initial weights before each backend's updates.
                       stages, // Run the complete processing sequence in dependency order.
                       {&parameters, &gradients, &predictions, &loss}, // Restore identical initial weights before each backend's updates.
                       [&model]() { // Restore identical initial weights before each backend's updates.
                           model.reset_parameters(); // Restore identical initial weights before each backend's updates.
                       }); // Restore identical initial weights before each backend's updates.
        const float final_loss = static_cast<const shdr::buffer<float>&>(model.loss) // Read the actual fitted training loss.
                                     .data()[0]; // Read the actual fitted training loss.
        if (!(final_loss < // Require a useful learning response on real image-label pairs.
              initial_loss)) { // Require a useful learning response on real image-label pairs.
            throw shdr::error( // Report the failed format, numerical or resource precondition.
                "real MNIST mini-batch training did not reduce cross entropy"); // Catch failed gradient or update behavior.
        } // Analytic training has a measurable real-data objective improvement.
        std::cout << (convolutional ? "MNIST compact CNN" : "MNIST two-feature ANN") // Report the completed real-data processing check.
                  << " mini-batch loss: " << initial_loss << " -> " << final_loss // Report the initial and final training mini-batch losses.
                  << '\n'; // Report the initial and final training mini-batch losses.
    } // Both small network variants receive real-data training coverage.
    shdr_examples::transformer_model model( // Allocate the existing fully trainable attention classifier.
        batch); // Allocate the existing fully trainable attention classifier.
    float* tokens = // Replace synthetic constructor sequences with real digit patch features.
        model.inputs // Replace synthetic constructor sequences with real digit patch features.
            .data(); // Replace synthetic constructor sequences with real digit patch features.
    unsigned* targets = model.labels.data(); // Store original zero-or-one digit labels.
    for (unsigned sample = 0; sample < batch; // Construct the same balanced real mini-batch as the other neural cases.
         ++sample) { // Construct the same balanced real mini-batch as the other neural cases.
        const unsigned index = // Select the same balanced original zero and one examples.
            selected_indices[sample]; // Select the same balanced original zero and one examples.
        targets[sample] = labels[index]; // Preserve original supervised class identities.
        for (unsigned row = 0; row < 4u; // Treat each seven-row image band as one sequence token.
             ++row) { // Treat each seven-row image band as one sequence token.
            for (unsigned column = 0; column < 4u; // Give each token four seven-column patch features.
                 ++column) { // Give each token four seven-column patch features.
                double total = 0.0; // Accumulate real intensity within this patch.
                for (unsigned y = 0; y < 7u; ++y) { // Visit every source row inside the patch.
                    for (unsigned x = 0; x < 7u; // Visit every source column inside the patch.
                         ++x) { // Visit every source column inside the patch.
                        total += images[index * 784u + (row * 7u + y) * 28u + column * 7u + // Include the original real pixel value.
                                        x]; // Include the original real pixel value.
                    } // The patch row has contributed all of its original pixels.
                } // The complete seven-by-seven real patch has been averaged.
                tokens[sample * 16u + row * 4u + column] = // Normalize patch mean into a small signed token feature.
                    float(total / (49.0 * 255.0) * 4.0 - // Normalize patch mean into a small signed token feature.
                          0.5); // Normalize patch mean into a small signed token feature.
            } // All four features of this real-image token are ready.
        } // All four tokens represent the original image's coarse spatial layout.
    } // Attention receives real image sequences and their original labels.
    model.prediction_pipeline().run(); // Measure initial real-data classification loss.
    const double initial = model.loss(); // Preserve the starting native attention cross entropy.
    shdr::pipeline stages; // Build linked analytic attention updates and final inference.
    for (unsigned step = 0; step < 12u; // Keep the real attention training demonstration bounded on small devices.
         ++step) { // Keep the real attention training demonstration bounded on small devices.
        stages += // Differentiate all 134 parameters and apply averaged SGD.
            model.training_pipeline(); // Differentiate all 134 parameters and apply averaged SGD.
    } // Every parameter update uses only the single-header resource and shader runtime.
    stages += // Produce final probabilities, loss and predicted classes.
        model.prediction_pipeline(); // Produce final probabilities, loss and predicted classes.
    data::checked_output<float> parameters(model.parameters, 0.003), // Keep the complete stage output for the backend comparison.
        gradients(model.gradients, 0.005); // Compare complete learned values and final derivatives.
    data::checked_output<vec4> predictions( // Keep the complete stage output for the backend comparison.
        model.predictions, 0.003); // Compare probabilities, sample loss and selected class records.
    data::evaluate(run, // Compare the complete real-data pipeline outputs on CPU and the requested GPU backend.
                   "MNIST patch-token transformer training/inference", // Use the exact diagnostic or comparison label for this case.
                   stages, // Run the complete processing sequence in dependency order.
                   {&parameters, &gradients, &predictions}, // Restore identical initial trainable weights for independent backend runs.
                   [&model]() { // Restore identical initial trainable weights for independent backend runs.
                       model.reset_parameters(); // Restore identical initial trainable weights for independent backend runs.
                   }); // Restore identical initial trainable weights for independent backend runs.
    const double final_loss = model.loss(); // Read the fitted mini-batch cross entropy.
    if (!(final_loss < // Require a measurable native training response to the real supervised data.
          initial)) { // Require a measurable native training response to the real supervised data.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "real MNIST transformer mini-batch training did not reduce cross entropy"); // Report failed attention training behavior.
    } // Measure the loss reduction from analytic transformer training on the selected batch.
    std::cout << "MNIST patch-token transformer mini-batch loss: " << initial << " -> " // Report the completed real-data processing check.
              << final_loss << '\n'; // Record the actual real-data learning result.
} // Separate complete held-out digit evaluation follows with the named deeper model.
void lenet( // Evaluate the pretrained named model against independent framework logits and published labels.
    runner& // Evaluate the pretrained named model against independent framework logits and published labels.
        run) { // Evaluate the pretrained named model against independent framework logits and published labels.
    const char* full_directory = std::getenv( // Allow explicit evaluation of the complete optional held-out split.
        "SHDR_FULL_MNIST_DIR"); // Allow explicit evaluation of the complete optional held-out split.
    const bool full = // Select the full split only when its cache directory was provided.
        full_directory && // Select the full split only when its cache directory was provided.
        *full_directory; // Select the full split only when its cache directory was provided.
    const auto images = // Load original test image bytes in dataset order.
        full ? data::bytes("mnist_full.bin", full_directory) // Load original test image bytes in dataset order.
             : data::bytes("mnist_test.bin"); // Load original test image bytes in dataset order.
    const auto labels = // Load the exact corresponding original labels.
        full ? data::bytes("mnist_full_labels.bin", full_directory) // Load the exact corresponding original labels.
             : data::bytes("mnist_labels.bin"); // Load the exact corresponding original labels.
    const auto parameters = data::floats( // Read documented plain float32 pretrained model weights.
        "lenet_weights.bin"); // Read documented plain float32 pretrained model weights.
    const auto reference = data::floats( // Read independent PyTorch logits for all ten thousand held-out images.
        "lenet_reference.bin"); // Read independent PyTorch logits for all ten thousand held-out images.
    if (images.size() != labels.size() * 784u || reference.size() != 10000u * 10u || // Validate all model evaluation shapes.
        labels.size() > 10000u) { // Validate all model evaluation shapes.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "MNIST LeNet fixtures have inconsistent image, label or reference counts"); // Reject incomplete or mismatched evaluation data.
    } // Images and independent references share original held-out dataset order.
    unsigned correct = // Count actual native classifier predictions matching original digit labels.
        0u; // Count actual native classifier predictions matching original digit labels.
    double maximum = 0.0; // Track the worst difference from independent framework logits.
    const auto start = std::chrono::steady_clock:: // Measure completed native evaluation including functional comparison work.
        now(); // Measure completed native evaluation including functional comparison work.
    for (unsigned offset = 0; offset < labels.size(); // Bound activation storage and dispatch sizes on every target.
         offset += 128u) { // Bound activation storage and dispatch sizes on every target.
        const unsigned batch = std::min(128u, // Include the final partial held-out batch.
                                        static_cast<unsigned>(labels.size()) - // Include the final partial held-out batch.
                                            offset); // Include the final partial held-out batch.
        shdr_examples::lenet_classifier model( // Build the same seven-stage named network for this bounded batch.
            batch); // Build the same seven-stage named network for this bounded batch.
        model.load_parameters(parameters); // Supply the exact trained weights to both backends.
        float* input = // Access native input storage for this held-out image batch.
            model.inputs.data(); // Access native input storage for this held-out image batch.
        for (unsigned i = 0; i < batch * 784u; ++i) { // Normalize every original held-out pixel.
            input[i] = images[offset * 784u + i] / // Match the documented training and reference preprocessing.
                       255.0f; // Match the documented training and reference preprocessing.
        } // First convolution supplies its own two-pixel zero border.
        run.run( // Execute the complete named model on the requested CPU or GPU backend.
            model // Execute the complete named model on the requested CPU or GPU backend.
                .inference_pipeline()); // Execute the complete named model on the requested CPU or GPU backend.
        const float* logits = static_cast<const shdr::buffer<float>&>(model.logits) // Read completed native classifier logits.
                                  .data(); // Read completed native classifier logits.
        for (unsigned sample = 0; sample < batch; // Evaluate each original held-out digit independently.
             ++sample) { // Evaluate each original held-out digit independently.
            unsigned prediction = 0u; // Begin argmax with class zero.
            for ( // Compare every classifier logit with the independent preparation framework.
                unsigned digit = 0; digit < 10u; // Compare every classifier logit with the independent preparation framework.
                ++digit) { // Compare every classifier logit with the independent preparation framework.
                const float actual = logits[sample * 10u + digit]; // Read this native class score.
                const float expected = // Select the matching original test image's reference score.
                    reference[(offset + sample) * 10u + // Select the matching original test image's reference score.
                              digit]; // Select the matching original test image's reference score.
                maximum = std::max( // Measure the greatest normalized difference from the framework logits.
                    maximum, // Measure the greatest normalized difference from the framework logits.
                    data::difference( // Measure the greatest normalized difference from the framework logits.
                        actual, // Measure the greatest normalized difference from the framework logits.
                        expected)); // Measure the greatest normalized difference from the framework logits.
                if (data::difference(actual, expected) > // Allow float reduction order while preserving a strict inference comparison.
                    0.0005) { // Allow float reduction order while preserving a strict inference comparison.
                    throw shdr::error( // Report the failed format, numerical or resource precondition.
                        "native LeNet logit differs from independent framework reference"); // Report a model layout or numerical implementation error.
                } // Native layer wiring matches independent deep inference for this class score.
                if (actual > // Apply deterministic first-maximum class selection.
                    logits[sample * 10u + // Apply deterministic first-maximum class selection.
                           prediction]) { // Apply deterministic first-maximum class selection.
                    prediction = digit; // Keep the highest-scoring digit class seen so far.
                } // Ties retain the earliest class index.
            } // All ten scores have an independent real-data inference reference.
            correct += prediction == labels[offset + sample] // Compare the native predicted digit with original ground truth.
                           ? 1u // Compare the native predicted digit with original ground truth.
                           : 0u; // Compare the native predicted digit with original ground truth.
        } // The full bounded batch contributes measured held-out accuracy.
    } // Every selected held-out image has been evaluated exactly once.
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start) // Record completed native evaluation duration.
                               .count(); // Record completed native evaluation duration.
    const double accuracy = // Calculate accuracy from correct labels in the official held-out test split.
        double(correct) / // Calculate accuracy from correct labels in the official held-out test split.
        labels // Calculate accuracy from correct labels in the official held-out test split.
            .size(); // Calculate accuracy from correct labels in the official held-out test split.
    if (accuracy < // Require the prepared model's useful real held-out behavior on every supported platform.
        0.95) { // Require the prepared model's useful real held-out behavior on every supported platform.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "native LeNet held-out accuracy fell below 95 percent"); // Catch a model or dataset preprocessing regression.
    } // The check is tied to a real public classification task with original labels.
    std::cout << "LeNet MNIST held-out: " << correct << '/' << labels.size() // Report the completed real-data processing check.
              << " correct, accuracy " << accuracy * 100.0 << "%, normalized reference error " // Report actual labels, numerical parity and completed runtime.
              << maximum << ", completed seconds " << seconds // Report actual labels, numerical parity and completed runtime.
              << '\n'; // Report actual labels, numerical parity and completed runtime.
} // Offline training remains reproducible and optional for C++ deployment.
} // namespace

int main(int argc, // Run every real-data algorithm family on the selected supported profile.
         char** argv) { // Run every real-data algorithm family on the selected supported profile.
    try { // Convert useful validation failures into a nonzero process result.
        runner run(argc, // Select ordinary CPU execution or one requested headless GPU profile.
                   argv); // Select ordinary CPU execution or one requested headless GPU profile.
        const auto source = image(); // Read and normalize the same public image once.
        basic(run, source); // Check linked, hybrid and asynchronous resources with real data.
        images(run, source); // Check image storage and an independently verified box filter.
        vision(run, source); // Check the complete real-image corner pipeline.
        spectral( // Check two-dimensional Fourier processing and three-dimensional real image stacks.
            run, // Select the same execution backend for this complete case.
            source); // Check two-dimensional Fourier processing and three-dimensional real image stacks.
        imaging( // Check all Bayer formats and lossless compression on real-derived image data.
            run); // Check all Bayer formats and lossless compression on real-derived image data.
        signals(run); // Check every FFT, window, FIR and biquad mode on a real waveform.
        events( // Check exact real event decoding, compact CPU consumption and normal-flow processing.
            run); // Check exact real event decoding, compact CPU consumption and normal-flow processing.
        neural(run); // Check native analytic training and inference on real digit-image features.
        shdr_examples::check_lenet_classifier( // Validate all seven named-model stages against an independent double oracle.
            run); // Validate all seven named-model stages against an independent double oracle.
        lenet(run); // Check deeper named-model inference and held-out MNIST accuracy.
        std::cout // Report the completed real-data processing check.
            << "All public real-data functional cases passed\n"; // Report completion only after every case succeeds.
        return 0; // Mark the complete real-data evaluation as successful.
    } // Unsupported GPU contexts receive the same explicit skip policy as the other examples.
#ifdef SHDR_HAVE_EGL // Catch the optional library's unsupported-context result when EGL is built.
    catch (const shdr_examples::context_unavailable& // Preserve unsupported hardware as a distinct test outcome.
               error) { // Preserve unsupported hardware as a distinct test outcome.
        std::cerr << "SKIP: " << error.what() // Explain which requested profile was unavailable.
                  << '\n'; // Explain which requested profile was unavailable.
        return 77; // Let CTest classify this unsupported GPU context as skipped.
    } // Algorithm or numerical failures remain real test failures below.
#endif // Finish the optional headless-context handling.
    catch (const std::exception& // Report any fixture, numerical or runtime validation failure.
               error) { // Report any fixture, numerical or runtime validation failure.
        std::cerr << error.what() // Keep the actual failure visible to the user and test runner.
                  << '\n'; // Keep the actual failure visible to the user and test runner.
        return 1; // Mark failed real-data processing as an error.
    } // Every failure path has a useful process result.
} // This one executable covers the real-data functionality of all example algorithm families.
