#include "runner.hpp" // Include runner.hpp declarations.

/// Link SSBO and image stages, checking a full two-dimensional CPU/GPU result.
int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Select CPU execution or the requested headless GPU profile.
        const unsigned width = 19, height = 13; // Choose dimensions that exercise partial final workgroups on both image axes.
        shdr::buffer<float> input(width * height), output(width * height); // Allocate distinct resources for the input, intermediate stages and final result.
        shdr::image2d<float, 1> original(width, height), filtered(width, height); // Allocate separate input and output images for neighborhood sampling.
        shdr_examples::randomize(input, 1102u, 0.0f, 23.0f); // Fill the input with reproducible random values.

        auto upload_owner = shdr::make_kernel("buffer_to_image_kernel"); // Create the linked buffer to image kernel instance.
        shdr::kernel& upload = *upload_owner; // Retain the owned shader instance for every stage that references it.
        upload.bind("input_values", input); // Copy the host buffer into the image shared with the blur stage.
        upload.bind("output_image", original); // Copy the host buffer into the image shared with the blur stage.

        auto blur_owner = shdr::make_kernel("blur_kernel"); // Create the linked blur kernel instance.
        shdr::kernel& blur = *blur_owner; // Retain the owned shader instance for every stage that references it.
        blur.bind("input_image", original); // Connect the immutable input image and the separate filtered image.
        blur.bind("output_image", filtered); // Connect the immutable input image and the separate filtered image.

        auto download_owner = shdr::make_kernel("image_to_buffer_kernel"); // Create the linked image to buffer kernel instance.
        shdr::kernel& download = *download_owner; // Retain the owned shader instance for every stage that references it.
        download.bind("input_image", filtered); // Copy the filtered image into the buffer used by the host result check.
        download.bind("output_values", output); // Copy the filtered image into the buffer used by the host result check.

        const shdr::extent groups((width + 7) / 8, (height + 7) / 8); // Round up both image axes to the shader's eight-by-eight local size.
        shdr::pipeline pipeline; // Declare pipeline for the following calculation.
        pipeline += shdr::dispatch(upload, groups); // Begin by transferring values between two shader resource types.
        pipeline += shdr::dispatch(blur, groups); // Consume the image produced by the preceding stage and filter its neighborhood.
        pipeline += shdr::dispatch(download, groups); // Return the filtered scalar pixels to a buffer resource.
        run.run(pipeline); // Execute the configured pipeline on the requested backend.

        const shdr::buffer<float>& read_input = input; // Use a constant buffer view so GPU writes synchronize before host reads.
        const shdr::buffer<float>& read_output = output; // Use a constant buffer view so GPU writes synchronize before host reads.
        for (unsigned y = 0; y < height; ++y) { // Iterate over y for this calculation.
            for (unsigned x = 0; x < width; ++x) { // Iterate over x for this calculation.
                float expected = 0.0f; // Start the independent scalar reference at zero.
                for (int dy = -1; dy <= 1; ++dy) { // Iterate over dy for this calculation.
                    for (int dx = -1; dx <= 1; ++dx) { // Iterate over dx for this calculation.
                        const unsigned sx = // Choose the reference sample column before applying the border rule.
                            unsigned(std::max(0, std::min(int(width) - 1, int(x) + dx))); // Clamp this reference coordinate to the nearest valid image edge.
                        const unsigned sy = // Choose the reference sample row before applying the border rule.
                            unsigned(std::max(0, std::min(int(height) - 1, int(y) + dy))); // Clamp this reference coordinate to the nearest valid image edge.
                        expected += read_input.data()[sy * width + sx]; // Accumulate the independently clamped three-by-three input neighborhood.
                    } // End this block.
                } // End this block.
                shdr_examples::require_close(read_output.data()[y * width + x], // Check the computed value against the independent reference.
                                             expected / 9.0f, // Average exactly nine clamped neighbor samples.
                                             "image pixel " + std::to_string(y * width + x)); // Identify the flattened pixel if this reference comparison fails.
            } // End this block.
        } // End this block.
        std::cout << (run.gpu() ? "GPU" : "CPU") << " buffer/image pipeline: " << width << " x " // Write this part of the example result or diagnostic.
                  << height << " pixels verified\n"; // Write this part of the example result or diagnostic.
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
