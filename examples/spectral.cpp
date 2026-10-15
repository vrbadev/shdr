/** @file spectral.cpp
 * Linked 2D/3D convolution and Fourier pipelines over seed-reproducible data.
 * Filters use true convolution with clamp-to-edge. The 2D direct DFT is a small
 * direct transform. An independent full 2D reference
 * checks the separable GPU/CPU stages and the inverse reconstruction.
 */
#include "runner.hpp" // Include runner.hpp declarations.
#include <algorithm> // Include algorithm declarations.
#include <complex> // Include complex declarations.
#include <random> // Include random declarations.
#include <vector> // Include vector declarations.

namespace { // Group these helpers within their namespace.
typedef shdr::glsl::vec2 sample; // Declare shdr for the following calculation.
typedef shdr::glsl::uvec2 size2; // Declare shdr for the following calculation.
typedef shdr::glsl::uvec3 size3; // Declare shdr for the following calculation.

int edge(int coordinate, unsigned size) { // Implement edge for the operations below.
    return std::max(0, std::min(coordinate, int(size) - 1)); // Return the result of this operation.
} // End this block.

// Accumulate the full reference products in double precision.
std::vector<sample> reference2d(const std::vector<sample>& input, // Declare reference2d with the parameters continued below.
                                size2 size, // Declare size for the following calculation.
                                const std::vector<float>& weights, // Declare weights for the following calculation.
                                size2 kernel_size) { // Declare kernel size for the following calculation.
    std::vector<sample> output(input.size()); // Construct output with these initial values.
    for (unsigned y = 0; y < size.y; ++y) // Iterate over y for this calculation.
        for (unsigned x = 0; x < size.x; ++x) { // Iterate over x for this calculation.
            double real = 0, imaginary = 0; // Use this floating-point value or resource in the following calculation.
            for (unsigned ky = 0; ky < kernel_size.y; ++ky) // Iterate over ky for this calculation.
                for (unsigned kx = 0; kx < kernel_size.x; ++kx) { // Iterate over kx for this calculation.
                    int qx = edge(int(x) + int(kernel_size.x / 2) - int(kx), size.x); // Set qx from this expression.
                    int qy = edge(int(y) + int(kernel_size.y / 2) - int(ky), size.y); // Set qy from this expression.
                    const sample value = input[unsigned(qx) + size.x * unsigned(qy)]; // Set value from this expression.
                    const double coefficient = weights[kx + kernel_size.x * ky]; // Set coefficient from this expression.
                    real += double(value.x) * coefficient; // Update real with this contribution.
                    imaginary += double(value.y) * coefficient; // Update imaginary with this contribution.
                } // End this block.
            output[x + size.x * y] = sample(float(real), float(imaginary)); // Set output[x + size.x * y] from this expression.
        } // End this block.
    return output; // Return output.
} // End this block.

std::vector<float> reference3d(const std::vector<float>& input, // Declare reference3d with the parameters continued below.
                               size3 size, // Declare size for the following calculation.
                               const std::vector<float>& weights, // Declare weights for the following calculation.
                               size3 kernel_size) { // Declare kernel size for the following calculation.
    std::vector<float> output(input.size()); // Construct output with these initial values.
    for (unsigned z = 0; z < size.z; ++z) // Iterate over z for this calculation.
        for (unsigned y = 0; y < size.y; ++y) // Iterate over y for this calculation.
            for (unsigned x = 0; x < size.x; ++x) { // Iterate over x for this calculation.
                double total = 0; // Use this floating-point value or resource in the following calculation.
                for (unsigned kz = 0; kz < kernel_size.z; ++kz) // Iterate over kz for this calculation.
                    for (unsigned ky = 0; ky < kernel_size.y; ++ky) // Iterate over ky for this calculation.
                        for (unsigned kx = 0; kx < kernel_size.x; ++kx) { // Iterate over kx for this calculation.
                            unsigned qx = // Set qx from this expression.
                                unsigned(edge(int(x) + int(kernel_size.x / 2) - int(kx), size.x)); // Apply unsigned to these arguments.
                            unsigned qy = // Set qy from this expression.
                                unsigned(edge(int(y) + int(kernel_size.y / 2) - int(ky), size.y)); // Apply unsigned to these arguments.
                            unsigned qz = // Set qz from this expression.
                                unsigned(edge(int(z) + int(kernel_size.z / 2) - int(kz), size.z)); // Apply unsigned to these arguments.
                            total += // Update total with this contribution.
                                double(input[qx + size.x * (qy + size.y * qz)]) * // Use this floating-point value or resource in the following calculation.
                                double(weights[kx + kernel_size.x * (ky + kernel_size.y * kz)]); // Use this floating-point value or resource in the following calculation.
                        } // End this block.
                output[x + size.x * (y + size.y * z)] = float(total); // Set output[x + size.x * (y + size.y * z)] from this expression.
            } // End this block.
    return output; // Return output.
} // End this block.

// Evaluate the full two-dimensional DFT directly as an independent reference.
std::vector<sample> reference_dft(const std::vector<sample>& input, size2 size) { // Implement reference dft for the operations below.
    std::vector<sample> output(input.size()); // Construct output with these initial values.
    const double tau = 6.283185307179586476925286766559; // Set tau from this expression.
    for (unsigned ky = 0; ky < size.y; ++ky) // Iterate over ky for this calculation.
        for (unsigned kx = 0; kx < size.x; ++kx) { // Iterate over kx for this calculation.
            std::complex<double> total(0, 0); // Construct total with these initial values.
            for (unsigned y = 0; y < size.y; ++y) // Iterate over y for this calculation.
                for (unsigned x = 0; x < size.x; ++x) { // Iterate over x for this calculation.
                    sample value = input[x + size.x * y]; // Set value from this expression.
                    double angle = -tau * (double(kx) * x / size.x + double(ky) * y / size.y); // Use this floating-point value or resource in the following calculation.
                    total += std::complex<double>(value.x, value.y) * std::polar(1.0, angle); // Update total with this contribution.
                } // End this block.
            output[kx + size.x * ky] = sample(float(total.real()), float(total.imag())); // Set output[kx + size.x * ky] from this expression.
        } // End this block.
    return output; // Return output.
} // End this block.

template <class T> shdr::buffer<T> resource(const std::vector<T>& values) { // Accept the value types or dimensions used by this template.
    shdr::buffer<T> result(values.size()); // Construct result with these initial values.
    std::copy(values.begin(), values.end(), result.data()); // Apply copy to these arguments.
    return result; // Return result.
} // End this block.

void check(const shdr::buffer<sample>& actual, // Declare check with the parameters continued below.
           const std::vector<sample>& expected, // Declare expected for the following calculation.
           const std::string& label, // Declare label for the following calculation.
           float tolerance = 0.0002f) { // Set tolerance from this expression.
    for (std::size_t i = 0; i < expected.size(); ++i) { // Iterate over i for this calculation.
        shdr_examples::require_close(actual.data()[i].x, // Check the computed value against the independent reference.
                                     expected[i].x, // Continue the expression using expected, i, x.
                                     label + " real[" + std::to_string(i) + "]", // Apply to string to these arguments.
                                     tolerance); // Continue the expression using tolerance.
        shdr_examples::require_close(actual.data()[i].y, // Check the computed value against the independent reference.
                                     expected[i].y, // Continue the expression using expected, i, y.
                                     label + " imaginary[" + std::to_string(i) + "]", // Apply to string to these arguments.
                                     tolerance); // Continue the expression using tolerance.
    } // End this block.
} // End this block.
void check(const shdr::buffer<float>& actual, // Declare check with the parameters continued below.
           const std::vector<float>& expected, // Declare expected for the following calculation.
           const std::string& label) { // Declare label for the following calculation.
    for (std::size_t i = 0; i < expected.size(); ++i) // Iterate over i for this calculation.
        shdr_examples::require_close( // Check the computed value against the independent reference.
            actual.data()[i], expected[i], label + "[" + std::to_string(i) + "]", 0.0002f); // Apply data to these arguments.
} // End this block.

std::unique_ptr<shdr::kernel> convolution2d(const shdr::buffer<sample>& input, // Declare convolution2d with the parameters continued below.
                                            const shdr::buffer<sample>& output, // Declare output for the following calculation.
                                            const shdr::buffer<float>& weights, // Declare weights for the following calculation.
                                            size2 dimensions, // Declare dimensions for the following calculation.
                                            size2 kernel_size) { // Declare kernel size for the following calculation.
    std::unique_ptr<shdr::kernel> shader = shdr::make_kernel("convolution2d_kernel"); // Create the linked convolution2d kernel instance.
    shader->bind("input_values", input).bind("output_values", output).bind("coefficients", weights); // Connect this shader port to its application resource.
    shader->set<size2>("dimensions", dimensions).set<size2>("kernel_size", kernel_size); // Supply this shader uniform with its typed application value.
    return shader; // Return shader.
} // End this block.
std::unique_ptr<shdr::kernel> convolution3d(const shdr::buffer<float>& input, // Declare convolution3d with the parameters continued below.
                                            const shdr::buffer<float>& output, // Declare output for the following calculation.
                                            const shdr::buffer<float>& weights, // Declare weights for the following calculation.
                                            size3 dimensions, // Declare dimensions for the following calculation.
                                            size3 kernel_size) { // Declare kernel size for the following calculation.
    std::unique_ptr<shdr::kernel> shader = shdr::make_kernel("convolution3d_kernel"); // Create the linked convolution3d kernel instance.
    shader->bind("input_values", input).bind("output_values", output).bind("coefficients", weights); // Connect this shader port to its application resource.
    shader->set<size3>("dimensions", dimensions).set<size3>("kernel_size", kernel_size); // Supply this shader uniform with its typed application value.
    return shader; // Return shader.
} // End this block.
std::unique_ptr<shdr::kernel> fourier(const char* name, // Declare fourier with the parameters continued below.
                                      const shdr::buffer<sample>& input, // Declare input for the following calculation.
                                      const shdr::buffer<sample>& output, // Declare output for the following calculation.
                                      size2 dimensions, // Declare dimensions for the following calculation.
                                      float direction, // Declare direction for the following calculation.
                                      float normalization) { // Declare normalization for the following calculation.
    std::unique_ptr<shdr::kernel> shader = shdr::make_kernel(name); // Create the linked shader instance.
    shader->bind("input_values", input).bind("output_values", output); // Connect this shader port to its application resource.
    shader->set<size2>("dimensions", dimensions) // Supply this shader uniform with its typed application value.
        .set<float>("direction", direction) // Supply this shader uniform with its typed application value.
        .set<float>("normalization", normalization); // Supply this shader uniform with its typed application value.
    return shader; // Return shader.
} // End this block.

void image_and_fourier(shdr_examples::runner& run, std::mt19937& random) { // Implement image and fourier for the operations below.
    const size2 dimensions(8, 8); // Construct dimensions with these initial values.
    const unsigned count = dimensions.x * dimensions.y; // Set count from this expression.
    std::uniform_real_distribution<float> distribution(-1, 1); // Construct distribution with these initial values.
    std::vector<sample> pixels(count); // Construct pixels with these initial values.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        pixels[i] = sample(distribution(random), 0); // Set pixels[i] from this expression.
    pixels[0].x += 3; // Edge and interior impulses make clamp behavior visible.
    pixels[3 + dimensions.x * 4].x += 5; // Update pixels[3 + dimensions.x * 4].x with this contribution.
    const std::vector<float> blur_coefficients = {0, 0.125f, 0, 0.125f, 0.5f, 0.125f, 0, 0.125f, 0}; // Set blur coefficients from this expression.
    const std::vector<float> derivative_coefficients = {-0.5f, 0, 0.5f}; // Set derivative coefficients from this expression.
    shdr::buffer<sample> input = resource(pixels), blurred(count), filtered(count), // Set input from this expression.
                         row_frequency(count), spectrum(count), inverse_rows(count), // Apply row frequency to these arguments.
                         reconstructed(count); // Apply reconstructed to these arguments.
    shdr::buffer<float> blur_weights = resource(blur_coefficients), // Set blur weights from this expression.
                        derivative_weights = resource(derivative_coefficients); // Set derivative weights from this expression.
    auto blur = convolution2d(input, blurred, blur_weights, dimensions, size2(3, 3)); // Set blur from this expression.
    auto derivative = convolution2d(blurred, filtered, derivative_weights, dimensions, size2(3, 1)); // Set derivative from this expression.
    auto rows = fourier("dft_rows_kernel", filtered, row_frequency, dimensions, -1, 1); // Set rows from this expression.
    auto columns = fourier("dft_columns_kernel", row_frequency, spectrum, dimensions, -1, 1); // Set columns from this expression.
    auto inverse_row = // Set inverse row from this expression.
        fourier("dft_rows_kernel", spectrum, inverse_rows, dimensions, 1, 1.0f / dimensions.x); // Apply fourier to these arguments.
    auto inverse_column = fourier( // Set inverse column from this expression.
        "dft_columns_kernel", inverse_rows, reconstructed, dimensions, 1, 1.0f / dimensions.y); // Provide this text argument for the operation above.
    const shdr::extent groups((dimensions.x + 7) / 8, (dimensions.y + 7) / 8); // Construct groups with these initial values.
    shdr::pipeline pipeline = // Set pipeline from this expression.
        shdr::dispatch(*blur, groups) >> shdr::dispatch(*derivative, groups) >> // Add a shader dispatch with the required workgroup counts.
        shdr::dispatch(*rows, groups) >> shdr::dispatch(*columns, groups) >> // Add a shader dispatch with the required workgroup counts.
        shdr::dispatch(*inverse_row, groups) >> shdr::dispatch(*inverse_column, groups); // Add a shader dispatch with the required workgroup counts.
    run.run(pipeline); // Execute the configured pipeline on the requested backend.
    const std::vector<sample> reference_blur = // Set reference blur from this expression.
        reference2d(pixels, dimensions, blur_coefficients, size2(3, 3)); // Apply reference2d to these arguments.
    const std::vector<sample> reference_filter = // Set reference filter from this expression.
        reference2d(reference_blur, dimensions, derivative_coefficients, size2(3, 1)); // Apply reference2d to these arguments.
    check(blurred, reference_blur, "2D convolution blur"); // Apply check to these arguments.
    check(filtered, reference_filter, "2D asymmetric convolution"); // Apply check to these arguments.
    check(spectrum, reference_dft(reference_filter, dimensions), "2D forward DFT", 0.002f); // Apply check to these arguments.
    check(reconstructed, reference_filter, "2D inverse DFT reconstruction", 0.0003f); // Apply check to these arguments.
    // Kernel resources are replaceable after composing the pipeline.
    const std::vector<float> identity_coefficients = {1}; // Set identity coefficients from this expression.
    shdr::buffer<float> identity_weights = resource(identity_coefficients); // Set identity weights from this expression.
    derivative->bind("coefficients", identity_weights).set<size2>("kernel_size", size2(1)); // Connect this shader port to its application resource.
    run.run(pipeline); // Execute the configured pipeline on the requested backend.
    check(filtered, reference_blur, "rebound convolution coefficients"); // Apply check to these arguments.
    check(reconstructed, reference_blur, "rebound inverse reconstruction", 0.0003f); // Apply check to these arguments.
    std::cout << "8x8 2D convolution -> convolution -> DFT rows/columns -> inverse: verified\n"; // Write this part of the example result or diagnostic.
} // End this block.

void volume(shdr_examples::runner& run, std::mt19937& random) { // Implement volume for the operations below.
    const size3 dimensions(5, 4, 3); // Construct dimensions with these initial values.
    const unsigned count = dimensions.x * dimensions.y * dimensions.z; // Set count from this expression.
    std::uniform_real_distribution<float> distribution(-1, 1); // Construct distribution with these initial values.
    std::vector<float> voxels(count); // Construct voxels with these initial values.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        voxels[i] = distribution(random); // Set voxels[i] from this expression.
    voxels[0] += 2; // Update voxels[0] with this contribution.
    voxels[2 + dimensions.x * (1 + dimensions.y)] += 4; // Update voxels[2 + dimensions.x * (1 + dimensions.y)] with this contribution.
    const std::vector<float> x_coefficients = {0.25f, 0.5f, 0.25f}; // Set x coefficients from this expression.
    const std::vector<float> y_coefficients = {-0.25f, 0.75f, 0.5f}; // Set y coefficients from this expression.
    const std::vector<float> z_coefficients = {0.2f, 0.3f, 0.5f}; // Set z coefficients from this expression.
    shdr::buffer<float> input = resource(voxels), along_x(count), along_y(count), output(count); // Set input from this expression.
    shdr::buffer<float> x_weights = resource(x_coefficients), y_weights = resource(y_coefficients), // Set x weights from this expression.
                        z_weights = resource(z_coefficients); // Set z weights from this expression.
    auto x = convolution3d(input, along_x, x_weights, dimensions, size3(3, 1, 1)); // Set x from this expression.
    auto y = convolution3d(along_x, along_y, y_weights, dimensions, size3(1, 3, 1)); // Set y from this expression.
    auto z = convolution3d(along_y, output, z_weights, dimensions, size3(1, 1, 3)); // Set z from this expression.
    const shdr::extent groups( // Declare groups with the parameters continued below.
        (dimensions.x + 3) / 4, (dimensions.y + 3) / 4, (dimensions.z + 3) / 4); // Continue the expression using dimensions, x, dimensions.
    shdr::pipeline pipeline = // Set pipeline from this expression.
        shdr::dispatch(*x, groups) >> shdr::dispatch(*y, groups) >> shdr::dispatch(*z, groups); // Add a shader dispatch with the required workgroup counts.
    run.run(pipeline); // Execute the configured pipeline on the requested backend.
    const std::vector<float> reference_x = // Set reference x from this expression.
        reference3d(voxels, dimensions, x_coefficients, size3(3, 1, 1)); // Apply reference3d to these arguments.
    const std::vector<float> reference_y = // Set reference y from this expression.
        reference3d(reference_x, dimensions, y_coefficients, size3(1, 3, 1)); // Apply reference3d to these arguments.
    const std::vector<float> reference_z = // Set reference z from this expression.
        reference3d(reference_y, dimensions, z_coefficients, size3(1, 1, 3)); // Apply reference3d to these arguments.
    check(along_x, reference_x, "3D X convolution"); // Apply check to these arguments.
    check(along_y, reference_y, "3D Y convolution"); // Apply check to these arguments.
    check(output, reference_z, "3D Z convolution"); // Apply check to these arguments.
    std::cout << "5x4x3 3D convolution X -> Y -> Z: verified\n"; // Write this part of the example result or diagnostic.
} // End this block.
} // namespace

int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Construct run with these initial values.
        std::mt19937 random(0x53484452u); // Construct random with these initial values.
        image_and_fourier(run, random); // Apply image and fourier to these arguments.
        volume(run, random); // Apply volume to these arguments.
        std::cout << (run.gpu() ? "GPU" : "CPU") << " spectral pipelines passed\n"; // Write this part of the example result or diagnostic.
        return 0; // Report successful completion.
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
    } catch (const shdr_examples::context_unavailable& error) { // Report the exception from the preceding operation.
        std::cerr << "SKIP: " << error.what() << '\n'; // Write this part of the example result or diagnostic.
        return 77; // Report an unavailable optional GPU backend.
#endif // Finish this conditional compilation branch.
    } catch (const std::exception& error) { // Report the exception from the preceding operation.
        std::cerr << error.what() << '\n'; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
