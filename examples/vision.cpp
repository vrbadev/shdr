#include "runner.hpp" // Include runner.hpp declarations.
#include <vector> // Include vector declarations.

namespace { // Group these helpers within their namespace.

/// Index a row-major image with the same explicit clamp-to-edge border convention.
std::size_t pixel(int x, int y, unsigned width, unsigned height) { // Implement pixel for the operations below.
    const unsigned cx = unsigned(std::max(0, std::min(int(width) - 1, x))); // Set cx from this expression.
    const unsigned cy = unsigned(std::max(0, std::min(int(height) - 1, y))); // Set cy from this expression.
    return std::size_t(cy) * width + cx; // Return std::size t(cy) * width + cx.
} // End this block.

/// Independent reference: apply the full 25-tap Gaussian convolution directly.
std::vector<float> // Continue the expression using std, vector, float.
gaussian_reference(const shdr::buffer<float>& input, unsigned width, unsigned height) { // Apply gaussian reference to these arguments.
    const int weights[5] = {1, 4, 6, 4, 1}; // Set weights[5] from this expression.
    std::vector<float> result(std::size_t(width) * height); // Construct result with these initial values.
    const float* source = input.data(); // Set source from this expression.
    for (unsigned y = 0; y < height; ++y) // Iterate over y for this calculation.
        for (unsigned x = 0; x < width; ++x) { // Iterate over x for this calculation.
            double sum = 0.0; // Use this floating-point value or resource in the following calculation.
            for (int dy = -2; dy <= 2; ++dy) // Iterate over dy for this calculation.
                for (int dx = -2; dx <= 2; ++dx) // Iterate over dx for this calculation.
                    sum += double(source[pixel(int(x) + dx, int(y) + dy, width, height)]) * // Update sum with this contribution.
                           weights[dx + 2] * weights[dy + 2]; // Continue the expression using weights, dx, weights.
            result[std::size_t(y) * width + x] = float(sum / 256.0); // Set result[std::size t(y) * width + x] from this expression.
        } // End this block.
    return result; // Return result.
} // End this block.

/// Apply the two Sobel coefficient tables independently of the shader's expanded formulas.
std::vector<glsl::vec2> // Continue the expression using std, vector, glsl.
gradient_reference(const std::vector<float>& input, unsigned width, unsigned height) { // Apply gradient reference to these arguments.
    const int sx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}; // Set sx[3][3] from this expression.
    const int sy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}; // Set sy[3][3] from this expression.
    std::vector<glsl::vec2> result(std::size_t(width) * height); // Construct result with these initial values.
    for (unsigned y = 0; y < height; ++y) // Iterate over y for this calculation.
        for (unsigned x = 0; x < width; ++x) { // Iterate over x for this calculation.
            double gx = 0.0, gy = 0.0; // Use this floating-point value or resource in the following calculation.
            for (int dy = -1; dy <= 1; ++dy) // Iterate over dy for this calculation.
                for (int dx = -1; dx <= 1; ++dx) { // Iterate over dx for this calculation.
                    const double value = input[pixel(int(x) + dx, int(y) + dy, width, height)]; // Set value from this expression.
                    gx += value * sx[dy + 1][dx + 1]; // Update gx with this contribution.
                    gy += value * sy[dy + 1][dx + 1]; // Update gy with this contribution.
                } // End this block.
            result[std::size_t(y) * width + x] = glsl::vec2(float(gx / 8.0), float(gy / 8.0)); // Set result[std::size t(y) * width + x] from this expression.
        } // End this block.
    return result; // Return result.
} // End this block.

/// Form and smooth a symmetric 2x2 tensor, then use det(M)-k*trace(M)^2.
std::vector<float> harris_reference(const std::vector<glsl::vec2>& gradients, // Declare harris reference with the parameters continued below.
                                    unsigned width, // Declare width for the following calculation.
                                    unsigned height, // Declare height for the following calculation.
                                    float k) { // Declare k for the following calculation.
    const int weights[3] = {1, 2, 1}; // Set weights[3] from this expression.
    std::vector<float> result(std::size_t(width) * height); // Construct result with these initial values.
    for (unsigned y = 0; y < height; ++y) // Iterate over y for this calculation.
        for (unsigned x = 0; x < width; ++x) { // Iterate over x for this calculation.
            double xx = 0.0, xy = 0.0, yy = 0.0; // Use this floating-point value or resource in the following calculation.
            for (int dy = -1; dy <= 1; ++dy) // Iterate over dy for this calculation.
                for (int dx = -1; dx <= 1; ++dx) { // Iterate over dx for this calculation.
                    const glsl::vec2& g = gradients[pixel(int(x) + dx, int(y) + dy, width, height)]; // Set g from this expression.
                    const double weight = double(weights[dx + 1] * weights[dy + 1]) / 16.0; // Set weight from this expression.
                    xx += double(g.x) * g.x * weight; // Update xx with this contribution.
                    xy += double(g.x) * g.y * weight; // Update xy with this contribution.
                    yy += double(g.y) * g.y * weight; // Update yy with this contribution.
                } // End this block.
            result[std::size_t(y) * width + x] = // Set result[std::size t(y) * width + x] from this expression.
                float(xx * yy - xy * xy - double(k) * (xx + yy) * (xx + yy)); // Apply float to these arguments.
        } // End this block.
    return result; // Return result.
} // End this block.

/// Bind dimensions shared by the independently dispatched image-processing kernels.
void dimensions(shdr::kernel& kernel, unsigned width, unsigned height) { // Implement dimensions for the operations below.
    kernel.set("width", width).set("height", height); // Apply set to these arguments.
} // End this block.

} // namespace

/// Gaussian smoothing -> Sobel -> structure tensor -> Harris -> nonmaximum suppression.
int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Construct run with these initial values.
        const unsigned width = 32, height = 32, count = width * height; // Set width from this expression.
        const float harris_k = 0.04f; // Set harris k from this expression.
        shdr::buffer<float> input(count), horizontal(count), blurred(count), response(count); // Construct input with these initial values.
        shdr::buffer<glsl::vec2> gradients(count); // Construct gradients with these initial values.
        shdr::buffer<glsl::vec4> tensors(count); // Construct tensors with these initial values.
        shdr::buffer<unsigned> corners(count); // Construct corners with these initial values.
        for (unsigned y = 0; y < height; ++y) // Iterate over y for this calculation.
            for (unsigned x = 0; x < width; ++x) // Iterate over x for this calculation.
                input[std::size_t(y) * width + x] = // Set input[std::size t(y) * width + x] from this expression.
                    x >= 8 && x < 24 && y >= 8 && y < 24 ? 1.0f : 0.0f; // Continue the expression using x, x, y.

        auto gaussian_x = shdr::make_kernel("gaussian_horizontal_kernel"); // Create the linked gaussian horizontal kernel instance.
        auto gaussian_y = shdr::make_kernel("gaussian_vertical_kernel"); // Create the linked gaussian vertical kernel instance.
        auto sobel = shdr::make_kernel("sobel_kernel"); // Create the linked sobel kernel instance.
        auto tensor = shdr::make_kernel("harris_tensor_kernel"); // Create the linked harris tensor kernel instance.
        auto harris = shdr::make_kernel("harris_response_kernel"); // Create the linked harris response kernel instance.
        auto nonmax = shdr::make_kernel("nonmax_kernel"); // Create the linked nonmax kernel instance.
        gaussian_x->bind("input_values", input).bind("output_values", horizontal); // Connect this shader port to its application resource.
        gaussian_y->bind("input_values", horizontal).bind("output_values", blurred); // Connect this shader port to its application resource.
        sobel->bind("input_values", blurred).bind("output_values", gradients); // Connect this shader port to its application resource.
        tensor->bind("input_values", gradients).bind("output_values", tensors).set("count", count); // Connect this shader port to its application resource.
        harris->bind("input_values", tensors) // Connect this shader port to its application resource.
            .bind("output_values", response) // Connect this shader port to its application resource.
            .set("harris_k", harris_k); // Apply set to these arguments.
        nonmax->bind("input_values", response) // Connect this shader port to its application resource.
            .bind("output_values", corners) // Connect this shader port to its application resource.
            .set("threshold", 0.00001f); // Apply set to these arguments.
        dimensions(*gaussian_x, width, height); // Apply dimensions to these arguments.
        dimensions(*gaussian_y, width, height); // Apply dimensions to these arguments.
        dimensions(*sobel, width, height); // Apply dimensions to these arguments.
        dimensions(*harris, width, height); // Apply dimensions to these arguments.
        dimensions(*nonmax, width, height); // Apply dimensions to these arguments.

        const shdr::extent groups((count + 63) / 64); // Construct groups with these initial values.
        auto pipeline = shdr::dispatch(*gaussian_x, groups) >> // Add a shader dispatch with the required workgroup counts.
                        shdr::dispatch(*gaussian_y, groups) >> shdr::dispatch(*sobel, groups) >> // Add a shader dispatch with the required workgroup counts.
                        shdr::dispatch(*tensor, groups) >> shdr::dispatch(*harris, groups) >> // Add a shader dispatch with the required workgroup counts.
                        shdr::dispatch(*nonmax, groups); // Add a shader dispatch with the required workgroup counts.
        run.run(pipeline); // Execute the configured pipeline on the requested backend.

        const std::vector<float> expected_blur = gaussian_reference(input, width, height); // Set expected blur from this expression.
        const std::vector<glsl::vec2> expected_gradients = // Set expected gradients from this expression.
            gradient_reference(expected_blur, width, height); // Apply gradient reference to these arguments.
        const std::vector<float> expected_response = // Set expected response from this expression.
            harris_reference(expected_gradients, width, height, harris_k); // Apply harris reference to these arguments.
        const shdr::buffer<float>& read_blur = blurred; // Use a constant buffer view so GPU writes synchronize before host reads.
        const shdr::buffer<glsl::vec2>& read_gradients = gradients; // Use a constant buffer view so GPU writes synchronize before host reads.
        const shdr::buffer<float>& read_response = response; // Use a constant buffer view so GPU writes synchronize before host reads.
        const shdr::buffer<unsigned>& read_corners = corners; // Use a constant buffer view so GPU writes synchronize before host reads.
        const float* blur_data = read_blur.data(); // Set blur data from this expression.
        const glsl::vec2* gradient_data = read_gradients.data(); // Set gradient data from this expression.
        const float* response_data = read_response.data(); // Set response data from this expression.
        const unsigned* corner_data = read_corners.data(); // Set corner data from this expression.
        unsigned corner_count = 0; // Set corner count from this expression.
        for (unsigned y = 0; y < height; ++y) // Iterate over y for this calculation.
            for (unsigned x = 0; x < width; ++x) { // Iterate over x for this calculation.
                const unsigned i = y * width + x; // Set i from this expression.
                shdr_examples::require_close( // Check the computed value against the independent reference.
                    blur_data[i], expected_blur[i], "Gaussian blur", 0.000001f); // Continue the expression using blur data, i, expected blur.
                shdr_examples::require_close( // Check the computed value against the independent reference.
                    gradient_data[i].x, expected_gradients[i].x, "Sobel dx", 0.000001f); // Continue the expression using gradient data, i, x.
                shdr_examples::require_close( // Check the computed value against the independent reference.
                    gradient_data[i].y, expected_gradients[i].y, "Sobel dy", 0.000001f); // Continue the expression using gradient data, i, y.
                shdr_examples::require_close( // Check the computed value against the independent reference.
                    response_data[i], expected_response[i], "Harris response", 0.0000001f); // Continue the expression using response data, i, expected response.
                // This smoothed square has one stable maximum one pixel inside each corner.
                const unsigned expected_corner = // Set expected corner from this expression.
                    (x == 9 || x == 22) && (y == 9 || y == 22) ? 1u : 0u; // Continue the expression using x, x, y.
                if (corner_data[i] != expected_corner) // Check this condition before entering the following branch.
                    throw std::runtime_error( // Report this failed operation.
                        "nonmaximum suppression produced an unexpected corner at " + // Provide this text argument for the operation above.
                        std::to_string(x) + "," + std::to_string(y)); // Apply to string to these arguments.
                corner_count += corner_data[i]; // Update corner count with this contribution.
            } // End this block.
        std::cout << (run.gpu() ? "GPU" : "CPU") << " vision pipeline: " << corner_count // Write this part of the example result or diagnostic.
                  << " Harris corners verified\n"; // Write this part of the example result or diagnostic.
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
