// Apply a valid convolution with optional zero padding and ReLU.
#ifdef __cplusplus // Include the runtime only when building the CPU adapter.
#include "shdr.h" // Share resource and math types with the application.
#endif // Finish header inclusion for the generated C++ build.
layout(local_size_x = 64) in; // Give each invocation one output feature value.
layout(std430, binding = 0) readonly buffer Inputs { // Read the previous layer in NCHW order.
    float input_values[]; // Store complete channel planes inside each image.
}; // Finish the input resource declaration.
layout(std430, binding = 1) readonly buffer Parameters { // Read flattened model weights and biases.
    float parameters[]; // Match the offline model parameter order.
}; // Finish the parameter resource declaration.
layout(std430, binding = 2) // Use the third tightly packed shader storage binding.
    writeonly buffer Outputs { // Store each next-layer feature in its own output position.
    float output_values[]; // Keep the next layer in the same NCHW order.
}; // Finish the output resource declaration.
uniform uint batch_count; // Receive the number of independent images.
uniform uint input_width; // Receive the width of each input channel plane.
uniform uint input_height; // Receive the height of each input channel plane.
uniform uint input_channels; // Receive the number of input feature planes.
uniform uint output_channels; // Receive the number of learned convolution filters.
uniform uint kernel_size; // Receive the side length of each square filter.
uniform uint padding; // Receive the number of implicit zero pixels on each edge.
uniform uint weight_offset; // Receive the first convolution weight in the flat model.
uniform uint bias_offset; // Receive the first convolution bias in the flat model.
void main() { // Compute the output feature value assigned to this invocation.
    uint width = // Compute the output width after kernel extent and zero padding.
        input_width + 2u * padding - kernel_size + 1u; // Find the valid padded output width.
    uint height = // Compute the output height after kernel extent and zero padding.
        input_height + 2u * padding - kernel_size + 1u; // Find the valid padded output height.
    uint plane = width * height, // Count values per output plane before identifying the image.
         index = gl_GlobalInvocationID.x; // Find this output within the flattened layer.
    if (index >= batch_count * output_channels * plane) // Ignore excess invocations beyond the final convolution output.
        return; // Ignore unused invocations after the layer.
    uint row = index / (output_channels * plane); // Find the independent image in this batch.
    uint channel = // Select the learned output feature plane.
        (index / plane) % output_channels; // Find the learned filter producing this feature.
    uint x = index % width, // Decode the horizontal and vertical output coordinates.
         y = (index % plane) / width; // Recover the pixel coordinates in the output plane.
    float value = // Begin convolution accumulation with the learned channel bias.
        parameters[bias_offset + channel]; // Start the feature value with its learned bias.
    for (uint source = 0u; source < input_channels; // Visit every input feature plane connected to this output.
         ++source) { // Accumulate contributions from every input plane.
        for (uint ky = 0u; ky < kernel_size; ++ky) { // Visit each row of the convolution kernel.
            for (uint kx = 0u; kx < kernel_size; // Visit every horizontal position in the learned kernel.
                 ++kx) { // Visit each column of the convolution kernel.
                int sx = int(x + kx) - // Move the kernel position into signed padded input coordinates.
                         int(padding); // Place the kernel column inside the unpadded input.
                int sy = // Move the vertical kernel position into signed padded input coordinates.
                    int(y + ky) - int(padding); // Place the kernel row inside the unpadded input.
                if (sx < 0 || sy < 0 || sx >= int(input_width) || sy >= int(input_height)) // Treat positions outside the original feature plane as zero.
                    continue; // Use zero outside the input image.
                uint sample_index = (row * input_channels + source) * input_width * input_height + // Locate this image and source channel in NCHW storage.
                                    uint(sy) * input_width + // Move to the selected row of the source feature plane.
                                    uint(sx); // Select the NCHW input feature.
                uint coefficient = // Locate this learned kernel coefficient.
                    weight_offset + // Begin at the first weight of this convolution layer.
                    ((channel * input_channels + source) * kernel_size + ky) * kernel_size + // Select the output channel, input channel and kernel row.
                    kx; // Select the matching learned kernel coefficient.
                value += parameters[coefficient] * // Multiply the learned coefficient by the selected source feature.
                         input_values[sample_index]; // Accumulate this convolution contribution.
            } // Finish this kernel row.
        } // Finish this source plane's kernel.
    } // Finish the sum over input feature planes.
    output_values[index] = // Store the completed feature with the chosen activation.
        max(value, 0.0); // Apply ReLU once after the complete learned projection.
} // Finish this convolution invocation.
