// Downsample independent channel planes with two-by-two average pooling.
#ifdef __cplusplus // Include the runtime only when building the CPU adapter.
#include "shdr.h" // Share resource and math types with the application.
#endif // Finish header inclusion for the generated C++ build.
layout(local_size_x = 64) in; // Give each invocation one pooled feature value.
layout(std430, binding = 0) readonly buffer Inputs { // Read the previous layer in NCHW order.
    float input_values[]; // Store the feature planes produced by convolution.
}; // Finish the input resource declaration.
layout(std430, // Use tightly packed shader storage for the following port.
       binding = 1) writeonly buffer Outputs { // Write a distinct downsampled feature layer.
    float output_values[]; // Preserve the image and channel ordering.
}; // Finish the output resource declaration.
uniform uint batch_count; // Receive the number of independent images.
uniform uint channels; // Receive the number of feature planes per image.
uniform uint input_width; // Receive the even input-plane width.
uniform uint input_height; // Receive the even input-plane height.
void main() { // Average the source square assigned to this output feature.
    uint width = input_width / 2u, height = input_height / 2u; // Halve the two spatial dimensions.
    uint plane = width * height, // Count values per output plane before identifying the image.
         index = gl_GlobalInvocationID.x; // Find this output in the flattened layer.
    if (index >= batch_count * channels * plane) // Ignore excess invocations beyond the pooled output.
        return; // Ignore unused invocations after the output layer.
    uint x = index % width, y = (index % plane) / width; // Recover this pooled pixel's coordinates.
    uint source = (index / plane) * input_width * input_height + 2u * y * input_width + // Locate the upper-left source pixel of this two-by-two pooling tile.
                  2u * x; // Select the top-left source pixel in its channel plane.
    float value = // Begin pooling with the two pixels in the upper source row.
        input_values[source] + input_values[source + 1u]; // Add the upper two source pixels.
    value += input_values[source + input_width] + // Add the two pixels in the second row of the pooling tile.
             input_values[source + input_width + 1u]; // Add the lower two source pixels.
    output_values[index] = value * 0.25; // Divide the complete sum by the four-pixel area.
} // Finish this average-pooling invocation.
