// Bilinear demosaicing of all four Bayer phases, with clamp-to-edge sampling.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in; // Cover a 8 by 8 group of output coordinates.
layout(std430, binding = 0) readonly buffer Samples { // Read one unpacked unsigned sensor value per pixel through buffer binding 0.
    uint samples[]; // Store one unpacked unsigned sensor value per pixel.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Colors { // Write normalized linear RGBA colors through buffer binding 1.
    vec4 colors[]; // Store normalized linear RGBA colors.
}; // Finish this declaration.
uniform uint width; // Set the row width of the flattened image buffer.
uniform uint height; // Set the number of rows in the image buffer.
uniform uint pattern; // Select the RGGB, BGGR, GRBG or GBRG Bayer phase.
uniform uint bit_depth; // Set the number of valid bits in each unpacked sensor sample.
uint color_at(uint x, uint y) { // Determine which sensor color the two-by-two Bayer phase records here.
    uint phase = (y % 2u) * 2u + x % 2u; // Locate this pixel inside its repeating two-by-two Bayer cell.
    uint red = pattern == 0u ? 0u : pattern == 1u ? 3u : pattern == 2u ? 1u : 2u; // Select the red cell position for the configured Bayer pattern.
    uint blue = 3u - red; // Place blue diagonally opposite red within that Bayer cell.
    return phase == red ? 0u : phase == blue ? 2u : 1u; // Return red, green or blue as channel index zero, one or two.
} // End this block.
void main() { // Reconstruct the missing color channels at one Bayer sensor pixel.
    uvec2 p = gl_GlobalInvocationID.xy; // Select the output pixel or voxel assigned to this invocation.
    if (p.x >= width || p.y >= height) // Ignore workgroup padding outside the sensor image.
        return; // Stop this invocation before performing another resource access.
    vec3 result = vec3(0.0); // Start the reconstructed RGB pixel at zero.
    uint center_color = color_at(p.x, p.y); // Identify the sensor channel already recorded at the center pixel.
    for (uint channel = 0u; channel < 3u; ++channel) { // Reconstruct each of the three output color channels.
        float value = 0.0; // Start the neighboring sample sum for this channel.
        float weight = 0.0; // Count how many matching-color samples enter the average.
        if (center_color == channel) // Preserve the channel that was measured directly at this pixel.
            value = float(samples[p.y * width + p.x]); // Read the measured center sample for its recorded color channel.
        else { // Interpolate the channel from neighboring same-color samples.
            for (int dy = -1; dy <= 1; ++dy) // Visit the three nearby sensor rows.
                for (int dx = -1; dx <= 1; ++dx) { // Visit the three nearby sensor columns.
                    uint x = uint(clamp(int(p.x) + dx, 0, int(width) - 1)); // Clamp the neighboring column at the sensor boundary.
                    uint y = uint(clamp(int(p.y) + dy, 0, int(height) - 1)); // Clamp the neighboring row at the sensor boundary.
                    if (color_at(x, y) == channel) { // Use only neighboring samples that measure this channel.
                        value += float(samples[y * width + x]); // Accumulate one neighboring sensor measurement of this channel.
                        weight += 1.0; // Count that measurement toward the interpolation average.
                    } // End this block.
                } // End this block.
            if (weight > 0.0) // Average the channel when at least one matching neighbor contributes.
                value /= weight; // Average the selected same-color neighbor measurements.
        } // End this block.
        result[channel] = value / float((1u << bit_depth) - 1u); // Normalize the reconstructed channel by the maximum valid sensor value.
    } // End this block.
    colors[p.y * width + p.x] = vec4(result, 1.0); // Store the reconstructed linear RGB pixel with opaque alpha.
} // End this block.
