// Decode byte-addressed Bayer8, little-endian Bayer16, MIPI RAW10 or MIPI RAW12.
// Each packed_words[] uint carries four consecutive bytes, least significant byte first.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Group 64 independent invocations along x.
layout(std430, binding = 0) readonly buffer Packed { // Read four packed bytes per uint in little-endian byte order through buffer binding 0.
    uint packed_words[]; // Store four packed bytes per uint in little-endian byte order.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer Samples { // Write one unpacked unsigned sensor value per pixel through buffer binding 1.
    uint samples[]; // Store one unpacked unsigned sensor value per pixel.
}; // Finish this declaration.
uniform uint count; // Limit work to the valid element count, including a partial final workgroup.
uniform uint packing; // Select Bayer8, little-endian Bayer16, MIPI RAW10 or MIPI RAW12 byte packing.
uniform uint bit_depth; // Set the number of valid bits in each unpacked sensor sample.
uint byte_at(uint position) { // Extract one byte from the little-endian packed uint buffer.
    return (packed_words[position / 4u] >> ((position % 4u) * 8u)) & 255u; // Shift the requested byte to the low bits and mask away the other three bytes.
} // End this block.
void main() { // Decode one packed sensor sample into an unsigned Bayer value.
    uint i = gl_GlobalInvocationID.x; // Select this invocation's global x coordinate.
    if (i >= count) // Stop padding invocations before reading or writing an element.
        return; // Stop this invocation before performing another resource access.
    uint value = 0u; // Initialize the unpacked sample before selecting its byte format.
    if (packing == 0u) // Decode a format with one complete sample in each byte.
        value = byte_at(i); // Read the Bayer8 sample directly.
    else if (packing == 1u) // Decode a little-endian format with two bytes per sample.
        value = byte_at(i * 2u) | (byte_at(i * 2u + 1u) << 8u); // Combine the low and high bytes of the Bayer16 sample.
    else if (packing == 2u) { // Decode four RAW10 samples sharing one low-bit byte.
        uint start = (i / 4u) * 5u; // Locate this four-sample RAW10 group of five bytes.
        value = (byte_at(start + i % 4u) << 2u) | ((byte_at(start + 4u) >> ((i % 4u) * 2u)) & 3u); // Combine the eight high bits with this sample's two low bits.
    } else { // Decode two RAW12 samples sharing one low-bit byte.
        uint start = (i / 2u) * 3u; // Locate this two-sample RAW12 group of three bytes.
        value = (byte_at(start + i % 2u) << 4u) | ((byte_at(start + 2u) >> ((i % 2u) * 4u)) & 15u); // Combine the eight high bits with this sample's four low bits.
    } // End this block.
    samples[i] = value & ((1u << bit_depth) - 1u); // Keep only the declared valid sensor bits in the unpacked sample.
} // End this block.
