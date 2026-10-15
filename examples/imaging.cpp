#include "runner.hpp" // Include runner.hpp declarations.
#include <algorithm> // Include algorithm declarations.
#include <random> // Include random declarations.
#include <vector> // Include vector declarations.

namespace { // Group these helpers within their namespace.
// Pixel phases follow the format descriptions in the native demosaicing shader.
unsigned channel(unsigned x, unsigned y, unsigned pattern) { // Implement channel for the operations below.
    const unsigned phase = (y % 2) * 2 + x % 2; // Set phase from this expression.
    const unsigned red = pattern == 0 ? 0 : pattern == 1 ? 3 : pattern == 2 ? 1 : 2; // Set red from this expression.
    return phase == red ? 0 : phase == 3 - red ? 2 : 1; // Return phase == red ? 0 : phase == 3 - red ? 2 : 1.
} // End this block.

// Pack bytes into standard uint storage, shaders extract bytes explicitly, so host endianness is irrelevant.
std::vector<unsigned> pack(const std::vector<unsigned>& samples, unsigned packing) { // Implement pack for the operations below.
    std::vector<unsigned> bytes; // Declare bytes for the following calculation.
    for (std::size_t i = 0; i < samples.size();) { // Iterate over i for this calculation.
        if (packing == 0) { // Check this condition before entering the following branch.
            bytes.push_back(samples[i++]); // Apply push back to these arguments.
        } else if (packing == 1) { // Check this condition before entering the following branch.
            unsigned value = samples[i++]; // Set value from this expression.
            bytes.push_back(value & 255); // Apply push back to these arguments.
            bytes.push_back(value >> 8); // Apply push back to these arguments.
        } else if (packing == 2) { // Check this condition before entering the following branch.
            unsigned lows = 0; // Set lows from this expression.
            for (unsigned j = 0; j < 4; ++j) { // Iterate over j for this calculation.
                bytes.push_back(samples[i + j] >> 2); // Apply push back to these arguments.
                lows |= (samples[i + j] & 3) << (2 * j); // Update lows with this contribution.
            } // End this block.
            bytes.push_back(lows); // Apply push back to these arguments.
            i += 4; // Update i with this contribution.
        } else { // Pack each RAW12 pair into two high bytes and one shared low-nibble byte.
            bytes.push_back(samples[i] >> 4); // Apply push back to these arguments.
            bytes.push_back(samples[i + 1] >> 4); // Apply push back to these arguments.
            bytes.push_back((samples[i] & 15) | ((samples[i + 1] & 15) << 4)); // Apply push back to these arguments.
            i += 2; // Update i with this contribution.
        } // End this block.
    } // End this block.
    std::vector<unsigned> words((bytes.size() + 3) / 4, 0); // Construct words with these initial values.
    for (std::size_t i = 0; i < bytes.size(); ++i) // Iterate over i for this calculation.
        words[i / 4] |= bytes[i] << ((i % 4) * 8); // Update words[i / 4] with this contribution.
    return words; // Return words.
} // End this block.

// Check every pixel against a host interpolation oracle for all 16 packing/phase combinations.
void demosaic(shdr_examples::runner& run) { // Implement demosaic for the operations below.
    const unsigned width = 8, height = 6, count = width * height; // Set width from this expression.
    const unsigned depths[] = {8, 16, 10, 12}; // Declare depths for the following calculation.
    std::mt19937 random(12345); // Construct random with these initial values.
    auto unpack = shdr::make_kernel("bayer_unpack_kernel"); // Create the linked bayer unpack kernel instance.
    auto debayer = shdr::make_kernel("debayer_kernel"); // Create the linked debayer kernel instance.
    auto luma = shdr::make_kernel("luminance_kernel"); // Create the linked luminance kernel instance.
    shdr::buffer<unsigned> decoded(count); // Construct decoded with these initial values.
    shdr::buffer<glsl::vec4> colors(count); // Construct colors with these initial values.
    shdr::buffer<float> gray(count); // Construct gray with these initial values.
    unpack->bind("samples", decoded).set("count", count); // Connect this shader port to its application resource.
    debayer->bind("samples", decoded) // Connect this shader port to its application resource.
        .bind("colors", colors) // Connect this shader port to its application resource.
        .set("width", width) // Apply set to these arguments.
        .set("height", height); // Apply set to these arguments.
    luma->bind("colors", colors).bind("gray", gray).set("count", count); // Connect this shader port to its application resource.
    const shdr::pipeline p = shdr::dispatch(*unpack, shdr::extent(1)) >> // Add a shader dispatch with the required workgroup counts.
                             shdr::dispatch(*debayer, shdr::extent(1, 1)) >> // Add a shader dispatch with the required workgroup counts.
                             shdr::dispatch(*luma, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
    for (unsigned packing = 0; packing < 4; ++packing) // Iterate over packing for this calculation.
        for (unsigned pattern = 0; pattern < 4; ++pattern) { // Iterate over pattern for this calculation.
            const unsigned maximum = (1u << depths[packing]) - 1u; // Set maximum from this expression.
            std::vector<unsigned> samples(count); // Construct samples with these initial values.
            for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
                samples[i] = random() & maximum; // Set samples[i] from this expression.
            const std::vector<unsigned> words = pack(samples, packing); // Set words from this expression.
            shdr::buffer<unsigned> packed(words.size()); // Construct packed with these initial values.
            std::copy(words.begin(), words.end(), packed.data()); // Apply copy to these arguments.
            unpack->bind("packed_words", packed) // Connect this shader port to its application resource.
                .set("packing", packing) // Apply set to these arguments.
                .set("bit_depth", depths[packing]); // Apply set to these arguments.
            debayer->set("pattern", pattern).set("bit_depth", depths[packing]); // Apply set to these arguments.
            run.run(p); // Execute the configured pipeline on the requested backend.
            const unsigned* actual_samples = // Set actual samples from this expression.
                static_cast<const shdr::buffer<unsigned>&>(decoded).data(); // Apply data to these arguments.
            const glsl::vec4* actual_colors = // Set actual colors from this expression.
                static_cast<const shdr::buffer<glsl::vec4>&>(colors).data(); // Apply data to these arguments.
            const float* actual_gray = static_cast<const shdr::buffer<float>&>(gray).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
            for (unsigned y = 0; y < height; ++y) // Iterate over y for this calculation.
                for (unsigned x = 0; x < width; ++x) { // Iterate over x for this calculation.
                    const unsigned i = y * width + x; // Set i from this expression.
                    if (actual_samples[i] != samples[i]) // Check this condition before entering the following branch.
                        throw std::runtime_error("packed Bayer decoding differs"); // Report packed Bayer decoding differs.
                    float rgb[3] = {0, 0, 0}; // Set rgb[3] from this expression.
                    for (unsigned c = 0; c < 3; ++c) { // Iterate over c for this calculation.
                        if (channel(x, y, pattern) == c) // Check this condition before entering the following branch.
                            rgb[c] = float(samples[i]) / float(maximum); // Set rgb[c] from this expression.
                        else { // Average neighboring measurements of the missing color channel.
                            double sum = 0; // Use this floating-point value or resource in the following calculation.
                            unsigned weight = 0; // Set weight from this expression.
                            for (int dy = -1; dy <= 1; ++dy) // Iterate over dy for this calculation.
                                for (int dx = -1; dx <= 1; ++dx) { // Iterate over dx for this calculation.
                                    const unsigned nx = static_cast<unsigned>( // Set nx from this expression.
                                        std::max(0, std::min(int(width) - 1, int(x) + dx))); // Apply max to these arguments.
                                    const unsigned ny = static_cast<unsigned>( // Set ny from this expression.
                                        std::max(0, std::min(int(height) - 1, int(y) + dy))); // Apply max to these arguments.
                                    if (channel(nx, ny, pattern) == c) { // Check this condition before entering the following branch.
                                        sum += samples[ny * width + nx]; // Update sum with this contribution.
                                        ++weight; // Continue the expression using weight.
                                    } // End this block.
                                } // End this block.
                            rgb[c] = weight ? float(sum / (weight * double(maximum))) : 0; // Set rgb[c] from this expression.
                        } // End this block.
                        shdr_examples::require_close( // Check the computed value against the independent reference.
                            actual_colors[i][c], rgb[c], "Bayer interpolation"); // Continue the expression using actual colors, i, c.
                    } // End this block.
                    shdr_examples::require_close(actual_colors[i].w, 1, "Bayer alpha"); // Check the computed value against the independent reference.
                    shdr_examples::require_close(actual_gray[i], // Check the computed value against the independent reference.
                                                 rgb[0] * 0.2126f + rgb[1] * 0.7152f + // Continue the expression using rgb, rgb.
                                                     rgb[2] * 0.0722f, // Continue the expression using rgb.
                                                 "Bayer luminance"); // Provide this text argument for the operation above.
                } // End this block.
        } // End this block.
    std::cout << "All four Bayer phases and Bayer8/Bayer16/RAW10/RAW12 packings verified\n"; // Write this part of the example result or diagnostic.
} // End this block.

// A multi-stage lossless compressor allocates capacity once and records compact stream length in offsets.
void compression(shdr_examples::runner& run) { // Implement compression for the operations below.
    const unsigned count = 513, block_size = 32, blocks = (count + block_size - 1) / block_size; // Set count from this expression.
    shdr::buffer<unsigned> values(count), sizes(blocks), offsets(blocks + 1), encoded(count * 2), // Declare values with the parameters continued below.
        restored(count); // Apply restored to these arguments.
    std::mt19937 random(54321); // Construct random with these initial values.
    unsigned* input = values.data(); // Set input from this expression.
    for (unsigned i = 0; i < count;) { // Iterate over i for this calculation.
        const unsigned value = random() % 256, span = 3 + random() % 12; // Set value from this expression.
        for (unsigned j = 0; j < span && i < count; ++j) // Iterate over j for this calculation.
            input[i++] = value; // Set input[i++] from this expression.
    } // End this block.
    auto counter = shdr::make_kernel("rle_count_kernel"); // Create the linked rle count kernel instance.
    auto prefix = shdr::make_kernel("rle_offsets_kernel"); // Create the linked rle offsets kernel instance.
    auto encoder = shdr::make_kernel("rle_encode_kernel"); // Create the linked rle encode kernel instance.
    auto decoder = shdr::make_kernel("rle_decode_kernel"); // Create the linked rle decode kernel instance.
    counter->bind("values", values) // Connect this shader port to its application resource.
        .bind("sizes", sizes) // Connect this shader port to its application resource.
        .set("count", count) // Apply set to these arguments.
        .set("block_size", block_size); // Apply set to these arguments.
    prefix->bind("sizes", sizes).bind("offsets", offsets).set("blocks", blocks); // Connect this shader port to its application resource.
    encoder->bind("values", values) // Connect this shader port to its application resource.
        .bind("offsets", offsets) // Connect this shader port to its application resource.
        .bind("encoded", encoded) // Connect this shader port to its application resource.
        .set("count", count) // Apply set to these arguments.
        .set("block_size", block_size); // Apply set to these arguments.
    decoder->bind("encoded", encoded) // Connect this shader port to its application resource.
        .bind("offsets", offsets) // Connect this shader port to its application resource.
        .bind("restored", restored) // Connect this shader port to its application resource.
        .set("count", count) // Apply set to these arguments.
        .set("block_size", block_size); // Apply set to these arguments.
    const shdr::extent groups((blocks + 63) / 64); // Construct groups with these initial values.
    const shdr::pipeline p = shdr::dispatch(*counter, groups) >> // Add a shader dispatch with the required workgroup counts.
                             shdr::dispatch(*prefix, shdr::extent(1)) >> // Add a shader dispatch with the required workgroup counts.
                             shdr::dispatch(*encoder, groups) >> shdr::dispatch(*decoder, groups); // Add a shader dispatch with the required workgroup counts.
    run.run(p); // Execute the configured pipeline on the requested backend.
    const unsigned* decoded = static_cast<const shdr::buffer<unsigned>&>(restored).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
    const unsigned* locations = static_cast<const shdr::buffer<unsigned>&>(offsets).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
    const unsigned* words = static_cast<const shdr::buffer<unsigned>&>(encoded).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
    unsigned expected_position = 0; // Set expected position from this expression.
    for (unsigned b = 0; b < blocks; ++b) { // Iterate over b for this calculation.
        if (locations[b] != expected_position) // Check this condition before entering the following branch.
            throw std::runtime_error("RLE prefix mismatch"); // Report RLE prefix mismatch.
        unsigned i = b * block_size, end = std::min(count, (b + 1) * block_size); // Set i from this expression.
        while (i < end) { // Find the next complete run in this tile.
            const unsigned value = input[i]; // Set value from this expression.
            unsigned span = 1; // Set span from this expression.
            while (i + span < end && input[i + span] == value) // Extend the run across consecutive copies of the same value.
                ++span; // Continue the expression using span.
            if (words[expected_position] != value || words[expected_position + 1] != span) // Check this condition before entering the following branch.
                throw std::runtime_error("RLE encoded pair mismatch"); // Report RLE encoded pair mismatch.
            expected_position += 2; // Update expected position with this contribution.
            i += span; // Update i with this contribution.
        } // End this block.
    } // End this block.
    if (locations[blocks] != expected_position) // Check this condition before entering the following branch.
        throw std::runtime_error("RLE compact length mismatch"); // Report RLE compact length mismatch.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        if (decoded[i] != input[i]) // Check this condition before entering the following branch.
            throw std::runtime_error("lossless RLE roundtrip mismatch"); // Report lossless RLE roundtrip mismatch.
    std::cout << "Lossless tiled RLE: " << count << " input words, " << expected_position // Write this part of the example result or diagnostic.
              << " encoded words + " << blocks + 1 << " offsets\n"; // Write this part of the example result or diagnostic.
} // End this block.
} // namespace

int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Construct run with these initial values.
        demosaic(run); // Apply demosaic to these arguments.
        compression(run); // Apply compression to these arguments.
        return 0; // Report successful completion.
    } // End this block.
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
    catch (const shdr_examples::context_unavailable& e) { // Handle the indicated exception from the preceding operation.
        std::cerr << "SKIP: " << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 77; // Report an unavailable optional GPU backend.
    } // End this block.
#endif // Finish this conditional compilation branch.
    catch (const std::exception& e) { // Handle the indicated exception from the preceding operation.
        std::cerr << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
