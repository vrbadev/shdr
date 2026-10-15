/** @file evt3.cpp
 * @brief Validate a streaming parallel EVT3 decoder against a scalar host oracle.
 * Checks direct/vector events, trigger order, camera bit, synchronization, wraps,
 * empty/odd-size batches, arbitrary block cuts, and state carried between calls.
 */
#include "runner.hpp" // Include runner.hpp declarations.
#include "evt3_support.hpp" // Include evt3_support.hpp declarations.
#include "evt3_flow_checks.hpp" // Include evt3_flow_checks.hpp declarations.
#include <cstdint> // Include cstdint declarations.
#include <vector> // Include vector declarations.

namespace { // Group these helpers within their namespace.
using shdr::glsl::uvec4; // Give this type or declaration a convenient local name.

// A scalar switch interpreter forms timestamps in uint64 arithmetic independently
// of the GPU block summaries, prefix passes, and packed output addressing.
struct reference_decoder { // Declare reference decoder.
    unsigned y = 0, base = 0, polarity = 0, high = 0, low = 0, wraps = 0; // Set y from this expression.
    bool synchronized = false; // Set synchronized from this expression.
    std::vector<uvec4> decode(const std::vector<std::uint16_t>& words) { // Implement decode for the operations below.
        std::vector<uvec4> result; // Declare result for the following calculation.
        for (std::uint16_t word : words) { // Visit each std required by this loop.
            const unsigned value = word & 4095u; // Set value from this expression.
            switch (word >> 12u) { // Select behavior from the value of this expression.
            case 0: // Handle this specific selection value.
                y = value; // Set y from this expression.
                break; // Finish this loop or selection immediately.
            case 3: // Handle this specific selection value.
                base = value & 2047u; // Set base from this expression.
                polarity = value >> 11u; // Set polarity from this expression.
                break; // Finish this loop or selection immediately.
            case 6: // Handle this specific selection value.
                low = value; // Set low from this expression.
                break; // Finish this loop or selection immediately.
            case 8: // Handle this specific selection value.
                if (synchronized && high > value && high - value >= 4085u) // Check this condition before entering the following branch.
                    ++wraps; // Continue the expression using wraps.
                high = value; // Set high from this expression.
                low = 0; // Set low from this expression.
                synchronized = true; // Set synchronized from this expression.
                break; // Finish this loop or selection immediately.
            case 2: // Handle this specific selection value.
                if (synchronized) // Check this condition before entering the following branch.
                    append(result, // Apply append to these arguments.
                           value & 2047u, // Continue the expression using value.
                           (y & 2047u) | ((value >> 11u) << 11u) | ((y >> 11u) << 12u)); // Continue the expression using y, value, y.
                break; // Finish this loop or selection immediately.
            case 4: // Handle this specific selection value.
            case 5: { // Handle this specific selection value.
                const unsigned width = (word >> 12u) == 4 ? 12u : 8u; // Set width from this expression.
                if (synchronized) // Check this condition before entering the following branch.
                    for (unsigned bit = 0; bit < width; ++bit) // Iterate over bit for this calculation.
                        if (value & (1u << bit)) // Check this condition before entering the following branch.
                            append(result, // Apply append to these arguments.
                                   (base + bit) & 2047u, // Continue the expression using base, bit.
                                   (y & 2047u) | (polarity << 11u) | ((y >> 11u) << 12u)); // Continue the expression using y, polarity, y.
                base = (base + width) & 2047u; // Set base from this expression.
                break; // Finish this loop or selection immediately.
            } // End this block.
            case 10: // Handle this specific selection value.
                if (synchronized) // Check this condition before entering the following branch.
                    append(result, value >> 8u, 0x80000000u | (value & 1u)); // Apply append to these arguments.
                break; // Finish this loop or selection immediately.
            default: // Skip extension and reserved word types after processing supported event words.
                break; // Finish this loop or selection immediately.
            } // End this block.
        } // End this block.
        return result; // Return result.
    } // End this block.
    void append(std::vector<uvec4>& result, unsigned x, unsigned yp) const { // Implement append for the operations below.
        const std::uint64_t t = // Set t from this expression.
            (std::uint64_t(wraps) * 16777216u) + (std::uint64_t(high) * 4096u) + low; // Apply uint64 t to these arguments.
        result.push_back(uvec4(x, yp, static_cast<unsigned>(t), static_cast<unsigned>(t >> 32))); // Apply push back to these arguments.
    } // End this block.
    void verify_state(const shdr::buffer<uvec4>& state) const { // Implement verify state for the operations below.
        const uvec4* actual = state.data(); // Set actual from this expression.
        if (actual[0] != uvec4(y, base, polarity, high) || // Check this condition before entering the following branch.
            actual[1] != uvec4(low, wraps, synchronized ? 1u : 0u, 0u)) // Apply uvec4 to these arguments.
            throw std::runtime_error("EVT3 persistent decoder state mismatch"); // Report EVT3 persistent decoder state mismatch.
    } // End this block.
}; // Finish this declaration.

void compare(shdr_examples::evt3_decoder& decoder, // Declare compare with the parameters continued below.
             const std::vector<uvec4>& expected, // Declare expected for the following calculation.
             reference_decoder& reference) { // Declare reference for the following calculation.
    if (decoder.event_count() != expected.size()) // Check this condition before entering the following branch.
        throw std::runtime_error("EVT3 output length mismatch"); // Report EVT3 output length mismatch.
    const uvec4* actual = static_cast<const shdr::buffer<uvec4>&>(decoder.events).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
    for (std::size_t i = 0; i < expected.size(); ++i) // Iterate over i for this calculation.
        if (actual[i] != expected[i]) // Check this condition before entering the following branch.
            throw std::runtime_error("EVT3 ordered event mismatch at " + std::to_string(i)); // Report EVT3 ordered event mismatch at .
    reference.verify_state(decoder.state); // Apply verify state to these arguments.
} // End this block.

void independent(shdr_examples::runner& runner) { // Implement independent for the operations below.
    std::vector<std::uint16_t> words = { // Set words from this expression.
        0x0007u, 0x3003u, 0x4fffu, 0x2002u, 0xa101u, // Defer these prefix events until a timestamp establishes synchronization.
        0x8ffeu, 0x6fffu, 0x0fffu, 0x3ffdu, 0x400bu, 0x5101u, // camera + X wrap, VECT8 ignores unused bits
        0x8fffu, 0x6002u, 0x2005u, 0x8000u, 0x6023u, 0xaff1u, 0xe012u, // Complete this expression or declaration.
        0xf123u, 0x7004u, 0x1000u, 0x4000u, 0x5fffu, // extensions/reserved/empty vectors
        0x8000u, 0x2007u // repeated TIME_HIGH resets low time
    }; // Finish this declaration.
    const unsigned block_sizes[] = {1, 3, 7, 64, 128}; // Declare block sizes for the following calculation.
    for (unsigned block_size : block_sizes) { // Visit each block size required by this loop.
        shdr_examples::evt3_decoder decoder(unsigned(words.size()), block_size); // Construct decoder with these initial values.
        decoder.load(words); // Apply load to these arguments.
        reference_decoder reference; // Declare reference for the following calculation.
        const std::vector<uvec4> expected = reference.decode(words); // Set expected from this expression.
        runner.run(decoder.reset_pipeline()); // Execute the configured pipeline on the requested backend.
        compare(decoder, expected, reference); // Apply compare to these arguments.
        runner.run(decoder.reset_pipeline()); // Execute the configured pipeline on the requested backend.
        compare(decoder, expected, reference); // Apply compare to these arguments.
    } // End this block.
    shdr_examples::evt3_decoder decoder(6013); // Construct decoder with these initial values.
    words = shdr_examples::evt3_random_words(6013, 897); // Set words from this expression.
    decoder.load(words); // Apply load to these arguments.
    reference_decoder reference; // Declare reference for the following calculation.
    const std::vector<uvec4> expected = reference.decode(words); // Set expected from this expression.
    runner.run(decoder.reset_pipeline()); // Execute the configured pipeline on the requested backend.
    compare(decoder, expected, reference); // Apply compare to these arguments.
} // End this block.

void streaming(shdr_examples::runner& runner) { // Implement streaming for the operations below.
    const std::vector<std::uint16_t> words = shdr_examples::evt3_random_words(2049, 293); // Set words from this expression.
    const unsigned chunks[] = {1, 17, 127, 128, 129}; // Declare chunks for the following calculation.
    for (unsigned chunk : chunks) { // Visit each chunk required by this loop.
        shdr_examples::evt3_decoder decoder(chunk, 7); // Construct decoder with these initial values.
        reference_decoder reference; // Declare reference for the following calculation.
        for (std::size_t begin = 0; begin < words.size(); begin += chunk) { // Iterate over begin for this calculation.
            const std::size_t end = std::min(words.size(), begin + chunk); // Set end from this expression.
            const std::vector<std::uint16_t> batch(words.begin() + begin, words.begin() + end); // Construct batch with these initial values.
            decoder.load(batch); // Apply load to these arguments.
            const std::vector<uvec4> expected = reference.decode(batch); // Set expected from this expression.
            runner.run(decoder.stream_pipeline()); // Execute the configured pipeline on the requested backend.
            compare(decoder, expected, reference); // Apply compare to these arguments.
        } // End this block.
        decoder.load(std::vector<std::uint16_t>()); // Apply load to these arguments.
        runner.run(decoder.stream_pipeline()); // Execute the configured pipeline on the requested backend.
        compare(decoder, std::vector<uvec4>(), reference); // Apply compare to these arguments.
    } // End this block.
    // The 256th wrap crosses uint32 timestamp range, pair representation keeps it exact.
    shdr_examples::evt3_decoder decoder(3, 1); // Construct decoder with these initial values.
    decoder.state.data()[0] = uvec4(2047u, 0u, 1u, 4095u); // Apply data to these arguments.
    decoder.state.data()[1] = uvec4(0u, 255u, 1u, 0u); // Apply data to these arguments.
    decoder.load(std::vector<std::uint16_t>{0x8000u, 0x6123u, 0x2001u}); // Apply load to these arguments.
    reference_decoder reference; // Declare reference for the following calculation.
    reference.y = 2047u; // Set reference.y from this expression.
    reference.polarity = 1u; // Set reference.polarity from this expression.
    reference.high = 4095u; // Set reference.high from this expression.
    reference.wraps = 255u; // Set reference.wraps from this expression.
    reference.synchronized = true; // Set reference.synchronized from this expression.
    const std::vector<uvec4> expected = // Set expected from this expression.
        reference.decode(std::vector<std::uint16_t>{0x8000u, 0x6123u, 0x2001u}); // Apply decode to these arguments.
    runner.run(decoder.stream_pipeline()); // Execute the configured pipeline on the requested backend.
    compare(decoder, expected, reference); // Apply compare to these arguments.
} // End this block.

// Consume CPU or GPU decoded events on the host in a compact downstream CD layout.
void compact_processing(shdr_examples::runner& runner) { // Implement compact processing for the operations below.
    const std::vector<std::uint16_t> words = { // Set words from this expression.
        0x8fffu, 0x6fffu, 0x02cfu, 0x2cffu, 0xa101u, 0x8000u, 0x6003u, 0x2002u}; // Complete this expression or declaration.
    shdr_examples::evt3_decoder decoder(unsigned(words.size()), 3); // Construct decoder with these initial values.
    decoder.load(words); // Apply load to these arguments.
    runner.run(decoder.reset_pipeline()); // Execute the configured pipeline on the requested backend.
    const std::vector<std::uint64_t> compact = // Set compact from this expression.
        shdr_examples::evt3_compact_cd_to64(decoder.events, decoder.event_count()); // Apply evt3 compact cd to64 to these arguments.
    if (compact.size() != 2u) // Check this condition before entering the following branch.
        throw std::runtime_error("compact CD consumer should omit trigger records"); // Report compact CD consumer should omit trigger records.
    const unsigned expected_x[] = {1279u, 2u}; // Declare expected x for the following calculation.
    const unsigned expected_p[] = {1u, 0u}; // Declare expected p for the following calculation.
    const std::uint64_t expected_t[] = {16777215u, 16777219u}; // Declare expected t for the following calculation.
    for (unsigned i = 0; i < 2u; ++i) { // Iterate over i for this calculation.
        if ((compact[i] & 2047u) != expected_x[i] || ((compact[i] >> 11u) & 1023u) != 719u || // Check this condition before entering the following branch.
            ((compact[i] >> 21u) & 1u) != expected_p[i] || (compact[i] >> 22u) != expected_t[i]) // Continue the expression using compact, i, expected p.
            throw std::runtime_error( // Report this failed operation.
                "64-bit CD consumer changed coordinates, polarity, or 42-bit timestamp"); // Provide this text argument for the operation above.
    } // End this block.
    shdr::buffer<uvec4> invalid(1); // Construct invalid with these initial values.
    invalid.data()[0] = uvec4(1u, 1024u, 0u, 0u); // Apply data to these arguments.
    bool caught = false; // Set caught from this expression.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::evt3_compact_cd_to64(invalid, 1); // Apply evt3 compact cd to64 to these arguments.
    } catch (const shdr::error&) { // Report the exception from the preceding operation.
        caught = true; // Set caught from this expression.
    } // End this block.
    if (!caught) // Check this condition before entering the following branch.
        throw std::runtime_error("compact CD consumer must validate Y capacity"); // Report compact CD consumer must validate Y capacity.
    invalid.data()[0] = uvec4(1u, 1u, 0u, 1024u); // Apply data to these arguments.
    caught = false; // Set caught from this expression.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::evt3_compact_cd_to64(invalid, 1); // Apply evt3 compact cd to64 to these arguments.
    } catch (const shdr::error&) { // Report the exception from the preceding operation.
        caught = true; // Set caught from this expression.
    } // End this block.
    if (!caught) // Check this condition before entering the following branch.
        throw std::runtime_error("compact CD consumer must validate timestamp capacity"); // Report compact CD consumer must validate timestamp capacity.
    std::cout << "CPU CD consumer: " << compact.size() // Write this part of the example result or diagnostic.
              << " compact uint64 words with 42-bit timestamps verified\n"; // Write this part of the example result or diagnostic.
} // End this block.
} // namespace

int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner runner(argc, argv); // Construct runner with these initial values.
        independent(runner); // Apply independent to these arguments.
        streaming(runner); // Apply streaming to these arguments.
        compact_processing(runner); // Apply compact processing to these arguments.
        shdr_examples::check_evt3_flow(runner); // Apply check evt3 flow to these arguments.
        std::cout // Write this part of the example result or diagnostic.
            << "EVT3 ordered direct/vector/trigger events, block and batch boundaries, and timestamps verified\n"; // Write this part of the example result or diagnostic.
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
