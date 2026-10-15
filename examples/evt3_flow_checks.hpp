/** @file evt3_flow_checks.hpp
 * @brief Synthetic event-plane fixtures and an independent optical-flow oracle.
 * A plane t=100*x+50*y microseconds has normal velocity (8000,4000) pixels/second.
 * Chain decoding and flow estimation through the same event buffers.
 */
#ifndef SHDR_EXAMPLES_EVT3_FLOW_CHECKS_HPP // Open this include guard or optional feature branch.
#define SHDR_EXAMPLES_EVT3_FLOW_CHECKS_HPP // Define SHDR_EXAMPLES_EVT3_FLOW_CHECKS_HPP for the declarations that follow.
#include "runner.hpp" // Include runner.hpp declarations.
#include "evt3_support.hpp" // Include evt3_support.hpp declarations.
#include <algorithm> // Include algorithm declarations.
#include <cmath> // Include cmath declarations.
#include <cstdint> // Include cstdint declarations.
#include <vector> // Include vector declarations.

namespace shdr_examples { // Group these helpers within their namespace.

/** @brief Build time-ordered direct EVT3 events for a translating temporal plane.
 * crossing_32 selects a time origin just before the low timestamp word wraps.
 * Synchronization-only words establish the preceding 24-bit wraps, so the same
 * fixture exercises the decoder and exact timestamp subtraction in the flow stage.
 */
inline std::vector<std::uint16_t> // Declare std for the following calculation.
evt3_planar_words(unsigned width = 32, unsigned height = 24, bool crossing_32 = false) { // Set width from this expression.
    if (!width || !height || width > 2048u || height > 2048u) // Check this condition before entering the following branch.
        throw shdr::error("event-plane dimensions must be between 1 and 2048"); // Report event-plane dimensions must be between 1 and 2048.
    struct pixel { // Declare pixel.
        unsigned x, y, time; // Declare x for the following calculation.
    }; // Finish this declaration.
    std::vector<pixel> pixels; // Declare pixels for the following calculation.
    for (unsigned y = 0; y < height; ++y) // Iterate over y for this calculation.
        for (unsigned x = 0; x < width; ++x) // Iterate over x for this calculation.
            pixels.push_back(pixel{x, y, 100u * x + 50u * y}); // Apply push back to these arguments.
    std::stable_sort(pixels.begin(), pixels.end(), [](const pixel& a, const pixel& b) { // Apply stable sort to these arguments.
        return a.time < b.time; // Return a.time < b.time.
    }); // Finish this declaration.
    std::vector<std::uint16_t> words; // Declare words for the following calculation.
    if (crossing_32) // Check this condition before entering the following branch.
        for (unsigned wrap = 0; wrap < 255u; ++wrap) { // Iterate over wrap for this calculation.
            words.push_back(0x8fffu); // Apply push back to these arguments.
            words.push_back(0x8000u); // Apply push back to these arguments.
        } // End this block.
    const std::uint64_t origin = crossing_32 ? (std::uint64_t(1) << 32) - 2000u : 0u; // Set origin from this expression.
    for (const pixel& p : pixels) { // Visit each p required by this loop.
        const std::uint64_t timestamp = origin + p.time; // Set timestamp from this expression.
        words.push_back(static_cast<std::uint16_t>(0x8000u | ((timestamp >> 12u) & 4095u))); // Apply push back to these arguments.
        words.push_back(static_cast<std::uint16_t>(0x6000u | (timestamp & 4095u))); // Apply push back to these arguments.
        words.push_back(static_cast<std::uint16_t>(p.y)); // Apply push back to these arguments.
        words.push_back(static_cast<std::uint16_t>(0x2000u | p.x)); // Apply push back to these arguments.
    } // End this block.
    return words; // Return words.
} // End this block.

/** @brief Attach a bounded local-plane flow stage to decoded event storage.
 * The decoder offsets provide the compact event count on the GPU. A dispatch
 * covering allocated capacity uses that device-resident count to bound its work.
 */
inline std::unique_ptr<shdr::kernel> evt3_flow_kernel(evt3_decoder& decoder, // Declare evt3 flow kernel with the parameters continued below.
                                                      shdr::buffer<shdr::glsl::vec4>& output, // Declare output for the following calculation.
                                                      unsigned history = 128u) { // Set history from this expression.
    auto shader = shdr::make_kernel("evt3_flow_kernel"); // Create the linked evt3 flow kernel instance.
    shader->bind("events", decoder.events) // Connect this shader port to its application resource.
        .bind("offsets", decoder.offsets) // Connect this shader port to its application resource.
        .bind("flow", output) // Connect this shader port to its application resource.
        .set<unsigned>("blocks", decoder.blocks()) // Supply this shader uniform with its typed application value.
        .set<float>("radius", 3.0f) // Supply this shader uniform with its typed application value.
        .set<unsigned>("history", history) // Supply this shader uniform with its typed application value.
        .set<unsigned>("time_window", 10000u); // Supply this shader uniform with its typed application value.
    return shader; // Return shader.
} // End this block.

namespace detail { // Group these helpers within their namespace.

// Solve the full three-parameter normal equations in double precision as an
// independent reference for the timestamp-plane slopes and intercept.
inline shdr::glsl::vec4 evt3_reference_flow(const std::vector<shdr::glsl::uvec4>& events, // Declare evt3 reference flow with the parameters continued below.
                                            unsigned index, // Declare index for the following calculation.
                                            float radius, // Declare radius for the following calculation.
                                            unsigned history, // Declare history for the following calculation.
                                            unsigned time_window) { // Declare time window for the following calculation.
    const shdr::glsl::uvec4 center = events[index]; // Set center from this expression.
    if (center.y & 0x80000000u) // Check this condition before entering the following branch.
        return shdr::glsl::vec4(0.0f); // Return shdr::glsl::vec4(0.0f).
    double normal[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}}; // Use this floating-point value or resource in the following calculation.
    const std::uint64_t current = (std::uint64_t(center.w) << 32) | center.z; // Set current from this expression.
    unsigned neighbors = 0; // Set neighbors from this expression.
    for (unsigned distance = 1; distance <= std::min(128u, history) && distance <= index; // Iterate over distance for this calculation.
         ++distance) { // Continue the expression using distance.
        const shdr::glsl::uvec4 previous = events[index - distance]; // Set previous from this expression.
        if ((previous.y & 0x80000000u) || (previous.y & 0x1800u) != (center.y & 0x1800u)) // Check this condition before entering the following branch.
            continue; // Skip the remaining work for this loop iteration.
        const std::uint64_t timestamp = (std::uint64_t(previous.w) << 32) | previous.z; // Set timestamp from this expression.
        if (timestamp > current || current - timestamp > time_window) // Check this condition before entering the following branch.
            continue; // Skip the remaining work for this loop iteration.
        const double dx = double(previous.x) - center.x, // Set dx from this expression.
                     dy = double(previous.y & 2047u) - double(center.y & 2047u); // Set dy from this expression.
        if (dx * dx + dy * dy > double(radius) * radius) // Check this condition before entering the following branch.
            continue; // Skip the remaining work for this loop iteration.
        const double row[3] = {dx, dy, 1.0}, dt = -double(current - timestamp); // Set row[3] from this expression.
        for (unsigned r = 0; r < 3; ++r) { // Iterate over r for this calculation.
            for (unsigned c = 0; c < 3; ++c) // Iterate over c for this calculation.
                normal[r][c] += row[r] * row[c]; // Update normal[r][c] with this contribution.
            normal[r][3] += row[r] * dt; // Update normal[r][3] with this contribution.
        } // End this block.
        ++neighbors; // Continue the expression using neighbors.
    } // End this block.
    if (neighbors < 6u) // Check this condition before entering the following branch.
        return shdr::glsl::vec4(0.0f); // Return shdr::glsl::vec4(0.0f).
    for (unsigned c = 0; c < 3; ++c) { // Iterate over c for this calculation.
        unsigned pivot = c; // Set pivot from this expression.
        for (unsigned r = c + 1; r < 3; ++r) // Iterate over r for this calculation.
            if (std::fabs(normal[r][c]) > std::fabs(normal[pivot][c])) // Check this condition before entering the following branch.
                pivot = r; // Set pivot from this expression.
        if (std::fabs(normal[pivot][c]) < 1e-9) // Check this condition before entering the following branch.
            return shdr::glsl::vec4(0.0f); // Return shdr::glsl::vec4(0.0f).
        for (unsigned j = c; j < 4; ++j) // Iterate over j for this calculation.
            std::swap(normal[pivot][j], normal[c][j]); // Apply swap to these arguments.
        const double divisor = normal[c][c]; // Set divisor from this expression.
        for (unsigned j = c; j < 4; ++j) // Iterate over j for this calculation.
            normal[c][j] /= divisor; // Update normal[c][j] with this contribution.
        for (unsigned r = 0; r < 3; ++r) // Iterate over r for this calculation.
            if (r != c) { // Check this condition before entering the following branch.
                const double multiplier = normal[r][c]; // Set multiplier from this expression.
                for (unsigned j = c; j < 4; ++j) // Iterate over j for this calculation.
                    normal[r][j] -= multiplier * normal[c][j]; // Update normal[r][j] with this contribution.
            } // End this block.
    } // End this block.
    const double a = normal[0][3], b = normal[1][3], gradient = a * a + b * b; // Set a from this expression.
    if (gradient <= 0.000001) // Check this condition before entering the following branch.
        return shdr::glsl::vec4(0.0f); // Return shdr::glsl::vec4(0.0f).
    return shdr::glsl::vec4( // Return shdr::glsl::vec4(.
        float(a * 1000000.0 / gradient), float(b * 1000000.0 / gradient), float(neighbors), 1.0f); // Apply float to these arguments.
} // End this block.

inline shdr::glsl::uvec4 evt3_flow_record(unsigned x, unsigned packed_y, std::uint64_t time) { // Implement evt3 flow record for the operations below.
    return shdr::glsl::uvec4( // Return shdr::glsl::uvec4(.
        x, packed_y, static_cast<unsigned>(time), static_cast<unsigned>(time >> 32)); // Continue the expression using x, packed y, static cast.
} // End this block.

// Interleaved camera/polarity records, triggers, a stale high timestamp word and
// an out-of-order timestamp make every neighborhood rejection observable.
inline void evt3_mixed_flow_check(runner& run) { // Implement evt3 mixed flow check for the operations below.
    const std::uint64_t origin = (std::uint64_t(1) << 32) - 200u; // Set origin from this expression.
    std::vector<shdr::glsl::uvec4> records; // Declare records for the following calculation.
    records.push_back(evt3_flow_record(2u, 2u, 0u)); // Apply push back to these arguments.
    for (unsigned y = 0; y < 5; ++y) // Iterate over y for this calculation.
        for (unsigned x = 0; x < 5; ++x) { // Iterate over x for this calculation.
            const unsigned relative = 100u * x + 50u * y; // Set relative from this expression.
            if (relative > 300u || (x == 2u && y == 2u)) // Check this condition before entering the following branch.
                continue; // Skip the remaining work for this loop iteration.
            const std::uint64_t time = origin + relative; // Set time from this expression.
            records.push_back(evt3_flow_record(x, y, time)); // Apply push back to these arguments.
            records.push_back(evt3_flow_record(x, y | 2048u, time - 17u)); // Apply push back to these arguments.
            records.push_back(evt3_flow_record(x, y | 4096u, time - 31u)); // Apply push back to these arguments.
            records.push_back(evt3_flow_record(1u, 0x80000001u, time)); // Apply push back to these arguments.
        } // End this block.
    std::stable_sort( // Apply stable sort to these arguments.
        records.begin(), records.end(), [](const shdr::glsl::uvec4& a, const shdr::glsl::uvec4& b) { // Apply begin to these arguments.
            return a.w < b.w || (a.w == b.w && a.z < b.z); // Return a.w < b.w || (a.w == b.w && a.z < b.z).
        }); // Finish this declaration.
    records.push_back( // Apply push back to these arguments.
        evt3_flow_record(2u, 1u, origin + 400u)); // Future data must be ignored by the final fit.
    records.push_back(evt3_flow_record(2u, 2u, origin + 300u)); // Apply push back to these arguments.
    shdr::buffer<shdr::glsl::uvec4> events(records.size()); // Construct events with these initial values.
    std::copy(records.begin(), records.end(), events.data()); // Apply copy to these arguments.
    shdr::buffer<unsigned> offsets{static_cast<unsigned>(records.size())}; // Declare offsets for the following calculation.
    shdr::buffer<shdr::glsl::vec4> output(records.size()); // Construct output with these initial values.
    auto shader = shdr::make_kernel("evt3_flow_kernel"); // Create the linked evt3 flow kernel instance.
    shader->bind("events", events) // Connect this shader port to its application resource.
        .bind("offsets", offsets) // Connect this shader port to its application resource.
        .bind("flow", output) // Connect this shader port to its application resource.
        .set<unsigned>("blocks", 0u) // Supply this shader uniform with its typed application value.
        .set<float>("radius", 3.0f) // Supply this shader uniform with its typed application value.
        .set<unsigned>("history", 128u) // Supply this shader uniform with its typed application value.
        .set<unsigned>("time_window", 250u); // Supply this shader uniform with its typed application value.
    run.run(shdr::pipeline(shdr::dispatch( // Execute the configured pipeline on the requested backend.
        *shader, shdr::extent((static_cast<unsigned>(records.size()) + 63u) / 64u)))); // Apply extent to these arguments.
    const shdr::glsl::vec4* actual = // Set actual from this expression.
        static_cast<const shdr::buffer<shdr::glsl::vec4>&>(output).data(); // Apply data to these arguments.
    for (unsigned index = 0; index < records.size(); ++index) { // Iterate over index for this calculation.
        const shdr::glsl::vec4 expected = evt3_reference_flow(records, index, 3.0f, 128u, 250u); // Set expected from this expression.
        require_close(actual[index].x, expected.x, "mixed-event flow X", 1.5f); // Check the computed value against the independent reference.
        require_close(actual[index].y, expected.y, "mixed-event flow Y", 1.5f); // Check the computed value against the independent reference.
        require_close(actual[index].z, expected.z, "mixed-event neighbor count", 0.0f); // Check the computed value against the independent reference.
        require_close(actual[index].w, expected.w, "mixed-event fit validity", 0.0f); // Check the computed value against the independent reference.
    } // End this block.
    require_close(actual[records.size() - 1u].x, 8000.0f, "filtered temporal-plane X", 1.5f); // Check the computed value against the independent reference.
    require_close(actual[records.size() - 1u].y, 4000.0f, "filtered temporal-plane Y", 1.5f); // Check the computed value against the independent reference.
    // Require two independent spatial directions for a two-dimensional plane fit.
    for (unsigned index = 0; index < records.size(); ++index) // Iterate over index for this calculation.
        records[index] = evt3_flow_record(index % 3u, 2u, origin + 50u * index); // Set records[index] from this expression.
    std::copy(records.begin(), records.end(), events.data()); // Apply copy to these arguments.
    shader->set<unsigned>("time_window", 10000u); // Supply this shader uniform with its typed application value.
    run.run(shdr::pipeline(shdr::dispatch( // Execute the configured pipeline on the requested backend.
        *shader, shdr::extent((static_cast<unsigned>(records.size()) + 63u) / 64u)))); // Apply extent to these arguments.
    actual = static_cast<const shdr::buffer<shdr::glsl::vec4>&>(output).data(); // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned index = 0; index < records.size(); ++index) // Iterate over index for this calculation.
        if (actual[index] != shdr::glsl::vec4(0.0f)) // Check this condition before entering the following branch.
            throw shdr::error("collinear events produced a valid plane fit"); // Report collinear events produced a valid plane fit.
    // This six-neighbor geometry has an exactly zero horizontal timestamp slope.
    // Early covariance division can invent a tiny slope and amplify it into false velocity.
    // The fitted vertical slope is one fifth of a microsecond per pixel.
    const std::uint64_t axis_time = (std::uint64_t(1) << 32) + 100u; // Keep exact timestamp subtraction near a low-word boundary.
    records = { // Reproduce the integer geometry of a difficult recorded-event neighborhood.
        evt3_flow_record(394u, 573u, axis_time - 1u), // Include the first event one microsecond before the center.
        evt3_flow_record(393u, 573u, axis_time - 1u), // Include the second older event on the same pixel row.
        evt3_flow_record(394u, 574u, axis_time), // Include a same-time neighbor four pixels to the right.
        evt3_flow_record(393u, 574u, axis_time), // Include a same-time neighbor three pixels to the right.
        evt3_flow_record(392u, 574u, axis_time), // Include a same-time neighbor two pixels to the right.
        evt3_flow_record(394u, 572u, axis_time), // Supply spatial variation that makes the plane fit nondegenerate.
        evt3_flow_record(390u, 575u, axis_time)}; // Place the evaluated center after all six preceding neighbors.
    std::copy(records.begin(), records.end(), events.data()); // Replace the input prefix with the zero-axis regression fixture.
    offsets.data()[0] = static_cast<unsigned>(records.size()); // Restrict shader output to the seven supplied records.
    shader->set<float>("radius", 5.0f); // Include the diagonal neighbor exactly five pixels from the center.
    run.run(shdr::pipeline(shdr::dispatch(*shader, shdr::extent(1u)))); // Fit the same regression neighborhood on the selected backend.
    actual = static_cast<const shdr::buffer<shdr::glsl::vec4>&>(output).data(); // Read completed flow through a constant buffer view.
    const unsigned center = static_cast<unsigned>(records.size() - 1u); // Select the event with all six regression neighbors.
    const shdr::glsl::vec4 expected = evt3_reference_flow(records, center, 5.0f, 128u, 10000u); // Solve the independent double-precision normal equations.
    require_close(expected.x, 0.0f, "zero-axis oracle horizontal velocity", 0.00001f); // Verify zero horizontal velocity in the independent fit.
    require_close(expected.y, 5000000.0f, "zero-axis oracle vertical velocity", 1.0f); // Verify the fixture's known normal velocity in pixels per second.
    require_close(actual[center].x, expected.x, "zero-axis flow X", 0.00001f); // Detect any false velocity caused by rounding a mathematically zero slope.
    require_close(actual[center].y, expected.y, "zero-axis flow Y", 1.0f); // Bound the vertical error relative to the large known velocity.
    require_close(actual[center].z, 6.0f, "zero-axis neighbor count", 0.0f); // Require all six intended neighbors to participate.
    require_close(actual[center].w, 1.0f, "zero-axis fit validity", 0.0f); // Require the noncollinear regression geometry to produce a valid fit.
} // End this block.

// Fused arithmetic can produce a positive rounded determinant for large collinear integer covariances.
// Exact pixel-direction tests must reject these neighborhoods before solving any timestamp slope.
inline void evt3_collinear_history_check(runner& run) { // Check long singular neighborhoods on the selected CPU or GPU backend.
    for (unsigned count : {64u, 65u, 128u, 129u}) { // Exercise both workgroup boundaries and the maximum neighbor history.
        const std::uint64_t origin = (std::uint64_t(1) << 32) + 1000u; // Keep all preceding timestamps within a short exact window.
        std::vector<shdr::glsl::uvec4> records; // Build a timestamp-ordered line of repeated pixel locations.
        for (int age = int(count) - 1; age > 0; --age) { // Emit the oldest event first and approach the evaluated center.
            const int position = age == 1 ? 1 : (age % 2 ? -2 : 2); // Break covariance power-of-two factors while staying on one line.
            const unsigned x = unsigned(100 + 2 * position), y = unsigned(100 + position); // Keep all pixels on the exact integer line X minus 100 equals twice Y minus 100.
            records.push_back(evt3_flow_record(x, y, origin - unsigned(age))); // Give every candidate a valid earlier timestamp.
        } // Finish the long collinear neighbor sequence.
        records.push_back(evt3_flow_record(100u, 100u, origin)); // Include the evaluated center on the same spatial line.
        shdr::buffer<shdr::glsl::uvec4> input(records.size()); // Allocate the complete singular neighborhood.
        std::copy(records.begin(), records.end(), input.data()); // Load exactly the generated pixel and timestamp records.
        shdr::buffer<unsigned> offsets{count}; // Provide the event count directly for this isolated flow-kernel fixture.
        shdr::buffer<shdr::glsl::vec4> output(count); // Receive one invalid zero flow record for every input event.
        auto shader = shdr::make_kernel("evt3_flow_kernel"); // Create the same native flow module used by decoded-event pipelines.
        shader->bind("events", input).bind("offsets", offsets).bind("flow", output); // Connect the direct regression input and its result resource.
        shader->set<unsigned>("blocks", 0u).set<unsigned>("history", 128u); // Read the single total offset and allow the maximum supported history.
        shader->set<float>("radius", 5.0f).set<unsigned>("time_window", 10000u); // Set a radius that includes every candidate while retaining the standard fit thresholds.
        const shdr::pipeline pipeline(shdr::dispatch(*shader, shdr::extent((count + 63u) / 64u))); // Cover all records and ignore padded invocations normally.
        for (unsigned duplicates = 0u; duplicates < 2u; ++duplicates) { // Check both a true spatial line and repeated center positions.
            if (duplicates != 0u) { // Replace the line with a neighborhood of identical pixel coordinates.
                for (unsigned index = 0u; index < count; ++index) { // Keep all timestamps ordered while removing spatial variation.
                    records[index] = evt3_flow_record(100u, 100u, origin - (count - 1u - index)); // Make every record a duplicate of the center's pixel.
                } // Finish the rank-zero neighborhood.
                std::copy(records.begin(), records.end(), input.data()); // Upload the changed positions for the second regression run.
            } // Retain the original line geometry during the first run.
            run.run(pipeline); // Execute the complete singular neighborhood on the selected backend.
            const shdr::glsl::vec4* actual = static_cast<const shdr::buffer<shdr::glsl::vec4>&>(output).data(); // Read synchronized results through the constant buffer interface.
            for (unsigned index = 0u; index < count; ++index) { // Check every prefix and the longest available neighborhood.
                const shdr::glsl::vec4 expected = evt3_reference_flow(records, index, 5.0f, 128u, 10000u); // Confirm singularity with the independent double-precision solver.
                if (expected != shdr::glsl::vec4(0.0f)) { // Require the regression fixture to remain mathematically degenerate.
                    throw shdr::error("collinear-history oracle unexpectedly accepted a plane"); // Report an invalid fixture or independent solver regression.
                } // The independent oracle rejects this spatial geometry.
                if (actual[index] != expected) { // Reject a fabricated valid flow or any nonzero invalid output component.
                    throw shdr::error("long collinear history produced a valid plane fit"); // Catch determinant residues caused by fused floating-point products.
                } // Both backends must publish the same exact invalid result.
            } // Finish all prefix checks for this singular geometry.
        } // Finish line and repeated-center regressions.
    } // Finish every bounded history size.
} // Finish exact spatial rank validation.
} // namespace detail

/** @brief Verify decoded planes and GPU/CPU flow against a scalar double oracle.
 * Both a small timestamp origin and a uint32-boundary crossing use one linked
 * decode-and-flow pipeline. Invalid fits publish zero and valid fits publish a
 * count with normal velocity. The oracle checks every compact output event.
 */
inline void check_evt3_flow(runner& run) { // Implement check evt3 flow for the operations below.
    for (unsigned crossing = 0; crossing < 2; ++crossing) { // Iterate over crossing for this calculation.
        const std::vector<std::uint16_t> words = evt3_planar_words(32, 24, crossing != 0); // Set words from this expression.
        evt3_decoder decoder(static_cast<unsigned>(words.size()), 127); // Construct decoder with these initial values.
        decoder.load(words); // Apply load to these arguments.
        shdr::buffer<shdr::glsl::vec4> output(decoder.events.size()); // Construct output with these initial values.
        auto shader = evt3_flow_kernel(decoder, output); // Set shader from this expression.
        shdr::pipeline pipeline = decoder.reset_pipeline(); // Set pipeline from this expression.
        pipeline += shdr::dispatch( // Add a shader dispatch with the required workgroup counts.
            *shader, shdr::extent((static_cast<unsigned>(output.size()) + 63u) / 64u)); // Apply extent to these arguments.
        run.run(pipeline); // Execute the configured pipeline on the requested backend.
        const unsigned count = decoder.event_count(); // Set count from this expression.
        if (count != 32u * 24u) // Check this condition before entering the following branch.
            throw shdr::error("temporal-plane event count mismatch"); // Report temporal-plane event count mismatch.
        const shdr::glsl::uvec4* decoded = // Set decoded from this expression.
            static_cast<const shdr::buffer<shdr::glsl::uvec4>&>(decoder.events).data(); // Apply data to these arguments.
        const std::vector<shdr::glsl::uvec4> events(decoded, decoded + count); // Construct events with these initial values.
        const shdr::glsl::vec4* actual = // Set actual from this expression.
            static_cast<const shdr::buffer<shdr::glsl::vec4>&>(output).data(); // Apply data to these arguments.
        unsigned valid = 0; // Set valid from this expression.
        for (unsigned index = 0; index < count; ++index) { // Iterate over index for this calculation.
            const shdr::glsl::vec4 expected = // Set expected from this expression.
                detail::evt3_reference_flow(events, index, 3.0f, 128u, 10000u); // Apply evt3 reference flow to these arguments.
            require_close(actual[index].x, expected.x, "event-plane flow X", 1.5f); // Check the computed value against the independent reference.
            require_close(actual[index].y, expected.y, "event-plane flow Y", 1.5f); // Check the computed value against the independent reference.
            require_close(actual[index].z, expected.z, "event-plane neighbor count", 0.0f); // Check the computed value against the independent reference.
            require_close(actual[index].w, expected.w, "event-plane fit validity", 0.0f); // Check the computed value against the independent reference.
            if (expected.w != 0.0f) { // Check this condition before entering the following branch.
                ++valid; // Continue the expression using valid.
                require_close( // Check the computed value against the independent reference.
                    actual[index].x, 8000.0f, "translating-plane normal velocity X", 1.5f); // Continue the expression using actual, index, x.
                require_close( // Check the computed value against the independent reference.
                    actual[index].y, 4000.0f, "translating-plane normal velocity Y", 1.5f); // Continue the expression using actual, index, y.
            } // End this block.
        } // End this block.
        if (valid < count / 2u) // Check this condition before entering the following branch.
            throw shdr::error("temporal plane produced too few valid flow fits"); // Report temporal plane produced too few valid flow fits.
        std::cout << "EVT3 normal flow: " << valid << '/' << count // Write this part of the example result or diagnostic.
                  << " valid events, velocity (8000,4000) pixels/s" // Write this part of the example result or diagnostic.
                  << (crossing ? " across a 32-bit timestamp boundary" : "") << '\n'; // Write this part of the example result or diagnostic.
    } // End this block.
    detail::evt3_mixed_flow_check(run); // Apply evt3 mixed flow check to these arguments.
    detail::evt3_collinear_history_check(run); // Reject long collinear and repeated-pixel neighborhoods independently of floating-point contraction.
} // End this block.
} // namespace shdr_examples
#endif // Finish this conditional compilation branch.
