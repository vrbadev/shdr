/** @file signal_filters_checks.hpp
 * @brief Independent signal-filter checks for CPU and GPU execution.
 * Double convolution and direct-form I recurrence provide output oracles for
 * shaders using float convolution and transposed direct-form II. Frequency
 * response checks also measure mixed tones after the initial transient.
 */
#ifndef SHDR_EXAMPLES_SIGNAL_FILTERS_CHECKS_HPP // Include this example helper only once.
#define SHDR_EXAMPLES_SIGNAL_FILTERS_CHECKS_HPP // Remember that this example helper is available.
#include "runner.hpp" // Select CPU execution or the requested graphics profile.
#include "signal_filters.hpp" // Use the reusable filter designers and pipeline helpers.
#include <complex> // Evaluate frequency responses with complex arithmetic.
#include <random> // Create repeatable random fixtures and initial parameters.

namespace shdr_examples { // Keep these example helpers in their own namespace.
namespace signal_checks { // Keep these example helpers in their own namespace.

inline void require(bool value, const char* message) { // Keep the test checks active in every build configuration.
    if (!value) // Handle this validation or boundary case explicitly.
        throw shdr::error(message); // Report the invalid design or failed check clearly.
} // Finish the require operation.
template <class Function> void rejects(Function function, const char* message) { // Accept the test function while keeping error checks reusable.
    bool caught = false; // Remember whether the expected validation error occurs.
    try { // Run the operation while preserving its failure diagnostic.
        function(); // Execute the operation whose rejection is being checked.
    } catch (const shdr::error&) { // Recognize the expected filter validation error.
        caught = true; // Record that the invalid input was rejected.
    } // Finish handling this reported failure.
    require(caught, message); // Require this design property or test invariant.
} // Finish the rejects operation.
inline std::complex<double> unit_phase(double frequency, double rate) { // Compute one complex sample of the frequency phase.
    return std::polar(1.0, -2.0 * signal_detail::pi() * frequency / rate); // Return the unit complex phase for this frequency.
} // Finish the unit_phase operation.
inline double fir_gain(const std::vector<float>& coefficients, double frequency, double rate) { // Evaluate FIR gain using complex frequency response.
    std::complex<double> gain(0, 0), phase(1, 0), step = unit_phase(frequency, rate); // Keep the complex phase and accumulated transfer response.
    for (float coefficient : coefficients) { // Visit every value needed by this calculation.
        gain += double(coefficient) * phase; // Add this tap at its complex frequency phase.
        phase *= step; // Advance to the next tap delay.
    } // Finish this channel, sample or feature pass.
    return std::abs(gain); // Return the transfer-function magnitude.
} // Finish the fir_gain operation.
inline double biquad_gain(const std::array<float, 5>& coefficients, double frequency, double rate) { // Evaluate the normalized biquad transfer function.
    const std::complex<double> phase = unit_phase(frequency, rate); // Keep the complex phase and accumulated transfer response.
    return std::abs( // Return the transfer-function magnitude.
        (double(coefficients[0]) + double(coefficients[1]) * phase + // Evaluate the feed-forward numerator at this frequency.
         double(coefficients[2]) * phase * phase) / // Include the second delayed feed-forward coefficient.
        (1.0 + double(coefficients[3]) * phase + double(coefficients[4]) * phase * phase)); // Divide by the normalized feedback denominator.
} // Finish the biquad_gain operation.
inline std::vector<float> fixture(unsigned count, unsigned channels) { // Build separate impulse, step and random test channels.
    std::vector<float> input(count * channels, 0.0f); // Allocate zero-filled, independent fixture channels.
    std::mt19937 random(12345u); // Use a fixed seed so every backend receives identical data.
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f); // Choose a bounded range for reproducible noise or weights.
    input[0] = 1.0f; // Put the impulse only in the first channel.
    for (unsigned i = 0; i < count; ++i) { // Visit every value needed by this calculation.
        if (channels > 1u) // Keep the step fixture separate from the impulse channel.
            input[count + i] = 1.0f; // Use a unit step for the second channel.
        for (unsigned channel = 2u; channel < channels; ++channel) // Process each channel independently.
            input[channel * count + i] = distribution(random); // Give the remaining channels reproducible random samples.
    } // Finish this channel, sample or feature pass.
    return input; // Return the complete channel-major input fixture.
} // Finish the fixture operation.
inline std::vector<double> fir_reference(const std::vector<float>& input, // Convolve with double accumulation for the independent oracle.
                                         unsigned count, // Receive the validated sample, channel or coefficient count.
                                         unsigned channels, // Receive the validated sample, channel or coefficient count.
                                         const std::vector<float>& coefficients) { // Use the caller-provided finite FIR coefficients.
    std::vector<double> output(input.size(), 0.0); // Allocate independent double oracle outputs.
    for (unsigned channel = 0; channel < channels; ++channel) // Process each channel independently.
        for (unsigned i = 0; i < count; ++i) // Visit every value needed by this calculation.
            for (unsigned tap = 0; tap < coefficients.size() && tap <= i; ++tap) // Use only causal taps that lie inside this batch.
                output[channel * count + i] += // Accumulate this causal tap in double precision.
                    double(coefficients[tap]) * input[channel * count + i - tap]; // Multiply this causal sample and coefficient in double precision.
    return output; // Return every independently computed filtered sample.
} // Finish the fir_reference operation.
inline std::vector<double> biquad_reference(const std::vector<float>& input, // Use a direct-form I double recurrence as an independent oracle.
                                            unsigned count, // Receive the validated sample, channel or coefficient count.
                                            unsigned channels, // Receive the validated sample, channel or coefficient count.
                                            const std::array<float, 5>& coefficients, // Use five normalized biquad coefficients.
                                            std::vector<shdr::glsl::vec2>* final_state = 0) { // Keep the independent expected recurrence pair for each channel.
    std::vector<double> output(input.size()); // Allocate independent double oracle outputs.
    if (final_state) // Record reference state only when the caller asks for it.
        final_state->resize(channels); // Reserve one final recurrence pair per channel.
    for (unsigned channel = 0; channel < channels; ++channel) { // Process each channel independently.
        double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0; // Start the independent direct-form I recurrence with zero history.
        for (unsigned i = 0; i < count; ++i) { // Visit every value needed by this calculation.
            const double x = input[channel * count + i]; // Read this input sample for the double recurrence.
            const double y = coefficients[0] * x + coefficients[1] * x1 + coefficients[2] * x2 - // Apply the independent direct-form I difference equation.
                             coefficients[3] * y1 - coefficients[4] * y2; // Subtract the two normalized feedback contributions.
            output[channel * count + i] = y; // Keep this independently computed filtered sample.
            x2 = x1; // Shift the input history by one sample.
            x1 = x; // Remember the current input for the next sample.
            y2 = y1; // Shift the output history by one sample.
            y1 = y; // Remember the current output for the next sample.
        } // Finish this channel, sample or feature pass.
        if (final_state) // Record reference state only when the caller asks for it.
            (*final_state)[channel] = // Convert direct-form I history into transposed recurrence state.
                shdr::glsl::vec2(float(coefficients[1] * x1 + coefficients[2] * x2 - // Convert the last input and output histories into the first state value.
                                       coefficients[3] * y1 - coefficients[4] * y2), // Include the feedback history in the first state value.
                                 float(coefficients[2] * x1 - coefficients[4] * y1)); // Convert the double reference value for the float comparison.
    } // Finish this channel, sample or feature pass.
    return output; // Return every independently computed filtered sample.
} // Finish the biquad_reference operation.
inline void compare(const shdr::buffer<float>& actual, // Check every filtered output against the double oracle.
                    const std::vector<double>& expected, // Receive the independently accumulated double outputs.
                    const char* label, // Give failures a descriptive check name.
                    float tolerance = 0.00003f) { // Allow only the stated float error in the result check.
    const float* output = actual.data(); // Read the resource after any required GPU result synchronization.
    for (unsigned i = 0; i < expected.size(); ++i) // Visit every value needed by this calculation.
        require_close(output[i], static_cast<float>(expected[i]), label, tolerance); // Compare this result with its independent expected value.
} // Finish the compare operation.
inline double tone_amplitude(const shdr::buffer<float>& samples, // Measure a tone by sine and cosine projection.
                             unsigned begin, // Begin assembling at the first sample in each channel.
                             unsigned count, // Receive the validated sample, channel or coefficient count.
                             double frequency, // Receive the tone, cutoff or center frequency.
                             double rate) { // Receive the positive sampling rate in frequency units.
    const float* values = samples.data(); // Read the resource after any required GPU result synchronization.
    double real = 0.0, imaginary = 0.0; // Start both tone projection accumulators at zero.
    for (unsigned i = 0; i < count; ++i) { // Visit every value needed by this calculation.
        const double angle = 2.0 * signal_detail::pi() * frequency * (begin + i) / rate; // Compute the absolute phase of this tone sample.
        real += values[begin + i] * std::cos(angle); // Accumulate the cosine projection of the filtered signal.
        imaginary += values[begin + i] * std::sin(angle); // Accumulate the sine projection of the filtered signal.
    } // Finish this channel, sample or feature pass.
    return 2.0 * std::sqrt(real * real + imaginary * imaginary) / count; // Build separate impulse, step and random test channels.
} // Finish the tone_amplitude operation.

inline void coefficient_checks() { // Check designer gains and invalid parameter handling.
    const filter_mode modes[] = {filter_mode::low_pass, // List every supported filter shape exercised by the checks.
                                 filter_mode::high_pass, // Include the high-pass response in this group of checks.
                                 filter_mode::band_pass, // Include the band-pass response in this group of checks.
                                 filter_mode::band_stop}; // Include the band-stop response in this group of checks.
    for (filter_mode mode : modes) { // Check every supported filter shape.
        const bool band = mode == filter_mode::band_pass || mode == filter_mode::band_stop; // Use two cutoff edges only for band-selective filters.
        const std::vector<float> fir = // Design the FIR coefficients used by this independent check.
            fir_coefficients(mode, 48000.0, band ? 3000.0 : 4000.0, band ? 9000.0 : 0.0, 65u); // Design the causal FIR coefficients for this configuration.
        for (unsigned i = 0; i < fir.size(); ++i) // Visit every value needed by this calculation.
            require_close(fir[i], fir[fir.size() - 1u - i], "FIR symmetry", 0.0000001f); // Compare this result with its independent expected value.
        const double dc = fir_gain(fir, 0.0, 48000.0); // Measure the FIR response at zero frequency.
        require_close(float(dc), // Compare this result with its independent expected value.
                      mode == filter_mode::low_pass || mode == filter_mode::band_stop ? 1.0f : 0.0f, // Expect unit DC gain for low-pass and band-stop designs.
                      "FIR DC gain", // Identify this check clearly in any failure message.
                      0.000002f); // Apply the declared float error allowance.
        if (mode == filter_mode::low_pass) // Choose the coefficient rule for this filter shape.
            require(fir_gain(fir, 1000, 48000) > .98 && fir_gain(fir, 12000, 48000) < .015, // Require this design property or test invariant.
                    "low-pass FIR selectivity"); // Identify this check clearly in any failure message.
        if (mode == filter_mode::high_pass) // Choose the coefficient rule for this filter shape.
            require(fir_gain(fir, 1000, 48000) < .015 && fir_gain(fir, 12000, 48000) > .98, // Require this design property or test invariant.
                    "high-pass FIR selectivity"); // Identify this check clearly in any failure message.
        if (mode == filter_mode::band_pass) // Choose the coefficient rule for this filter shape.
            require(fir_gain(fir, 6000, 48000) > .99 && fir_gain(fir, 1000, 48000) < .015 && // Require this design property or test invariant.
                        fir_gain(fir, 16000, 48000) < .015, // Require attenuation above the band-pass upper edge.
                    "band-pass FIR selectivity"); // Identify this check clearly in any failure message.
        if (mode == filter_mode::band_stop) // Choose the coefficient rule for this filter shape.
            require(fir_gain(fir, 6000, 48000) < .000002 && fir_gain(fir, 1000, 48000) > .98 && // Require this design property or test invariant.
                        fir_gain(fir, 16000, 48000) > .98, // Require gain above the rejected band-stop interval.
                    "band-stop FIR selectivity"); // Identify this check clearly in any failure message.
        const std::array<float, 5> biquad = biquad_coefficients(mode, 48000.0, 6000.0); // Design normalized coefficients for this biquad check.
        if (mode == filter_mode::low_pass) // Choose the coefficient rule for this filter shape.
            require_close( // Compare this result with its independent expected value.
                float(biquad_gain(biquad, 0, 48000)), 1.0f, "biquad low-pass DC", 0.000002f); // Convert the double reference value for the float comparison.
        if (mode == filter_mode::high_pass) // Choose the coefficient rule for this filter shape.
            require_close(float(biquad_gain(biquad, 24000, 48000)), // Compare this result with its independent expected value.
                          1.0f, // Expect unit gain at this normalization frequency.
                          "biquad high-pass Nyquist", // Identify this check clearly in any failure message.
                          0.000002f); // Apply the declared float error allowance.
        if (mode == filter_mode::band_pass) // Choose the coefficient rule for this filter shape.
            require_close(float(biquad_gain(biquad, 6000, 48000)), // Compare this result with its independent expected value.
                          1.0f, // Expect unit gain at this normalization frequency.
                          "biquad band-pass center", // Identify this check clearly in any failure message.
                          0.000002f); // Apply the declared float error allowance.
        if (mode == filter_mode::band_stop) // Choose the coefficient rule for this filter shape.
            require(biquad_gain(biquad, 6000, 48000) < .000002, "biquad notch center"); // Require this design property or test invariant.
    } // Finish this channel, sample or feature pass.
    const std::vector<float> average = // Design the FIR coefficients used by this independent check.
        fir_coefficients(filter_mode::moving_average, 48000, 0, 0, 5); // Design the five equal weights used by the average check.
    for (float value : average) // Inspect each stored float coefficient.
        require_close(value, 0.2f, "moving-average coefficients", 0.0f); // Compare this result with its independent expected value.
    rejects( // Confirm that invalid configuration is rejected before dispatch.
        [] { // Define one invalid configuration to test.
            fir_coefficients(filter_mode::low_pass, 48000, 4000, 0, 4); // Try an even tap count that the designer must reject.
        }, // Finish the invalid-configuration test operation.
        "even FIR taps accepted"); // Identify this check clearly in any failure message.
    rejects( // Confirm that invalid configuration is rejected before dispatch.
        [] { // Define one invalid configuration to test.
            fir_coefficients(filter_mode::low_pass, 0, 4000); // Try an invalid zero sample rate.
        }, // Finish the invalid-configuration test operation.
        "zero sample rate accepted"); // Identify this check clearly in any failure message.
    rejects( // Confirm that invalid configuration is rejected before dispatch.
        [] { // Define one invalid configuration to test.
            fir_coefficients(filter_mode::low_pass, 48000, 24000); // Try a cutoff at Nyquist, outside the supported open interval.
        }, // Finish the invalid-configuration test operation.
        "Nyquist cutoff accepted"); // Identify this check clearly in any failure message.
    rejects( // Confirm that invalid configuration is rejected before dispatch.
        [] { // Define one invalid configuration to test.
            fir_coefficients(filter_mode::band_pass, 48000, 9000, 3000); // Try band edges in the wrong order.
        }, // Finish the invalid-configuration test operation.
        "descending band edges accepted"); // Identify this check clearly in any failure message.
    rejects( // Confirm that invalid configuration is rejected before dispatch.
        [] { // Define one invalid configuration to test.
            biquad_coefficients(filter_mode::low_pass, 48000, 4000, 0); // Exercise the positive-quality validation with a zero value.
        }, // Finish the invalid-configuration test operation.
        "zero Q accepted"); // Identify this check clearly in any failure message.
    rejects( // Confirm that invalid configuration is rejected before dispatch.
        [] { // Define one invalid configuration to test.
            biquad_coefficients(filter_mode::low_pass, 48000, 1e-20); // Try a frequency whose rounded feedback loses strict stability.
        }, // Finish the invalid-configuration test operation.
        "unstable rounded biquad accepted"); // Identify this check clearly in any failure message.
} // Finish the coefficient_checks operation.

inline void fir_outputs(runner& run) { // Check causal FIR outputs and measured mixed-tone gains.
    const unsigned count = 257, channels = 3; // Choose a non-workgroup-aligned sample count and separate channels.
    const std::vector<float> source = fixture(count, channels); // Keep the shared impulse, step and seeded random inputs.
    shdr::buffer<float> input(source.size()), output(source.size()); // Allocate independent float resources for this processing check.
    std::copy(source.begin(), source.end(), input.data()); // Copy these host values into the shared resource.
    const filter_mode modes[] = {filter_mode::low_pass, // List every supported filter shape exercised by the checks.
                                 filter_mode::high_pass, // Include the high-pass response in this group of checks.
                                 filter_mode::band_pass, // Include the band-pass response in this group of checks.
                                 filter_mode::band_stop, // Include the band-stop response in this group of checks.
                                 filter_mode::moving_average}; // Include equal-weight moving-average filtering.
    for (filter_mode mode : modes) { // Check every supported filter shape.
        const bool band = mode == filter_mode::band_pass || mode == filter_mode::band_stop; // Use two cutoff edges only for band-selective filters.
        const std::vector<float> coefficients = // Design the FIR coefficients used by this independent check.
            fir_coefficients(mode, 48000, band ? 3000 : 4000, band ? 9000 : 0, 65); // Design the causal FIR coefficients for this configuration.
        fir_pipeline filter(input, output, count, channels, coefficients); // Connect the FIR check to its designed coefficients.
        run.run(filter.processing_pipeline()); // Execute the linked stages on the selected backend.
        compare(output, // Compare every filtered sample with the independent oracle.
                fir_reference(source, count, channels, coefficients), // Supply the independently accumulated causal convolution.
                "FIR impulse/step/random oracle"); // Identify this check clearly in any failure message.
        run.run(filter.processing_pipeline()); // Execute the linked stages on the selected backend.
        compare( // Compare every filtered sample with the independent oracle.
            output, fir_reference(source, count, channels, coefficients), "FIR batch prefix reset"); // Verify that another FIR run starts with a fresh zero prefix.
        rejects( // Confirm that invalid configuration is rejected before dispatch.
            [&] { // Use the existing resources to test invalid rebinding.
                filter.bind(input, input, count); // Try unsafe input and output storage aliasing.
            }, // Finish the invalid-configuration test operation.
            "FIR aliased storage accepted"); // Identify this check clearly in any failure message.
        rejects( // Confirm that invalid configuration is rejected before dispatch.
            [&] { // Use the existing resources to test invalid rebinding.
                filter.bind(input, output, count + 1u); // Try a count that exceeds the allocated channel capacity.
            }, // Finish the invalid-configuration test operation.
            "undersized FIR storage accepted"); // Identify this check clearly in any failure message.
    } // Finish this channel, sample or feature pass.
    const unsigned tone_count = 5312; // Reserve a steady-state tone window after the filter transient.
    shdr::buffer<float> tones(tone_count), filtered(tone_count); // Allocate independent float resources for this processing check.
    const double frequencies[3] = {1000, 6000, 12000}, amplitudes[3] = {.6, .4, .3}; // Mix separated passband and stopband tones with known amplitudes.
    for (unsigned i = 0; i < tone_count; ++i) { // Visit each temporal sample in the tone mixture.
        double value = 0.0; // Start this sample or projected-feature accumulator at zero.
        for (unsigned tone = 0; tone < 3; ++tone) // Visit each frequency in the mixed-tone fixture.
            value += amplitudes[tone] * // Add this sinusoidal component to the mixed-tone input.
                     std::cos(2.0 * signal_detail::pi() * frequencies[tone] * i / 48000.0); // Compute this sinusoid at the shared sampling rate.
        tones.data()[i] = static_cast<float>(value); // Store the same tone mixture for every backend.
    } // Finish this channel, sample or feature pass.
    for (filter_mode mode : modes) { // Check every supported filter shape.
        const bool band = mode == filter_mode::band_pass || mode == filter_mode::band_stop; // Use two cutoff edges only for band-selective filters.
        const std::vector<float> coefficients = // Design the FIR coefficients used by this independent check.
            fir_coefficients(mode, 48000, band ? 3000 : 4000, band ? 9000 : 0, 65); // Design the causal FIR coefficients for this configuration.
        fir_pipeline filter(tones, filtered, tone_count, 1, coefficients); // Connect the FIR check to its designed coefficients.
        run.run(filter.processing_pipeline()); // Execute the linked stages on the selected backend.
        for (unsigned tone = 0; tone < 3; ++tone) // Visit each frequency in the mixed-tone fixture.
            require_close( // Compare this result with its independent expected value.
                float(tone_amplitude(filtered, 512, 4800, frequencies[tone], 48000)), // Convert the double reference value for the float comparison.
                float(amplitudes[tone] * fir_gain(coefficients, frequencies[tone], 48000)), // Convert the double reference value for the float comparison.
                "FIR mixed-tone gain", // Identify this check clearly in any failure message.
                0.00005f); // Apply the declared float error allowance.
    } // Finish this channel, sample or feature pass.
} // Finish the fir_outputs operation.

inline void biquad_outputs(runner& run) { // Check biquad outputs, retained state and chunk boundaries.
    const unsigned count = 257, channels = 3; // Choose a non-workgroup-aligned sample count and separate channels.
    const std::vector<float> source = fixture(count, channels); // Keep the shared impulse, step and seeded random inputs.
    shdr::buffer<float> input(source.size()), output(source.size()); // Allocate independent float resources for this processing check.
    std::copy(source.begin(), source.end(), input.data()); // Copy these host values into the shared resource.
    const filter_mode modes[] = {filter_mode::low_pass, // List every supported filter shape exercised by the checks.
                                 filter_mode::high_pass, // Include the high-pass response in this group of checks.
                                 filter_mode::band_pass, // Include the band-pass response in this group of checks.
                                 filter_mode::band_stop}; // Include the band-stop response in this group of checks.
    for (filter_mode mode : modes) { // Check every supported filter shape.
        const std::array<float, 5> coefficients = biquad_coefficients(mode, 48000, 6000); // Design normalized coefficients for this biquad check.
        std::vector<shdr::glsl::vec2> expected_state; // Keep one pair of recurrence values for each channel.
        const std::vector<double> expected = // Compute the independent full-block double output oracle.
            biquad_reference(source, count, channels, coefficients, &expected_state); // Compute the independent recurrence output and final state.
        biquad_pipeline filter(input, output, count, channels, coefficients); // Connect the biquad check with fresh zero history.
        run.run(filter.processing_pipeline()); // Execute the linked stages on the selected backend.
        compare(output, expected, "biquad direct-form I oracle"); // Compare every filtered sample with the independent oracle.
        const shdr::glsl::vec2* state = // Read the resource after any required GPU result synchronization.
            static_cast<const shdr::buffer<shdr::glsl::vec2>&>(filter.state).data(); // Read the retained state after any required GPU synchronization.
        for (unsigned channel = 0; channel < channels; ++channel) // Process each channel independently.
            for (unsigned component = 0; component < 2; ++component) // Visit every value needed by this calculation.
                require_close(state[channel][component], // Compare this result with its independent expected value.
                              expected_state[channel][component], // Supply the independently converted recurrence value.
                              "biquad state oracle", // Identify this check clearly in any failure message.
                              0.00003f); // Apply the declared float error allowance.
        filter.reset_state(); // Start a fresh filter history for this independent comparison.
        run.run(filter.processing_pipeline()); // Execute the linked stages on the selected backend.
        compare(output, expected, "biquad explicit state reset"); // Compare every filtered sample with the independent oracle.

        shdr::buffer<float> chunk_input(channels), chunk_output(channels); // Allocate independent float resources for this processing check.
        biquad_pipeline streaming(chunk_input, chunk_output, 1, channels, coefficients); // Keep one stateful filter instance across variable chunks.
        const unsigned chunks[] = {1, 17, 64, 3, 129, 43}; // Use uneven chunks that exercise state across many boundaries.
        unsigned begin = 0; // Begin assembling at the first sample in each channel.
        std::vector<double> assembled(source.size()); // Collect streamed chunks in their full-signal positions.
        unsigned chunk_index = 0; // Begin the CPU and GPU transition pattern at its first chunk.
        for (unsigned chunk : chunks) { // Process the next variable-size temporal block.
            chunk_input = shdr::buffer<float>(chunk * channels); // Allocate this chunk with its channel-major stride.
            chunk_output = shdr::buffer<float>(chunk * channels); // Allocate this chunk with its channel-major stride.
            for (unsigned channel = 0; channel < channels; ++channel) // Process each channel independently.
                for (unsigned i = 0; i < chunk; ++i) // Process the next variable-size temporal block.
                    chunk_input.data()[channel * chunk + i] = source[channel * count + begin + i]; // Copy this channel segment into the next temporal chunk.
            streaming.bind(chunk_input, chunk_output, chunk); // Connect this temporal chunk while retaining previous channel state.
            if (run.gpu() && chunk_index % 2u) // Use the requested GPU path or explicit CPU transition.
                streaming.processing_pipeline().run(); // Process this chunk on the CPU using the same retained state.
            else // Process this chunk through the selected runner backend.
                run.run(streaming.processing_pipeline()); // Execute the linked stages on the selected backend.
            const float* actual = static_cast<const shdr::buffer<float>&>(chunk_output).data(); // Read the resource after any required GPU result synchronization.
            for (unsigned channel = 0; channel < channels; ++channel) // Process each channel independently.
                for (unsigned i = 0; i < chunk; ++i) // Process the next variable-size temporal block.
                    assembled[channel * count + begin + i] = actual[channel * chunk + i]; // Place the chunk output in the complete reference signal.
            begin += chunk; // Advance to the next temporal chunk boundary.
            ++chunk_index; // Alternate the CPU and GPU transition test on the next chunk.
        } // Finish this channel, sample or feature pass.
        require(begin == count, "stream fixture did not cover the entire signal"); // Require this design property or test invariant.
        for (unsigned i = 0; i < assembled.size(); ++i) // Visit every value needed by this calculation.
            require_close(float(assembled[i]), // Compare this result with its independent expected value.
                          float(expected[i]), // Convert the computed host value to the shader float type.
                          "biquad streaming chunk equivalence", // Identify this check clearly in any failure message.
                          0.00003f); // Apply the declared float error allowance.
    } // Finish this channel, sample or feature pass.
    const unsigned tone_count = 5312; // Reserve a steady-state tone window after the filter transient.
    shdr::buffer<float> tones(tone_count), filtered(tone_count); // Allocate independent float resources for this processing check.
    const double frequencies[3] = {1000, 6000, 12000}, amplitudes[3] = {.6, .4, .3}; // Mix separated passband and stopband tones with known amplitudes.
    for (unsigned i = 0; i < tone_count; ++i) { // Visit each temporal sample in the tone mixture.
        double value = 0.0; // Start this sample or projected-feature accumulator at zero.
        for (unsigned tone = 0; tone < 3; ++tone) // Visit each frequency in the mixed-tone fixture.
            value += amplitudes[tone] * // Add this sinusoidal component to the mixed-tone input.
                     std::cos(2.0 * signal_detail::pi() * frequencies[tone] * i / 48000.0); // Compute this sinusoid at the shared sampling rate.
        tones.data()[i] = static_cast<float>(value); // Store the same tone mixture for every backend.
    } // Finish this channel, sample or feature pass.
    for (filter_mode mode : modes) { // Check every supported filter shape.
        const std::array<float, 5> coefficients = biquad_coefficients(mode, 48000, 6000); // Design normalized coefficients for this biquad check.
        biquad_pipeline filter(tones, filtered, tone_count, 1, coefficients); // Connect the biquad check with fresh zero history.
        run.run(filter.processing_pipeline()); // Execute the linked stages on the selected backend.
        for (unsigned tone = 0; tone < 3; ++tone) // Visit each frequency in the mixed-tone fixture.
            require_close( // Compare this result with its independent expected value.
                float(tone_amplitude(filtered, 512, 4800, frequencies[tone], 48000)), // Convert the double reference value for the float comparison.
                float(amplitudes[tone] * biquad_gain(coefficients, frequencies[tone], 48000)), // Convert the double reference value for the float comparison.
                "biquad mixed-tone gain", // Identify this check clearly in any failure message.
                0.00005f); // Apply the declared float error allowance.
    } // Finish this channel, sample or feature pass.
} // Finish the biquad_outputs operation.
} // namespace signal_checks

/** @brief Check designed gains, causal outputs, independent channels and streamed state. */
inline void check_signal_filters(runner& run) { // Run all filter design and processing checks.
    signal_checks::coefficient_checks(); // Run the corresponding independent correctness check.
    signal_checks::fir_outputs(run); // Run the corresponding independent correctness check.
    signal_checks::biquad_outputs(run); // Run the corresponding independent correctness check.
    std::cout // Print the verified result or the reason a check failed.
        << "FIR low/high/band-pass/band-stop/average and stateful biquads: impulse, step, random and mixed tones verified\n"; // Complete the readable result message.
} // Finish the check_signal_filters operation.
} // namespace shdr_examples
#endif // Finish the conditional compilation block.
