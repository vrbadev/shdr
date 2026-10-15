/** @file fft_checks.hpp
 * @brief FFT, window, and frequency-filter checks against independent signal references.
 * Small direct DFTs provide an independent numerical reference for the FFT.
 * Known tone mixtures check pass bands, stop bands, and reconstructed amplitudes.
 */
#ifndef SHDR_EXAMPLES_FFT_CHECKS_HPP // Include the reference helpers only once.
#define SHDR_EXAMPLES_FFT_CHECKS_HPP // Mark this validation header as included.
#include "runner.hpp" // Select the CPU or requested GPU backend.
#include "fft_support.hpp" // Build reusable transform and filtering pipelines.
#include <complex> // Accumulate independent reference transforms in double precision.
#include <random> // Generate the same input on every backend.

namespace shdr_examples { // Keep validation utilities with the other examples.
namespace fft_checks_detail { // Separate reference calculations from shader implementations.
typedef shdr::glsl::vec2 complex_value; // Store real and imaginary values in the shader layout.
inline std::vector<complex_value> // Return complex spectra for all input channels.
direct_dft( // Accept the original channel-major sample vector.
    const std::vector<complex_value>& values, // Accept the original channel-major sample vector.
    unsigned length) { // Calculate each frequency independently.
    std::vector<complex_value> result(values.size()); // Keep the same channel-major shape.
    const double tau = 6.283185307179586476925286766559; // Use a double-precision phase reference.
    for (unsigned channel = 0; // Start one independent transform for each channel.
         channel < values.size() / length; // Start one independent transform for each channel.
         ++channel) { // Transform each channel as an independent signal.
        for (unsigned bin = 0; bin < length; ++bin) { // Evaluate one unshifted frequency bin.
            std::complex<double> total(0.0, // Begin the real part of the reference sum at zero.
                                       0.0); // Accumulate with more precision than the shaders.
            for (unsigned position = 0; // Visit every time-domain position for this bin.
                 position < length; // Visit every time-domain position for this bin.
                 ++position) { // Read every value in this reference block.
                const complex_value value = // Read one original complex sample.
                    values[channel * length + position]; // Select the original complex input.
                const double phase = // Calculate this sample’s phase in radians.
                    -tau * bin * position / length; // Follow the unscaled forward DFT convention.
                total += std::complex<double>( // Multiply the complex sample by its unit-magnitude phase.
                             value.x, // Multiply the complex sample by its unit-magnitude phase.
                             value.y) * // Multiply the complex sample by its unit-magnitude phase.
                         std::polar(1.0, phase); // Add this input's complex frequency contribution.
            } // This reference sums all complex Fourier terms directly.
            result[channel * length + bin] = // Store this bin in the channel-major reference output.
                complex_value( // Store this bin in the channel-major reference output.
                    float(total.real()), // Convert the accumulated real component to float.
                    float(total.imag())); // Compare the final reference with float shader outputs.
        } // Continue with the next frequency in this channel.
    } // Independent channels have separate reference spectra.
    return result; // Supply a complete spectrum for verification.
} // The reference has quadratic work and is used only on small test blocks.
inline void compare(const shdr::buffer<complex_value>& // Read the completed transform through a const resource.
                        actual, // Read the completed transform through a const resource.
                    const std::vector<complex_value>& // Accept the independent reference values to compare.
                        expected, // Accept the independent reference values to compare.
                    const char* label) { // Check every complex value.
    const complex_value* values = actual.data(); // Download GPU results only after processing.
    for (unsigned i = 0; i < expected.size(); ++i) { // Compare both components at each position.
        require_close( // Check the real component with an absolute and relative tolerance.
            values[i].x, // Read the shader’s real output.
            expected[i].x, // Use the direct reference’s real result.
            label, // Keep the caller’s diagnostic label.
            0.0003f + // Allow a small absolute float error.
                0.00001f * // Add tolerance proportional to the expected amplitude.
                    std::fabs( // Use the magnitude so negative values receive the same tolerance.
                        expected[i].x)); // Allow float arithmetic error in the real component.
        require_close( // Check the imaginary component separately.
            values[i].y, // Read the shader’s imaginary output.
            expected[i].y, // Use the direct reference’s imaginary result.
            label, // Preserve the same diagnostic label.
            0.0003f + 0.00001f * // Combine absolute and relative imaginary error limits.
                          std::fabs(expected[i].y)); // Use the same tolerance for imaginary values.
    } // A mismatch identifies a failed transform or signal response.
} // Const access preserves the input values for subsequent dispatches.
inline double // Evaluate scalar filter gains with double precision.
low_gain(double frequency, // Accept the positive frequency in hertz.
         double cutoff, // Accept the transition center in hertz.
         double width) { // Describe the expected block-filter response in double precision.
    if (width == 0.0) // Select the hard-cutoff response for a zero transition width.
        return frequency <= cutoff ? 1.0 : 0.0; // Match the documented hard cutoff convention.
    const double position = // Normalize distance from the transition’s lower edge.
        (frequency - cutoff + width * 0.5) / // Normalize distance from the transition’s lower edge.
        width; // Express frequency within the cosine transition.
    if (position <= 0.0) // Detect the full-pass region.
        return 1.0; // Keep values below the transition.
    if (position >= 1.0) // Detect the full-stop region.
        return 0.0; // Reject values above the transition.
    return 0.5 + // Use a raised cosine within the transition.
           0.5 * // Use a raised cosine within the transition.
               std::cos( // Use a raised cosine within the transition.
                   3.1415926535897932384626433832795 * // Use a raised cosine within the transition.
                   position); // Evaluate the smooth gain independently of the shader.
} // Negative-frequency bins use this same real gain.
} // namespace fft_checks_detail

/** @brief Check small batched transforms, reconstruction, input aliasing, and invalid sizes. */
inline void // Expose the transform validation routine.
check_fft1d(runner& run) { // Use independent reference transforms on the selected backend.
    const unsigned lengths[] = {// List the small block sizes that receive direct references.
                                1, // Include this documented transform length in the independent reference checks.
                                2, // Include this documented transform length in the independent reference checks.
                                4, // Include this documented transform length in the independent reference checks.
                                8, // Include this documented transform length in the independent reference checks.
                                16, // Include this documented transform length in the independent reference checks.
                                64, // Include this documented transform length in the independent reference checks.
                                128}; // Cover the copy case and several butterfly depths.
    std::mt19937 random(12345u); // Reproduce inputs on different machines.
    std::uniform_real_distribution<float> values( // Begin a uniform distribution at minus one.
        -1.0f, // Begin a uniform distribution at minus one.
        1.0f); // Keep reference amplitudes well bounded.
    for (unsigned length : lengths) { // Validate each supported block size.
        const unsigned channels = 3, // Test three separate channel transforms per length.
            count = length * channels; // Include independently transformed channels.
        std::vector<shdr::glsl::vec2> original( // Allocate the original complex fixture vector.
            count); // Retain unmodified values for reference checks.
        shdr::buffer<shdr::glsl::vec2> input(count), // Allocate input storage and forward-transform output.
            spectrum(count), // Allocate input storage and forward-transform output.
            restored(count); // Allocate reusable transform resources.
        for (unsigned i = 0; i < count; // Visit every sample across the three channels.
             ++i) { // Fill real and imaginary components with random input.
            original[i] = shdr::glsl::vec2( // Assign both complex components of this input.
                values(random), // Generate the real component independently.
                values(random)); // Generate both real and imaginary sample components.
        } // Different channels start with different values.
        original[0] += shdr::glsl::vec2(3.0f, -2.0f); // Include a strong complex impulse.
        std::copy(original.begin(), // Start copying at the original vector’s first value.
                  original.end(), // Stop at the end of the original vector.
                  input.data()); // Supply the same original values to either backend.
        fft1d forward(input, spectrum, length, channels); // Plan an unscaled forward transform.
        fft1d inverse(spectrum, // Read the forward spectrum as inverse-transform input.
                      restored, // Write reconstructed values into separate storage.
                      length, // Use the same length for both transforms.
                      channels, // Reconstruct all three channels independently.
                      true); // Plan inverse reconstruction with 1/N scaling.
        shdr::pipeline pipeline; // Link both transforms before running.
        forward.append_to(pipeline); // First produce the frequency spectrum.
        inverse.append_to(pipeline); // Reconstruct from that spectrum on the same backend.
        run.run(pipeline); // Execute every stage in dependency order.
        fft_checks_detail::compare( // Compare the completed forward spectrum.
            spectrum, // Compare the completed forward spectrum.
            fft_checks_detail::direct_dft( // Compute a direct double-precision reference from original samples.
                original, // Use the preserved original values as the reference.
                length), // Compute a direct double-precision reference from original samples.
            "1D FFT reference"); // Validate the actual forward FFT.
        fft_checks_detail::compare( // Check the reconstructed signal after the inverse stage.
            restored, // Read the inverse-transform output.
            original, // Use the original signal as the reconstruction reference.
            "1D inverse FFT reconstruction"); // Validate normalization and inverse phases.
        double input_energy = 0.0, // Start the original signal’s energy sum at zero.
            spectrum_energy = 0.0; // Check the transform's energy convention.
        const shdr::glsl::vec2* bins = // Access forward bins through a read-only pointer.
            static_cast<const shdr::buffer<shdr::glsl::vec2>&>( // Keep the spectrum resource const during readback.
                spectrum) // Keep the spectrum resource const during readback.
                .data(); // Read completed bins through const resource access.
        for (unsigned i = 0; i < count; ++i) { // Sum squared magnitudes across all channels.
            input_energy += // Accumulate each original sample’s squared magnitude.
                double(original[i].x) * // Square the original real component in double precision.
                    original[i].x + // Square the original real component in double precision.
                double(original[i].y) * original[i].y; // Measure original signal energy.
            spectrum_energy += // Accumulate each spectrum bin’s squared magnitude.
                double(bins[i].x) * // Square the spectrum’s real component in double precision.
                    bins[i].x + // Square the spectrum’s real component in double precision.
                double(bins[i].y) * bins[i].y; // Measure the unscaled spectrum energy.
        } // Parseval's identity predicts an N-fold increase for this forward convention.
        if (std::fabs(spectrum_energy / length - // Compare normalized spectral energy with original energy.
                      input_energy) > // Compare normalized spectral energy with original energy.
            0.00005 * input_energy) { // Allow the accumulated float error.
            throw shdr::error( // Report a failed energy identity.
                "FFT energy normalization differs from Parseval's identity"); // Report an incorrect transform convention.
        } // The transform's scaling is now independently checked.
        std::copy(original.begin(), // Begin restoring the original sample vector.
                  original.end(), // Copy its complete range before the aliasing check.
                  input.data()); // Restore the original values before testing aliases.
        fft1d in_place( // Construct a transform with shared input and output storage.
            input, // Read samples from the caller’s input buffer.
            input, // Write results back into that same buffer.
            length, // Keep the original transform length.
            channels); // Use working buffers to support a shared input and output resource.
        run.run(in_place.processing_pipeline()); // Perform the same FFT with caller storage reused.
        fft_checks_detail::compare( // Compare the aliased transform with an independent result.
            input, // Read the buffer overwritten by the transform.
            fft_checks_detail::direct_dft( // Recompute the direct reference from preserved original values.
                original, length), // Recompute the direct reference from preserved original values.
            "1D in-place FFT"); // Verify that an in-place transform preserves the expected result.
    } // All small sizes agree with an independent direct transform.
    shdr::buffer<shdr::glsl::vec2> short_input( // Allocate a deliberately small input buffer for invalid plans.
        4), // Allocate a deliberately small input buffer for invalid plans.
        short_output(4); // Exercise dimension checks with small buffers.
    for (unsigned length : {0u, 3u, 8u}) { // Reject zero, non-power-of-two, and undersized buffers.
        bool rejected = false; // Record whether validation caught the invalid plan.
        try { // Attempt construction so validation failures can be caught.
            fft1d invalid(short_input, // Ask the public constructor to validate this plan.
                          short_output, // Ask the public constructor to validate this plan.
                          length); // Ask the public constructor to validate this plan.
        } // Try to construct a plan with this invalid length or capacity.
        catch (const shdr::error&) { // Catch the library’s dimension or capacity error.
            rejected = true; // Remember that this unsafe plan was rejected.
        } // Dimension failures should be named runtime errors.
        if (!rejected) // Fail if an invalid plan escaped its checks.
            throw shdr::error( // Report acceptance of an unsafe transform shape.
                "invalid FFT dimensions were accepted"); // Keep unsafe plans out of application loops.
    } // Valid plans have consistent lengths and resource capacities.
    std::cout // Write the transform validation result to standard output.
        << "1D FFT: batched complex spectra, inverse, energy, and input aliasing verified\n"; // Summarize the reference checks.
} // The next examples use these transforms as signal-processing stages.

/** @brief Validate all block-frequency filter modes with known tones and cosine transitions. */
inline void check_frequency_filters( // Expose the frequency-response validation routine.
    runner& run) { // Compare reconstructed signals with their known frequency response.
    const unsigned length = 128, channels = 2, // Use two channels of 128 samples each.
        count = length * channels; // Use bin-aligned tones in two independent channels.
    const double tau = // Keep a double-precision full-turn phase constant.
        6.283185307179586476925286766559; // Generate reference sinusoidal inputs in double precision.
    const double frequencies[] = {// List tone frequencies used to probe the filter edges.
                                  3.0, // Include pass, stop, and exact cutoff frequencies.
                                  8.0, // Include pass, stop, and exact cutoff frequencies.
                                  11.0, // Include pass, stop, and exact cutoff frequencies.
                                  20.0, // Include pass, stop, and exact cutoff frequencies.
                                  25.0}; // Include pass, stop, and exact cutoff frequencies.
    const double amplitudes[] = {// Assign a distinguishable amplitude to every tone.
                                 0.7, // Give each component an identifiable amplitude.
                                 0.2, // Give each component an identifiable amplitude.
                                 0.5, // Give each component an identifiable amplitude.
                                 0.3, // Give each component an identifiable amplitude.
                                 0.4}; // Give each component an identifiable amplitude.
    shdr::buffer<shdr::glsl::vec2> input(count), // Allocate original channel-major complex samples.
        output(count); // Keep original and filtered signals separate.
    for (unsigned channel = 0; channel < channels; // Build each channel with its own phase offset.
         ++channel) { // Generate a different phase for each channel.
        for (unsigned position = 0; // Visit every sample in this channel’s block.
             position < length; // Visit every sample in this channel’s block.
             ++position) { // Form the complete real-valued tone mixture.
            double value = 0.3; // Include DC so high-pass rejection is checked.
            for (unsigned tone = 0; tone < 5; ++tone) { // Add all known components.
                value += amplitudes[tone] * // Weight this tone by its known amplitude.
                         std::cos(tau * frequencies[tone] * position / // Generate its cosine with a bin-aligned phase.
                                      length + // Generate its cosine with a bin-aligned phase.
                                  0.31 * channel); // Keep every component aligned with FFT bins.
            } // The frequencies share the sample rate of 128 hertz.
            input.data()[channel * length + // Write the complete tone mixture into this sample.
                         position] = // Write the complete tone mixture into this sample.
                shdr::glsl::vec2(float(value), 0.0f); // Store a real input in complex FFT storage.
        } // Channel layout matches the transform helper.
    } // Every filter receives the same original signal.
    for (unsigned selected = 0; selected < 4; // Select each of the four public filter modes.
         ++selected) { // Exercise low-pass, high-pass, band-pass, and band-stop modes.
        for (float width : {0.0f, 6.0f}) { // Check both hard bin edges and smooth transitions.
            const frequency_filter // Convert the test index into the public filter enumeration.
                mode = // Convert the test index into the public filter enumeration.
                static_cast<frequency_filter>(selected); // Use the public mode type.
            fft_filter filter(input, // Read the original signal in the filtering pipeline.
                              output, // Write reconstructed filtered samples into separate storage.
                              length, // Use 128 samples per transform.
                              channels, // Process both channels with the same frequency response.
                              128.0f, // Set a sample rate of 128 hertz.
                              mode, // Choose the current low, high, band or notch response.
                              8.0f, // Place the lower cutoff at eight hertz.
                              20.0f, // Place the upper band edge at twenty hertz.
                              width); // Build the complete transform-filter-inverse pipeline.
            run.run(filter.processing_pipeline()); // Keep intermediate spectra in their bound processing resources.
            std::vector<shdr::glsl::vec2> expected( // Allocate a complete analytic output vector.
                count); // Assemble the known output from tone amplitudes.
            const auto response = // Build an independent scalar response function.
                [&](double frequency) { // Evaluate the documented gain for any component.
                    const double lower = fft_checks_detail:: // Evaluate the low-pass response at the lower cutoff.
                        low_gain( // Evaluate the low-pass response at the lower cutoff.
                            frequency, // Describe the lower cutoff.
                            8.0, // Describe the lower cutoff.
                            width); // Describe the lower cutoff.
                    if (selected == 0) // Handle the low-pass mode.
                        return lower; // Low-pass keeps frequencies below this cutoff.
                    if (selected == 1) // Handle the complementary high-pass mode.
                        return 1.0 - lower; // High-pass uses its complementary response.
                    const double band = // Combine the two cutoff responses into a pass band.
                        (1.0 - lower) * // Reject frequencies retained by the lower low-pass edge.
                        fft_checks_detail::low_gain( // Retain frequencies below the upper edge.
                            frequency, // Combine both edges for a band.
                            20.0, // Combine both edges for a band.
                            width); // Combine both edges for a band.
                    return selected == 2 // Choose the pass band or its complement.
                               ? band // Return the pass-band response for band-pass mode.
                               : 1.0 - band; // Band-stop keeps the complement of band-pass.
                }; // All responses are symmetric for positive and negative frequencies.
            for (unsigned channel = 0; // Reconstruct each channel with its original phase.
                 channel < channels; // Reconstruct each channel with its original phase.
                 ++channel) { // Preserve each channel's phase in the expected output.
                for (unsigned position = 0; // Generate each analytic time-domain sample.
                     position < length; // Generate each analytic time-domain sample.
                     ++position) { // Reconstruct the expected time-domain signal directly.
                    double value = 0.3 * response(0.0); // Apply the expected DC gain.
                    for (unsigned tone = 0; // Apply the filter response to every known tone.
                         tone < 5; // Apply the filter response to every known tone.
                         ++tone) { // Apply the expected gain to each tone.
                        value += // Scale the tone by both amplitude and expected gain.
                            amplitudes[tone] * // Scale the tone by both amplitude and expected gain.
                            response( // Scale the tone by both amplitude and expected gain.
                                frequencies // Scale the tone by both amplitude and expected gain.
                                    [tone]) * // Scale the tone by both amplitude and expected gain.
                            std::cos( // Preserve the tone’s phase in the reconstructed sample.
                                tau * frequencies[tone] * position / // Preserve the tone’s phase in the reconstructed sample.
                                    length + // Preserve the tone’s phase in the reconstructed sample.
                                0.31 * channel); // Keep each tone's original phase.
                    } // A real symmetric response has an expected imaginary component of zero.
                    expected[channel * length + // Store the analytic channel-major output value.
                             position] = // Store the analytic channel-major output value.
                        shdr::glsl::vec2(float(value), 0.0f); // Store the analytic filtered signal.
                } // Continue through this channel's reconstructed block.
            } // Apply the same response independently to both channels.
            fft_checks_detail:: // Compare the actual filtered signal with its analytic response.
                compare( // Compare the actual filtered signal with its analytic response.
                    output, // Read the reconstructed filter output.
                    expected, // Use the independently assembled tone mixture.
                    "FFT signal filter response"); // Compare every reconstructed value.
        } // The response checks include the transition edges.
    } // All four block-filter modes now have numerical coverage.
    std::cout // Write the filter validation result to standard output.
        << "FFT filters: low-pass, high-pass, band-pass, and band-stop tone responses verified\n"; // Describe the completed signal checks.
} // Circular block filtering remains distinct from a streaming FIR or IIR filter.

/** @brief Check rectangular, Hann, Hamming, and Blackman windows before a Fourier transform. */
inline void check_signal_windows(runner& run) { // Verify window values and their linked spectra.
    const unsigned length = 32; // Use a small block for an independent direct DFT.
    shdr::buffer<shdr::glsl::vec2> input(length), // Allocate original and window-weighted complex values.
        windowed(length), // Allocate original and window-weighted complex values.
        spectrum(length); // Keep the original signal and both outputs.
    for (unsigned i = 0; i < length; ++i) // Create every sample in the small window fixture.
        input.data()[i] = // Assign the next complex fixture value.
            shdr::glsl::vec2(float(i % 7) - 3.0f, 0.25f); // Include both complex components.
    auto window = shdr::make_kernel("signal_window_kernel"); // Create the native window stage once.
    window // Connect original samples to the window shader.
        ->bind("input_values", input) // Connect original samples to the window shader.
        .bind("output_values", windowed); // Link caller storage to windowed FFT input.
    window->set("block_length", length).set("channels", 1u); // Set one complete block.
    fft1d transform(windowed, spectrum, length); // Plan an FFT of the weighted values.
    shdr::pipeline pipeline( // Start a pipeline with the window dispatch.
        shdr::dispatch(*window, shdr::extent(1))); // Begin the chain with input weighting.
    transform.append_to( // Append the FFT plan to the same linked pipeline.
        pipeline); // Pass the windowed resource directly to the FFT.
    for (unsigned mode = 0; mode < 4; // Select each supported amplitude window.
         ++mode) { // Reuse the linked resources for every supported window.
        window->set("mode", mode); // Select the next amplitude response.
        run.run(pipeline); // Execute the window and transform on the same backend.
        std::vector<shdr::glsl::vec2> // Allocate independent window-weighted reference values.
            expected( // Allocate independent window-weighted reference values.
                length); // Retain an independently weighted reference block.
        for (unsigned i = 0; i < length; ++i) { // Compute the reference gain at each position.
            const double phase = // Calculate phase from the sample’s position within the window.
                6.283185307179586476925286766559 * // Calculate phase from the sample’s position within the window.
                i / // Calculate phase from the sample’s position within the window.
                (length - 1u); // Sweep the symmetric window endpoints.
            double gain = 1.0; // Rectangular weighting preserves input amplitudes.
            if (mode == 1) // Select the Hann formula.
                gain = 0.5 - 0.5 * std::cos(phase); // Evaluate a Hann window.
            if (mode == 2) // Select the Hamming formula.
                gain = 0.54 - 0.46 * std::cos(phase); // Evaluate a Hamming window.
            if (mode == 3) // Select the Blackman formula.
                gain = 0.42 - 0.5 * std::cos(phase) + // Combine the first two Blackman terms.
                       0.08 * std::cos(2.0 * phase); // Evaluate a Blackman window.
            expected[i] = // Scale both complex components by the analytic window gain.
                static_cast<const shdr::buffer<shdr::glsl::vec2>&>(input) // Scale both complex components by the analytic window gain.
                    .data()[i] * // Scale both complex components by the analytic window gain.
                float(gain); // Apply the reference to both complex components.
        } // The windowed signal intentionally differs from the original input.
        fft_checks_detail::compare( // Compare window output with independent weighted samples.
            windowed, // Validate the weighting stage itself.
            expected, // Use the independently calculated expected values.
            "signal window"); // Validate the weighting stage itself.
        fft_checks_detail::compare( // Compare the linked FFT output after windowing.
            spectrum, // Compare the linked FFT output after windowing.
            fft_checks_detail::direct_dft( // Transform the independent window reference by direct DFT.
                expected, length), // Transform the independent window reference by direct DFT.
            "windowed FFT spectrum"); // Validate the complete linked chain.
    } // Every window can be selected dynamically after pipeline construction.
    std::cout // Write the window validation result to standard output.
        << "Signal windows: rectangular, Hann, Hamming, and Blackman stages verified\n"; // Report the spectral-analysis checks.
} // Applications can bind these windowed spectra to further GPU consumers.
} // namespace shdr_examples
#endif // Finish the FFT validation include guard.
