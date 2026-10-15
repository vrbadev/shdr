/** @file signal_filters.hpp
 * @brief Causal FIR and stateful biquad examples over ordinary float buffers.
 * Frequencies use the same units as the sample rate, normally hertz. Buffers
 * store complete channels consecutively: channel*count+sample. Filter classes
 * expose pipelines that can be linked directly to other shdr dispatches.
 */
#ifndef SHDR_EXAMPLES_SIGNAL_FILTERS_HPP // Include this example helper only once.
#define SHDR_EXAMPLES_SIGNAL_FILTERS_HPP // Remember that this example helper is available.
#include "shdr.h" // Use the same header for CPU and GPU resource wiring.
#include <algorithm> // Use standard copying, limits and zero filling.
#include <array> // Use fixed-size host coefficient and prediction arrays.
#include <cmath> // Use ordinary scalar math in host designers and oracles.
#include <vector> // Use owned host arrays for coefficients and reference results.

namespace shdr_examples { // Keep these example helpers in their own namespace.

/** @brief Frequency-selective modes shared by the FIR and biquad designers. */
enum class filter_mode { low_pass, high_pass, band_pass, band_stop, moving_average }; // Name the supported filter shapes explicitly.

namespace signal_detail { // Keep these example helpers in their own namespace.
inline double pi() { // Return the angle constant used by the filter designers.
    return 3.1415926535897932384626433832795; // Use a double-precision angle constant.
} // Finish the pi operation.
inline void sample_rate(double rate) { // Require a finite positive sampling rate.
    if (!std::isfinite(rate) || rate <= 0.0) // Reject a nonfinite or out-of-range value.
        throw shdr::error("filter sample rate must be finite and positive"); // Report the invalid design or failed check clearly.
} // Finish the sample_rate operation.
inline void cutoff(double frequency, double rate) { // Require a frequency strictly below Nyquist.
    if (!std::isfinite(frequency) || frequency <= 0.0 || frequency >= rate / 2.0) // Reject a nonfinite or out-of-range value.
        throw shdr::error("filter frequency must be strictly between zero and Nyquist"); // Report the invalid design or failed check clearly.
} // Finish the cutoff operation.
inline std::vector<double> low_pass(double normalized, unsigned taps) { // Build a Hamming-windowed sinc with unit DC gain.
    std::vector<double> result(taps); // Keep filter design coefficients in double precision until normalization.
    const int center = static_cast<int>(taps / 2u); // Find the symmetry center of the odd-length FIR.
    double sum = 0.0; // Start the scalar accumulator at zero.
    for (unsigned i = 0; i < taps; ++i) { // Use only causal taps that lie inside this batch.
        const int offset = static_cast<int>(i) - center; // Measure this tap relative to the symmetry center.
        const double sinc = offset ? std::sin(2.0 * pi() * normalized * offset) / (pi() * offset) // Evaluate the ideal low-pass sinc around the center tap.
                                   : 2.0 * normalized; // Use the finite sinc limit at its center tap.
        const double window = 0.54 - 0.46 * std::cos(2.0 * pi() * i / (taps - 1u)); // Apply the Hamming window to reduce sidelobes.
        sum += result[i] = sinc * window; // Store the windowed tap and accumulate its DC gain.
    } // Finish this channel, sample or feature pass.
    if (!std::isfinite(sum) || std::fabs(sum) < 1e-12) // Reject a nonfinite or out-of-range value.
        throw shdr::error("FIR design has negligible normalization gain"); // Report the invalid design or failed check clearly.
    for (double& value : result) // Apply the design adjustment to every coefficient.
        value /= sum; // Normalize this tap by the computed passband gain.
    return result; // Return the finite designed filter coefficients.
} // Finish the low_pass operation.
inline void normalize(std::vector<double>& values, double normalized) { // Set the requested passband gain to one.
    double gain = 0.0; // Start the scalar accumulator at zero.
    const int center = static_cast<int>(values.size() / 2u); // Find the symmetry center of the odd-length FIR.
    for (unsigned i = 0; i < values.size(); ++i) // Visit every value needed by this calculation.
        gain += values[i] * std::cos(2.0 * pi() * normalized * (static_cast<int>(i) - center)); // Sum the zero-phase response at the selected frequency.
    if (!std::isfinite(gain) || std::fabs(gain) < 1e-12) // Reject a nonfinite or out-of-range value.
        throw shdr::error("FIR design has negligible normalization gain"); // Report the invalid design or failed check clearly.
    for (double& value : values) // Apply the design adjustment to every coefficient.
        value /= gain; // Normalize this tap by the computed passband gain.
} // Finish the normalize operation.
inline void validate_biquad(const std::array<float, 5>& coefficients) { // Reject nonfinite or unstable float coefficients.
    for (float value : coefficients) // Inspect each stored float coefficient.
        if (!std::isfinite(value)) // Reject a nonfinite or out-of-range value.
            throw shdr::error("biquad coefficients must be finite"); // Report the invalid design or failed check clearly.
    const double a1 = coefficients[3], a2 = coefficients[4]; // Read the normalized feedback coefficients in double precision.
    if (std::fabs(a2) >= 1.0 || 1.0 + a1 + a2 <= 0.0 || 1.0 - a1 + a2 <= 0.0) // Handle this validation or boundary case explicitly.
        throw shdr::error("biquad denominator must remain stable at float precision"); // Report the invalid design or failed check clearly.
} // Finish the validate_biquad operation.
inline unsigned channels(unsigned value) { // Validate the number of independent channels.
    if (!value || value > 65536u) // Handle this validation or boundary case explicitly.
        throw shdr::error("filter channel count must be between 1 and 65536"); // Report the invalid design or failed check clearly.
    return value; // Return the validated channel count.
} // Finish the channels operation.
inline void buffers(const shdr::buffer<float>& input, // Validate channel-major storage and prevent unsafe aliasing.
                    const shdr::buffer<float>& output, // Receive distinct output storage with matching channel capacity.
                    unsigned count, // Receive the validated sample, channel or coefficient count.
                    unsigned channel_count) { // Receive the validated sample, channel or coefficient count.
    if (!count || count > 0xffffffffu / channel_count) // Handle this validation or boundary case explicitly.
        throw shdr::error("filter count must be positive and fit GLSL uint indexing"); // Report the invalid design or failed check clearly.
    const std::size_t needed = static_cast<std::size_t>(count) * channel_count; // Use host-size arithmetic to count every channel sample.
    if (input.size() < needed || output.size() < needed) // Check that both buffers contain every channel sample.
        throw shdr::error("filter buffers do not contain all channel samples"); // Report the invalid design or failed check clearly.
    if (input.state() == output.state()) // Reject input and output that share the same storage.
        throw shdr::error("filter input and output must use distinct storage"); // Report the invalid design or failed check clearly.
} // Finish the buffers operation.
} // namespace signal_detail

/** @brief Design an odd-length linear-phase FIR using a Hamming-windowed sinc.
 * Low-pass coefficients have unit DC gain. High-pass uses spectral inversion
 * and unit Nyquist gain. Band-pass subtracts two low-pass designs and has unit
 * midpoint gain.  band-stop inverts that band-pass and has unit DC gain.
 * Moving-average returns equal weights and ignores both cutoffs. Other modes
 * require cutoffs strictly inside Nyquist.  band edges must increase. Tap count
 * must be odd and between 3 and 4095. The causal group delay is (taps-1)/2.
 */
inline std::vector<float> fir_coefficients(filter_mode mode, // Design the requested causal linear-phase FIR.
                                           double sample_rate, // Receive the positive sampling rate in frequency units.
                                           double cutoff1, // Receive the cutoff or band-edge frequency.
                                           double cutoff2 = 0.0, // Receive the cutoff or band-edge frequency.
                                           unsigned taps = 31u) { // Receive the validated sample, channel or coefficient count.
    signal_detail::sample_rate(sample_rate); // Validate the sample-rate units before designing coefficients.
    if (taps < 3u || taps > 4095u || taps % 2u == 0u) // Handle this validation or boundary case explicitly.
        throw shdr::error("FIR tap count must be odd and between 3 and 4095"); // Report the invalid design or failed check clearly.
    if (mode == filter_mode::moving_average) // Choose the coefficient rule for this filter shape.
        return std::vector<float>(taps, 1.0f / taps); // Give every moving-average tap the same weight.
    signal_detail::cutoff(cutoff1, sample_rate); // Validate this frequency against the sampling rate.
    std::vector<double> values; // Keep filter design coefficients in double precision until normalization.
    if (mode == filter_mode::low_pass || mode == filter_mode::high_pass) { // Choose the coefficient rule for this filter shape.
        values = signal_detail::low_pass(cutoff1 / sample_rate, taps); // Build the appropriate low-pass prototype.
        if (mode == filter_mode::high_pass) { // Choose the coefficient rule for this filter shape.
            for (double& value : values) // Apply the design adjustment to every coefficient.
                value = -value; // Invert this coefficient for spectral inversion.
            values[taps / 2u] += 1.0; // Add the centered impulse to complete spectral inversion.
            signal_detail::normalize(values, 0.5); // Scale the chosen passband response to unity.
        } // Finish handling this condition.
    } else if (mode == filter_mode::band_pass || mode == filter_mode::band_stop) { // Use the alternative filter-design case.
        signal_detail::cutoff(cutoff2, sample_rate); // Validate this frequency against the sampling rate.
        if (cutoff1 >= cutoff2) // Require the lower band edge to precede the upper edge.
            throw shdr::error("FIR band edges must be strictly increasing"); // Report the invalid design or failed check clearly.
        const std::vector<double> lower = signal_detail::low_pass(cutoff1 / sample_rate, taps); // Keep the lower-edge low-pass prototype for subtraction.
        values = signal_detail::low_pass(cutoff2 / sample_rate, taps); // Build the appropriate low-pass prototype.
        for (unsigned i = 0; i < taps; ++i) // Use only causal taps that lie inside this batch.
            values[i] -= lower[i]; // Subtract the lower low-pass response to form a band-pass.
        signal_detail::normalize(values, 0.5 * (cutoff1 / sample_rate + cutoff2 / sample_rate)); // Scale the chosen passband response to unity.
        if (mode == filter_mode::band_stop) { // Choose the coefficient rule for this filter shape.
            for (double& value : values) // Apply the design adjustment to every coefficient.
                value = -value; // Invert this coefficient for spectral inversion.
            values[taps / 2u] += 1.0; // Add the centered impulse to complete spectral inversion.
            signal_detail::normalize(values, 0.0); // Scale the chosen passband response to unity.
        } // Finish handling this condition.
    } else // Use the alternative filter-design case.
        throw shdr::error("unknown FIR filter mode"); // Report the invalid design or failed check clearly.
    std::vector<float> result(values.begin(), values.end()); // Convert the designed coefficients to GPU-compatible floats.
    for (float value : result) // Inspect each stored float coefficient.
        if (!std::isfinite(value)) // Reject a nonfinite or out-of-range value.
            throw shdr::error("FIR coefficients must remain finite at float precision"); // Report the invalid design or failed check clearly.
    return result; // Return the finite designed filter coefficients.
} // Finish handling this condition.

/** @brief Design a normalized RBJ/W3C low-pass, high-pass, band-pass or notch biquad.
 * Return order is b0,b1,b2,a1,a2 with a0 normalized to one. Band-pass has unit
 * center-frequency gain.  band_stop selects a notch. Frequency is the cutoff or
 * center in sample-rate units, and Q must be finite and positive. Designs whose
 * rounded float feedback coefficients lose strict stability are rejected.
 * Formula reference: https://www.w3.org/TR/audio-eq-cookbook/ .
 */
inline std::array<float, 5> biquad_coefficients(filter_mode mode, // Design normalized coefficients from the RBJ formulas.
                                                double sample_rate, // Receive the positive sampling rate in frequency units.
                                                double frequency, // Receive the tone, cutoff or center frequency.
                                                double q = 0.7071067811865475244) { // Receive the positive biquad quality factor.
    signal_detail::sample_rate(sample_rate); // Validate the sample-rate units before designing coefficients.
    signal_detail::cutoff(frequency, sample_rate); // Validate this frequency against the sampling rate.
    if (!std::isfinite(q) || q <= 0.0) // Reject a nonfinite or out-of-range value.
        throw shdr::error("biquad Q must be finite and positive"); // Report the invalid design or failed check clearly.
    const double omega = 2.0 * signal_detail::pi() * (frequency / sample_rate); // Convert the cutoff frequency into radians per sample.
    const double cosine = std::cos(omega), alpha = std::sin(omega) / (2.0 * q), a0 = 1.0 + alpha; // Compute the RBJ frequency and damping terms.
    double b0 = 0.0, b1 = 0.0, b2 = 0.0; // Prepare the three feed-forward numerator coefficients.
    if (mode == filter_mode::low_pass) { // Choose the coefficient rule for this filter shape.
        b0 = b2 = (1.0 - cosine) / 2.0; // Use symmetric outer numerator coefficients for this filter.
        b1 = 1.0 - cosine; // Set the middle numerator coefficient for this filter.
    } else if (mode == filter_mode::high_pass) { // Use the alternative filter-design case.
        b0 = b2 = (1.0 + cosine) / 2.0; // Use symmetric outer numerator coefficients for this filter.
        b1 = -(1.0 + cosine); // Set the middle numerator coefficient for this filter.
    } else if (mode == filter_mode::band_pass) { // Use the alternative filter-design case.
        b0 = alpha; // Give the band-pass unit gain at its center frequency.
        b2 = -alpha; // Use the opposite outer coefficient for the band-pass zero.
    } else if (mode == filter_mode::band_stop) { // Use the alternative filter-design case.
        b0 = b2 = 1.0; // Use symmetric outer numerator coefficients for this filter.
        b1 = -2.0 * cosine; // Set the middle numerator coefficient for this filter.
    } else // Use the alternative filter-design case.
        throw shdr::error("biquad supports low-pass, high-pass, band-pass and band-stop modes"); // Report the invalid design or failed check clearly.
    const std::array<float, 5> result = {{static_cast<float>(b0 / a0), // Normalize every coefficient by a0 before storing floats.
                                          static_cast<float>(b1 / a0), // Convert the computed host value to the shader float type.
                                          static_cast<float>(b2 / a0), // Convert the computed host value to the shader float type.
                                          static_cast<float>(-2.0 * cosine / a0), // Convert the computed host value to the shader float type.
                                          static_cast<float>((1.0 - alpha) / a0)}}; // Convert the computed host value to the shader float type.
    signal_detail::validate_biquad(result); // Require stable finite feedback coefficients.
    return result; // Return the finite designed filter coefficients.
} // Finish handling this condition.

/** @brief One causal batch FIR dispatch over channel-major float resources.
 * Each execution starts every channel with an implicit zero prefix.
 * Repeating the pipeline processes a fresh batch. Input and output
 * require distinct storage. Public coefficients can be edited before
 * the next dispatch, retaining the original coefficient count.
 */
class fir_pipeline { // Bundle FIR coefficients and one causal batch dispatch.
    unsigned channels_; // Remember the fixed number of independent signal channels.
    std::unique_ptr<shdr::kernel> shader_; // Own the compiled filter instance and its resource bindings.
    shdr::pipeline pipeline_; // Keep the filter dispatch ready for repeated execution.

  public: // Expose the resources and processing helpers.
    shdr::buffer<float> coefficients; // Keep filter coefficients in a resource shared by CPU and GPU.
    /** @brief Bind a batch and copy a nonempty set of finite causal FIR weights. */
    fir_pipeline(const shdr::buffer<float>& input, // Own coefficients and connect the FIR batch resources.
                 const shdr::buffer<float>& output, // Receive distinct output storage with matching channel capacity.
                 unsigned count, // Receive the validated sample, channel or coefficient count.
                 unsigned channels, // Receive the validated sample, channel or coefficient count.
                 const std::vector<float>& weights) // Use the caller-provided finite FIR coefficients.
        : channels_(signal_detail::channels(channels)), // Validate the channel count before allocating per-channel state.
          shader_(shdr::make_kernel("signal_fir_kernel")), coefficients(weights.size()) { // Create the compiled native filter instance.
        if (weights.empty() || weights.size() > 4095u) // Handle this validation or boundary case explicitly.
            throw shdr::error("FIR weights require between 1 and 4095 coefficients"); // Report the invalid design or failed check clearly.
        for (float value : weights) // Inspect each stored float coefficient.
            if (!std::isfinite(value)) // Reject a nonfinite or out-of-range value.
                throw shdr::error("FIR weights must be finite"); // Report the invalid design or failed check clearly.
        std::copy(weights.begin(), weights.end(), coefficients.data()); // Copy these host values into the shared resource.
        shader_->bind("coefficients", coefficients) // Connect the shader to its coefficient, state or sample resource.
            .set<unsigned>("taps", static_cast<unsigned>(weights.size())) // Set the shader dimensions or scalar processing parameter.
            .set<unsigned>("channels", channels_); // Set the shader dimensions or scalar processing parameter.
        bind(input, output, count); // Connect the initial input and output batches.
    } // Finish the fir_pipeline operation.
    /** @brief Design coefficients and bind their batch pipeline in one step. */
    fir_pipeline(const shdr::buffer<float>& input, // Own coefficients and connect the FIR batch resources.
                 const shdr::buffer<float>& output, // Receive distinct output storage with matching channel capacity.
                 unsigned count, // Receive the validated sample, channel or coefficient count.
                 unsigned channels, // Receive the validated sample, channel or coefficient count.
                 filter_mode mode, // Receive the requested filter shape.
                 double rate, // Receive the positive sampling rate in frequency units.
                 double cutoff1, // Receive the cutoff or band-edge frequency.
                 double cutoff2 = 0.0, // Receive the cutoff or band-edge frequency.
                 unsigned taps = 31u) // Receive the validated sample, channel or coefficient count.
        : fir_pipeline(input, // Reuse the constructor with newly designed coefficients.
                       output, // Pass the distinct output resource to the wiring constructor.
                       count, // Pass the positive per-channel sample count.
                       channels, // Pass the fixed number of independent channels.
                       fir_coefficients(mode, rate, cutoff1, cutoff2, taps)) { // Design the causal FIR coefficients for this configuration.
    } // Finish the fir_coefficients operation.
    /** @brief Rebind channel-major batches and update their sample count. */
    void bind(const shdr::buffer<float>& input, const shdr::buffer<float>& output, unsigned count) { // Connect the next input and output block.
        signal_detail::buffers(input, output, count, channels_); // Validate dimensions and distinct storage before rebinding.
        shader_->bind("input_values", input) // Connect the shader to its coefficient, state or sample resource.
            .bind("output_values", output) // Connect the next named resource to the same shader instance.
            .set<unsigned>("count", count); // Set the shader dimensions or scalar processing parameter.
        const unsigned total = count * channels_; // Remember the fixed number of independent signal channels.
        pipeline_ = shdr::pipeline( // Retain the dispatch with the required work group count.
            shdr::dispatch(*shader_, shdr::extent(total / 64u + (total % 64u ? 1u : 0u)))); // Cover each independent FIR sample or biquad channel with a work group.
    } // Finish the bind operation.
    /** @brief Return the prepared wiring for CPU run(), GPU run(device) or async submission. */
    const shdr::pipeline& processing_pipeline() const { // Expose the reusable filter processing dispatch.
        return pipeline_; // Keep the filter dispatch ready for repeated execution.
    } // Finish the processing_pipeline operation.
}; // Finish the fir_pipeline resource helper.

/** @brief One stateful biquad dispatch, parallel across independent channels.
 * Each channel processes its temporal block sequentially. State contains one
 * vec2(z1,z2) per channel for transposed direct-form II. State
 * persists across bind() and runs, including explicit CPU/GPU transitions.
 * Give independent streams or concurrent jobs separate filter instances/state.
 */
class biquad_pipeline { // Bundle biquad coefficients and persistent channel history.
    unsigned channels_; // Remember the fixed number of independent signal channels.
    std::unique_ptr<shdr::kernel> shader_; // Own the compiled filter instance and its resource bindings.
    shdr::pipeline pipeline_; // Keep the filter dispatch ready for repeated execution.

  public: // Expose the resources and processing helpers.
    shdr::buffer<float> coefficients; // Keep filter coefficients in a resource shared by CPU and GPU.
    shdr::buffer<shdr::glsl::vec2> state; // Keep one pair of recurrence values for each channel.
    /** @brief Bind normalized stable coefficients and zero-initialized channel state. */
    biquad_pipeline(const shdr::buffer<float>& input, // Own coefficients and connect the persistent biquad state.
                    const shdr::buffer<float>& output, // Receive distinct output storage with matching channel capacity.
                    unsigned count, // Receive the validated sample, channel or coefficient count.
                    unsigned channels, // Receive the validated sample, channel or coefficient count.
                    const std::array<float, 5>& weights) // Use five normalized biquad coefficients.
        : channels_(signal_detail::channels(channels)), // Validate the channel count before allocating per-channel state.
          shader_(shdr::make_kernel("signal_biquad_kernel")), coefficients(5), state(channels_) { // Create the compiled native filter instance.
        signal_detail::validate_biquad(weights); // Require stable finite feedback coefficients.
        std::copy(weights.begin(), weights.end(), coefficients.data()); // Copy these host values into the shared resource.
        shader_->bind("coefficients", coefficients) // Connect the shader to its coefficient, state or sample resource.
            .bind("state", state) // Connect the next named resource to the same shader instance.
            .set<unsigned>("channels", channels_); // Set the shader dimensions or scalar processing parameter.
        bind(input, output, count); // Connect the initial input and output batches.
        pipeline_ = shdr::pipeline( // Retain the dispatch with the required work group count.
            shdr::dispatch(*shader_, shdr::extent(channels_ / 64u + (channels_ % 64u ? 1u : 0u)))); // Cover each independent FIR sample or biquad channel with a work group.
    } // Finish the biquad_pipeline operation.
    /** @brief Design coefficients and bind their streaming pipeline in one step. */
    biquad_pipeline(const shdr::buffer<float>& input, // Own coefficients and connect the persistent biquad state.
                    const shdr::buffer<float>& output, // Receive distinct output storage with matching channel capacity.
                    unsigned count, // Receive the validated sample, channel or coefficient count.
                    unsigned channels, // Receive the validated sample, channel or coefficient count.
                    filter_mode mode, // Receive the requested filter shape.
                    double rate, // Receive the positive sampling rate in frequency units.
                    double frequency, // Receive the tone, cutoff or center frequency.
                    double q = 0.7071067811865475244) // Receive the positive biquad quality factor.
        : biquad_pipeline( // Reuse the constructor with newly designed coefficients.
              input, output, count, channels, biquad_coefficients(mode, rate, frequency, q)) { // Reuse the resource wiring with freshly designed normalized coefficients.
    } // Finish the biquad_coefficients operation.
    /** @brief Rebind the next channel-major block, preserving every channel's state. */
    void bind(const shdr::buffer<float>& input, const shdr::buffer<float>& output, unsigned count) { // Connect the next input and output block.
        signal_detail::buffers(input, output, count, channels_); // Validate dimensions and distinct storage before rebinding.
        shader_->bind("input_values", input) // Connect the shader to its coefficient, state or sample resource.
            .bind("output_values", output) // Connect the next named resource to the same shader instance.
            .set<unsigned>("count", count); // Set the shader dimensions or scalar processing parameter.
    } // Finish the bind operation.
    /** @brief Explicitly discard channel history and clear its host state.
     * Resetting writes zero values to host state. The next GPU dispatch
     * uploads the cleared state through the ordinary resource synchronization.
     */
    void reset_state() { // Replace the channel history with zero-filled host state.
        state.mark_host_modified(); // Discard old GPU history before clearing every host state value.
        shdr::glsl::vec2* values = state.data(); // Acquire the host storage that this initialization will overwrite.
        std::fill(values, values + state.size(), shdr::glsl::vec2(0.0f)); // Clear every stored channel recurrence value.
    } // Finish the reset_state operation.
    /** @brief Return the state-preserving CPU/GPU processing dispatch. */
    const shdr::pipeline& processing_pipeline() const { // Expose the reusable filter processing dispatch.
        return pipeline_; // Keep the filter dispatch ready for repeated execution.
    } // Finish the processing_pipeline operation.
}; // Finish the biquad_pipeline resource helper.
} // namespace shdr_examples
#endif // Finish the conditional compilation block.
