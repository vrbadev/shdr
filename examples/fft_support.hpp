/** @file fft_support.hpp
 * @brief Batched complex FFT and block signal-processing examples.
 * A vec2 stores real and imaginary parts. Channels occupy consecutive blocks.
 * Dispatch boundaries order the dependent FFT stages.
 */
#ifndef SHDR_EXAMPLES_FFT_SUPPORT_HPP // Prevent duplicate helper definitions.
#define SHDR_EXAMPLES_FFT_SUPPORT_HPP // Mark this helper as included.
#include "shdr.h" // Use the single-header compute runtime.
#include <cmath> // Compute shared phase tables once on the host.
#include <memory> // Keep each stage's kernel alive with its pipeline.
#include <vector> // Store the kernels for all butterfly stages.

namespace shdr_examples { // Keep example helpers separate from the runtime API.
namespace fft_detail { // Share dimension checks between signal helpers.
inline unsigned power_of_two(unsigned length) { // Require a positive power-of-two transform length.
    if (!length || (length & (length - 1u))) { // Test that exactly one length bit is set.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "FFT length must be a positive power of two"); // Report the unsupported length before allocation.
    } // A length of one uses the permutation copy directly.
    return length; // Use the checked length in member initialization.
} // Use the requested block length exactly.
inline unsigned channel_count(unsigned channels) { // Require at least one independent signal.
    if (!channels) { // Require at least one signal channel before creating dispatches.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "signal channel count must be positive"); // Explain the invalid dimension.
    } // Positive channel counts are checked for overflow with their length next.
    return channels; // Preserve the requested channel count.
} // Every helper uses the same channel-major layout.
inline unsigned elements(unsigned length, // Check shader index arithmetic before allocation.
                         unsigned channels) { // Check shader index arithmetic before allocation.
    if (length > 0xffffffffu / channels) { // Validate the product capacity within shader unsigned indexing.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "signal dimensions exceed GLSL uint indexing"); // Keep host and shader indexing consistent.
    } // The product now fits the shader's unsigned type.
    return length * channels; // Return the total number of complex values.
} // Buffer constructors also check allocation byte counts.
inline void // Inspect the input's allocated size.
buffers(const shdr::buffer<shdr::glsl::vec2>& input, // Inspect the input's allocated size.
        const shdr::buffer<shdr::glsl::vec2>& output, // Read the output capacity from host resource metadata.
        unsigned count) { // Read the output capacity from host resource metadata.
    if (input.size() < count || // Require storage for every requested channel.
        output.size() < count) { // Require storage for every requested channel.
        throw shdr::error( // Report the failed format, numerical or resource precondition.
            "signal buffers do not contain every requested channel"); // Fail before a dispatch could access missing values.
    } // Extra capacity remains available to other application code.
} // Dimension checks use the resource capacity metadata.
inline void append(shdr::pipeline& destination, // Link existing stages in their execution order.
                   const shdr::pipeline& source) { // Link existing stages in their execution order.
    for (const shdr::stage& stage : // Retain the references already stored by each helper.
         source.stages()) { // Retain the references already stored by each helper.
        destination += stage; // Place this dispatch after all previously appended work.
    } // The helper objects must remain alive while their stages execute.
} // This avoids copying any input or intermediate values.
} // namespace fft_detail

/** @brief An out-of-place radix-two complex FFT with O(N log N) work.
 * Forward transforms are unscaled. Inverse transforms include 1/N.
 * Input and output may share storage because working buffers separate stages.
 * Keep this object alive while using its pipeline or an appended dispatch.
 */
class fft1d { // Own one fixed-size transform plan and its working resources.
    unsigned length_, channels_, // Keep dimensions checked and consistent across all stages.
        count_; // Keep dimensions checked and consistent across all stages.
    shdr::buffer<shdr::glsl::vec2> first_, second_, // Alternate working buffers and share one phase table.
        twiddles_; // Alternate working buffers and share one phase table.
    std::vector<std::unique_ptr<shdr::kernel>> // Retain each permutation and butterfly kernel.
        kernels_; // Retain each permutation and butterfly kernel.
    shdr::pipeline pipeline_; // Preserve the dependency order established at construction.
  public: // Expose execution while keeping working resources private.
    /** @brief Plan a batched forward or inverse transform over channel-major buffers. */
    fft1d(const shdr::buffer<shdr::glsl::vec2>& input, // Bind the original complex values.
          const shdr::buffer<shdr::glsl::vec2>& output, // Bind the final transform result.
          unsigned length, // Select block dimensions and transform direction.
          unsigned channels = 1, // Select block dimensions and transform direction.
          bool inverse = false) // Select block dimensions and transform direction.
        : length_(fft_detail::power_of_two(length)), // Check the radix-two requirement first.
          channels_(fft_detail::channel_count( // Check the channel count before computing products.
              channels)), // Check the channel count before computing products.
          count_( // Keep every shader index representable.
              fft_detail::elements(length_, channels_)), // Keep every shader index representable.
          first_(count_), second_(count_), // Allocate working storage once.
          twiddles_(std::max(1u, length_ / 2u)) { // Allocate working storage once.
        fft_detail::buffers(input, output, count_); // Validate caller-owned resource capacities.
        const double tau = // Use one full turn for complex phases.
            6.283185307179586476925286766559; // Use one full turn for complex phases.
        shdr::glsl::vec2* phases = twiddles_.data(); // Fill the host table before GPU preparation.
        for (unsigned k = 0; k < twiddles_.size(); // Compute the phases needed by every butterfly stage.
             ++k) { // Compute the phases needed by every butterfly stage.
            const double angle = -tau * double(k) / length_; // Store forward-transform phases.
            phases[k] = shdr::glsl::vec2( // Store matching float values for both backends.
                float(std::cos(angle)), // Store matching float values for both backends.
                float(std::sin(angle))); // Store matching float values for both backends.
        } // Butterflies read these precomputed phase values.
        unsigned bits = 0; // Count the binary digits in a block position.
        for (unsigned size = length_; size > 1u; size /= 2u) { // Repeated halving measures log2(N).
            ++bits; // Add one bit for this power of two.
        } // Length one has zero permutation bits.
        std::unique_ptr<shdr::kernel> permutation = // Create the index-ordering stage.
            shdr::make_kernel("fft_bit_reverse_kernel"); // Create the index-ordering stage.
        permutation->bind("input_values", input); // Read the application input.
        permutation->bind("output_values", // Length one needs only a copy.
                          length_ == 1u ? output : first_); // Length one needs only a copy.
        permutation->set("block_length", length_) // Supply the per-channel block length.
            .set("channels", channels_) // Supply the channels processing parameter.
            .set("bits", bits); // Supply the permutation dimensions.
        pipeline_ += shdr::dispatch( // Dispatch this configured shader with the following work group geometry.
            *permutation,
            shdr::extent( // Use quotient and remainder to cover every value.
                count_ / 64u + // Use quotient and remainder to cover every value.
                (count_ % 64u != 0u))); // Use quotient and remainder to cover every value.
        kernels_.push_back( // Keep the stage alive after this constructor returns.
            std::move(permutation)); // Keep the stage alive after this constructor returns.
        bool read_first = true; // The permutation initially fills the first working buffer.
        for (unsigned span = 2u; // Grow butterfly groups from two values to the full block.
             span <= length_ && // Grow butterfly groups from two values to the full block.
             length_ > 1u;) { // Grow butterfly groups from two values to the full block.
            std::unique_ptr<shdr::kernel> butterfly = // Give each stage independent uniforms.
                shdr::make_kernel("fft_butterfly_kernel"); // Give each stage independent uniforms.
            butterfly->bind("input_values", // Read the preceding stage's output.
                            read_first ? first_ : second_); // Read the preceding stage's output.
            butterfly->bind( // Send the last stage to caller storage.
                "output_values", // Use the exact diagnostic or comparison label for this case.
                span == length_ // Send the last stage to caller storage.
                    ? output // Send the last stage to caller storage.
                    : (read_first ? second_ : first_)); // Send the last stage to caller storage.
            butterfly->bind("twiddles", twiddles_); // Reuse the same phase table at every span.
            butterfly->set("block_length", length_) // Supply the per-channel block length.
                .set("channels", channels_) // Supply the channels processing parameter.
                .set("span", span); // Set this stage's butterfly width.
            butterfly->set("direction", // Conjugate the phase table for an inverse transform.
                           inverse ? -1.0f // Conjugate the phase table for an inverse transform.
                                   : 1.0f); // Conjugate the phase table for an inverse transform.
            butterfly->set("normalization", // Scale exactly once for the inverse.
                           inverse && span == length_ // Scale exactly once for the inverse.
                               ? 1.0f / length_ // Scale exactly once for the inverse.
                               : 1.0f); // Scale exactly once for the inverse.
            pipeline_ += shdr::dispatch( // Dispatch this configured shader with the following work group geometry.
                *butterfly,
                shdr::extent( // Use quotient and remainder to cover every butterfly output.
                    count_ / 64u + // Use quotient and remainder to cover every butterfly output.
                    (count_ % 64u != // Use quotient and remainder to cover every butterfly output.
                     0u))); // Use quotient and remainder to cover every butterfly output.
            kernels_.push_back( // Retain the stage's source, bindings, and uniforms.
                std::move(butterfly)); // Retain the stage's source, bindings, and uniforms.
            read_first = !read_first; // Exchange the working-buffer roles for the next stage.
            if (span == length_) // Stop before the final span could overflow on doubling.
                break; // Stop before the final span could overflow on doubling.
            span *= 2u; // Double the butterfly group size.
        } // The last dispatch has written the complete transformed block.
    } // Preparation and execution are separate from plan construction.
    /** @brief Return the fixed transform stages for CPU, GPU, or asynchronous execution. */
    const shdr::pipeline& // Return the existing plan and its bound resources.
    processing_pipeline() const { // Return the existing plan and its bound resources.
        return pipeline_; // Run this pipeline on the CPU or a GPU device.
    } // GPU preparation can compile and upload the plan before a hot loop.
    /** @brief Append all FFT stages to a larger ordered pipeline. */
    void append_to( // Keep original kernel ownership in this FFT plan.
        shdr::pipeline& destination) const { // Keep original kernel ownership in this FFT plan.
        fft_detail::append(destination, // Link the transform after earlier application work.
                           pipeline_); // Link the transform after earlier application work.
    } // Values remain in their bound CPU or GPU buffers.
}; // End the reusable complex-transform helper.

/** @brief Frequency responses available for independent FFT signal blocks. */
enum class frequency_filter { // Match the native mask shader's mode values.
    low_pass, // Match the native mask shader's mode values.
    high_pass, // Match the native mask shader's mode values.
    band_pass, // Match the native mask shader's mode values.
    band_stop // Match the native mask shader's mode values.
}; // Match the native mask shader's mode values.

/** @brief Forward FFT followed by a symmetric spectral mask and an inverse FFT.
 * This is circular block filtering. Use padding and overlap for a linear stream.
 * A positive transition width joins pass and stop bands with a cosine response.
 */
class fft_filter { // Own all resources needed to filter one fixed-size batch.
    unsigned length_, channels_, count_; // Store checked FFT dimensions.
    shdr::buffer<shdr::glsl::vec2> spectrum_, // Retain spectra for later inspection or linked processing.
        masked_; // Retain spectra for later inspection or linked processing.
    fft1d forward_, inverse_; // Reuse the same transform implementation in both directions.
    std::unique_ptr<shdr::kernel> mask_; // Keep the frequency-selection dispatch alive.
    shdr::pipeline pipeline_; // Link forward transform, mask, and inverse transform.
  public: // Expose the complete processing chain and optional spectral results.
    /** @brief Construct a frequency-selective filter using hertz for rate and cutoffs. */
    fft_filter(const shdr::buffer<shdr::glsl::vec2>& // Bind one or more real or complex input blocks.
                   input, // Bind one or more real or complex input blocks.
               const shdr::buffer<shdr::glsl::vec2>& output, // Bind reconstructed filtered blocks.
               unsigned length, // Define block geometry and frequency units.
               unsigned channels, // Define block geometry and frequency units.
               float sample_rate, // Define block geometry and frequency units.
               frequency_filter mode, // Select one cutoff or two band edges.
               float lower_hz, // Select one cutoff or two band edges.
               float upper_hz = 0.0f, // Select one cutoff or two band edges.
               float transition_hz = 0.0f) // Select one cutoff or two band edges.
        : length_( // Reject a block length unsupported by the FFT.
              fft_detail::power_of_two(length)), // Reject a block length unsupported by the FFT.
          channels_(fft_detail::channel_count(channels)), // Require at least one channel.
          count_( // Check index products before allocation.
              fft_detail::elements(length_, channels_)), // Check index products before allocation.
          spectrum_(count_), masked_(count_), // Allocate reusable frequency-domain buffers.
          forward_(input, spectrum_, length_, channels_), // Plan the forward transform.
          inverse_( // Plan normalized inverse reconstruction.
              masked_, output, length_, channels_, true), // Plan normalized inverse reconstruction.
          mask_(shdr::make_kernel("signal_frequency_mask_kernel")) { // Create the response stage.
        if (!std::isfinite(sample_rate) || // Frequency units need a valid positive rate.
            sample_rate <= 0.0f) { // Frequency units need a valid positive rate.
            throw shdr::error( // Report the failed format, numerical or resource precondition.
                "FFT filter sample rate must be finite and positive"); // Reject invalid conversion from bins to hertz.
        } // Nyquist is half this sample rate.
        if (!std::isfinite(lower_hz) || lower_hz <= 0.0f || // Validate the first cutoff.
            lower_hz >= sample_rate * 0.5f) { // Validate the first cutoff.
            throw shdr::error( // Report the failed format, numerical or resource precondition.
                "FFT filter cutoff must lie strictly between zero and Nyquist"); // Keep the response inside representable frequencies.
        } // Low-pass and high-pass use this single cutoff.
        if (mode >= frequency_filter::band_pass && // Validate both edges of a band.
            (!std::isfinite(upper_hz) || upper_hz <= lower_hz || // Validate both edges of a band.
             upper_hz >= sample_rate * 0.5f)) { // Validate both edges of a band.
            throw shdr::error( // Report the failed format, numerical or resource precondition.
                "FFT filter band edges must increase below Nyquist"); // Reject reversed or missing edges.
        } // The band can now be evaluated symmetrically.
        if (!std::isfinite(transition_hz) || // Require a finite nonnegative transition width.
            transition_hz < 0.0f) { // Require a finite nonnegative transition width.
            throw shdr::error( // Report the failed format, numerical or resource precondition.
                "FFT filter transition must be finite and nonnegative"); // Zero selects a hard bin response.
        } // Wider transitions deliberately soften the filter response.
        const unsigned selected = // Match the host enumeration to the GLSL uniform.
            static_cast<unsigned>(mode); // Match the host enumeration to the GLSL uniform.
        if (selected > 3u) // Reject values outside the documented responses.
            throw shdr::error( // Report the failed format, numerical or resource precondition.
                "unknown FFT filter mode"); // Reject values outside the documented responses.
        mask_->bind("input_values", spectrum_) // Connect the input_values shader port to spectrum_ storage.
            .bind("output_values", masked_); // Link the transform result to the mask.
        mask_->set("block_length", length_) // Supply the per-channel block length.
            .set("channels", channels_) // Supply the channels processing parameter.
            .set("mode", selected); // Supply channel and bin dimensions.
        mask_->set("sample_rate", sample_rate) // Supply the sample_rate processing parameter.
            .set("lower_hz", lower_hz) // Supply the lower_hz processing parameter.
            .set("upper_hz", upper_hz); // Describe the response in hertz.
        mask_->set("transition_hz", transition_hz); // Select hard or smooth cutoff behavior.
        forward_.append_to(pipeline_); // First compute the unshifted spectrum.
        pipeline_ += shdr::dispatch( // Dispatch this configured shader with the following work group geometry.
            *mask_,
            shdr::extent(count_ / 64u + // Dispatch the mask over every resident spectral value.
                         (count_ % 64u != // Dispatch the mask over every resident spectral value.
                          0u))); // Dispatch the mask over every resident spectral value.
        inverse_.append_to(pipeline_); // Reconstruct the filtered signal on the same backend.
    } // The plan can be prepared once and executed repeatedly.
    /** @brief Return the full transform, mask, and reconstruction pipeline. */
    const shdr::pipeline& // Expose all linked stages as one operation.
    processing_pipeline() const { // Expose all linked stages as one operation.
        return pipeline_; // CPU and GPU execution use the same plan.
    } // The final transform writes to the chosen output, including aliased input storage.
    /** @brief Append the complete block filter to another processing chain. */
    void append_to( // Preserve dependencies when linking further stages.
        shdr::pipeline& destination) const { // Preserve dependencies when linking further stages.
        fft_detail::append(destination, pipeline_); // Add every dispatch in processing order.
    } // The filter object owns its kernels and must remain alive.
    /** @brief Inspect the original unshifted FFT spectrum when needed. */
    const shdr::buffer<shdr::glsl::vec2>& // Prefer a const reference for optional downloads.
    spectrum() const { // Prefer a const reference for optional downloads.
        return spectrum_; // GPU data is read back only when its data pointer is requested.
    } // Downstream GPU kernels can bind this resource directly.
    /** @brief Inspect the spectrum after applying the frequency response. */
    const shdr::buffer<shdr::glsl::vec2>& // Keep the masked values available as a shared resource.
    filtered_spectrum() const { // Keep the masked values available as a shared resource.
        return masked_; // Return the shared masked spectrum for another consumer.
    } // Reading on the host remains an explicit synchronization point.
}; // End the frequency-domain block-filter helper.
} // namespace shdr_examples
#endif // Finish the FFT helper include guard.
