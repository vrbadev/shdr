// Apply a symmetric low-pass, high-pass, band-pass, or band-stop FFT response.
#ifdef __cplusplus // Include the runtime only in the generated C++ build.
#include "shdr.h" // Use the common vec2 buffer representation.
#endif // Finish header inclusion for the generated C++ build.
layout(local_size_x = 64) in; // Each invocation scales one spectrum bin.
layout(std430, binding = 0) readonly buffer Input { // Read an unshifted FFT spectrum.
    vec2 input_values[]; // Read an unshifted FFT spectrum.
}; // Read an unshifted FFT spectrum.
layout(std430, binding = 1) writeonly buffer Output { // Preserve the input for other consumers.
    vec2 output_values[]; // Preserve the input for other consumers.
}; // Preserve the input for other consumers.
uniform uint block_length; // Number of bins per channel.
uniform uint channels; // Number of independent channel spectra.
uniform uint mode; // Select low-pass 0, high-pass 1, band-pass 2, or band-stop 3.
uniform float sample_rate; // Convert bin indices into hertz.
uniform float lower_hz; // Set the single cutoff or the lower edge of a band.
uniform float upper_hz; // Set the upper edge for a band filter.
uniform float transition_hz; // Use zero for hard edges or a positive cosine transition.
float low_response(float frequency, float cutoff) { // Describe the gain below one cutoff.
    if (transition_hz == 0.0) // Include the cutoff bin in a hard low-pass response.
        return frequency <= cutoff ? 1.0 // Include the cutoff bin in a hard low-pass response.
                                   : 0.0; // Include the cutoff bin in a hard low-pass response.
    float begin = cutoff - transition_hz * 0.5; // Place half the transition below the cutoff.
    float end = cutoff + transition_hz * 0.5; // Place the other half above the cutoff.
    if (frequency <= begin) // Keep bins inside the pass band.
        return 1.0; // Keep bins inside the pass band.
    if (frequency >= end) // Remove bins inside the stop band.
        return 0.0; // Remove bins inside the stop band.
    float phase = // Map the transition to one half cosine period.
        (frequency - begin) / transition_hz; // Map the transition to one half cosine period.
    return 0.5 * // Join both plateaus with a smooth gain.
           (1.0 + cos(3.14159265358979323846 * phase)); // Join both plateaus with a smooth gain.
} // The same response is used for positive and negative frequencies.
void main() { // Filtering stays on the selected compute backend.
    uint index = gl_GlobalInvocationID.x; // Select one channel and bin.
    if (index >= block_length * channels) // Ignore dispatch padding.
        return; // Ignore dispatch padding.
    uint bin = index % block_length; // Find this bin within its channel.
    float frequency = float(min(bin, block_length - bin)) * sample_rate / // Fold negative frequencies onto their magnitude.
                      float(block_length); // Fold negative frequencies onto their magnitude.
    float lower = low_response(frequency, lower_hz); // Evaluate the first cutoff.
    float gain = lower; // Start with a low-pass response.
    if (mode == 1u) // Spectral inversion produces a high-pass response.
        gain = 1.0 - lower; // Spectral inversion produces a high-pass response.
    if (mode >= 2u) { // A band filter needs two cutoff responses.
        float band = (1.0 - lower) * // Keep the frequencies between both edges.
                     low_response(frequency, upper_hz); // Keep the frequencies between both edges.
        gain = mode == 2u ? band : 1.0 - band; // Invert the band to create a band-stop response.
    } // The gain remains real so real-input conjugate symmetry is preserved.
    output_values[index] = input_values[index] * gain; // Keep phase and scale only the amplitude.
} // The inverse FFT converts this spectrum back into a signal block.
