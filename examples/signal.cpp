/** @file signal.cpp
 * @brief Run FFT and time-domain signal pipelines on the CPU or a headless GPU.
 * Independent reference checks validate complex transforms and filter responses.
 */
#include "runner.hpp" // Select the backend from the example command line.
#include "fft_checks.hpp" // Check FFT, spectral masks, and input windows.
#include "signal_filters_checks.hpp" // Check causal FIR and streaming biquad filters.
int main(int argc, char** argv) { // Run every signal example with the same selected backend.
    try { // Report dimension or numerical errors as a failed example.
        shdr_examples::runner run(argc, argv); // Use CPU by default or the requested GPU profile.
        shdr_examples::check_fft1d(run); // Validate the forward and inverse complex transforms.
        shdr_examples::check_frequency_filters(run); // Validate all frequency-domain filter modes.
        shdr_examples::check_signal_windows(run); // Validate windowed spectral analysis.
        shdr_examples::check_signal_filters( // Validate FIR responses and persistent biquad state.
            run); // Validate FIR responses and persistent biquad state.
        std::cout << (run.gpu() ? "GPU" : "CPU") // Report the completed real-data processing check.
                  << " signal pipelines passed\n"; // Show which backend completed the checks.
        return 0; // All processing and reference comparisons succeeded.
#ifdef SHDR_HAVE_EGL // The optional EGL backend can identify unavailable contexts.
    } catch (const shdr_examples::context_unavailable& // Report an unavailable optional profile with the skip status.
                 error) { // Report an unavailable optional profile with the skip status.
        std::cerr << "SKIP: " << error.what() << '\n'; // Explain why this backend was unavailable.
        return 77; // Report an unavailable optional context with the CTest skip code.
#endif // Finish the optional graphics-context exception handler.
    } catch (const std::exception& error) { // Preserve all actual processing failures.
        std::cerr << error.what() << '\n'; // Print the numerical or resource diagnostic.
        return 1; // Mark the example as failed.
    } // Exception handling keeps the command-line result explicit.
} // The executable contains its native shader sources after the CMake build.
