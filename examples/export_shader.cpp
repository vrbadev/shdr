#include "shdr.h" // Include shdr.h declarations.
#include <iostream> // Include iostream declarations.
#include <stdexcept> // Include stdexcept declarations.
#include <string> // Include string declarations.

/// Export a shader as standalone GLSL for the selected standards-compliant profile.
int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        if (argc != 3) // Check this condition before entering the following branch.
            throw std::invalid_argument("usage: export_shader KERNEL PROFILE"); // Report usage: export_shader KERNEL PROFILE.
        const std::string name(argv[2]); // Construct name with these initial values.
        shdr::profile profile; // Declare profile for the following calculation.
        if (name == "es31") // Check this condition before entering the following branch.
            profile = shdr::profile::es31; // Set profile from this expression.
        else if (name == "es32") // Select the GLES 3.2 source profile.
            profile = shdr::profile::es32; // Set profile from this expression.
        else if (name == "gl43") // Select the desktop GL 4.3 source profile.
            profile = shdr::profile::gl43; // Set profile from this expression.
        else if (name == "gl45") // Select the desktop GL 4.5 source profile.
            profile = shdr::profile::gl45; // Set profile from this expression.
        else if (name == "gl46") // Select the desktop GL 4.6 source profile.
            profile = shdr::profile::gl46; // Set profile from this expression.
        else // Reject profile names outside the supported list.
            throw std::invalid_argument("unknown profile: " + name); // Report unknown profile: .
        std::cout << shdr::make_kernel(argv[1])->glsl(profile); // Create the linked shader instance.
        return 0; // Report successful completion.
    } catch (const std::exception& error) { // Report the exception from the preceding operation.
        std::cerr << error.what() << "\n"; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
