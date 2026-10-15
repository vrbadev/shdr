#include "shdr.h" // Include shdr.h declarations.

#if defined(_WIN32) // Select the platform or build feature required by this branch.
#define SHDR_PLUGIN_EXPORT __declspec(dllexport) // Define SHDR_PLUGIN_EXPORT for the declarations that follow.
#else // Select the alternative implementation for this build.
#define SHDR_PLUGIN_EXPORT __attribute__((visibility("default"))) // Define SHDR_PLUGIN_EXPORT for the declarations that follow.
#endif // Finish this conditional compilation branch.

// The factory and destructor form this example's small C ABI.
// Both sides still use the same C++ compiler/runtime and the same shdr header.
extern "C" { // Expose predictable symbol names for the runtime module loader.
SHDR_PLUGIN_EXPORT shdr::kernel* shdr_create_kernel() { // Export a factory for the native affine shader compiled into this module.
    return shdr::make_kernel("affine_kernel").release(); // Transfer ownership of the registered kernel to the module caller.
} // End this block.

SHDR_PLUGIN_EXPORT void shdr_destroy_kernel(shdr::kernel* value) { // Export a destructor matching the factory's object allocation.
    delete value; // Destroy the complete shader object while this module is still loaded.
} // End this block.
} // End this block.
