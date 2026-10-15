#include "runner.hpp" // Include runner.hpp declarations.
#include <dlfcn.h> // Include dlfcn.h declarations.

namespace { // Group these helpers within their namespace.

/// Own a shader plugin until its kernel, stages and GPU program have been destroyed.
class plugin { // Declare plugin.
    void* handle_; // Declare handle for the following calculation.
    typedef shdr::kernel* (*create_fn)(); // Describe the exported factory's stable function signature.
    typedef void (*destroy_fn)(shdr::kernel*); // Describe the exported matching destructor signature.
    destroy_fn destroy_; // Destroy the ABI object using the same module that allocated it.
    shdr::kernel* kernel_; // Keep the loaded module's dual-compiled shader instance.

    template <class Function> Function symbol(const char* name) { // Accept the value types or dimensions used by this template.
        dlerror(); // Clear any earlier loader error before looking up this symbol.
        void* address = dlsym(handle_, name); // Look up the requested exported function in the loaded module.
        const char* error = dlerror(); // Check whether this symbol lookup produced a loader error.
        if (error) // Check this condition before entering the following branch.
            throw std::runtime_error(error); // Report this failed operation.
        return reinterpret_cast<Function>(address); // Use the POSIX loader result with its documented exported function type.
    } // End this block.

  public: // Set public access for the following class members.
    explicit plugin(const char* path) : handle_(0), destroy_(0), kernel_(0) { // Implement plugin for the operations below.
        handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL); // Load the module and resolve its dependencies before creating its shader.
        if (!handle_) // Check this condition before entering the following branch.
            throw std::runtime_error(dlerror()); // Report this failed operation.
        try { // Run the operation while retaining the error handling below.
            create_fn create = symbol<create_fn>("shdr_create_kernel"); // Resolve the plugin factory through its exported C entry point.
            destroy_ = symbol<destroy_fn>("shdr_destroy_kernel"); // Find the matching module destructor before taking ownership.
            kernel_ = create(); // Create the module's compiled CPU kernel and embedded GPU variant.
            if (!kernel_) // Check this condition before entering the following branch.
                throw std::runtime_error("plugin returned a null kernel"); // Report plugin returned a null kernel.
        } catch (...) { // Report the exception from the preceding operation.
            if (kernel_ && destroy_) // Check this condition before entering the following branch.
                destroy_(kernel_); // Release the module-owned kernel through its exported destructor.
            dlclose(handle_); // Release the module only after its kernel has been destroyed.
            throw; // Report this failed operation.
        } // End this block.
    } // End this block.

    ~plugin() { // Destroy the shader before unloading the code implementing its virtual methods.
        if (kernel_) // Check this condition before entering the following branch.
            destroy_(kernel_); // Release the kernel while its module code is still loaded.
        if (handle_) // Check this condition before entering the following branch.
            dlclose(handle_); // Release the module only after its kernel has been destroyed.
    } // End this block.
    shdr::kernel& kernel() { // Implement kernel for the operations below.
        return *kernel_; // Expose the same kernel interface used by statically linked stages.
    } // End this block.
    plugin(const plugin&) = delete; // Keep one owner for the loaded module and its kernel.
    plugin& operator=(const plugin&) = delete; // Prevent two wrappers from unloading the same module.
}; // Finish this declaration.

} // namespace

/// Load a CPU shader dynamically and link its corresponding GPU shader identically.
int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Select CPU execution or the requested headless GPU profile.
        plugin library(SHDR_PLUGIN_PATH); // Keep the plugin loaded longer than every pipeline stage referring to its kernel.
        const unsigned count = 193; // Use a nonmultiple of the workgroup size to exercise the shader bounds guards.
        shdr::buffer<float> input(count), scaled(count), clipped(count), result(1); // Allocate distinct resources for the input, intermediate stages and final result.
        shdr_examples::randomize(input, 1104u, 0.0f, 24.0f); // Fill the input with reproducible random values.

        // Generic name-based binding supports a kernel whose C++ class is unknown here.
        shdr::kernel& affine = library.kernel(); // Use the loaded module's dual-compiled kernel through the common base interface.
        affine.bind("input_values", input); // Read the resource produced by the preceding stage or the original host input.
        affine.bind("output_values", scaled); // Write the shared resource consumed by the next stage or the final host check.
        affine.set<unsigned>("count", count); // Limit the shader to the actual number of valid buffer elements.
        affine.set<float>("scale", 0.5f); // Configure the first stage's multiplicative gain.
        affine.set<float>("offset", -2.0f); // Configure the bias added after the first stage scales each input.

        auto clip_owner = shdr::make_kernel("clamp_kernel"); // Create the linked clamp kernel instance.
        shdr::kernel& clip = *clip_owner; // Retain the owned shader instance for every stage that references it.
        clip.bind("input_values", scaled); // Read the resource produced by the preceding stage or the original host input.
        clip.bind("output_values", clipped); // Write the shared resource consumed by the next stage or the final host check.
        clip.set<unsigned>("count", count); // Limit the shader to the actual number of valid buffer elements.
        clip.set<float>("lower", 0.0f); // Set the lower clipping bound in the second stage.
        clip.set<float>("upper", 5.0f); // Set the upper clipping bound in the second stage.

        auto sum_owner = shdr::make_kernel("sum_kernel"); // Create the linked sum kernel instance.
        shdr::kernel& sum = *sum_owner; // Retain the owned shader instance for every stage that references it.
        sum.bind("input_values", clipped); // Read the resource produced by the preceding stage or the original host input.
        sum.bind("output_values", result); // Write the shared resource consumed by the next stage or the final host check.
        sum.set<unsigned>("count", count); // Limit the shader to the actual number of valid buffer elements.

        shdr::pipeline pipeline = shdr::dispatch(affine, shdr::extent((count + 63) / 64)) >> // Start the pipeline with the affine shader and enough 64-wide groups.
                                  shdr::dispatch(clip, shdr::extent((count + 31) / 32)) >> // Link the clipping shader using its own 32-wide group size.
                                  shdr::dispatch(sum, shdr::extent(1)); // Return to the selected backend for the final one-invocation reduction.
        run.run(pipeline); // Execute the configured pipeline on the requested backend.

        float expected = 0.0f; // Start the independent scalar reference at zero.
        const shdr::buffer<float>& read_input = input; // Use a constant buffer view so GPU writes synchronize before host reads.
        for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
            expected += std::min(5.0f, std::max(0.0f, read_input.data()[i] * 0.5f - 2.0f)); // Apply the same affine and clipping formula directly to the original input.
        const shdr::buffer<float>& read_result = result; // Use a constant buffer view so GPU writes synchronize before host reads.
        shdr_examples::require_close(read_result.data()[0], expected, "dynamic shader sum"); // Check the computed value against the independent reference.
        std::cout << (run.gpu() ? "GPU" : "CPU") // Write this part of the example result or diagnostic.
                  << " dynamic shader pipeline: " << read_result.data()[0] << "\n"; // Write this part of the example result or diagnostic.
        return 0; // Report successful completion.
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
    } catch (const shdr_examples::context_unavailable& error) { // Report the exception from the preceding operation.
        std::cerr << "SKIP: " << error.what() << "\n"; // Write this part of the example result or diagnostic.
        return 77; // Report an unavailable optional GPU backend.
#endif // Finish this conditional compilation branch.
    } catch (const std::exception& error) { // Report the exception from the preceding operation.
        std::cerr << error.what() << "\n"; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
