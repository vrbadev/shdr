#include "examples/runner.hpp" // Include examples/runner.hpp declarations.
#include <atomic> // Include atomic declarations.
#include <chrono> // Include chrono declarations.
#include <cstddef> // Include cstddef declarations.
#include <thread> // Include thread declarations.

namespace { // Group these helpers within their namespace.
#if defined(_WIN32) // Select the platform or build feature required by this branch.
#define SHDR_TEST_GL_CALL __stdcall // Define SHDR_TEST_GL_CALL for the declarations that follow.
#else // Select the alternative implementation for this build.
#define SHDR_TEST_GL_CALL // Define SHDR_TEST_GL_CALL for the declarations that follow.
#endif // Finish this conditional compilation branch.

void require(bool value, const char* message) { // Check that this operation satisfies the expected behavior.
    if (!value) // Check this condition before entering the following branch.
        throw std::runtime_error(message); // Report this failed operation.
} // End this block.

template <class Function> void must_throw(Function function, const char* message) { // Accept the value types or dimensions used by this template.
    bool caught = false; // Set caught from this expression.
    try { // Run the operation while retaining the error handling below.
        function(); // Apply function to these arguments.
    } catch (const shdr::error&) { // Report the exception from the preceding operation.
        caught = true; // Set caught from this expression.
    } // End this block.
    require(caught, message); // Check that this operation satisfies the expected behavior.
} // End this block.

// Count real driver synchronization and readback calls during submission polling.
// Check zero-timeout waits and the resources retained by pending work.
struct gl_calls { // Declare gl calls.
    shdr::detail::gl_api& gl; // Declare gl for the following calculation.
    shdr::detail::gl_api::Finish_type finish; // Declare finish for the following calculation.
    shdr::detail::gl_api::MapBufferRange_type map; // Declare map for the following calculation.
    shdr::detail::gl_api::ClientWaitSync_type wait; // Declare wait for the following calculation.
    shdr::detail::gl_api::DispatchCompute_type dispatch; // Declare dispatch for the following calculation.
    std::atomic<unsigned> finishes, maps, waits, blocking_waits, dispatches; // Declare finishes for the following calculation.
    static gl_calls*& active() { // Declare gl calls for the following calculation.
        static gl_calls* value = 0; // Set value from this expression.
        return value; // Return value.
    } // End this block.
    static void SHDR_TEST_GL_CALL observe_finish() { // Declare SHDR TEST GL CALL for the following calculation.
        ++active()->finishes; // Apply active to these arguments.
        active()->finish(); // Apply active to these arguments.
    } // End this block.
    static void* SHDR_TEST_GL_CALL observe_map(shdr::detail::GLenum target, // Declare SHDR TEST GL CALL for the following calculation.
                                               shdr::detail::GLintptr offset, // Declare offset for the following calculation.
                                               shdr::detail::GLsizeiptr size, // Declare size for the following calculation.
                                               shdr::detail::GLbitfield access) { // Declare access for the following calculation.
        ++active()->maps; // Apply active to these arguments.
        return active()->map(target, offset, size, access); // Return active()->map(target, offset, size, access).
    } // End this block.
    static shdr::detail::GLenum SHDR_TEST_GL_CALL observe_wait(shdr::detail::GLsync fence, // Declare SHDR TEST GL CALL for the following calculation.
                                                               shdr::detail::GLbitfield flags, // Declare flags for the following calculation.
                                                               shdr::detail::GLuint64 timeout) { // Declare timeout for the following calculation.
        ++active()->waits; // Apply active to these arguments.
        if (timeout) // Check this condition before entering the following branch.
            ++active()->blocking_waits; // Apply active to these arguments.
        return active()->wait(fence, flags, timeout); // Return active()->wait(fence, flags, timeout).
    } // End this block.
    static void SHDR_TEST_GL_CALL observe_dispatch(shdr::detail::GLuint x, // Declare SHDR TEST GL CALL for the following calculation.
                                                   shdr::detail::GLuint y, // Declare y for the following calculation.
                                                   shdr::detail::GLuint z) { // Declare z for the following calculation.
        ++active()->dispatches; // Apply active to these arguments.
        active()->dispatch(x, y, z); // Apply active to these arguments.
    } // End this block.
    explicit gl_calls(shdr::device& device) // Declare gl calls with the parameters continued below.
        : gl(device.state()->gl), finish(gl.Finish), map(gl.MapBufferRange), // Declare gl with the parameters continued below.
          wait(gl.ClientWaitSync), dispatch(gl.DispatchCompute), finishes(0), maps(0), waits(0), // Apply wait to these arguments.
          blocking_waits(0), dispatches(0) { // Apply blocking waits to these arguments.
        require(active() == 0, "nested driver observer"); // Check that this operation satisfies the expected behavior.
        active() = this; // Apply active to these arguments.
        gl.Finish = observe_finish; // Set gl.Finish from this expression.
        gl.MapBufferRange = observe_map; // Set gl.Map Buffer Range from this expression.
        gl.ClientWaitSync = observe_wait; // Set gl.Client Wait Sync from this expression.
        gl.DispatchCompute = observe_dispatch; // Set gl.Dispatch Compute from this expression.
    } // End this block.
    ~gl_calls() { // Apply gl calls to these arguments.
        gl.Finish = finish; // Set gl.Finish from this expression.
        gl.MapBufferRange = map; // Set gl.Map Buffer Range from this expression.
        gl.ClientWaitSync = wait; // Set gl.Client Wait Sync from this expression.
        gl.DispatchCompute = dispatch; // Set gl.Dispatch Compute from this expression.
        active() = 0; // Apply active to these arguments.
    } // End this block.
}; // Finish this declaration.
#undef SHDR_TEST_GL_CALL // Continue the expression using undef, SHDR TEST GL CALL.

template <class Function> Function resolve(const char* name) { // Accept the value types or dimensions used by this template.
    void* address = shdr_examples::egl_context::resolve(name); // Set address from this expression.
    if (!address) // Check this condition before entering the following branch.
        throw std::runtime_error(std::string("missing GL entry point ") + name); // Report missing GL entry point .
    return reinterpret_cast<Function>(address); // Return reinterpret cast<Function>(address).
} // End this block.

/// An application's independently owned SSBO, borrowed temporarily by shdr.
class raw_buffer { // Declare raw buffer.
    typedef void (*gen_fn)(int, unsigned*); // Construct void with these initial values.
    typedef void (*bind_fn)(unsigned, unsigned); // Construct void with these initial values.
    typedef void (*data_fn)(unsigned, std::ptrdiff_t, const void*, unsigned); // Construct void with these initial values.
    typedef void (*delete_fn)(int, const unsigned*); // Construct void with these initial values.
    unsigned name_; // Declare name for the following calculation.
    delete_fn destroy_; // Declare destroy for the following calculation.

  public: // Set public access for the following class members.
    raw_buffer(const float* values, std::size_t count) // Apply raw buffer to these arguments.
        : name_(0), destroy_(resolve<delete_fn>("glDeleteBuffers")) { // Implement name for the operations below.
        resolve<gen_fn>("glGenBuffers")(1, &name_); // Provide this text argument for the operation above.
        resolve<bind_fn>("glBindBuffer")(0x90D2, name_); // GL_SHADER_STORAGE_BUFFER
        resolve<data_fn>("glBufferData")(0x90D2, // Provide this text argument for the operation above.
                                         std::ptrdiff_t(count * sizeof(float)), // Apply ptrdiff t to these arguments.
                                         values, // Continue the expression using values.
                                         0x88E8); // GL_DYNAMIC_DRAW
        typedef unsigned (*error_fn)(); // Construct unsigned with these initial values.
        if (resolve<error_fn>("glGetError")() != 0) { // Check this condition before entering the following branch.
            destroy_(1, &name_); // Apply destroy to these arguments.
            throw std::runtime_error("failed to allocate application's SSBO"); // Report failed to allocate application's SSBO.
        } // End this block.
    } // End this block.
    ~raw_buffer() { // Apply raw buffer to these arguments.
        destroy_(1, &name_); // Apply destroy to these arguments.
    } // End this block.
    unsigned name() const { // Implement name for the operations below.
        return name_; // Return name.
    } // End this block.
    void set_first(float value) { // Implement set first for the operations below.
        typedef void (*subdata_fn)(unsigned, std::ptrdiff_t, std::ptrdiff_t, const void*); // Construct void with these initial values.
        resolve<bind_fn>("glBindBuffer")(0x90D2, name_); // Provide this text argument for the operation above.
        resolve<subdata_fn>("glBufferSubData")(0x90D2, 0, sizeof(float), &value); // Apply sizeof to these arguments.
    } // End this block.
    raw_buffer(const raw_buffer&) = delete; // Apply raw buffer to these arguments.
    raw_buffer& operator=(const raw_buffer&) = delete; // Set operator from this expression.
}; // Finish this declaration.

void test_external_buffers(shdr::device& device) { // Implement test external buffers for the operations below.
    const unsigned count = 33; // Set count from this expression.
    float values[count], zero[count] = {}; // Set zero[count] from this expression.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        values[i] = float(i) - 8.0f; // Set values[i] from this expression.
    raw_buffer application_input(values, count), application_output(zero, count); // Construct application input with these initial values.
    { // Begin this local scope.
        shdr::buffer<float> input = // Set input from this expression.
            shdr::buffer<float>::from_gl(device, application_input.name(), count); // Apply from gl to these arguments.
        shdr::buffer<float> output = // Set output from this expression.
            shdr::buffer<float>::from_gl(device, application_output.name(), count); // Apply from gl to these arguments.
        if (input.gl_handle(device) != application_input.name() || // Check this condition before entering the following branch.
            output.gl_handle(device) != application_output.name()) // Apply gl handle to these arguments.
            throw std::runtime_error("borrowing changed application's GL handles"); // Report borrowing changed application's GL handles.
        auto affine_owner = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
        shdr::kernel& affine = *affine_owner; // Set affine from this expression.
        affine.bind("input_values", input); // Connect this shader port to its application resource.
        affine.bind("output_values", output); // Connect this shader port to its application resource.
        affine.set<unsigned>("count", count); // Supply this shader uniform with its typed application value.
        affine.set<float>("scale", 2.0f); // Supply this shader uniform with its typed application value.
        affine.set<float>("offset", 1.0f); // Supply this shader uniform with its typed application value.

        shdr::pipeline pipeline; // Declare pipeline for the following calculation.
        pipeline += shdr::dispatch(affine, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
        pipeline.run(device); // Execute the configured pipeline on the requested backend.
        const shdr::buffer<float>& read_output = output; // Use a constant buffer view so GPU writes synchronize before host reads.
        for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
            shdr_examples::require_close( // Check the computed value against the independent reference.
                read_output.data()[i], values[i] * 2 + 1, "external GPU buffer"); // Apply data to these arguments.

        input.data()[0] = 123.0f; // Apply data to these arguments.
        pipeline.run(device); // Execute the configured pipeline on the requested backend.
        shdr_examples::require_close( // Check the computed value against the independent reference.
            read_output.data()[0], 247.0f, "host edit uploaded to external buffer"); // Apply data to these arguments.
        // The CPU version can use those same externally owned resources dynamically.
        affine.set<float>("scale", 3.0f); // Supply this shader uniform with its typed application value.
        pipeline.run(); // Execute the configured pipeline on the requested backend.
        shdr_examples::require_close( // Check the computed value against the independent reference.
            read_output.data()[0], 370.0f, "CPU dispatch using borrowed GL resource"); // Apply data to these arguments.
        output.upload(device); // Apply upload to these arguments.
        application_output.set_first(321.0f); // Apply set first to these arguments.
        output.mark_gpu_modified(device); // Apply mark gpu modified to these arguments.
        shdr_examples::require_close(read_output.data()[0], 321.0f, "application GL edit readback"); // Check the computed value against the independent reference.

        float* retained = input.data(); // Set retained from this expression.
        input.upload(device); // Apply upload to these arguments.
        retained[1] = 80.0f; // Set retained[1] from this expression.
        input.mark_host_modified(); // Apply mark host modified to these arguments.
        pipeline.run(device); // Execute the configured pipeline on the requested backend.
        shdr_examples::require_close( // Check the computed value against the independent reference.
            read_output.data()[1], 241.0f, "explicit retained host pointer dirty mark"); // Apply data to these arguments.
    } // End this block.
    typedef unsigned char (*is_buffer_fn)(unsigned); // Declare unsigned for the following calculation.
    is_buffer_fn is_buffer = resolve<is_buffer_fn>("glIsBuffer"); // Set is buffer from this expression.
    if (!is_buffer(application_input.name()) || !is_buffer(application_output.name())) // Check this condition before entering the following branch.
        throw std::runtime_error("shdr destroyed an application's borrowed SSBO"); // Report shdr destroyed an application's borrowed SSBO.
} // End this block.

void test_shader_errors(shdr::device& device) { // Implement test shader errors for the operations below.
    // Dedicated test subclasses supply malformed programs for compile-failure checks.
    struct missing_kernel : shdr::kernel { // Declare missing kernel.
        missing_kernel() : shdr::kernel("missing", shdr::extent(1)) { // Apply missing kernel to these arguments.
        } // End this block.
        void main() override { // Run this example and validate its results.
        } // End this block.
    } missing; // Continue the expression using missing.
    struct broken_kernel : shdr::kernel { // Declare broken kernel.
        broken_kernel() : shdr::kernel("broken", shdr::extent(1)) { // Apply broken kernel to these arguments.
            set_embedded_source("layout(local_size_x=1) in; void main(){invalid_glsl;}\n"); // Apply set embedded source to these arguments.
        } // End this block.
        void main() override { // Run this example and validate its results.
        } // End this block.
    } broken; // Continue the expression using broken.
    shdr::pipeline pipeline; // Declare pipeline for the following calculation.
    pipeline += shdr::dispatch(missing, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
    bool failed = false; // Set failed from this expression.
    try { // Run the operation while retaining the error handling below.
        pipeline.run(device); // Execute the configured pipeline on the requested backend.
    } catch (const shdr::error&) { // Report the exception from the preceding operation.
        failed = true; // Set failed from this expression.
    } // End this block.
    if (!failed) // Check this condition before entering the following branch.
        throw std::runtime_error("missing GPU shader source was accepted"); // Report missing GPU shader source was accepted.
    shdr::pipeline invalid; // Declare invalid for the following calculation.
    invalid += shdr::dispatch(broken, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
    failed = false; // Set failed from this expression.
    try { // Run the operation while retaining the error handling below.
        invalid.run(device); // Execute the configured pipeline on the requested backend.
    } catch (const shdr::error&) { // Report the exception from the preceding operation.
        failed = true; // Set failed from this expression.
    } // End this block.
    if (!failed) // Check this condition before entering the following branch.
        throw std::runtime_error("invalid GPU shader source was accepted"); // Report invalid GPU shader source was accepted.

    shdr::buffer<float> input(4), output(4); // Construct input with these initial values.
    auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shader->bind("input_values", input) // Connect this shader port to its application resource.
        .bind("output_values", output) // Connect this shader port to its application resource.
        .set<unsigned>("count", 4u) // Supply this shader uniform with its typed application value.
        .set<float>("scale", 1.0f) // Supply this shader uniform with its typed application value.
        .set<float>("offset", 2.0f); // Supply this shader uniform with its typed application value.
    shdr::pipeline recovery; // Declare recovery for the following calculation.
    recovery += shdr::dispatch(*shader, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
    recovery.run(device); // Verify successful dispatch on the same device after the compile failure.
    const shdr::buffer<float>& read_output = output; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned i = 0; i < 4; ++i) // Iterate over i for this calculation.
        shdr_examples::require_close( // Check the computed value against the independent reference.
            read_output.data()[i], 2.0f, "shader recovery after compile error"); // Apply data to these arguments.
} // End this block.

void test_typed_images(shdr::device& device) { // Implement test typed images for the operations below.
    const unsigned width = 5, height = 3, pixels = width * height; // Set width from this expression.
    shdr::image2d<float, 1> f(width, height); // Construct f with these initial values.
    shdr::image2d<int, 1> i(width, height); // Construct i with these initial values.
    shdr::image2d<unsigned, 1> u(width, height); // Construct u with these initial values.
    shdr::image2d<float, 4> f4(width, height); // Construct f4 with these initial values.
    shdr::image2d<int, 4> i4(width, height); // Construct i4 with these initial values.
    shdr::image2d<unsigned, 4> u4(width, height); // Construct u4 with these initial values.
    for (unsigned p = 0; p < pixels; ++p) { // Iterate over p for this calculation.
        f.data()[p] = float(p); // Apply data to these arguments.
        i.data()[p] = int(p); // Apply data to these arguments.
        u.data()[p] = p; // Apply data to these arguments.
        for (unsigned c = 0; c < 4; ++c) { // Iterate over c for this calculation.
            f4.data()[4 * p + c] = float(4 * p + c); // Apply data to these arguments.
            i4.data()[4 * p + c] = int(4 * p + c); // Apply data to these arguments.
            u4.data()[4 * p + c] = 4 * p + c; // Apply data to these arguments.
        } // End this block.
    } // End this block.
    // Borrow a texture owned by another resource wrapper to exercise from_gl.
    shdr::image2d<float, 1> borrowed_f = // Set borrowed f from this expression.
        shdr::image2d<float, 1>::from_gl(device, f.gl_handle(device), width, height); // Apply from gl to these arguments.
    auto scalar_owner = shdr::make_kernel("scalar_images_kernel"); // Create the linked scalar images kernel instance.
    shdr::kernel& scalar = *scalar_owner; // Set scalar from this expression.
    scalar.bind("float_scalar", borrowed_f); // Connect this shader port to its application resource.
    scalar.bind("int_scalar", i); // Connect this shader port to its application resource.
    scalar.bind("uint_scalar", u); // Connect this shader port to its application resource.

    // Input and output may alias here because each invocation handles its own pixel.
    auto rgba_f_owner = shdr::make_kernel("rgba_float_image_kernel"); // Create the linked rgba float image kernel instance.
    shdr::kernel& rgba_f = *rgba_f_owner; // Set rgba f from this expression.
    rgba_f.bind("input_image", f4); // Connect this shader port to its application resource.
    rgba_f.bind("output_image", f4); // Connect this shader port to its application resource.

    auto rgba_i_owner = shdr::make_kernel("rgba_int_image_kernel"); // Create the linked rgba int image kernel instance.
    shdr::kernel& rgba_i = *rgba_i_owner; // Set rgba i from this expression.
    rgba_i.bind("input_image", i4); // Connect this shader port to its application resource.
    rgba_i.bind("output_image", i4); // Connect this shader port to its application resource.

    auto rgba_u_owner = shdr::make_kernel("rgba_uint_image_kernel"); // Create the linked rgba uint image kernel instance.
    shdr::kernel& rgba_u = *rgba_u_owner; // Set rgba u from this expression.
    rgba_u.bind("input_image", u4); // Connect this shader port to its application resource.
    rgba_u.bind("output_image", u4); // Connect this shader port to its application resource.

    shdr::pipeline pipeline = // Set pipeline from this expression.
        shdr::dispatch(scalar, shdr::extent(2, 1)) >> shdr::dispatch(rgba_f, shdr::extent(2, 1)) >> // Add a shader dispatch with the required workgroup counts.
        shdr::dispatch(rgba_i, shdr::extent(2, 1)) >> shdr::dispatch(rgba_u, shdr::extent(2, 1)); // Add a shader dispatch with the required workgroup counts.
    pipeline.run(device); // Execute the configured pipeline on the requested backend.
    f.mark_gpu_modified(device); // The application informs the original wrapper of alias writes.
    const shdr::image2d<float, 1>& rf = f; // Set rf from this expression.
    const shdr::image2d<int, 1>& ri = i; // Set ri from this expression.
    const shdr::image2d<unsigned, 1>& ru = u; // Set ru from this expression.
    const shdr::image2d<float, 4>& rf4 = f4; // Set rf4 from this expression.
    const shdr::image2d<int, 4>& ri4 = i4; // Set ri4 from this expression.
    const shdr::image2d<unsigned, 4>& ru4 = u4; // Set ru4 from this expression.
    for (unsigned p = 0; p < pixels; ++p) { // Iterate over p for this calculation.
        shdr_examples::require_close(rf.data()[p], float(p) + 0.5f, "r32f image readback"); // Check the computed value against the independent reference.
        if (ri.data()[p] != int(p) - 2 || ru.data()[p] != p + 3) // Check this condition before entering the following branch.
            throw std::runtime_error("integer scalar image readback"); // Report integer scalar image readback.
        for (unsigned c = 0; c < 4; ++c) { // Iterate over c for this calculation.
            shdr_examples::require_close(rf4.data()[4 * p + c], // Check the computed value against the independent reference.
                                         float(4 * p + c) + float(c + 1) / 4, // Apply float to these arguments.
                                         "rgba32f image readback"); // Provide this text argument for the operation above.
            if (ri4.data()[4 * p + c] != int(4 * p + c) - int(c + 1) || // Check this condition before entering the following branch.
                ru4.data()[4 * p + c] != 4 * p + c + c + 1) // Apply data to these arguments.
                throw std::runtime_error("integer RGBA image readback"); // Report integer RGBA image readback.
        } // End this block.
    } // End this block.
} // End this block.

/// Resource ownership retains a headless context after its device wrapper is released.
void test_headless_context_lifetimes(shdr::device& original) { // Implement test headless context lifetimes for the operations below.
    std::unique_ptr<shdr::headless_device> first( // Declare first with the parameters continued below.
        new shdr::headless_device(original.shader_profile())); // Declare shdr for the following calculation.
    std::unique_ptr<shdr::buffer<float>> retained(new shdr::buffer<float>{3.0f, 7.0f}); // Construct retained with these initial values.
    retained->upload(*first); // Apply upload to these arguments.
    retained->mark_gpu_modified(*first); // Apply mark gpu modified to these arguments.

    shdr::headless_device second(original.shader_profile()); // Construct second with these initial values.
    shdr::buffer<float> input{2.0f, 4.0f}, output(2); // Declare input for the following calculation.
    auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shader->bind("input_values", input) // Connect this shader port to its application resource.
        .bind("output_values", output) // Connect this shader port to its application resource.
        .set<unsigned>("count", 2u) // Supply this shader uniform with its typed application value.
        .set<float>("scale", 3.0f) // Supply this shader uniform with its typed application value.
        .set<float>("offset", 1.0f); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline = shdr::dispatch(*shader, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
    pipeline.run(second); // Execute the configured pipeline on the requested backend.
    first.reset(); // Apply reset to these arguments.

    const shdr::buffer<float>& read_retained = *retained; // Use a constant buffer view so GPU writes synchronize before host reads.
    shdr_examples::require_close(read_retained.data()[0], 3.0f, "retained context readback"); // Check the computed value against the independent reference.
    shdr_examples::require_close(read_retained.data()[1], 7.0f, "retained context readback"); // Check the computed value against the independent reference.
    retained.reset(); // Apply reset to these arguments.
    second.synchronize(); // Wait for submitted GPU work before continuing.
    if (second.renderer().empty()) // Check this condition before entering the following branch.
        throw std::runtime_error("second headless context lost its renderer"); // Report second headless context lost its renderer.
    pipeline.run(second); // Execute the configured pipeline on the requested backend.
    const shdr::buffer<float>& read_output = output; // Use a constant buffer view so GPU writes synchronize before host reads.
    shdr_examples::require_close(read_output.data()[0], 7.0f, "second context remains usable"); // Check the computed value against the independent reference.
    shdr_examples::require_close(read_output.data()[1], 13.0f, "second context remains usable"); // Check the computed value against the independent reference.
    original.synchronize(); // Wait for submitted GPU work before continuing.
} // End this block.

void configure_affine(shdr::kernel& shader, // Declare configure affine with the parameters continued below.
                      const shdr::buffer<float>& input, // Declare input for the following calculation.
                      const shdr::buffer<float>& output, // Declare output for the following calculation.
                      unsigned count) { // Declare count for the following calculation.
    shader.bind("input_values", input) // Connect this shader port to its application resource.
        .bind("output_values", output) // Connect this shader port to its application resource.
        .set<unsigned>("count", count) // Supply this shader uniform with its typed application value.
        .set<float>("scale", 2.0f) // Supply this shader uniform with its typed application value.
        .set<float>("offset", 1.0f); // Supply this shader uniform with its typed application value.
} // End this block.

void test_gpu_submissions(shdr::device& device) { // Implement test gpu submissions for the operations below.
    const unsigned count = 257; // Set count from this expression.
    shdr::buffer<float> input(count), output(count); // Construct input with these initial values.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        input.data()[i] = float(i); // Apply data to these arguments.
    auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    configure_affine(*shader, input, output, count); // Apply configure affine to these arguments.
    shdr::pipeline pipeline(shdr::dispatch(*shader, shdr::extent(5))); // Add a shader dispatch with the required workgroup counts.
    pipeline.prepare(device); // Apply prepare to these arguments.
    gl_calls observed(device); // Construct observed with these initial values.
    shdr::submission submitted = pipeline.submit(device); // Submit work and retain its completion handle.
    require(submitted.valid(), "nonempty GPU submission has no completion handle"); // Check that this operation satisfies the expected behavior.
    require(observed.dispatches == 1 && observed.finishes == 0 && observed.maps == 0, // Check that this operation satisfies the expected behavior.
            "GPU submission performed an implicit finish or readback"); // Provide this text argument for the operation above.
    const shdr::wait_status zero = submitted.wait(std::chrono::nanoseconds(0)); // Set zero from this expression.
    require(zero == shdr::wait_status::ready || zero == shdr::wait_status::timeout, // Check that this operation satisfies the expected behavior.
            "zero timeout status"); // Provide this text argument for the operation above.
    require(observed.blocking_waits == 0, "zero timeout performed a blocking driver wait"); // Check that this operation satisfies the expected behavior.
    must_throw( // Check that this operation satisfies the expected behavior.
        [&] { // Complete this expression or declaration.
            submitted.wait(std::chrono::nanoseconds(-1)); // Apply wait to these arguments.
        }, // Finish this declaration.
        "negative timeout accepted"); // Provide this text argument for the operation above.
    const unsigned queries = observed.waits; // Set queries from this expression.
    std::atomic<bool> wrong_thread_rejected(false); // Construct wrong thread rejected with these initial values.
    std::thread wrong_thread([&] { // Implement wrong thread for the operations below.
        try { // Run the operation while retaining the error handling below.
            submitted.ready(); // Apply ready to these arguments.
        } catch (const shdr::error&) { // Report the exception from the preceding operation.
            wrong_thread_rejected = true; // Set wrong thread rejected from this expression.
        } // End this block.
    }); // Finish this declaration.
    wrong_thread.join(); // Apply join to these arguments.
    require(wrong_thread_rejected && observed.waits == queries, // Check that this operation satisfies the expected behavior.
            "completion query used GL on the wrong thread"); // Provide this text argument for the operation above.

    shdr::executor workers(1, 2); // Construct workers with these initial values.
    must_throw( // Check that this operation satisfies the expected behavior.
        [&] { // Complete this expression or declaration.
            workers.submit(pipeline); // Submit work and retain its completion handle.
        }, // Finish this declaration.
        "CPU executor accepted newer GPU data"); // Provide this text argument for the operation above.
    require(observed.maps == 0, "CPU submission rejection downloaded GPU data"); // Check that this operation satisfies the expected behavior.
    shdr::pipeline mixed = shdr::dispatch(*shader, shdr::extent(5)) >> // Add a shader dispatch with the required workgroup counts.
                           shdr::dispatch(*shader, shdr::extent(5), shdr::backend::cpu); // Add a shader dispatch with the required workgroup counts.
    must_throw( // Check that this operation satisfies the expected behavior.
        [&] { // Complete this expression or declaration.
            mixed.submit(device); // Submit work and retain its completion handle.
        }, // Finish this declaration.
        "GPU submission accepted a CPU stage"); // Provide this text argument for the operation above.
    require(observed.dispatches == 1, "mixed submission dispatched before rejecting its CPU stage"); // Check that this operation satisfies the expected behavior.

    shdr::submission moved = std::move(submitted); // Set moved from this expression.
    require(!submitted.valid() && submitted.ready() && moved.valid(), // Check that this operation satisfies the expected behavior.
            "GPU completion move semantics"); // Provide this text argument for the operation above.
    const shdr::wait_status bounded = moved.wait(std::chrono::milliseconds(5)); // Set bounded from this expression.
    require(bounded == shdr::wait_status::ready || bounded == shdr::wait_status::timeout, // Check that this operation satisfies the expected behavior.
            "bounded completion wait status"); // Provide this text argument for the operation above.
    moved.wait(); // Blocking is explicit at this call.
    const shdr::buffer<float>& result = output; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        shdr_examples::require_close(result.data()[i], float(i) * 2 + 1, "async GPU result"); // Check the computed value against the independent reference.
    workers.submit(pipeline).get(); // Host residency permits a CPU snapshot after readback.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        shdr_examples::require_close( // Check the computed value against the independent reference.
            result.data()[i], float(i) * 2 + 1, "async CPU after explicit readback"); // Apply data to these arguments.
    shdr::submission empty = shdr::pipeline().submit(device); // Submit work and retain its completion handle.
    require(!empty.valid() && empty.ready(), "empty pipeline issued GPU work"); // Check that this operation satisfies the expected behavior.
} // End this block.

void test_dropped_submissions(shdr::device& device) { // Implement test dropped submissions for the operations below.
    std::weak_ptr<shdr::detail::resource_state> input_lifetime, output_lifetime; // Declare input lifetime for the following calculation.
    std::weak_ptr<shdr::detail::program_state> program_lifetime; // Declare program lifetime for the following calculation.
    shdr::submission completion; // Declare completion for the following calculation.
    gl_calls observed(device); // Construct observed with these initial values.
    { // Begin this local scope.
        shdr::buffer<float> input{2.0f, 4.0f}, output(2); // Declare input for the following calculation.
        auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
        configure_affine(*shader, input, output, 2); // Apply configure affine to these arguments.
        shdr::pipeline pipeline(shdr::dispatch(*shader, shdr::extent(1))); // Add a shader dispatch with the required workgroup counts.
        completion = pipeline.submit(device); // Submit work and retain its completion handle.
        input_lifetime = input.state(); // Set input lifetime from this expression.
        output_lifetime = output.state(); // Set output lifetime from this expression.
        program_lifetime = shader->program_state(); // Set program lifetime from this expression.
    } // End this block.
    require(!input_lifetime.expired() && !output_lifetime.expired() && !program_lifetime.expired(), // Check that this operation satisfies the expected behavior.
            "GPU submission did not retain its resources and program"); // Provide this text argument for the operation above.
    const unsigned queries = observed.waits; // Set queries from this expression.
    std::thread discard( // Declare discard with the parameters continued below.
        [](shdr::submission dropped) { // Continue the expression using shdr, submission, dropped.
            (void)dropped; // Continue the expression using void, dropped.
        }, // Finish this declaration.
        std::move(completion)); // Apply move to these arguments.
    discard.join(); // Apply join to these arguments.
    require(observed.finishes == 0 && observed.maps == 0 && observed.waits == queries, // Check that this operation satisfies the expected behavior.
            "discarding a GPU submission called GL or waited on the worker thread"); // Provide this text argument for the operation above.
    require(!input_lifetime.expired() && !output_lifetime.expired() && !program_lifetime.expired(), // Check that this operation satisfies the expected behavior.
            "discarding a pending submission released resources before retirement"); // Provide this text argument for the operation above.
    const std::chrono::steady_clock::time_point deadline = // Set deadline from this expression.
        std::chrono::steady_clock::now() + std::chrono::seconds(10); // Apply now to these arguments.
    while (!program_lifetime.expired() && std::chrono::steady_clock::now() < deadline) { // Poll deferred retirement until retained program ownership is released or the deadline arrives.
        device.collect_garbage(); // Apply collect garbage to these arguments.
        std::this_thread::yield(); // Apply yield to these arguments.
    } // End this block.
    require(input_lifetime.expired() && output_lifetime.expired() && program_lifetime.expired(), // Check that this operation satisfies the expected behavior.
            "completed discarded submission retained resources indefinitely"); // Provide this text argument for the operation above.
    require(observed.finishes == 0 && observed.maps == 0 && observed.blocking_waits == 0, // Check that this operation satisfies the expected behavior.
            "deferred GPU retirement performed a blocking wait or readback"); // Provide this text argument for the operation above.
} // End this block.

void test_submission_context_lifetimes(shdr::device& original) { // Implement test submission context lifetimes for the operations below.
    for (unsigned order = 0; order < 2; ++order) { // Iterate over order for this calculation.
        std::unique_ptr<shdr::headless_device> device( // Declare device with the parameters continued below.
            new shdr::headless_device(original.shader_profile())); // Declare shdr for the following calculation.
        std::weak_ptr<shdr::detail::device_state> device_lifetime = device->state(); // Set device lifetime from this expression.
        std::weak_ptr<shdr::detail::resource_state> resource_lifetime; // Declare resource lifetime for the following calculation.
        std::weak_ptr<shdr::detail::program_state> program_lifetime; // Declare program lifetime for the following calculation.
        shdr::submission completion; // Declare completion for the following calculation.
        { // Begin this local scope.
            shdr::buffer<float> input{3.0f, 7.0f}, output(2); // Declare input for the following calculation.
            auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
            configure_affine(*shader, input, output, 2); // Apply configure affine to these arguments.
            shdr::pipeline pipeline(shdr::dispatch(*shader, shdr::extent(1))); // Add a shader dispatch with the required workgroup counts.
            completion = pipeline.submit(*device); // Submit work and retain its completion handle.
            resource_lifetime = output.state(); // Set resource lifetime from this expression.
            program_lifetime = shader->program_state(); // Set program lifetime from this expression.
        } // End this block.
        if (order == 0) { // Check this condition before entering the following branch.
            completion = shdr::submission(); // Set completion from this expression.
            device.reset(); // Apply reset to these arguments.
        } else { // Destroy the device wrapper first while the live submission retains its context.
            device.reset(); // Apply reset to these arguments.
            require(!device_lifetime.expired(), "live submission lost its owned context"); // Check that this operation satisfies the expected behavior.
            completion = shdr::submission(); // Set completion from this expression.
        } // End this block.
        require(resource_lifetime.expired() && program_lifetime.expired() && // Check that this operation satisfies the expected behavior.
                    device_lifetime.expired(), // Apply expired to these arguments.
                "device and submission destruction formed a resource ownership cycle"); // Provide this text argument for the operation above.
    } // End this block.
    original.collect_garbage(); // Apply collect garbage to these arguments.
} // End this block.

} // namespace

int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Construct run with these initial values.
        if (!run.device()) // Check this condition before entering the following branch.
            throw std::invalid_argument("test_gpu requires --gpu PROFILE"); // Report test_gpu requires --gpu PROFILE.
        test_external_buffers(*run.device()); // Apply test external buffers to these arguments.
        test_shader_errors(*run.device()); // Apply test shader errors to these arguments.
        test_typed_images(*run.device()); // Apply test typed images to these arguments.
        test_headless_context_lifetimes(*run.device()); // Apply test headless context lifetimes to these arguments.
        test_gpu_submissions(*run.device()); // Apply test gpu submissions to these arguments.
        test_dropped_submissions(*run.device()); // Apply test dropped submissions to these arguments.
        test_submission_context_lifetimes(*run.device()); // Apply test submission context lifetimes to these arguments.
        std::cout << "GPU ownership, typed images and async completion tests passed\n"; // Write this part of the example result or diagnostic.
        return 0; // Report successful completion.
    } catch (const shdr_examples::context_unavailable& error) { // Report the exception from the preceding operation.
        std::cerr << "SKIP: " << error.what() << "\n"; // Write this part of the example result or diagnostic.
        return 77; // Report an unavailable optional GPU backend.
    } catch (const std::exception& error) { // Report the exception from the preceding operation.
        std::cerr << error.what() << "\n"; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
