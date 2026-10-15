#include "shdr.h" // Include shdr.h declarations.
#include <cmath> // Include cmath declarations.
#include <iostream> // Include iostream declarations.
#include <stdexcept> // Include stdexcept declarations.
#include <string> // Include string declarations.

// Write distinct 3D coordinates and validate every compute invocation builtin.
SHDR_BEGIN(coordinates_kernel, 2, 3, 2) // Apply SHDR BEGIN to these arguments.
SHDR_BUFFER(0, uint, coordinates, writeonly) // Apply SHDR BUFFER to these arguments.
SHDR_BUFFER(1, uint, valid, writeonly) // Apply SHDR BUFFER to these arguments.
SHDR_MAIN() { // Apply SHDR MAIN to these arguments.
    uint width = gl_NumWorkGroups.x * gl_WorkGroupSize.x; // Set width from this expression.
    uint height = gl_NumWorkGroups.y * gl_WorkGroupSize.y; // Set height from this expression.
    uint index = gl_GlobalInvocationID.x + // Set index from this expression.
                 width * (gl_GlobalInvocationID.y + height * gl_GlobalInvocationID.z); // Continue the expression using width, gl Global Invocation ID, y.
    coordinates[index] = // Set coordinates[index] from this expression.
        gl_GlobalInvocationID.x + 100u * gl_GlobalInvocationID.y + 10000u * gl_GlobalInvocationID.z; // Continue the expression using gl Global Invocation ID, x, gl Global Invocation ID.
    uint local_index = // Set local index from this expression.
        gl_LocalInvocationID.x + // Continue the expression using gl Local Invocation ID, x.
        gl_WorkGroupSize.x * (gl_LocalInvocationID.y + gl_WorkGroupSize.y * gl_LocalInvocationID.z); // Continue the expression using gl Work Group Size, x, gl Local Invocation ID.
    valid[index] = uint( // Set valid[index] from this expression.
        gl_LocalInvocationIndex == local_index && // Continue the expression using gl Local Invocation Index, local index.
        gl_GlobalInvocationID.x == gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x && // Continue the expression using gl Global Invocation ID, x, gl Work Group ID.
        gl_GlobalInvocationID.y == gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y && // Continue the expression using gl Global Invocation ID, y, gl Work Group ID.
        gl_GlobalInvocationID.z == gl_WorkGroupID.z * gl_WorkGroupSize.z + gl_LocalInvocationID.z); // Continue the expression using gl Global Invocation ID, z, gl Work Group ID.
} // End this block.
SHDR_END() // Apply SHDR END to these arguments.

namespace { // Group these helpers within their namespace.

void require(bool condition, const std::string& message) { // Check that this operation satisfies the expected behavior.
    if (!condition) // Check this condition before entering the following branch.
        throw std::runtime_error(message); // Report this failed operation.
} // End this block.

void near(float actual, float expected, const std::string& message) { // Implement near for the operations below.
    require(std::isfinite(actual) && std::fabs(actual - expected) < 0.0001f, message); // Check that this operation satisfies the expected behavior.
} // End this block.

template <class Function> void throws(Function function, const std::string& message) { // Accept the value types or dimensions used by this template.
    bool caught = false; // Set caught from this expression.
    try { // Run the operation while retaining the error handling below.
        function(); // Apply function to these arguments.
    } catch (const std::exception&) { // Report the exception from the preceding operation.
        caught = true; // Set caught from this expression.
    } // End this block.
    require(caught, message); // Check that this operation satisfies the expected behavior.
} // End this block.

void test_borrowed_rebinding_and_repeat() { // Implement test borrowed rebinding and repeat for the operations below.
    float first_values[] = {1, 2, 3, 4, 5}; // Declare first values for the following calculation.
    float second_values[] = {10, 20, 30, 40, 50}; // Declare second values for the following calculation.
    float output_values[5] = {}; // Set output values[5] from this expression.
    shdr::buffer<float> first(first_values, 5), second(second_values, 5), output(output_values, 5); // Construct first with these initial values.
    auto affine_owner = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shdr::kernel& affine = *affine_owner; // Set affine from this expression.
    affine.bind("input_values", first); // Connect this shader port to its application resource.
    affine.bind("output_values", output); // Connect this shader port to its application resource.
    affine.set<unsigned>("count", 5u); // Supply this shader uniform with its typed application value.
    affine.set<float>("scale", 2.0f); // Supply this shader uniform with its typed application value.
    affine.set<float>("offset", 3.0f); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline; // Declare pipeline for the following calculation.
    pipeline += shdr::dispatch(affine, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
    pipeline.run(); // Execute the configured pipeline on the requested backend.
    for (unsigned i = 0; i < 5; ++i) // Iterate over i for this calculation.
        near(output_values[i], first_values[i] * 2 + 3, "borrowed host output"); // Apply near to these arguments.

    // Ports and uniforms remain dynamically bindable after pipeline construction.
    affine.bind("input_values", second); // Connect this shader port to its application resource.
    affine.set<float>("scale", 0.5f); // Supply this shader uniform with its typed application value.
    affine.set<float>("offset", -1.0f); // Supply this shader uniform with its typed application value.
    pipeline.run(); // Execute the configured pipeline on the requested backend.
    for (unsigned i = 0; i < 5; ++i) // Iterate over i for this calculation.
        near(output_values[i], second_values[i] * 0.5f - 1, "rebound host output"); // Apply near to these arguments.

    affine.bind("input_values", // Connect this shader port to its application resource.
                output); // Per-index input/output aliasing is legal for this shader.
    affine.set<float>("scale", 1.0f); // Supply this shader uniform with its typed application value.
    affine.set<float>("offset", 2.0f); // Supply this shader uniform with its typed application value.
    pipeline.run(); // Execute the configured pipeline on the requested backend.
    for (unsigned i = 0; i < 5; ++i) // Iterate over i for this calculation.
        near(output_values[i], second_values[i] * 0.5f + 1, "aliased host output"); // Apply near to these arguments.
} // End this block.

void test_complete_pipeline_composition() { // Verify algorithm-block composition, order and self-append ownership.
    shdr::buffer<float> values{2.0f}; // Keep one shared in-place value so dispatch order is observable.
    auto first = std::shared_ptr<shdr::kernel>(shdr::make_kernel("affine_kernel").release()); // Retain the first shader through shared dispatch ownership.
    auto second = std::shared_ptr<shdr::kernel>(shdr::make_kernel("affine_kernel").release()); // Retain the second independently configured shader.
    first->bind("input_values", values).bind("output_values", values).set("count", 1u).set("scale", 2.0f).set("offset", 1.0f); // Define x becoming 2x plus one.
    second->bind("input_values", values).bind("output_values", values).set("count", 1u).set("scale", 3.0f).set("offset", -2.0f); // Define x becoming 3x minus two.
    const shdr::stage a = shdr::dispatch(first, shdr::extent(1)); // Keep ownership in the first dispatch.
    const shdr::stage b = shdr::dispatch(second, shdr::extent(1), shdr::backend::cpu); // Keep the explicit CPU choice in the second dispatch.
    const shdr::pipeline first_block(a), second_block(b); // Build independently reusable processing blocks.
    const shdr::pipeline joined = first_block >> second_block; // Join complete blocks with the public composition operator.
    require(joined.size() == 2u && joined.stages()[1].target == shdr::backend::cpu, "joined block order and backend choice"); // Verify retained order and backend metadata.
    joined.run(); // Apply 2 to 5 and then 5 to 13 in the linked resources.
    near(static_cast<const shdr::buffer<float>&>(values).data()[0], 13.0f, "joined block execution order"); // Make reversed dispatch order fail this check.
    values.data()[0] = 2.0f; // Restore the original resource for prepend validation.
    (a >> second_block).run(); // Compose an individual dispatch before a complete processing block.
    near(static_cast<const shdr::buffer<float>&>(values).data()[0], 13.0f, "stage before complete block"); // Verify the mixed composition overload.
    shdr::pipeline repeated = first_block; // Start with one retained affine dispatch.
    repeated += repeated; // Append safely from the same vector even if capacity grows.
    require(repeated.size() == 2u && first_block.size() == 1u, "self append preserves source blocks"); // Verify a finite self-append copy and the separate source block.
    first.reset(); // Release the caller's first shader reference before execution.
    values.data()[0] = 2.0f; // Restore the input before the repeated retained dispatches.
    repeated.snapshot().run(); // Check that deferred snapshot composition retains shader ownership.
    near(static_cast<const shdr::buffer<float>&>(values).data()[0], 11.0f, "self appended retained stages"); // Verify two first-stage executions in sequence.
    shdr::pipeline appended = first_block; // Exercise ordinary complete-block append separately.
    appended += second_block; // Append the second block through the public overload.
    appended += shdr::pipeline(); // Treat an empty processing block as an ordinary no-op.
    values.data()[0] = 2.0f; // Restore the same original input for append validation.
    appended.run(); // Execute both retained linked stages after the caller released one owner.
    near(static_cast<const shdr::buffer<float>&>(values).data()[0], 13.0f, "complete block append"); // Verify dispatch ownership and order together.
} // All three composition forms preserve resource sharing and dispatch lifetimes.

void test_three_dimensional_dispatch() { // Implement test three dimensional dispatch for the operations below.
    shdr::buffer<unsigned> coordinates(4 * 6 * 4), valid(4 * 6 * 4); // Construct coordinates with these initial values.
    coordinates_kernel shader; // Declare shader for the following calculation.
    shader.coordinates = coordinates; // Set shader.coordinates from this expression.
    shader.valid = valid; // Set shader.valid from this expression.
    shdr::pipeline pipeline; // Declare pipeline for the following calculation.
    pipeline += shdr::dispatch(shader, shdr::extent(2, 2, 2)); // Add a shader dispatch with the required workgroup counts.
    pipeline.run(); // Execute the configured pipeline on the requested backend.
    const shdr::buffer<unsigned>& read_coordinates = coordinates; // Use a constant buffer view so GPU writes synchronize before host reads.
    const shdr::buffer<unsigned>& read_valid = valid; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned z = 0; z < 4; ++z) // Iterate over z for this calculation.
        for (unsigned y = 0; y < 6; ++y) // Iterate over y for this calculation.
            for (unsigned x = 0; x < 4; ++x) { // Iterate over x for this calculation.
                const unsigned index = x + 4 * (y + 6 * z); // Set index from this expression.
                require(read_coordinates.data()[index] == x + 100 * y + 10000 * z, // Check that this operation satisfies the expected behavior.
                        "3D global invocation ID"); // Provide this text argument for the operation above.
                require(read_valid.data()[index] == 1, "3D local and workgroup builtins"); // Check that this operation satisfies the expected behavior.
            } // End this block.
} // End this block.

void test_dynamic_binding_errors() { // Implement test dynamic binding errors for the operations below.
    auto shader_owner = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shdr::kernel& shader = *shader_owner; // Set shader from this expression.
    shdr::pipeline pipeline; // Declare pipeline for the following calculation.
    pipeline += shdr::dispatch(shader, shdr::extent(1)); // Add a shader dispatch with the required workgroup counts.
    throws( // Report this failed operation.
        [&] { // Complete this expression or declaration.
            pipeline.run(); // Execute the configured pipeline on the requested backend.
        }, // Finish this declaration.
        "unbound resources must report an error"); // Provide this text argument for the operation above.
    shdr::buffer<float> values(4); // Construct values with these initial values.
    shdr::kernel& generic = shader; // Set generic from this expression.
    throws( // Report this failed operation.
        [&] { // Complete this expression or declaration.
            generic.bind("misspelled", values); // Connect this shader port to its application resource.
        }, // Finish this declaration.
        "unknown buffer port must report an error"); // Provide this text argument for the operation above.
    throws( // Report this failed operation.
        [&] { // Complete this expression or declaration.
            generic.set<float>("misspelled", 1.0f); // Supply this shader uniform with its typed application value.
        }, // Finish this declaration.
        "unknown uniform must report an error"); // Provide this text argument for the operation above.
    throws( // Report this failed operation.
        [&] { // Complete this expression or declaration.
            generic.set<float>("count", 1.0f); // Supply this shader uniform with its typed application value.
        }, // Finish this declaration.
        "uniform type mismatch must report an error"); // Provide this text argument for the operation above.

    generic.bind("input_values", values); // Connect this shader port to its application resource.
    generic.bind("output_values", values); // Connect this shader port to its application resource.
    generic.set<unsigned>("count", 4u); // Supply this shader uniform with its typed application value.
    generic.set<float>("scale", 1.0f); // Supply this shader uniform with its typed application value.
    generic.set<float>("offset", 2.0f); // Supply this shader uniform with its typed application value.
    pipeline.run(); // Execute the configured pipeline on the requested backend.
    const shdr::buffer<float>& read_values = values; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned i = 0; i < 4; ++i) // Iterate over i for this calculation.
        near(read_values.data()[i], 2, "name-based resource and uniform binding"); // Apply near to these arguments.

    generic.set<unsigned>("count", 5u); // Supply this shader uniform with its typed application value.
    throws( // Report this failed operation.
        [&] { // Complete this expression or declaration.
            pipeline.run(); // Execute the configured pipeline on the requested backend.
        }, // Finish this declaration.
        "CPU shader out-of-range SSBO access must report an error"); // Provide this text argument for the operation above.
    generic.set<unsigned>("count", 4u); // Supply this shader uniform with its typed application value.
    shdr::pipeline gpu_only; // Declare gpu only for the following calculation.
    gpu_only += shdr::dispatch(shader, shdr::extent(1), shdr::backend::gpu); // Add a shader dispatch with the required workgroup counts.
    throws( // Report this failed operation.
        [&] { // Complete this expression or declaration.
            gpu_only.run(); // Execute the configured pipeline on the requested backend.
        }, // Finish this declaration.
        "explicit GPU stage requires a GPU device"); // Provide this text argument for the operation above.
} // End this block.

void test_linked_buffer_image_pipeline() { // Implement test linked buffer image pipeline for the operations below.
    const unsigned width = 7, height = 5; // Set width from this expression.
    shdr::buffer<float> input(width * height), output(width * height); // Construct input with these initial values.
    shdr::image2d<float, 1> original(width, height), filtered(width, height); // Construct original with these initial values.
    for (unsigned i = 0; i < input.size(); ++i) // Iterate over i for this calculation.
        input.data()[i] = float(i); // Apply data to these arguments.
    auto into_image_owner = shdr::make_kernel("buffer_to_image_kernel"); // Create the linked buffer to image kernel instance.
    shdr::kernel& into_image = *into_image_owner; // Set into image from this expression.
    into_image.bind("input_values", input); // Connect this shader port to its application resource.
    into_image.bind("output_image", original); // Connect this shader port to its application resource.
    auto blur_owner = shdr::make_kernel("blur_kernel"); // Create the linked blur kernel instance.
    shdr::kernel& blur = *blur_owner; // Set blur from this expression.
    blur.bind("input_image", original); // Connect this shader port to its application resource.
    blur.bind("output_image", filtered); // Connect this shader port to its application resource.
    auto into_buffer_owner = shdr::make_kernel("image_to_buffer_kernel"); // Create the linked image to buffer kernel instance.
    shdr::kernel& into_buffer = *into_buffer_owner; // Set into buffer from this expression.
    into_buffer.bind("input_image", filtered); // Connect this shader port to its application resource.
    into_buffer.bind("output_values", output); // Connect this shader port to its application resource.
    shdr::pipeline pipeline = shdr::dispatch(into_image, shdr::extent(1, 1)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(blur, shdr::extent(1, 1)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(into_buffer, shdr::extent(1, 1)); // Add a shader dispatch with the required workgroup counts.
    pipeline.run(); // Execute the configured pipeline on the requested backend.
    const shdr::buffer<float>& read_output = output; // Use a constant buffer view so GPU writes synchronize before host reads.
    near(read_output.data()[0], 8.0f / 3.0f, "clamped blur upper corner"); // Apply near to these arguments.
    near(read_output.data()[17], 17.0f, "clamped blur center"); // Apply near to these arguments.
    near(read_output.data()[34], 34.0f - 8.0f / 3.0f, "clamped blur lower corner"); // Apply near to these arguments.
} // End this block.

} // namespace

int main() { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        test_borrowed_rebinding_and_repeat(); // Apply test borrowed rebinding and repeat to these arguments.
        test_complete_pipeline_composition(); // Validate complete algorithm-block composition and retained ownership.
        test_three_dimensional_dispatch(); // Apply test three dimensional dispatch to these arguments.
        test_dynamic_binding_errors(); // Apply test dynamic binding errors to these arguments.
        test_linked_buffer_image_pipeline(); // Apply test linked buffer image pipeline to these arguments.
        std::cout << "CPU runtime tests passed\n"; // Write this part of the example result or diagnostic.
        return 0; // Report successful completion.
    } catch (const std::exception& error) { // Report the exception from the preceding operation.
        std::cerr << error.what() << "\n"; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
