/** @file async.cpp
 * Three resource slots keep CPU workers or GPU dispatches in flight while an
 * application polls completions. Binding and uniform snapshots let one producer
 * kernel be reused. Resource slots are recycled after their jobs complete.
 */
#include "runner.hpp" // Include runner.hpp declarations.
#include <chrono> // Include chrono declarations.
#include <future> // Include future declarations.
#include <thread> // Include thread declarations.
#include <vector> // Include vector declarations.

namespace { // Group these helpers within their namespace.
const unsigned element_count = 257; // Set element count from this expression.
const unsigned frame_count = 32; // Set frame count from this expression.
const unsigned slot_count = 3; // Set slot count from this expression.
struct slot { // Declare slot.
    shdr::buffer<float> input, output; // Declare input for the following calculation.
    shdr::submission gpu_done; // Declare gpu done for the following calculation.
    std::future<void> cpu_done; // Declare cpu done for the following calculation.
    bool occupied; // Declare occupied for the following calculation.
    unsigned frame; // Declare frame for the following calculation.
    slot() : input(element_count), output(element_count), occupied(false), frame(0) { // Apply slot to these arguments.
    } // End this block.
}; // Finish this declaration.

void check(const slot& resource, unsigned index) { // Implement check for the operations below.
    const shdr::buffer<float>& result = resource.output; // Use a constant buffer view so GPU writes synchronize before host reads.
    const float scale = float(resource.frame + 1); // Set scale from this expression.
    const float offset = -float(resource.frame); // Set offset from this expression.
    for (unsigned i = 0; i < element_count; ++i) // Iterate over i for this calculation.
        shdr_examples::require_close( // Check the computed value against the independent reference.
            result.data()[i], (float(i) + float(index * 10)) * scale + offset, "async ring result"); // Apply data to these arguments.
} // End this block.

void run_ring(shdr_examples::runner& runner) { // Implement run ring for the operations below.
    std::vector<slot> slots(slot_count); // Construct slots with these initial values.
    for (unsigned index = 0; index < slot_count; ++index) // Iterate over index for this calculation.
        for (unsigned i = 0; i < element_count; ++i) // Iterate over i for this calculation.
            slots[index].input.data()[i] = float(i + index * 10); // Apply data to these arguments.
    auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shader->set<unsigned>("count", element_count); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline(shdr::dispatch(*shader, shdr::extent((element_count + 63) / 64))); // Add a shader dispatch with the required workgroup counts.
    std::unique_ptr<shdr::executor> workers; // Declare workers for the following calculation.
    if (!runner.gpu()) // Check this condition before entering the following branch.
        workers.reset(new shdr::executor(2, slot_count)); // Apply reset to these arguments.
    if (runner.gpu()) { // Check this condition before entering the following branch.
        // Initial program compilation and every slot's allocation/upload happen
        // before entering the polling loop.
        for (unsigned index = 0; index < slot_count; ++index) { // Iterate over index for this calculation.
            shader->bind("input_values", slots[index].input) // Connect this shader port to its application resource.
                .bind("output_values", slots[index].output); // Connect this shader port to its application resource.
            pipeline.prepare(*runner.device()); // Apply prepare to these arguments.
        } // End this block.
    } // End this block.
    unsigned produced = 0, completed = 0; // Set produced from this expression.
    while (completed < frame_count) { // Pump the resource ring until every submitted frame completes.
        bool progress = false; // Set progress from this expression.
        for (unsigned index = 0; index < slot_count; ++index) { // Iterate over index for this calculation.
            slot& resource = slots[index]; // Set resource from this expression.
            if (resource.occupied) { // Check this condition before entering the following branch.
                const bool ready = runner.gpu() // Set ready from this expression.
                                       ? resource.gpu_done.ready() // Apply ready to these arguments.
                                       : resource.cpu_done.wait_for(std::chrono::nanoseconds(0)) == // Declare resource for the following calculation.
                                             std::future_status::ready; // Continue the expression using std, future status, ready.
                if (!ready) // Check this condition before entering the following branch.
                    continue; // Skip the remaining work for this loop iteration.
                if (!runner.gpu()) // Check this condition before entering the following branch.
                    resource.cpu_done.get(); // Apply get to these arguments.
                check(resource, index); // Apply check to these arguments.
                resource.occupied = false; // Set resource.occupied from this expression.
                ++completed; // Continue the expression using completed.
                progress = true; // Set progress from this expression.
            } // End this block.
            if (produced < frame_count) { // Check this condition before entering the following branch.
                resource.frame = produced++; // Set resource.frame from this expression.
                shader->bind("input_values", resource.input).bind("output_values", resource.output); // Connect this shader port to its application resource.
                shader->set<float>("scale", float(resource.frame + 1)) // Supply this shader uniform with its typed application value.
                    .set<float>("offset", -float(resource.frame)); // Supply this shader uniform with its typed application value.
                if (runner.gpu()) // Check this condition before entering the following branch.
                    resource.gpu_done = pipeline.submit(*runner.device()); // Submit work and retain its completion handle.
                else // Queue the frame snapshot on the CPU worker pool.
                    resource.cpu_done = workers->submit(pipeline); // Submit work and retain its completion handle.
                resource.occupied = true; // Set resource.occupied from this expression.
                progress = true; // Set progress from this expression.
            } // End this block.
        } // End this block.
        if (runner.gpu()) // Check this condition before entering the following branch.
            runner.device()->collect_garbage(); // Apply device to these arguments.
        // An event loop can perform other application work between pump passes.
        if (!progress) // Check this condition before entering the following branch.
            std::this_thread::yield(); // Apply yield to these arguments.
    } // End this block.
    if (workers) // Check this condition before entering the following branch.
        workers->shutdown(); // Apply shutdown to these arguments.
    if (runner.gpu()) { // Check this condition before entering the following branch.
        for (unsigned index = 0; index < slot_count; ++index) // Iterate over index for this calculation.
            if (slots[index].gpu_done.valid()) // Check this condition before entering the following branch.
                slots[index].gpu_done.wait(std::chrono::nanoseconds(0)); // Apply wait to these arguments.
        runner.device()->collect_garbage(); // Apply device to these arguments.
    } // End this block.
    std::cout << (runner.gpu() ? "GPU" : "CPU") << " async ring: " << completed // Write this part of the example result or diagnostic.
              << " frames verified\n"; // Write this part of the example result or diagnostic.
} // End this block.
} // namespace

int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner runner(argc, argv); // Construct runner with these initial values.
        run_ring(runner); // Apply run ring to these arguments.
        return 0; // Report successful completion.
#ifdef SHDR_HAVE_EGL // Enable this branch when its feature macro is defined.
    } catch (const shdr_examples::context_unavailable& e) { // Report the exception from the preceding operation.
        std::cerr << "SKIP: " << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 77; // Report an unavailable optional GPU backend.
#endif // Finish this conditional compilation branch.
    } catch (const std::exception& e) { // Report the exception from the preceding operation.
        std::cerr << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
