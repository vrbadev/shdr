/** @file test_async.cpp
 * Queue, snapshot and completion checks that use promises as deterministic
 * coordination points. The checks remain enabled in Release builds.
 */
#include "shdr.h" // Include shdr.h declarations.
#include <atomic> // Include atomic declarations.
#include <chrono> // Include chrono declarations.
#include <future> // Include future declarations.
#include <iostream> // Include iostream declarations.
#include <stdexcept> // Include stdexcept declarations.

namespace { // Group these helpers within their namespace.
void require(bool condition, const char* message) { // Check that this operation satisfies the expected behavior.
    if (!condition) // Check this condition before entering the following branch.
        throw std::runtime_error(message); // Report this failed operation.
} // End this block.
template <class Function> void must_throw(Function function, const char* message) { // Accept the value types or dimensions used by this template.
    bool caught = false; // Set caught from this expression.
    try { // Run the operation while retaining the error handling below.
        function(); // Apply function to these arguments.
    } catch (const std::exception&) { // Report the exception from the preceding operation.
        caught = true; // Set caught from this expression.
    } // End this block.
    require(caught, message); // Check that this operation satisfies the expected behavior.
} // End this block.
struct release_guard { // Declare release guard.
    std::promise<void>* value; // Declare value for the following calculation.
    explicit release_guard(std::promise<void>& promise) : value(&promise) { // Implement release guard for the operations below.
    } // End this block.
    void release() { // Implement release for the operations below.
        if (value) { // Check this condition before entering the following branch.
            value->set_value(); // Apply set value to these arguments.
            value = 0; // Set value from this expression.
        } // End this block.
    } // End this block.
    ~release_guard() { // Apply release guard to these arguments.
        try { // Run the operation while retaining the error handling below.
            release(); // Apply release to these arguments.
        } catch (...) { // Report the exception from the preceding operation.
        } // End this block.
    } // End this block.
}; // Finish this declaration.

void queue_and_errors() { // Implement queue and errors for the operations below.
    shdr::executor pool(1, 1); // Construct pool with these initial values.
    std::promise<void> entered, released; // Declare entered for the following calculation.
    release_guard unlock(released); // Construct unlock with these initial values.
    std::shared_future<void> release = released.get_future().share(); // Set release from this expression.
    std::future<void> active = pool.enqueue([&] { // Set active from this expression.
        entered.set_value(); // Apply set value to these arguments.
        release.wait(); // Apply wait to these arguments.
    }); // Finish this declaration.
    entered.get_future().wait(); // Apply get future to these arguments.
    std::atomic<unsigned> calls(0); // Construct calls with these initial values.
    std::future<void> queued = pool.enqueue([&] { // Set queued from this expression.
        ++calls; // Continue the expression using calls.
    }); // Finish this declaration.
    require(pool.pending() == 1 && pool.capacity() == 1 && pool.workers() == 1, // Check that this operation satisfies the expected behavior.
            "bounded pending queue"); // Provide this text argument for the operation above.
    must_throw( // Check that this operation satisfies the expected behavior.
        [&] { // Complete this expression or declaration.
            pool.enqueue([] { // Apply enqueue to these arguments.
            }); // Finish this declaration.
        }, // Finish this declaration.
        "full queue must reject a new job"); // Provide this text argument for the operation above.
    require(queued.wait_for(std::chrono::nanoseconds(0)) == std::future_status::timeout, // Check that this operation satisfies the expected behavior.
            "queued future is not falsely complete"); // Provide this text argument for the operation above.
    unlock.release(); // Apply release to these arguments.
    active.get(); // Apply get to these arguments.
    queued.get(); // Apply get to these arguments.
    require(calls == 1, "accepted job executes exactly once"); // Check that this operation satisfies the expected behavior.
    std::future<void> failed = pool.enqueue([] { // Set failed from this expression.
        throw std::runtime_error("task failure"); // Report task failure.
    }); // Finish this declaration.
    must_throw( // Check that this operation satisfies the expected behavior.
        [&] { // Complete this expression or declaration.
            failed.get(); // Apply get to these arguments.
        }, // Finish this declaration.
        "worker errors propagate through futures"); // Provide this text argument for the operation above.
    pool.enqueue([&] { // Apply enqueue to these arguments.
            ++calls; // Continue the expression using calls.
        }) // End this block.
        .get(); // Apply get to these arguments.
    require(calls == 2, "worker survives a throwing job"); // Check that this operation satisfies the expected behavior.
    std::future<void> forbidden_shutdown = pool.enqueue([&] { // Set forbidden shutdown from this expression.
        pool.shutdown(); // Apply shutdown to these arguments.
    }); // Finish this declaration.
    must_throw( // Check that this operation satisfies the expected behavior.
        [&] { // Complete this expression or declaration.
            forbidden_shutdown.get(); // Apply get to these arguments.
        }, // Finish this declaration.
        "worker shutdown must report an error rather than deadlock"); // Provide this text argument for the operation above.
    pool.shutdown(); // Apply shutdown to these arguments.
    must_throw( // Check that this operation satisfies the expected behavior.
        [&] { // Complete this expression or declaration.
            pool.enqueue([] { // Apply enqueue to these arguments.
            }); // Finish this declaration.
        }, // Finish this declaration.
        "shutdown rejects subsequent work"); // Provide this text argument for the operation above.
    must_throw( // Check that this operation satisfies the expected behavior.
        [] { // Complete this expression or declaration.
            shdr::executor invalid(0, 1); // Construct invalid with these initial values.
        }, // Finish this declaration.
        "zero worker count must fail"); // Provide this text argument for the operation above.
    must_throw( // Check that this operation satisfies the expected behavior.
        [] { // Complete this expression or declaration.
            shdr::executor invalid(1, 0); // Construct invalid with these initial values.
        }, // Finish this declaration.
        "zero capacity must fail"); // Provide this text argument for the operation above.
} // End this block.

void future_lifetime() { // Implement future lifetime for the operations below.
    shdr::executor pool(1, 2); // Construct pool with these initial values.
    std::promise<void> entered, released; // Declare entered for the following calculation.
    release_guard unlock(released); // Construct unlock with these initial values.
    std::shared_future<void> release = released.get_future().share(); // Set release from this expression.
    std::atomic<bool> completed(false); // Construct completed with these initial values.
    { // Begin this local scope.
        std::future<void> dropped = pool.enqueue([&] { // Set dropped from this expression.
            entered.set_value(); // Apply set value to these arguments.
            release.wait(); // Apply wait to these arguments.
            completed = true; // Set completed from this expression.
        }); // Finish this declaration.
        entered.get_future().wait(); // Apply get future to these arguments.
    } // End this block.
    // Reaching this point while release is pending verifies prompt future destruction.
    // Explicit shutdown drains the worker after its release signal.
    require(!completed, "destroying a future must not finish a waiting job"); // Check that this operation satisfies the expected behavior.
    unlock.release(); // Apply release to these arguments.
    pool.shutdown(); // Apply shutdown to these arguments.
    require(completed, "shutdown drains work after its future was discarded"); // Check that this operation satisfies the expected behavior.

    shdr::executor cleanup(1, 2); // Construct cleanup with these initial values.
    std::shared_ptr<int> captured(new int(1)); // Construct captured with these initial values.
    std::weak_ptr<int> lifetime = captured; // Set lifetime from this expression.
    std::future<void> finished = cleanup.enqueue([captured] { // Set finished from this expression.
    }); // Finish this declaration.
    captured.reset(); // Apply reset to these arguments.
    finished.get(); // Apply get to these arguments.
    require(lifetime.expired(), "job captures are released before completion becomes visible"); // Check that this operation satisfies the expected behavior.
} // End this block.

void pipeline_snapshots() { // Implement pipeline snapshots for the operations below.
    shdr::executor pool(1, 4); // Construct pool with these initial values.
    std::promise<void> entered, released; // Declare entered for the following calculation.
    release_guard unlock(released); // Construct unlock with these initial values.
    std::shared_future<void> release = released.get_future().share(); // Set release from this expression.
    std::future<void> gate = pool.enqueue([&] { // Set gate from this expression.
        entered.set_value(); // Apply set value to these arguments.
        release.wait(); // Apply wait to these arguments.
    }); // Finish this declaration.
    entered.get_future().wait(); // Apply get future to these arguments.
    shdr::buffer<float> first(4), second(4), first_output(4), second_output(4); // Construct first with these initial values.
    for (unsigned i = 0; i < 4; ++i) { // Iterate over i for this calculation.
        first.data()[i] = float(i); // Apply data to these arguments.
        second.data()[i] = float(i + 10); // Apply data to these arguments.
    } // End this block.
    auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shader->bind("input_values", first).bind("output_values", first_output); // Connect this shader port to its application resource.
    shader->set<unsigned>("count", 4).set<float>("scale", 2).set<float>("offset", 1); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline(shdr::dispatch(*shader, shdr::extent(1))); // Add a shader dispatch with the required workgroup counts.
    std::future<void> first_job = pool.submit(pipeline); // Submit work and retain its completion handle.
    shader->bind("input_values", second).bind("output_values", second_output); // Connect this shader port to its application resource.
    shader->set<float>("scale", 3).set<float>("offset", -1); // Supply this shader uniform with its typed application value.
    std::future<void> second_job = pool.submit(pipeline); // Submit work and retain its completion handle.
    shader.reset(); // The accepted jobs own their independent kernel snapshots.
    unlock.release(); // Apply release to these arguments.
    gate.get(); // Apply get to these arguments.
    first_job.get(); // Apply get to these arguments.
    second_job.get(); // Apply get to these arguments.
    const shdr::buffer<float>& read_first = first_output; // Use a constant buffer view so GPU writes synchronize before host reads.
    const shdr::buffer<float>& read_second = second_output; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned i = 0; i < 4; ++i) { // Iterate over i for this calculation.
        require(read_first.data()[i] == float(i) * 2 + 1, // Check that this operation satisfies the expected behavior.
                "uniforms and bindings of first snapshot"); // Provide this text argument for the operation above.
        require(read_second.data()[i] == float(i + 10) * 3 - 1, // Check that this operation satisfies the expected behavior.
                "uniforms and bindings of rebound snapshot"); // Provide this text argument for the operation above.
    } // End this block.
    auto gpu_shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    gpu_shader->bind("input_values", first).bind("output_values", first_output); // Connect this shader port to its application resource.
    shdr::pipeline gpu_only(shdr::dispatch(*gpu_shader, shdr::extent(1), shdr::backend::gpu)); // Add a shader dispatch with the required workgroup counts.
    must_throw( // Check that this operation satisfies the expected behavior.
        [&] { // Complete this expression or declaration.
            pool.submit(gpu_only); // Submit work and retain its completion handle.
        }, // Finish this declaration.
        "CPU workers must reject GPU-only work"); // Provide this text argument for the operation above.
} // End this block.

void parallel_jobs() { // Implement parallel jobs for the operations below.
    shdr::executor pool(2, 8); // Construct pool with these initial values.
    shdr::buffer<float> input(257), output_a(257), output_b(257); // Construct input with these initial values.
    for (unsigned i = 0; i < 257; ++i) // Iterate over i for this calculation.
        input.data()[i] = float(i); // Apply data to these arguments.
    auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shader->bind("input_values", input).bind("output_values", output_a); // Connect this shader port to its application resource.
    shader->set<unsigned>("count", 257).set<float>("scale", 2).set<float>("offset", 0); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline(shdr::dispatch(*shader, shdr::extent(5))); // Add a shader dispatch with the required workgroup counts.
    std::future<void> a = pool.submit(pipeline); // Submit work and retain its completion handle.
    shader->bind("output_values", output_b).set<float>("scale", 3); // Connect this shader port to its application resource.
    std::future<void> b = pool.submit(pipeline); // Submit work and retain its completion handle.
    a.get(); // Apply get to these arguments.
    b.get(); // Apply get to these arguments.
    const shdr::buffer<float>& read_a = output_a; // Use a constant buffer view so GPU writes synchronize before host reads.
    const shdr::buffer<float>& read_b = output_b; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned i = 0; i < 257; ++i) { // Iterate over i for this calculation.
        require(read_a.data()[i] == float(i) * 2, "parallel first kernel invocation state"); // Check that this operation satisfies the expected behavior.
        require(read_b.data()[i] == float(i) * 3, "parallel second kernel invocation state"); // Check that this operation satisfies the expected behavior.
    } // End this block.
    shdr::submission empty; // Declare empty for the following calculation.
    require(empty.ready() && empty.wait(std::chrono::nanoseconds(0)) == shdr::wait_status::ready, // Check that this operation satisfies the expected behavior.
            "empty GPU completion handle"); // Provide this text argument for the operation above.
} // End this block.
} // namespace

int main() { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        queue_and_errors(); // Apply queue and errors to these arguments.
        future_lifetime(); // Apply future lifetime to these arguments.
        pipeline_snapshots(); // Apply pipeline snapshots to these arguments.
        parallel_jobs(); // Apply parallel jobs to these arguments.
        std::cout << "Async CPU queue and snapshot checks passed\n"; // Write this part of the example result or diagnostic.
        return 0; // Report successful completion.
    } catch (const std::exception& e) { // Report the exception from the preceding operation.
        std::cerr << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
