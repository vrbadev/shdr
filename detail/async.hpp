/** @file async.hpp
 * @brief Nonblocking GPU completion handles and bounded CPU work queues.
 * Prepare GPU pipelines before a hot loop, keep a ring of independent resource
 * sets, and poll completion before reading or recycling a result. GPU calls
 * use the device's context thread. CPU workers execute independent snapshots
 * and require host-ready resources. Complete GPU readback on the context thread
 * before submitting a CPU job.
 */
#ifndef SHDR_ASYNC_HPP // Skip this section if it was already included.
#define SHDR_ASYNC_HPP /* Mark this section as included. */
#ifndef SHDR_RUNTIME_HPP // Skip this section if it was already included.
#include "math.hpp" // Use the portable GLSL scalar, vector, and matrix types.
#include "runtime.hpp" // Use the kernel, resource, and ordered pipeline definitions.
#endif // Finish this conditional section.
#include <chrono> // Express explicit waits and zero-time completion polls.
#include <condition_variable> // Wake CPU workers when jobs arrive or shutdown begins.
#include <deque> // Keep pending CPU jobs in submission order.
#include <exception> // Pass worker failures back through completion futures.
#include <future> // Give callers a completion result for each CPU job.
#include <mutex> // Protect shared work queues and deferred deletion.
#include <thread> // Track the context thread and start CPU workers.
#include <utility> // Move job and resource ownership between runtime objects.

namespace shdr { // Group these definitions in the shdr namespace.

/** @brief Result of a completion query: completed work or an expired timeout. */
enum class wait_status { ready, timeout }; // Name the two outcomes of a GPU completion query.

namespace detail { // Group these definitions in the detail namespace.
struct submission_resources { // Keep the submission resources values and operations together.
    std::vector<std::shared_ptr<resource_state>> resources; // Keep the resource storage retained until queued work completes.
    std::vector<std::shared_ptr<program_state>> programs; // Keep the compiled programs retained by a submission.
}; // End this scope in async.

struct submission_state { // Keep a GPU fence and its resources alive until retirement.
    std::shared_ptr<device_state> owner; // Keep the device that owns the GL resource.
    GLsync fence; // Keep the GL completion fence.
    std::shared_ptr<submission_resources> retained; // Keep the submission resource bundle.
    bool completed; // Cache whether this submission fence has already signaled.
    explicit submission_state(const std::shared_ptr<device_state>& d) // Retain the device, fence, and queued resource bundle.
        : owner(d), fence(0), retained(std::make_shared<submission_resources>()), completed(false) { // Retain the device and create an empty pending-completion resource bundle.
    } // End this scope in submission state.
    ~submission_state() noexcept { // Retire completed fences and defer pending fence cleanup.
        if (!fence) // Finish immediately for an empty fence.
            return; // Finish this operation.
        const GLsync sync = fence; // Copy the fence handle into the deferred cleanup callback.
        const std::shared_ptr<submission_resources> keep = retained; // Retain buffers and programs until deferred cleanup releases this bundle.
        try { // Keep cleanup available if an allocation, shader, or driver operation fails.
            if (completed) { // Retire a fence whose completion has already been observed.
                owner->retire([sync, keep](gl_api& gl) { // Delete on the context thread or defer deletion to that thread.
                    gl.DeleteSync(sync); // Delete the GL completion fence.
                    (void)keep; // Keep the resource references captured for this cleanup.
                }); // End this scope in submission state.
            } else { // Handle the alternative resource or completion case.
 // The context thread checks this retirement entry with a zero timeout.
                owner->defer_poll( // Retain a completion check until its fence signals.
                    [sync, keep](gl_api& gl) -> bool { // Retain these captures until the queued callback finishes.
                        const GLenum status = gl.ClientWaitSync(sync, 0, 0); // Poll the discarded fence with zero flags and a zero timeout.
                        if (status == TIMEOUT_EXPIRED) // Check the driver result before marking GPU completion.
                            return false; // Keep the discarded submission pending while its fence is unsignaled.
                        gl.DeleteSync(sync); // Delete the GL completion fence.
                        (void)keep; // Keep the resource references captured for this cleanup.
                        return true; // Remove the retired fence and release its retained resource bundle.
                    }, // End this scope in submission state.
                    [sync](gl_api& gl) { // Keep only the fence handle for nonblocking shutdown cancellation.
                        gl.DeleteSync(sync); // Delete the GL completion fence.
                    }); // End this scope in submission state.
            } // End this scope in submission state.
        } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
 // Destruction defers pending fence cleanup. GL command references keep
 // already queued operations valid if allocating a retirement entry fails.
        } // End this scope in submission state.
    } // End this scope in submission state.
    wait_status query(GLuint64 timeout) { // Check completion with the requested driver timeout.
        owner->require_thread(); // Require the device context thread and activate its headless context.
        if (completed) // Reuse the cached completion result after checking the context thread.
            return wait_status::ready; // Tell the caller that queued GPU commands have completed.
        const GLenum result = owner->gl.ClientWaitSync(fence, 0, timeout); // Check a GPU fence with the caller-selected timeout.
        if (result == ALREADY_SIGNALED || result == CONDITION_SATISFIED) { // Check the driver result before marking GPU completion.
            completed = true; // Set whether the GPU fence has already signaled.
            return wait_status::ready; // Tell the caller that queued GPU commands have completed.
        } // End this scope in query.
        if (result == TIMEOUT_EXPIRED) // Check the driver result before marking GPU completion.
            return wait_status::timeout; // Tell the caller that the explicit wait expired before completion.
        owner->gl.check("query GPU submission"); // Check for a driver error from the preceding GL operation.
        throw error("GPU completion query failed"); // Report gpu completion query failed.
    } // End this scope in query.
}; // End this scope in async.
} // namespace detail

/** @brief A movable handle for one queued GPU pipeline.
 * ready() performs a zero-timeout fence query. wait(timeout) bounds the requested
 * driver wait. wait() explicitly waits for completion. Destroying a handle
 * schedules pending fence and resource cleanup on the device context thread.
 * Device garbage collection releases those retained resources after completion.
 * Keep the GL context alive until retained work and resources have been retired.
 */
class submission { // Keep the submission values and operations together.
    std::unique_ptr<detail::submission_state> state_; // Own this completion handle and its retained GPU resources.
    explicit submission(std::unique_ptr<detail::submission_state> state) // Accept sole ownership of this fence and its retained resource bundle.
        : state_(std::move(state)) { // Move the fence state into this completion handle.
    } // End this scope in async.
    friend class pipeline; // Allow the related pipeline or resource helper to access this retained state.

  public: // Expose the operations callers can use.
    /** @brief Construct an empty, already-complete handle. */
    submission() { // Create an empty handle that is already complete.
    } // End this scope in async.
    submission(submission&&) noexcept = default; // Move the completion state into the new handle.
    submission& operator=(submission&&) noexcept = default; // Retire the old handle and move in the new completion state.
    submission(const submission&) = delete; // Prevent copying this object and duplicating its retained identity.
    submission& operator=(const submission&) = delete; // Prevent copying this object and duplicating its retained identity.
    /** @brief Return whether this handle represents submitted GPU commands. */
    bool valid() const { // Check whether this handle represents actual submitted GPU work.
        return bool(state_); // Return whether this handle retains a submission state.
    } // End this scope in async.
    /** @brief Query completion with a zero timeout on the device context thread. */
    bool ready() { // Check whether queued work has completed using a zero-timeout poll.
        return !state_ || state_->query(0) == wait_status::ready; // Check whether queued work has completed using a zero-timeout poll.
    } // End this scope in ready.
    /** @brief Request a driver wait with a nonnegative timeout in nanoseconds. */
    wait_status wait(std::chrono::nanoseconds timeout) { // Apply the explicit completion wait selected by the caller.
        if (timeout.count() < 0) // Require a nonnegative wait duration.
            throw error("GPU submission timeout must be nonnegative"); // Report gpu submission timeout must be nonnegative.
        if (!state_) // Treat an empty handle as already completed.
            return wait_status::ready; // Tell the caller that queued GPU commands have completed.
        return state_->query(static_cast<detail::GLuint64>(timeout.count())); // Apply the explicit completion wait selected by the caller.
    } // End this scope in wait.
    /** @brief Explicitly wait until submitted commands complete. */
    wait_status wait() { // Apply the explicit completion wait selected by the caller.
        if (!state_) // Treat an empty handle as already completed.
            return wait_status::ready; // Tell the caller that queued GPU commands have completed.
        while (state_->query((std::numeric_limits<detail::GLuint64>::max)()) == // Use the caller's explicit blocking wait until the fence completes.
               wait_status::timeout) { // Repeat only if the implementation still reports an incomplete wait.
        } // End this scope in wait.
        return wait_status::ready; // Tell the caller that queued GPU commands have completed.
    } // End this scope in wait.
}; // End this scope in async.

/** @brief Enqueue all GPU stages and return a fence for polling or explicit waits.
 * Each stage must support GPU execution. Call prepare()
 * before entering a latency-sensitive loop to compile shaders and allocate/upload
 * resources in advance. Subsequent host edits can still require an upload.
 * Uniform and binding changes after return affect subsequent submissions.
 */
inline submission pipeline::submit(device& d) const { // Queue compute work and return its completion handle.
    d.state()->require_thread(); // Require the device context thread and activate its headless context.
    const std::vector<stage>& commands = stages(); // Read the ordered stages that this submission will queue.
    for (std::size_t i = 0; i < commands.size(); ++i) // Visit each ordered GPU stage in this submission.
        if (commands[i].target == backend::cpu) // Reject explicit CPU stages before queueing any GPU commands.
            throw error("GPU submission cannot contain CPU stages"); // Require GPU stages for GPU submission.
    if (commands.empty()) // Skip fence creation for an empty pipeline.
        return submission(); // Represent an empty pipeline with an already completed handle.
    std::unique_ptr<detail::submission_state> completion(new detail::submission_state(d.state())); // Retain the device, fence, and queued resource bundle.
    completion->retained->programs.reserve(commands.size()); // Reserve room to retain one compiled program per stage.
    prepare(d); // Compile the program and prepare its bound GPU resources.
    for (std::size_t i = 0; i < commands.size(); ++i) { // Visit each ordered GPU stage in this submission.
        const std::vector<std::shared_ptr<detail::resource_state>> resources = // Set the resource storage retained until queued work completes.
            commands[i].shader->resources(); // Collect the shared storage used by this shader.
        completion->retained->resources.insert( // Retain this entry in the ordered resource or work collection.
            completion->retained->resources.end(), resources.begin(), resources.end()); // Keep the end of the current range.
        completion->retained->programs.push_back(commands[i].shader->program_state()); // Retain this entry in the ordered resource or work collection.
    } // End this scope in submit.
    try { // Keep cleanup available if an allocation, shader, or driver operation fails.
        run(d); // Queue all validated GPU stages in pipeline order.
    } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
 // A later dispatch can fail after an earlier dispatch was queued.
        completion->fence = d.state()->gl.FenceSync(detail::SYNC_GPU_COMMANDS_COMPLETE, 0); // Insert a completion fence after queued compute work.
        d.state()->gl.Flush(); // Flush queued GPU commands for driver execution.
        throw; // Rethrow the original failure after arranging the necessary cleanup.
    } // End this scope in submit.
    completion->fence = d.state()->gl.FenceSync(detail::SYNC_GPU_COMMANDS_COMPLETE, 0); // Insert a completion fence after queued compute work.
    if (!completion->fence) { // Check that the driver allocated a valid completion fence.
        d.state()->gl.check("create GPU submission fence"); // Expose retained runtime state to the resource and backend helpers.
        throw error("cannot create GPU submission fence"); // Report completion-fence allocation failure.
    } // End this scope in submit.
    d.state()->gl.Flush(); // Flush queued GPU commands for driver execution.
    d.state()->gl.check("submit GPU pipeline"); // Expose retained runtime state to the resource and backend helpers.
    return submission(std::move(completion)); // Queue compute work and return its completion handle.
} // End this scope in submit.

/** @brief A bounded CPU executor for independent pipeline jobs.
 * Each submit() snapshots kernels, uniforms and resource bindings. Snapshots
 * have independent invocation state, while buffers and images retain shared
 * storage. Give concurrent jobs distinct writable resources, such as ring slots.
 * Queue saturation throws immediately. Futures report results or exceptions.
 * Destroying a future releases its result handle, while the executor retains
 * the job until completion. shutdown() drains accepted jobs and joins workers.
 * Executor destruction also drains and joins them.
 */
class executor { // Keep the executor values and operations together.
    struct job { // Keep the job values and operations together.
        std::function<void()> function; // Keep the application function executed by a CPU worker.
        std::promise<void> completion; // Keep the completion result that the caller will receive.
        explicit job(const std::function<void()>& fn) : function(fn) { // Keep the accepted CPU function and its promise result.
        } // End this scope in async.
    }; // End this scope in async.
    struct queue_state { // Share the bounded queue between its producer and workers.
        std::mutex mutex; // Keep the lock protecting shared state.
        std::condition_variable changed; // Wake sleeping workers when work or shutdown state changes.
        std::deque<std::shared_ptr<job>> jobs; // Keep accepted CPU work awaiting an available worker.
        std::vector<std::thread::id> worker_ids; // Keep the worker thread identifiers used to reject self-joining.
        std::size_t pending, capacity; // Keep the current pending count and the maximum queue size.
        bool closing; // Keep whether the CPU executor is draining for shutdown.
        explicit queue_state(std::size_t n) : pending(0), capacity(n), closing(false) { // Start an empty queue with the chosen pending-job capacity.
        } // End this scope in async.
    }; // End this scope in async.
    std::shared_ptr<queue_state> state_; // Share the bounded queue between the executor and its workers.
    std::vector<std::thread> workers_; // Keep the threads that execute accepted CPU jobs.
    std::mutex shutdown_mutex_; // Keep the lock that serializes executor shutdown.

    static void consume(const std::shared_ptr<queue_state>& state) { // Take accepted jobs from the queue and execute them on this worker.
        for (;;) { // Keep this worker available until its queue is drained for shutdown.
            std::shared_ptr<job> next; // Keep the CPU job selected by this worker.
            { // Begin the protected scope for this operation.
                std::unique_lock<std::mutex> lock(state->mutex); // Hold the mutex while this shared state is accessed.
                state->changed.wait(lock, [&] { // Release the queue lock while sleeping until work or shutdown arrives.
                    return state->closing || !state->jobs.empty(); // Wake the worker for a queued job or an explicit shutdown.
                }); // End this scope in consume.
                if (state->jobs.empty()) // Exit only when shutdown has awakened an empty queue.
                    return; // Finish this operation.
                next = state->jobs.front(); // Set the CPU job selected by this worker.
                state->jobs.pop_front(); // Remove this accepted job from the pending queue.
                --state->pending; // Release its pending slot before executing the job.
            } // End this scope in consume.
            try { // Keep cleanup available if an allocation, shader, or driver operation fails.
                next->function(); // Execute the selected application function or CPU pipeline snapshot.
                next->function = std::function<void()>(); // Release kernel and resource captures before publishing completion.
                next->completion.set_value(); // Publish successful completion after the job releases its captures.
            } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
                const std::exception_ptr failure = std::current_exception(); // Set the exception captured from a CPU job.
                next->function = std::function<void()>(); // Release kernel and resource captures before publishing completion.
                next->completion.set_exception(failure); // Deliver the worker failure through the caller completion future.
            } // End this scope in consume.
        } // End this scope in consume.
    } // End this scope in consume.
    void reserve() { // Reserve capacity and report a full queue immediately.
        std::lock_guard<std::mutex> lock(state_->mutex); // Hold the mutex while this shared state is accessed.
        if (state_->closing) // Check the queue state before accepting or executing work.
            throw error("CPU executor is shut down"); // Report cpu executor is shut down.
        if (state_->pending == state_->capacity) // Check the queue state before accepting or executing work.
            throw error("CPU executor queue is full"); // Report cpu executor queue is full.
        ++state_->pending; // Count the accepted job against the pending queue capacity.
    } // End this scope in reserve.
    void abandon() noexcept { // Release the reserved slot after a submission failure.
        std::lock_guard<std::mutex> lock(state_->mutex); // Hold the mutex while this shared state is accessed.
        --state_->pending; // Return the queue slot reserved by an unsuccessful submission.
    } // End this scope in abandon.
    std::future<void> append_reserved(const std::function<void()>& function) { // Attach the accepted CPU job to its reserved queue slot.
        std::shared_ptr<job> work(new job(function)); // Keep the application function executed by a CPU worker.
        std::future<void> result = work->completion.get_future(); // Set the value returned by this operation.
        { // Begin the protected scope for this operation.
            std::lock_guard<std::mutex> lock(state_->mutex); // Hold the mutex while this shared state is accessed.
            if (state_->closing) // Check the queue state before accepting or executing work.
                throw error("CPU executor is shut down"); // Report cpu executor is shut down.
            state_->jobs.push_back(work); // Retain this entry in the ordered resource or work collection.
        } // End this scope in append reserved.
        state_->changed.notify_one(); // Wake workers so they can observe new queue or shutdown state.
        return result; // Return the value returned by this operation.
    } // End this scope in append reserved.

  public: // Expose the operations callers can use.
    /** @brief Start workers with room for at most capacity pending jobs.
     * Capacity counts jobs waiting for a worker. Both arguments must be positive.
     */
    explicit executor(std::size_t workers = 1, std::size_t capacity = 64) // Start a fixed number of workers with a bounded pending-job queue.
        : state_(std::make_shared<queue_state>(capacity)) { // Keep the maximum number of pending CPU jobs.
        if (!workers || !capacity) // Require positive worker and queue counts.
            throw error("CPU executor requires positive worker and queue counts"); // Report cpu executor requires positive worker and queue counts.
        workers_.reserve(workers); // Allocate space for every worker thread handle.
        state_->worker_ids.reserve(workers); // Allocate space for every worker thread identifier.
        try { // Keep cleanup available if an allocation, shader, or driver operation fails.
            for (std::size_t i = 0; i < workers; ++i) { // Visit this axis of the dispatch or current workgroup.
                const std::shared_ptr<queue_state> state = state_; // Set the shared device or worker state.
                workers_.push_back(std::thread([state] { // Retain this entry in the ordered resource or work collection.
                    consume(state); // Take accepted jobs from the queue and execute them on this worker.
                })); // End this scope in executor.
                state_->worker_ids.push_back(workers_.back().get_id()); // Retain this entry in the ordered resource or work collection.
            } // End this scope in executor.
        } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
            shutdown(); // Drain accepted CPU work and join the workers from an application thread.
            throw; // Rethrow the original failure after arranging the necessary cleanup.
        } // End this scope in executor.
    } // End this scope in executor.
    ~executor() { // Drain accepted work and join the CPU workers at explicit pool teardown.
        shutdown(); // Drain accepted CPU work and join the workers from an application thread.
    } // End this scope in executor.
    executor(const executor&) = delete; // Prevent copying this object and duplicating its retained identity.
    executor& operator=(const executor&) = delete; // Prevent copying this object and duplicating its retained identity.
    /** @brief Snapshot a host-ready CPU pipeline, enqueue it and return a future.
     * Each stage must support CPU execution and have current host data.
     */
    std::future<void> submit(const pipeline& source) { // Queue compute work and return its completion handle.
        reserve(); // Reserve capacity and report a full queue immediately.
        try { // Keep cleanup available if an allocation, shader, or driver operation fails.
            source.require_host_ready(); // Validate CPU stages and host-ready resources before queueing.
            const pipeline copy = source.snapshot(); // Capture isolated kernel interfaces and the current bindings and uniforms.
            return append_reserved([copy] { // Queue compute work and return its completion handle.
                copy.run(); // Execute the retained CPU snapshot on its worker thread.
            }); // End this scope in submit.
        } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
            abandon(); // Release the reserved slot after a submission failure.
            throw; // Rethrow the original failure after arranging the necessary cleanup.
        } // End this scope in submit.
    } // End this scope in submit.
    /** @brief Enqueue an application CPU function on the same bounded queue.
     * Run host operations in the callback and keep device operations on their context thread.
     */
    template <class Function> std::future<void> enqueue(Function function) { // Select the scalar, dimension, or access type at compile time.
        reserve(); // Reserve capacity and report a full queue immediately.
        try { // Keep cleanup available if an allocation, shader, or driver operation fails.
            return append_reserved(std::function<void()>(function)); // Queue the application callback on the bounded CPU executor.
        } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
            abandon(); // Release the reserved slot after a submission failure.
            throw; // Rethrow the original failure after arranging the necessary cleanup.
        } // End this scope in enqueue.
    } // End this scope in enqueue.
    /** @brief Return the count of queued or currently being constructed jobs. */
    std::size_t pending() const { // Read the count of jobs waiting for a worker.
        std::lock_guard<std::mutex> lock(state_->mutex); // Hold the mutex while this shared state is accessed.
        return state_->pending; // Read the count of jobs waiting for a worker.
    } // End this scope in pending.
    /** @brief Return the maximum number of pending jobs. */
    std::size_t capacity() const { // Return the configured pending-job limit.
        return state_->capacity; // Return the configured pending-job limit.
    } // End this scope in capacity.
    /** @brief Return the number of worker threads. */
    std::size_t workers() const { // Return the number of worker threads.
        return workers_.size(); // Return the number of worker threads.
    } // End this scope in workers.
    /** @brief Drain accepted jobs and join all workers. later submissions fail.
     * Call from an application thread so it can join every worker.
     */
    void shutdown() { // Drain accepted CPU work and join the workers from an application thread.
        if (std::find(state_->worker_ids.begin(), // Choose this branch when std::find(state ->worker ids.begin(),.
                      state_->worker_ids.end(), // Keep the end of the current range.
                      std::this_thread::get_id()) != state_->worker_ids.end()) // Reject joining the current executor worker from its own job.
            throw error("CPU executor shutdown requires an application thread"); // Report cpu executor shutdown requires an application thread.
        std::lock_guard<std::mutex> lifecycle(shutdown_mutex_); // Hold the mutex while this shared state is accessed.
        { // Begin the protected scope for this operation.
            std::lock_guard<std::mutex> lock(state_->mutex); // Hold the mutex while this shared state is accessed.
            state_->closing = true; // Set whether the CPU executor is draining for shutdown.
        } // End this scope in shutdown.
        state_->changed.notify_all(); // Wake workers so they can observe new queue or shutdown state.
        for (std::size_t i = 0; i < workers_.size(); ++i) // Visit each CPU worker during lifecycle handling.
            if (workers_[i].joinable()) // Choose this branch when workers [i].joinable().
                workers_[i].join(); // Join this CPU worker after accepted work has drained.
    } // End this scope in shutdown.
}; // End this scope in async.
} // namespace shdr
#endif // SHDR_ASYNC_HPP
