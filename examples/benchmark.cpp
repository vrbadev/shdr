/** @file benchmark.cpp
 * Seeded CPU/GPU comparisons for the example compute pipelines. Warmup excludes
 * compilation and allocation. Steady-state timings include hybrid backend transfers.
 * Final host readback and correctness comparisons occur after timed execution.
 */
#include "runner.hpp" // Include runner.hpp declarations.
#include "evt3_support.hpp" // Include evt3_support.hpp declarations.
#include "evt3_flow_checks.hpp" // Include evt3_flow_checks.hpp declarations.
#include "neural_support.hpp" // Include neural_support.hpp declarations.
#include "transformer_support.hpp" // Include transformer_support.hpp declarations.
#include "fft_support.hpp" // Include fft_support.hpp declarations.
#include "signal_filters.hpp" // Include signal_filters.hpp declarations.
#include "lenet_support.hpp" // Build the complete seven-stage image classifier.
#include <chrono> // Include chrono declarations.
#include <cstdlib> // Include cstdlib declarations.
#include <functional> // Include functional declarations.
#include <future> // Include future declarations.
#include <iomanip> // Include iomanip declarations.
#include <sstream> // Include sstream declarations.
#include <vector> // Include vector declarations.
#ifdef SHDR_PLUGIN_PATH // Enable this branch when its feature macro is defined.
#include <dlfcn.h> // Include dlfcn.h declarations.
#endif // Finish this conditional compilation branch.

namespace { // Group these helpers within their namespace.
using namespace shdr::glsl; // Allow concise GLSL type and builtin names in this scope.
const unsigned random_seed = 12345u; // Set random seed from this expression.

unsigned setting(const char* name, unsigned fallback, unsigned maximum) { // Implement setting for the operations below.
    const char* value = std::getenv(name); // Set value from this expression.
    if (!value) // Check this condition before entering the following branch.
        return fallback; // Return fallback.
    char* end = 0; // Set end from this expression.
    const unsigned long number = std::strtoul(value, &end, 10); // Set number from this expression.
    if (!*value || *end || !number || number > maximum) // Check this condition before entering the following branch.
        throw std::invalid_argument(std::string(name) + " must be between 1 and " + // Report  must be between 1 and .
                                    std::to_string(maximum)); // Apply to string to these arguments.
    return unsigned(number); // Return unsigned(number).
} // End this block.

shdr::extent groups(const shdr::kernel& kernel, unsigned x, unsigned y = 1, unsigned z = 1) { // Set y from this expression.
    const shdr::extent local = kernel.local_size(); // Set local from this expression.
    return shdr::extent( // Return shdr::extent(.
        (x + local.x - 1) / local.x, (y + local.y - 1) / local.y, (z + local.z - 1) / local.z); // Continue the expression using x, local, x.
} // End this block.

void same(float actual, float expected, double absolute, double relative) { // Compare CPU and GPU values using the selected tolerance.
    const double difference = std::fabs(double(actual) - double(expected)); // Set difference from this expression.
    if (!std::isfinite(actual) || !std::isfinite(expected) || // Check this condition before entering the following branch.
        difference > // Continue the expression using difference.
            absolute + relative * std::max(std::fabs(double(actual)), std::fabs(double(expected)))) // Apply max to these arguments.
        throw std::runtime_error("CPU/GPU float mismatch: " + std::to_string(actual) + " versus " + // Report CPU/GPU float mismatch: .
                                 std::to_string(expected)); // Apply to string to these arguments.
} // End this block.
void same(unsigned actual, unsigned expected, double, double) { // Compare CPU and GPU values using the selected tolerance.
    if (actual != expected) // Check this condition before entering the following branch.
        throw std::runtime_error("CPU/GPU uint mismatch"); // Report CPU/GPU uint mismatch.
} // End this block.
template <class T, std::size_t N> // Accept the value types or dimensions used by this template.
void same(const shdr::glsl::vec<T, N>& actual, // Compare CPU and GPU values using the selected tolerance.
          const shdr::glsl::vec<T, N>& expected, // Declare expected for the following calculation.
          double absolute, // Use this floating-point value or resource in the following calculation.
          double relative) { // Use this floating-point value or resource in the following calculation.
    for (std::size_t c = 0; c < N; ++c) // Iterate over c for this calculation.
        same(actual[c], expected[c], absolute, relative); // Compare CPU and GPU values using the selected tolerance.
} // End this block.

struct comparison { // Declare comparison.
    virtual ~comparison() { // Apply comparison to these arguments.
    } // End this block.
    virtual void capture() = 0; // Construct capture with these initial values.
    virtual void verify() = 0; // Construct verify with these initial values.
}; // Finish this declaration.
template <class T> class output_check : public comparison { // Accept the value types or dimensions used by this template.
    const shdr::buffer<T>& resource_; // Declare resource for the following calculation.
    std::vector<T> expected_; // Declare expected for the following calculation.
    double absolute_, relative_; // Use this floating-point value or resource in the following calculation.

  public: // Set public access for the following class members.
    explicit output_check(const shdr::buffer<T>& resource, // Declare output check with the parameters continued below.
                          double absolute = 1e-4, // Use this floating-point value or resource in the following calculation.
                          double relative = 1e-4) // Use this floating-point value or resource in the following calculation.
        : resource_(resource), absolute_(absolute), relative_(relative) { // Implement resource for the operations below.
    } // End this block.
    void capture() override { // Retain a host copy of this CPU output after its timed execution.
        const T* data = resource_.data(); // Set data from this expression.
        expected_.assign(data, data + resource_.size()); // Copy the complete CPU reference while the resource remains unmodified.
    } // End this block.
    void verify() override { // Synchronize this GPU output and compare each element outside the timed section.
        const T* data = resource_.data(); // Set data from this expression.
        for (std::size_t i = 0; i < expected_.size(); ++i) { // Iterate over i for this calculation.
            try { // Run the operation while retaining the error handling below.
                same(data[i], expected_[i], absolute_, relative_); // Compare CPU and GPU values using the selected tolerance.
            } catch (const std::exception& error) { // Report the exception from the preceding operation.
                throw std::runtime_error(std::string(error.what()) + " at element " + // Report  at element .
                                         std::to_string(i)); // Apply to string to these arguments.
            } // End this block.
        } // End this block.
    } // End this block.
}; // Finish this declaration.

struct timing { // Declare timing.
    std::string name, size; // Declare name for the following calculation.
    double cpu, gpu; // Use this floating-point value or resource in the following calculation.
}; // Finish this declaration.
class benchmark { // Declare benchmark.
    shdr::device* device_; // Declare device for the following calculation.
    unsigned iterations_; // Declare iterations for the following calculation.
    std::vector<timing> timings_; // Declare timings for the following calculation.
    std::vector<std::string> metadata_; // Declare metadata for the following calculation.

  public: // Set public access for the following class members.
    benchmark(shdr::device* device, unsigned iterations) // Apply benchmark to these arguments.
        : device_(device), iterations_(iterations) { // Implement device for the operations below.
    } // End this block.
    void note(const std::string& text) { // Implement note for the operations below.
        metadata_.push_back(text); // Apply push back to these arguments.
    } // End this block.
    void measure(const std::string& name, // Measure one reusable pipeline after both backends have been warmed up.
                 const std::string& size, // Declare size for the following calculation.
                 const shdr::pipeline& pipeline, // Declare pipeline for the following calculation.
                 std::initializer_list<comparison*> checks) { // Declare checks for the following calculation.
        measure_operations( // Apply measure operations to these arguments.
            name, // Continue the expression using name.
            size, // Continue the expression using size.
            [&pipeline](bool) { // Continue the expression using pipeline, bool.
                pipeline.run(); // Execute the configured pipeline on the requested backend.
            }, // Finish this declaration.
            [&pipeline, this](bool) { // Continue the expression using pipeline, this, bool.
                pipeline.run(*device_); // Execute the configured pipeline on the requested backend.
            }, // Finish this declaration.
            checks); // Continue the expression using checks.
    } // End this block.
    void measure_operations(const std::string& name, // Measure completed resident operations with reset and validation outside the timed loop.
                            const std::string& size, // Declare size for the following calculation.
                            const std::function<void(bool)>& cpu_operation, // Declare cpu operation for the following calculation.
                            const std::function<void(bool)>& gpu_operation, // Declare gpu operation for the following calculation.
                            std::initializer_list<comparison*> checks) { // Declare checks for the following calculation.
        typedef std::chrono::steady_clock clock; // Measure elapsed time with the monotonic steady-clock counter.
        cpu_operation(true); // Warm up CPU execution and reset stateful operations to their documented initial state.
        clock::time_point begin = clock::now(); // Begin timing only after the CPU warmup has finished.
        for (unsigned i = 0; i < iterations_; ++i) // Iterate over i for this calculation.
            cpu_operation(false); // Run the next CPU operation with the state produced by its predecessor.
        const double cpu = // Calculate the mean completed CPU operation time in milliseconds.
            std::chrono::duration<double, std::milli>(clock::now() - begin).count() / iterations_; // Apply now to these arguments.
        for (comparison* check : checks) // Visit each check required by this loop.
            check->capture(); // Save the CPU outputs after timing so host comparison work is excluded.
        double gpu = 0.0; // Initialize the GPU timing field for the optional device run.
        if (device_) { // Check this condition before entering the following branch.
            gpu_operation(true); // Compile, upload and warm up GPU resources before the timer starts.
            device_->synchronize(); // Finish all warmup GPU commands before starting the timer.
            begin = clock::now(); // Begin timing only after all warmup GPU commands have completed.
            for (unsigned i = 0; i < iterations_; ++i) // Iterate over i for this calculation.
                gpu_operation(false); // Run the next resident GPU operation before collecting final results.
            device_->synchronize(); // Wait for the timed GPU commands to complete before stopping the timer.
            gpu = std::chrono::duration<double, std::milli>(clock::now() - begin).count() / // Convert completed GPU elapsed time into milliseconds.
                  iterations_; // Divide total GPU time by the number of complete operations.
            for (comparison* check : checks) { // Visit each check required by this loop.
                try { // Run the operation while retaining the error handling below.
                    check->verify(); // Read outputs after timing and compare them with the retained CPU reference.
                } catch (const std::exception& error) { // Report the exception from the preceding operation.
                    throw std::runtime_error(name + ": " + error.what()); // Report : .
                } // End this block.
            } // End this block.
        } // End this block.
        timings_.push_back(timing{name, size, cpu, gpu}); // Retain the workload and both completed-operation timings for the results table.
    } // End this block.
    void print(unsigned scale) const { // Write the reproducible input settings, renderer and completed-operation timing table.
        std::cout << "Seed: " << random_seed << ", iterations: " << iterations_ // Write this part of the example result or diagnostic.
                  << ", scale: " << scale << ".\n"; // Write this part of the example result or diagnostic.
        if (device_) // Check this condition before entering the following branch.
            std::cout << "GPU renderer: " << device_->renderer() << ".\n"; // Write this part of the example result or diagnostic.
        for (const std::string& text : metadata_) // Visit each std required by this loop.
            std::cout << text << "\n"; // Write this part of the example result or diagnostic.
        std::cout // Write this part of the example result or diagnostic.
            << "Steady-state resident timings, compilation, first uploads and final readback excluded.\n" // Write this part of the example result or diagnostic.
            << "Hybrid timings include transfers between CPU and GPU stages.\n\n" // Write this part of the example result or diagnostic.
            << "| Pipeline | Workload | CPU ms | GPU ms | CPU/GPU |\n" // Write this part of the example result or diagnostic.
            << "| --- | --- | ---: | ---: | ---: |\n" // Write this part of the example result or diagnostic.
            << std::fixed << std::setprecision(3); // Write this part of the example result or diagnostic.
        for (const timing& row : timings_) { // Visit each row required by this loop.
            std::cout << "| " << row.name << " | " << row.size << " | " << row.cpu << " | "; // Write this part of the example result or diagnostic.
            if (device_) // Check this condition before entering the following branch.
                std::cout << row.gpu << " | " << row.cpu / row.gpu; // Write this part of the example result or diagnostic.
            else // Print placeholders for the CPU execution mode.
                std::cout << "- | -"; // Write this part of the example result or diagnostic.
            std::cout << " |\n"; // Write this part of the example result or diagnostic.
        } // End this block.
    } // End this block.
}; // Finish this declaration.

void random_complex(shdr::buffer<vec2>& values, unsigned seed) { // Implement random complex for the operations below.
    std::mt19937 generator(seed); // Construct generator with these initial values.
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f); // Construct distribution with these initial values.
    vec2* data = values.data(); // Set data from this expression.
    for (std::size_t i = 0; i < values.size(); ++i) // Iterate over i for this calculation.
        data[i] = vec2(distribution(generator), distribution(generator)); // Set data[i] from this expression.
} // End this block.
std::string dimensions(unsigned x, unsigned y, unsigned z = 0) { // Set z from this expression.
    return std::to_string(x) + " x " + std::to_string(y) + (z ? " x " + std::to_string(z) : ""); // Return the result of this operation.
} // End this block.

void buffer_pipeline(benchmark& bench, unsigned count, bool hybrid, shdr::kernel* external = 0) { // Set external from this expression.
    shdr::buffer<float> input(count), scaled(count), clipped(count), result(1); // Construct input with these initial values.
    shdr_examples::randomize(input, random_seed, -4.0f, 12.0f); // Fill the input with reproducible random values.
    auto owned_affine = // Set owned affine from this expression.
        external ? std::unique_ptr<shdr::kernel>() : shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shdr::kernel& affine = external ? *external : *owned_affine; // Set affine from this expression.
    affine.bind("input_values", input) // Connect this shader port to its application resource.
        .bind("output_values", scaled) // Connect this shader port to its application resource.
        .set<unsigned>("count", count) // Supply this shader uniform with its typed application value.
        .set<float>("scale", 2.0f) // Supply this shader uniform with its typed application value.
        .set<float>("offset", 1.0f); // Supply this shader uniform with its typed application value.
    auto clamp = shdr::make_kernel("clamp_kernel"); // Create the linked clamp kernel instance.
    clamp->bind("input_values", scaled) // Connect this shader port to its application resource.
        .bind("output_values", clipped) // Connect this shader port to its application resource.
        .set<unsigned>("count", count) // Supply this shader uniform with its typed application value.
        .set<float>("lower", 0.0f) // Supply this shader uniform with its typed application value.
        .set<float>("upper", 8.0f); // Supply this shader uniform with its typed application value.
    auto sum = shdr::make_kernel("sum_kernel"); // Create the linked sum kernel instance.
    sum->bind("input_values", clipped).bind("output_values", result).set<unsigned>("count", count); // Connect this shader port to its application resource.
    shdr::pipeline pipeline = // Set pipeline from this expression.
        shdr::dispatch(affine, groups(affine, count)) >> // Add a shader dispatch with the required workgroup counts.
        shdr::dispatch(*clamp, // Add a shader dispatch with the required workgroup counts.
                       groups(*clamp, count), // Apply groups to these arguments.
                       hybrid ? shdr::backend::cpu : shdr::backend::automatic) >> // Continue the expression using hybrid, shdr, backend.
        shdr::dispatch(*sum, groups(*sum, 1)); // Add a shader dispatch with the required workgroup counts.
    output_check<float> check_result(result), check_clipped(clipped); // Construct check result with these initial values.
    bench.measure(hybrid     ? "Hybrid affine/clamp/sum" // Apply measure to these arguments.
                  : external ? "Dynamic library stages" // Declare external for the following calculation.
                             : "Buffer affine/clamp/sum", // Provide this text argument for the operation above.
                  std::to_string(count) + " floats", // Apply to string to these arguments.
                  pipeline, // Continue the expression using pipeline.
                  {&check_result, &check_clipped}); // Continue the expression using check result, check clipped.
} // End this block.

void image_pipeline(benchmark& bench, // Declare image pipeline with the parameters continued below.
                    unsigned side, // Declare side for the following calculation.
                    shdr::device* device = 0, // Set device from this expression.
                    bool borrow = false) { // Set borrow from this expression.
    shdr::buffer<float> input(side * side), output(side * side); // Construct input with these initial values.
    shdr_examples::randomize(input, random_seed, 0.0f, 1.0f); // Fill the input with reproducible random values.
    // The application retains ownership of a texture temporarily borrowed by shdr.
    shdr::image2d<float, 1> application_image(side, side), filtered(side, side); // Construct application image with these initial values.
    shdr::image2d<float, 1> original = // Set original from this expression.
        borrow && device ? shdr::image2d<float, 1>::from_gl( // Apply from gl to these arguments.
                               *device, application_image.gl_handle(*device), side, side) // Apply gl handle to these arguments.
                         : application_image; // Declare application image for the following calculation.
    auto into = shdr::make_kernel("buffer_to_image_kernel"); // Create the linked buffer to image kernel instance.
    into->bind("input_values", input).bind("output_image", original); // Connect this shader port to its application resource.
    auto blur = shdr::make_kernel("blur_kernel"); // Create the linked blur kernel instance.
    blur->bind("input_image", original).bind("output_image", filtered); // Connect this shader port to its application resource.
    auto out = shdr::make_kernel("image_to_buffer_kernel"); // Create the linked image to buffer kernel instance.
    out->bind("input_image", filtered).bind("output_values", output); // Connect this shader port to its application resource.
    shdr::pipeline pipeline = shdr::dispatch(*into, groups(*into, side, side)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(*blur, groups(*blur, side, side)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(*out, groups(*out, side, side)); // Add a shader dispatch with the required workgroup counts.
    output_check<float> check(output); // Construct check with these initial values.
    bench.measure(borrow ? "Borrowed image blur" : "Buffer/image blur", // Apply measure to these arguments.
                  dimensions(side, side), // Apply dimensions to these arguments.
                  pipeline, // Continue the expression using pipeline.
                  {&check}); // Continue the expression using check.
} // End this block.

void bayer_pipeline(benchmark& bench, unsigned side) { // Implement bayer pipeline for the operations below.
    const unsigned count = side * side, words = (count / 4u * 5u + 3u) / 4u; // Set count from this expression.
    shdr::buffer<unsigned> packed(words), samples(count); // Construct packed with these initial values.
    shdr::buffer<vec4> colors(count); // Construct colors with these initial values.
    shdr::buffer<float> gray(count); // Construct gray with these initial values.
    std::mt19937 generator(random_seed); // Construct generator with these initial values.
    for (unsigned i = 0; i < words; ++i) // Iterate over i for this calculation.
        packed.data()[i] = generator(); // Apply data to these arguments.
    auto unpack = shdr::make_kernel("bayer_unpack_kernel"); // Create the linked bayer unpack kernel instance.
    unpack->bind("packed_words", packed) // Connect this shader port to its application resource.
        .bind("samples", samples) // Connect this shader port to its application resource.
        .set<unsigned>("count", count) // Supply this shader uniform with its typed application value.
        .set<unsigned>("packing", 2u) // Supply this shader uniform with its typed application value.
        .set<unsigned>("bit_depth", 10u); // Supply this shader uniform with its typed application value.
    auto debayer = shdr::make_kernel("debayer_kernel"); // Create the linked debayer kernel instance.
    debayer->bind("samples", samples) // Connect this shader port to its application resource.
        .bind("colors", colors) // Connect this shader port to its application resource.
        .set<unsigned>("width", side) // Supply this shader uniform with its typed application value.
        .set<unsigned>("height", side) // Supply this shader uniform with its typed application value.
        .set<unsigned>("pattern", 0u) // Supply this shader uniform with its typed application value.
        .set<unsigned>("bit_depth", 10u); // Supply this shader uniform with its typed application value.
    auto luminance = shdr::make_kernel("luminance_kernel"); // Create the linked luminance kernel instance.
    luminance->bind("colors", colors).bind("gray", gray).set<unsigned>("count", count); // Connect this shader port to its application resource.
    shdr::pipeline pipeline = shdr::dispatch(*unpack, groups(*unpack, count)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(*debayer, groups(*debayer, side, side)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(*luminance, groups(*luminance, count)); // Add a shader dispatch with the required workgroup counts.
    output_check<unsigned> sample_check(samples); // Construct sample check with these initial values.
    output_check<vec4> color_check(colors); // Construct color check with these initial values.
    output_check<float> gray_check(gray); // Construct gray check with these initial values.
    bench.measure("RAW10/debayer/luminance", // Apply measure to these arguments.
                  dimensions(side, side), // Apply dimensions to these arguments.
                  pipeline, // Continue the expression using pipeline.
                  {&sample_check, &color_check, &gray_check}); // Continue the expression using sample check, color check, gray check.
} // End this block.

void evt3_pipeline(benchmark& bench, unsigned word_count) { // Implement evt3 pipeline for the operations below.
    shdr_examples::evt3_decoder decoder(word_count); // Construct decoder with these initial values.
    decoder.load(shdr_examples::evt3_random_words(word_count, random_seed)); // Apply load to these arguments.
    output_check<uvec4> event_check(decoder.events), state_check(decoder.state); // Construct event check with these initial values.
    output_check<unsigned> offset_check(decoder.offsets); // Construct offset check with these initial values.
    bench.measure("EVT3 event decoding", // Apply measure to these arguments.
                  std::to_string(word_count) + " packed words", // Apply to string to these arguments.
                  decoder.reset_pipeline(), // Apply reset pipeline to these arguments.
                  {&event_check, &state_check, &offset_check}); // Continue the expression using event check, state check, offset check.
    bench.note("EVT3 compact output: " + std::to_string(decoder.event_count()) + " events from " + // Apply note to these arguments.
               std::to_string(word_count) + " words, five stages include state reset."); // Apply to string to these arguments.
} // End this block.

void evt3_optical_flow(benchmark& bench) { // Implement evt3 optical flow for the operations below.
    const std::vector<std::uint16_t> plane = shdr_examples::evt3_planar_words(32, 24); // Set plane from this expression.
    std::vector<std::uint16_t> words; // Declare words for the following calculation.
    std::mt19937 random(random_seed); // Construct random with these initial values.
    for (std::size_t i = 0; i < plane.size(); i += 4) { // Iterate over i for this calculation.
        // Seeded spatial sampling varies neighborhoods while preserving the known temporal plane.
        if (random() % 5u == 0u) // Check this condition before entering the following branch.
            continue; // Skip the remaining work for this loop iteration.
        words.insert(words.end(), plane.begin() + i, plane.begin() + i + 4); // Apply insert to these arguments.
    } // End this block.
    shdr_examples::evt3_decoder decoder(static_cast<unsigned>(words.size())); // Construct decoder with these initial values.
    decoder.load(words); // Apply load to these arguments.
    shdr::buffer<vec4> flow(decoder.events.size()); // Construct flow with these initial values.
    auto shader = shdr_examples::evt3_flow_kernel(decoder, flow, 64u); // Set shader from this expression.
    shdr::pipeline pipeline = decoder.reset_pipeline(); // Set pipeline from this expression.
    pipeline += shdr::dispatch(*shader, groups(*shader, static_cast<unsigned>(flow.size()))); // Add a shader dispatch with the required workgroup counts.
    output_check<uvec4> event_check(decoder.events); // Construct event check with these initial values.
    output_check<unsigned> offset_check(decoder.offsets); // Construct offset check with these initial values.
    output_check<vec4> flow_check(flow, 1e-3, 1e-3); // Construct flow check with these initial values.
    bench.measure("EVT3 + optical flow", // Apply measure to these arguments.
                  "32 x 24 sampled plane, history 64", // Provide this text argument for the operation above.
                  pipeline, // Continue the expression using pipeline.
                  {&event_check, &offset_check, &flow_check}); // Continue the expression using event check, offset check, flow check.
    bench.note("Optical flow input: " + std::to_string(decoder.event_count()) + // Apply note to these arguments.
               " seeded samples on a known temporal plane, six linked stages."); // Provide this text argument for the operation above.
} // End this block.

void binary_neural_pipeline(benchmark& bench, shdr::device* device, bool convolutional) { // Implement binary neural pipeline for the operations below.
    shdr_examples::binary_network model(convolutional, 32); // Construct model with these initial values.
    const std::string name = convolutional ? "CNN" : "ANN"; // Set name from this expression.
    const std::string size = std::to_string(model.batch_count()) + " samples, " + // Set size from this expression.
                             std::to_string(model.parameter_count()) + " parameters"; // Apply to string to these arguments.
    output_check<float> parameter_check(model.parameters), gradient_check(model.gradients); // Construct parameter check with these initial values.
    bench.measure_operations( // Apply measure operations to these arguments.
        name + " training", // Provide this text argument for the operation above.
        size, // Continue the expression using size.
        [&](bool warmup) { // Continue the expression using bool, warmup.
            if (warmup) // Check this condition before entering the following branch.
                model.reset_parameters(); // Restore the same initial parameter vector before this backend's warmup update.
            model.training_pipeline().run(); // Compute all sample gradients and apply one averaged SGD parameter update.
        }, // Finish this declaration.
        [&](bool warmup) { // Continue the expression using bool, warmup.
            if (warmup) // Check this condition before entering the following branch.
                model.reset_parameters(); // Restore the same initial parameter vector before this backend's warmup update.
            model.training_pipeline().run(*device); // Compute all sample gradients and apply one averaged SGD parameter update.
        }, // Finish this declaration.
        {&parameter_check, &gradient_check}); // Continue the expression using parameter check, gradient check.
    output_check<float> prediction_check(model.predictions); // Construct prediction check with these initial values.
    bench.measure(name + " inference", size, model.prediction_pipeline(), {&prediction_check}); // Apply measure to these arguments.
} // End this block.

void transformer_pipeline(benchmark& bench, shdr::device* device) { // Implement transformer pipeline for the operations below.
    shdr_examples::transformer_model model(8); // Construct model with these initial values.
    model.reset_inputs(random_seed); // Apply reset inputs to these arguments.
    const std::string size = "8 sequences, 4 tokens x 4, 134 parameters"; // Set size from this expression.
    output_check<float> parameter_check(model.parameters, 1e-4, 1e-3), // Declare parameter check with the parameters continued below.
        gradient_check(model.gradients, 1e-4, 1e-3); // Apply gradient check to these arguments.
    bench.measure_operations( // Apply measure operations to these arguments.
        "Transformer training", // Provide this text argument for the operation above.
        size, // Continue the expression using size.
        [&](bool warmup) { // Continue the expression using bool, warmup.
            if (warmup) // Check this condition before entering the following branch.
                model.reset_parameters(random_seed); // Restore the same initial parameter vector before this backend's warmup update.
            model.training_pipeline().run(); // Compute all sample gradients and apply one averaged SGD parameter update.
        }, // Finish this declaration.
        [&](bool warmup) { // Continue the expression using bool, warmup.
            if (warmup) // Check this condition before entering the following branch.
                model.reset_parameters(random_seed); // Restore the same initial parameter vector before this backend's warmup update.
            model.training_pipeline().run(*device); // Compute all sample gradients and apply one averaged SGD parameter update.
        }, // Finish this declaration.
        {&parameter_check, &gradient_check}); // Continue the expression using parameter check, gradient check.
    output_check<vec4> prediction_check(model.predictions, 1e-4, 1e-3); // Construct prediction check with these initial values.
    bench.measure("Transformer inference", size, model.prediction_pipeline(), {&prediction_check}); // Apply measure to these arguments.
    bench.note( // Apply note to these arguments.
        "Neural training timings measure one mini-batch gradient/SGD update, inference measures the complete forward pipeline."); // Provide this text argument for the operation above.
} // End this block.

void fft_round_trip(benchmark& bench) { // Implement fft round trip for the operations below.
    const unsigned length = 1024, channels = 8; // Set length from this expression.
    shdr::buffer<vec2> input(length * channels), spectrum(length * channels), // Declare input with the parameters continued below.
        output(length * channels); // Apply output to these arguments.
    random_complex(input, random_seed); // Apply random complex to these arguments.
    shdr_examples::fft1d forward(input, spectrum, length, channels); // Construct forward with these initial values.
    shdr_examples::fft1d inverse(spectrum, output, length, channels, true); // Construct inverse with these initial values.
    shdr::pipeline pipeline = forward.processing_pipeline(); // Set pipeline from this expression.
    inverse.append_to(pipeline); // Apply append to to these arguments.
    output_check<vec2> spectrum_check(spectrum, 1e-4, 1e-3), output_comparison(output, 1e-4, 1e-3); // Construct spectrum check with these initial values.
    bench.measure( // Apply measure to these arguments.
        "FFT forward/inverse", "8 x 1024 complex", pipeline, {&spectrum_check, &output_comparison}); // Provide this text argument for the operation above.
    const shdr::buffer<vec2>& read_input = input; // Use a constant buffer view so GPU writes synchronize before host reads.
    const shdr::buffer<vec2>& read_output = output; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned i = 0; i < length * channels; ++i) // Iterate over i for this calculation.
        same(read_output.data()[i], read_input.data()[i], 1e-4, 1e-3); // Compare CPU and GPU values using the selected tolerance.
} // End this block.

void fft_band_pass(benchmark& bench) { // Implement fft band pass for the operations below.
    const unsigned length = 1024, channels = 8; // Set length from this expression.
    shdr::buffer<vec2> input(length * channels), output(length * channels); // Construct input with these initial values.
    random_complex(input, random_seed); // Apply random complex to these arguments.
    shdr_examples::fft_filter filter(input, // Declare filter with the parameters continued below.
                                     output, // Continue the expression using output.
                                     length, // Continue the expression using length.
                                     channels, // Continue the expression using channels.
                                     48000.0f, // Complete this expression or declaration.
                                     shdr_examples::frequency_filter::band_pass, // Continue the expression using shdr examples, frequency filter, band pass.
                                     4000.0f, // Complete this expression or declaration.
                                     10000.0f); // Complete this expression or declaration.
    output_check<vec2> check(output, 1e-4, 1e-3); // Construct check with these initial values.
    bench.measure( // Apply measure to these arguments.
        "FFT band-pass", "8 x 1024 complex, 4-10 kHz", filter.processing_pipeline(), {&check}); // Apply processing pipeline to these arguments.
} // End this block.

void fir_low_pass(benchmark& bench) { // Implement fir low pass for the operations below.
    const unsigned count = 2048, channels = 8; // Set count from this expression.
    shdr::buffer<float> input(count * channels), output(count * channels); // Construct input with these initial values.
    shdr_examples::randomize(input, random_seed, -1.0f, 1.0f); // Fill the input with reproducible random values.
    shdr_examples::fir_pipeline filter(input, // Declare filter with the parameters continued below.
                                       output, // Continue the expression using output.
                                       count, // Continue the expression using count.
                                       channels, // Continue the expression using channels.
                                       shdr_examples::filter_mode::low_pass, // Continue the expression using shdr examples, filter mode, low pass.
                                       48000.0, // Complete this expression or declaration.
                                       4000.0, // Complete this expression or declaration.
                                       0.0, // Complete this expression or declaration.
                                       65); // Complete this expression or declaration.
    output_check<float> check(output); // Construct check with these initial values.
    bench.measure( // Apply measure to these arguments.
        "FIR low-pass", "8 x 2048 floats, 65 taps", filter.processing_pipeline(), {&check}); // Apply processing pipeline to these arguments.
} // End this block.

void biquad_low_pass(benchmark& bench, shdr::device* device) { // Implement biquad low pass for the operations below.
    const unsigned count = 2048, channels = 8; // Set count from this expression.
    shdr::buffer<float> input(count * channels), output(count * channels); // Construct input with these initial values.
    shdr_examples::randomize(input, random_seed, -1.0f, 1.0f); // Fill the input with reproducible random values.
    shdr_examples::biquad_pipeline filter( // Declare filter with the parameters continued below.
        input, output, count, channels, shdr_examples::filter_mode::low_pass, 48000.0, 4000.0); // Continue the expression using input, output, count.
    output_check<float> output_comparison(output); // Construct output comparison with these initial values.
    output_check<vec2> state_comparison(filter.state); // Construct state comparison with these initial values.
    bench.measure_operations( // Apply measure operations to these arguments.
        "Biquad low-pass", // Provide this text argument for the operation above.
        "8 x 2048 floats, persistent state", // Provide this text argument for the operation above.
        [&](bool warmup) { // Continue the expression using bool, warmup.
            if (warmup) // Check this condition before entering the following branch.
                filter.reset_state(); // Restore identical zero-delay state before this backend processes its first signal block.
            filter.processing_pipeline().run(); // Execute the configured pipeline on the requested backend.
        }, // Finish this declaration.
        [&](bool warmup) { // Continue the expression using bool, warmup.
            if (warmup) // Check this condition before entering the following branch.
                filter.reset_state(); // Restore identical zero-delay state before this backend processes its first signal block.
            filter.processing_pipeline().run(*device); // Execute the configured pipeline on the requested backend.
        }, // Finish this declaration.
        {&output_comparison, &state_comparison}); // Continue the expression using output comparison, state comparison.
} // End this block.

/// Compare completed asynchronous batches, with readback confined to warmup validation.
void async_pipeline(benchmark& bench, shdr::device* device) { // Implement async pipeline for the operations below.
    const unsigned count = 257, frame_count = 32, slot_count = 3; // Set count from this expression.
    struct slot { // Declare slot.
        shdr::buffer<float> input, output; // Declare input for the following calculation.
        std::future<void> cpu_done; // Declare cpu done for the following calculation.
        shdr::submission gpu_done; // Declare gpu done for the following calculation.
        bool occupied; // Declare occupied for the following calculation.
        unsigned frame; // Declare frame for the following calculation.
        slot() : input(count), output(count), occupied(false), frame(0) { // Apply slot to these arguments.
        } // End this block.
    }; // Finish this declaration.
    std::vector<slot> slots(slot_count); // Allocate three disjoint input and output slots before measuring the ring.
    for (unsigned i = 0; i < slot_count; ++i) // Iterate over i for this calculation.
        shdr_examples::randomize(slots[i].input, random_seed + i, -1.0f, 1.0f); // Fill the input with reproducible random values.
    auto shader = shdr::make_kernel("affine_kernel"); // Create the linked affine kernel instance.
    shader->set<unsigned>("count", count); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline = shdr::dispatch(*shader, groups(*shader, count)); // Add a shader dispatch with the required workgroup counts.
    shdr::executor workers(2, slot_count); // Use two CPU workers with enough bounded queue capacity for the three live slots.
    if (device) { // Check this condition before entering the following branch.
        for (slot& resource : slots) { // Visit each resource required by this loop.
            shader->bind("input_values", resource.input).bind("output_values", resource.output); // Connect this shader port to its application resource.
            pipeline.prepare(*device); // Prepare all slot resources and compile the GPU program before ring timing.
        } // End this block.
    } // End this block.
    const auto execute = [&](bool gpu, bool validate) { // Set execute from this expression.
        unsigned produced = 0, completed = 0; // Set produced from this expression.
        while (completed < frame_count) { // Pump the three resource slots until the complete frame batch finishes.
            bool progress = false; // Set progress from this expression.
            for (unsigned index = 0; index < slot_count; ++index) { // Iterate over index for this calculation.
                slot& resource = slots[index]; // Set resource from this expression.
                if (resource.occupied) { // Check this condition before entering the following branch.
                    const bool ready = // Set ready from this expression.
                        gpu ? resource.gpu_done.ready() // Apply ready to these arguments.
                            : resource.cpu_done.wait_for(std::chrono::nanoseconds(0)) == // Declare resource for the following calculation.
                                  std::future_status::ready; // Continue the expression using std, future status, ready.
                    if (!ready) // Keep this slot occupied until its preceding operation actually finishes.
                        continue; // Skip the remaining work for this loop iteration.
                    if (!gpu) // Check this condition before entering the following branch.
                        resource.cpu_done.get(); // Observe CPU worker exceptions before considering this slot reusable.
                    if (validate) { // Check every completed warmup frame while keeping readback outside the timed ring.
                        const shdr::buffer<float>& input = resource.input; // Use a constant buffer view so GPU writes synchronize before host reads.
                        const shdr::buffer<float>& output = resource.output; // Use a constant buffer view so GPU writes synchronize before host reads.
                        for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
                            same(output.data()[i], // Compare CPU and GPU values using the selected tolerance.
                                 input.data()[i] * float(resource.frame + 1) - // Apply data to these arguments.
                                     float(resource.frame), // Apply float to these arguments.
                                 1e-4, // Complete this expression or declaration.
                                 1e-4); // Complete this expression or declaration.
                    } // End this block.
                    resource.occupied = false; // Release this slot only after its prior frame has completed.
                    ++completed; // Advance the completed-frame counter after observing its fence or future.
                    progress = true; // Set progress from this expression.
                } // End this block.
                if (produced < frame_count && produced % slot_count == index) { // Assign frames round-robin so both backends receive the same slot inputs.
                    resource.frame = produced++; // Record the frame identity before changing its shader uniforms.
                    shader->bind("input_values", resource.input) // Connect this shader port to its application resource.
                        .bind("output_values", resource.output) // Connect this shader port to its application resource.
                        .set<float>("scale", float(resource.frame + 1)) // Supply this shader uniform with its typed application value.
                        .set<float>("offset", -float(resource.frame)); // Supply this shader uniform with its typed application value.
                    if (gpu) // Check this condition before entering the following branch.
                        resource.gpu_done = pipeline.submit(*device); // Queue the frame on GPU and retain its zero-timeout completion fence.
                    else // Queue the frame snapshot on the CPU worker pool.
                        resource.cpu_done = workers.submit(pipeline); // Queue a snapshot of the frame bindings for the bounded CPU executor.
                    resource.occupied = true; // Protect this slot from reuse until its submission or future is ready.
                    progress = true; // Set progress from this expression.
                } // End this block.
            } // End this block.
            if (gpu) // Check this condition before entering the following branch.
                device->collect_garbage(); // Poll and retire completed GPU fence records on the producer thread.
            if (!progress) // Check this condition before entering the following branch.
                std::this_thread::yield(); // Yield the CPU when all live slots are still busy.
        } // End this block.
    }; // Finish this declaration.
    output_check<float> a(slots[0].output), b(slots[1].output), c(slots[2].output); // Construct a with these initial values.
    bench.measure_operations( // Apply measure operations to these arguments.
        "Async ring", // Provide this text argument for the operation above.
        "32 x 257 floats, 3 slots", // Provide this text argument for the operation above.
        [&](bool validate) { // Continue the expression using bool, validate.
            execute(false, validate); // Apply execute to these arguments.
        }, // Finish this declaration.
        [&](bool validate) { // Continue the expression using bool, validate.
            execute(true, validate); // Apply execute to these arguments.
        }, // Finish this declaration.
        {&a, &b, &c}); // Continue the expression using a, b, c.
    workers.shutdown(); // Apply shutdown to these arguments.
} // End this block.

void convolution_2d(benchmark& bench, unsigned side) { // Implement convolution 2d for the operations below.
    shdr::buffer<vec2> input(side * side), output(side * side); // Construct input with these initial values.
    shdr::buffer<float> coefficients(25); // Construct coefficients with these initial values.
    random_complex(input, random_seed); // Apply random complex to these arguments.
    shdr_examples::randomize(coefficients, random_seed + 1, -0.04f, 0.04f); // Fill the input with reproducible random values.
    auto convolution = shdr::make_kernel("convolution2d_kernel"); // Create the linked convolution2d kernel instance.
    convolution->bind("input_values", input) // Connect this shader port to its application resource.
        .bind("output_values", output) // Connect this shader port to its application resource.
        .bind("coefficients", coefficients) // Connect this shader port to its application resource.
        .set<uvec2>("dimensions", uvec2(side, side)) // Supply this shader uniform with its typed application value.
        .set<uvec2>("kernel_size", uvec2(5, 5)); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline; // Declare pipeline for the following calculation.
    pipeline += shdr::dispatch(*convolution, groups(*convolution, side, side)); // Add a shader dispatch with the required workgroup counts.
    output_check<vec2> check(output); // Construct check with these initial values.
    bench.measure("2D convolution 5 x 5", dimensions(side, side) + " vec2", pipeline, {&check}); // Apply measure to these arguments.
} // End this block.

void convolution_3d(benchmark& bench, unsigned side) { // Implement convolution 3d for the operations below.
    shdr::buffer<float> input(side * side * side), output(side * side * side), coefficients(27); // Construct input with these initial values.
    shdr_examples::randomize(input, random_seed, -1.0f, 1.0f); // Fill the input with reproducible random values.
    shdr_examples::randomize(coefficients, random_seed + 1, -1.0f / 27, 1.0f / 27); // Fill the input with reproducible random values.
    auto convolution = shdr::make_kernel("convolution3d_kernel"); // Create the linked convolution3d kernel instance.
    convolution->bind("input_values", input) // Connect this shader port to its application resource.
        .bind("output_values", output) // Connect this shader port to its application resource.
        .bind("coefficients", coefficients) // Connect this shader port to its application resource.
        .set<uvec3>("dimensions", uvec3(side, side, side)) // Supply this shader uniform with its typed application value.
        .set<uvec3>("kernel_size", uvec3(3, 3, 3)); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline; // Declare pipeline for the following calculation.
    pipeline += shdr::dispatch(*convolution, groups(*convolution, side, side, side)); // Add a shader dispatch with the required workgroup counts.
    output_check<float> check(output); // Construct check with these initial values.
    bench.measure("3D convolution 3 x 3 x 3", dimensions(side, side, side), pipeline, {&check}); // Apply measure to these arguments.
} // End this block.

void fourier_pipeline(benchmark& bench) { // Implement fourier pipeline for the operations below.
    const unsigned side = 32, // Set side from this expression.
                   count = side * side; // Direct DFT remains small even when other workloads scale.
    shdr::buffer<vec2> input(count), rows(count), spectrum(count), inverse_rows(count), // Declare input with the parameters continued below.
        restored(count); // Apply restored to these arguments.
    random_complex(input, random_seed); // Apply random complex to these arguments.
    auto forward_rows = shdr::make_kernel("dft_rows_kernel"); // Create the linked dft rows kernel instance.
    auto forward_columns = shdr::make_kernel("dft_columns_kernel"); // Create the linked dft columns kernel instance.
    auto reverse_rows = shdr::make_kernel("dft_rows_kernel"); // Create the linked dft rows kernel instance.
    auto reverse_columns = shdr::make_kernel("dft_columns_kernel"); // Create the linked dft columns kernel instance.
    shdr::kernel* kernels[] = { // Declare kernels for the following calculation.
        forward_rows.get(), forward_columns.get(), reverse_rows.get(), reverse_columns.get()}; // Apply get to these arguments.
    shdr::buffer<vec2>* inputs[] = {&input, &rows, &spectrum, &inverse_rows}; // Declare inputs for the following calculation.
    shdr::buffer<vec2>* outputs[] = {&rows, &spectrum, &inverse_rows, &restored}; // Declare outputs for the following calculation.
    shdr::pipeline pipeline; // Declare pipeline for the following calculation.
    for (unsigned i = 0; i < 4; ++i) { // Iterate over i for this calculation.
        kernels[i] // Continue the expression using kernels, i.
            ->bind("input_values", *inputs[i]) // Connect this shader port to its application resource.
            .bind("output_values", *outputs[i]) // Connect this shader port to its application resource.
            .set<uvec2>("dimensions", uvec2(side, side)) // Supply this shader uniform with its typed application value.
            .set<float>("direction", i < 2 ? -1.0f : 1.0f) // Supply this shader uniform with its typed application value.
            .set<float>("normalization", i < 2 ? 1.0f : 1.0f / side); // Supply this shader uniform with its typed application value.
        pipeline += shdr::dispatch(*kernels[i], groups(*kernels[i], side, side)); // Add a shader dispatch with the required workgroup counts.
    } // End this block.
    output_check<vec2> spectral_check(spectrum, 1e-3, 1e-3), // Declare spectral check with the parameters continued below.
        reconstruction_check(restored, 1e-3, 1e-3); // Apply reconstruction check to these arguments.
    bench.measure("Forward/inverse direct DFT", // Apply measure to these arguments.
                  dimensions(side, side) + " complex", // Apply dimensions to these arguments.
                  pipeline, // Continue the expression using pipeline.
                  {&spectral_check, &reconstruction_check}); // Continue the expression using spectral check, reconstruction check.
    const shdr::buffer<vec2>& read_input = input; // Use a constant buffer view so GPU writes synchronize before host reads.
    const shdr::buffer<vec2>& read_restored = restored; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        same(read_restored.data()[i], read_input.data()[i], 1e-3, 1e-3); // Compare CPU and GPU values using the selected tolerance.
} // End this block.

void vision_pipeline(benchmark& bench, unsigned side) { // Implement vision pipeline for the operations below.
    const unsigned count = side * side; // Set count from this expression.
    shdr::buffer<float> input(count), horizontal(count), smooth(count), response(count); // Construct input with these initial values.
    shdr::buffer<vec2> gradient(count); // Construct gradient with these initial values.
    shdr::buffer<vec4> tensor(count); // Construct tensor with these initial values.
    shdr::buffer<unsigned> peaks(count); // Construct peaks with these initial values.
    shdr_examples::randomize(input, random_seed, 0.0f, 1.0f); // Fill the input with reproducible random values.
    auto h = shdr::make_kernel("gaussian_horizontal_kernel"), // Create the linked gaussian horizontal kernel instance.
         v = shdr::make_kernel("gaussian_vertical_kernel"); // Create the linked gaussian vertical kernel instance.
    auto sobel = shdr::make_kernel("sobel_kernel"), // Create the linked sobel kernel instance.
         outer = shdr::make_kernel("harris_tensor_kernel"); // Create the linked harris tensor kernel instance.
    auto harris = shdr::make_kernel("harris_response_kernel"), // Create the linked harris response kernel instance.
         nonmax = shdr::make_kernel("nonmax_kernel"); // Create the linked nonmax kernel instance.
    h->bind("input_values", input).bind("output_values", horizontal); // Connect this shader port to its application resource.
    v->bind("input_values", horizontal).bind("output_values", smooth); // Connect this shader port to its application resource.
    sobel->bind("input_values", smooth).bind("output_values", gradient); // Connect this shader port to its application resource.
    outer->bind("input_values", gradient) // Connect this shader port to its application resource.
        .bind("output_values", tensor) // Connect this shader port to its application resource.
        .set<unsigned>("count", count); // Supply this shader uniform with its typed application value.
    harris->bind("input_values", tensor) // Connect this shader port to its application resource.
        .bind("output_values", response) // Connect this shader port to its application resource.
        .set<float>("harris_k", 0.04f); // Supply this shader uniform with its typed application value.
    nonmax->bind("input_values", response) // Connect this shader port to its application resource.
        .bind("output_values", peaks) // Connect this shader port to its application resource.
        .set<float>("threshold", 1e-8f); // Supply this shader uniform with its typed application value.
    shdr::kernel* kernels[] = { // Declare kernels for the following calculation.
        h.get(), v.get(), sobel.get(), outer.get(), harris.get(), nonmax.get()}; // Apply get to these arguments.
    shdr::pipeline pipeline; // Declare pipeline for the following calculation.
    for (unsigned i = 0; i < 6; ++i) { // Iterate over i for this calculation.
        if (i != 3) // Check this condition before entering the following branch.
            kernels[i]->set<unsigned>("width", side).set<unsigned>("height", side); // Supply this shader uniform with its typed application value.
        pipeline += shdr::dispatch(*kernels[i], groups(*kernels[i], count)); // Add a shader dispatch with the required workgroup counts.
    } // End this block.
    output_check<float> smooth_check(smooth), response_check(response); // Construct smooth check with these initial values.
    output_check<vec2> gradient_check(gradient); // Construct gradient check with these initial values.
    output_check<unsigned> peak_check(peaks); // Construct peak check with these initial values.
    bench.measure("Gaussian/Sobel/Harris/NMS", // Apply measure to these arguments.
                  dimensions(side, side), // Apply dimensions to these arguments.
                  pipeline, // Continue the expression using pipeline.
                  {&smooth_check, &gradient_check, &response_check, &peak_check}); // Continue the expression using smooth check, gradient check, response check.
} // End this block.

void lenet_pipeline(benchmark& bench) { // Compare all seven layers of the named image classifier.
    shdr_examples::lenet_classifier model(32); // Allocate a resident batch of thirty-two grayscale images.
    shdr_examples::randomize(model.inputs, random_seed, 0.0f, 1.0f); // Use normalized seeded image values.
    shdr_examples::randomize(model.parameters, random_seed + 1u, -0.025f, 0.025f); // Keep randomized learned parameters small enough for stable comparisons.
    output_check<float> convolution_check(model.conv2, 1e-4, 1e-3); // Check the second convolution features after each backend completes.
    output_check<float> hidden_check(model.hidden2, 1e-4, 1e-3); // Check the final hidden representation as well as the head.
    output_check<float> logits_check(model.logits, 1e-4, 1e-3); // Retain the CPU reference for every class score.
    bench.measure("LeNet classifier", // Identify this full convolutional model in the timing table.
                  "32 x 28 x 28, 61,706 parameters", // State the complete resident workload size.
                  model.inference_pipeline(), // Execute two convolution and pool pairs followed by three dense stages.
                  {&convolution_check, &hidden_check, &logits_check}); // Verify internal features and final scores outside timing.
} // Finish the complete classifier comparison.

void compression_pipeline(benchmark& bench, unsigned count) { // Implement compression pipeline for the operations below.
    const unsigned block_size = 256, blocks = (count + block_size - 1) / block_size; // Set block size from this expression.
    shdr::buffer<unsigned> input(count), sizes(blocks), offsets(blocks + 1), encoded(count * 2), // Declare input with the parameters continued below.
        restored(count); // Apply restored to these arguments.
    std::mt19937 generator(random_seed); // Construct generator with these initial values.
    std::uniform_int_distribution<unsigned> run_length(4, 16), value(0, 4095); // Construct run length with these initial values.
    unsigned position = 0; // Set position from this expression.
    while (position < count) { // Fill the input with seeded runs until the requested element count is reached.
        const unsigned run = run_length(generator), sample = value(generator); // Set run from this expression.
        for (unsigned i = 0; i < run && position < count; ++i) // Iterate over i for this calculation.
            input.data()[position++] = sample; // Apply data to these arguments.
    } // End this block.
    auto tally = shdr::make_kernel("rle_count_kernel"), // Create the linked rle count kernel instance.
         prefix = shdr::make_kernel("rle_offsets_kernel"); // Create the linked rle offsets kernel instance.
    auto encode = shdr::make_kernel("rle_encode_kernel"), // Create the linked rle encode kernel instance.
         decode = shdr::make_kernel("rle_decode_kernel"); // Create the linked rle decode kernel instance.
    tally->bind("values", input) // Connect this shader port to its application resource.
        .bind("sizes", sizes) // Connect this shader port to its application resource.
        .set<unsigned>("count", count) // Supply this shader uniform with its typed application value.
        .set<unsigned>("block_size", block_size); // Supply this shader uniform with its typed application value.
    prefix->bind("sizes", sizes).bind("offsets", offsets).set<unsigned>("blocks", blocks); // Connect this shader port to its application resource.
    encode->bind("values", input) // Connect this shader port to its application resource.
        .bind("offsets", offsets) // Connect this shader port to its application resource.
        .bind("encoded", encoded) // Connect this shader port to its application resource.
        .set<unsigned>("count", count) // Supply this shader uniform with its typed application value.
        .set<unsigned>("block_size", block_size); // Supply this shader uniform with its typed application value.
    decode->bind("encoded", encoded) // Connect this shader port to its application resource.
        .bind("offsets", offsets) // Connect this shader port to its application resource.
        .bind("restored", restored) // Connect this shader port to its application resource.
        .set<unsigned>("count", count) // Supply this shader uniform with its typed application value.
        .set<unsigned>("block_size", block_size); // Supply this shader uniform with its typed application value.
    shdr::pipeline pipeline = shdr::dispatch(*tally, groups(*tally, blocks)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(*prefix, groups(*prefix, 1)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(*encode, groups(*encode, blocks)) >> // Add a shader dispatch with the required workgroup counts.
                              shdr::dispatch(*decode, groups(*decode, blocks)); // Add a shader dispatch with the required workgroup counts.
    output_check<unsigned> restored_check(restored), offsets_check(offsets), encoded_check(encoded); // Construct restored check with these initial values.
    bench.measure("Lossless tiled RLE round trip", // Apply measure to these arguments.
                  std::to_string(count) + " uints", // Apply to string to these arguments.
                  pipeline, // Continue the expression using pipeline.
                  {&restored_check, &offsets_check, &encoded_check}); // Continue the expression using restored check, offsets check, encoded check.
    const shdr::buffer<unsigned>& read_input = input; // Use a constant buffer view so GPU writes synchronize before host reads.
    const shdr::buffer<unsigned>& read_restored = restored; // Use a constant buffer view so GPU writes synchronize before host reads.
    for (unsigned i = 0; i < count; ++i) // Iterate over i for this calculation.
        same(read_restored.data()[i], read_input.data()[i], 0, 0); // Compare CPU and GPU values using the selected tolerance.
    const shdr::buffer<unsigned>& read_offsets = offsets; // Use a constant buffer view so GPU writes synchronize before host reads.
    const unsigned stream_words = read_offsets.data()[blocks]; // Set stream words from this expression.
    std::ostringstream metadata; // Declare metadata for the following calculation.
    metadata << "Lossless RLE: " << stream_words << " payload words + " << blocks + 1 // Provide this text argument for the operation above.
             << " offset words, " << std::fixed << std::setprecision(2) // Write this part of the example result or diagnostic.
             << double(count) / double(stream_words + blocks + 1) << "x size reduction."; // Write this part of the example result or diagnostic.
    bench.note(metadata.str()); // Apply note to these arguments.
} // End this block.

#ifdef SHDR_PLUGIN_PATH // Enable this branch when its feature macro is defined.
void dynamic_pipeline(benchmark& bench, unsigned count) { // Implement dynamic pipeline for the operations below.
    void* library = dlopen(SHDR_PLUGIN_PATH, RTLD_NOW | RTLD_LOCAL); // Set library from this expression.
    if (!library) // Check this condition before entering the following branch.
        throw std::runtime_error(dlerror()); // Report this failed operation.
    typedef shdr::kernel* (*create_function)(); // Declare shdr for the following calculation.
    typedef void (*destroy_function)(shdr::kernel*); // Construct void with these initial values.
    create_function create = // Set create from this expression.
        reinterpret_cast<create_function>(dlsym(library, "shdr_create_kernel")); // Apply dlsym to these arguments.
    destroy_function destroy = // Set destroy from this expression.
        reinterpret_cast<destroy_function>(dlsym(library, "shdr_destroy_kernel")); // Apply dlsym to these arguments.
    if (!create || !destroy) { // Check this condition before entering the following branch.
        dlclose(library); // Apply dlclose to these arguments.
        throw std::runtime_error("incomplete benchmark plugin"); // Report incomplete benchmark plugin.
    } // End this block.
    shdr::kernel* shader = 0; // Set shader from this expression.
    try { // Run the operation while retaining the error handling below.
        shader = create(); // Set shader from this expression.
        if (!shader) // Check this condition before entering the following branch.
            throw std::runtime_error("benchmark plugin returned no shader"); // Report an empty result from the plugin factory.
        buffer_pipeline(bench, count, false, shader); // Apply buffer pipeline to these arguments.
        destroy(shader); // Apply destroy to these arguments.
        shader = 0; // Set shader from this expression.
        dlclose(library); // Apply dlclose to these arguments.
    } catch (...) { // Report the exception from the preceding operation.
        if (shader) // Check this condition before entering the following branch.
            destroy(shader); // Apply destroy to these arguments.
        dlclose(library); // Apply dlclose to these arguments.
        throw; // Report this failed operation.
    } // End this block.
} // End this block.
#endif // Finish this conditional compilation branch.
} // namespace

int main(int argc, char** argv) { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        shdr_examples::runner run(argc, argv); // Construct run with these initial values.
        const unsigned iterations = setting("SHDR_BENCH_ITERATIONS", 3, 1000); // Set iterations from this expression.
        const unsigned scale = setting("SHDR_BENCH_SCALE", 1, 4); // Set scale from this expression.
        const unsigned side = 128 * scale; // Set side from this expression.
        benchmark bench(run.device(), iterations); // Construct bench with these initial values.
        buffer_pipeline(bench, side * side, false); // Apply buffer pipeline to these arguments.
        image_pipeline(bench, side); // Apply image pipeline to these arguments.
        image_pipeline(bench, side, run.device(), true); // Apply image pipeline to these arguments.
        buffer_pipeline(bench, side * side, true); // Apply buffer pipeline to these arguments.
#ifdef SHDR_PLUGIN_PATH // Enable this branch when its feature macro is defined.
        dynamic_pipeline(bench, side * side); // Apply dynamic pipeline to these arguments.
#endif // Finish this conditional compilation branch.
        bayer_pipeline(bench, side); // Apply bayer pipeline to these arguments.
        convolution_2d(bench, side); // Apply convolution 2d to these arguments.
        convolution_3d(bench, 24 * scale); // Apply convolution 3d to these arguments.
        fourier_pipeline(bench); // Apply fourier pipeline to these arguments.
        vision_pipeline(bench, side); // Apply vision pipeline to these arguments.
        compression_pipeline(bench, 65536 * scale); // Apply compression pipeline to these arguments.
        evt3_pipeline(bench, 65536 * scale); // Apply evt3 pipeline to these arguments.
        async_pipeline(bench, run.device()); // Apply async pipeline to these arguments.
        evt3_optical_flow(bench); // Apply evt3 optical flow to these arguments.
        binary_neural_pipeline(bench, run.device(), false); // Apply binary neural pipeline to these arguments.
        binary_neural_pipeline(bench, run.device(), true); // Apply binary neural pipeline to these arguments.
        transformer_pipeline(bench, run.device()); // Apply transformer pipeline to these arguments.
        fft_round_trip(bench); // Apply fft round trip to these arguments.
        fir_low_pass(bench); // Apply fir low pass to these arguments.
        biquad_low_pass(bench, run.device()); // Apply biquad low pass to these arguments.
        fft_band_pass(bench); // Apply fft band pass to these arguments.
        lenet_pipeline(bench); // Compare the named image classifier on both backends.
        bench.print(scale); // Apply print to these arguments.
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
