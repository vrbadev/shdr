# shdr 1.0.0

Write a compute shader once, then run it on the CPU, an OpenGL GPU, or a mixture
of both. The shdr library provides a small C++11 runtime, typed resources, and
ordered pipelines. Native GLSL shaders are compiled into CPU adapters and embedded GPU
programs during the build. The executable contains both versions. An application
creates kernels, binds data, and runs them.

The runtime is the single header [shdr.h](shdr.h). Native shader builds also use
[cmake/shdr.cmake](cmake/shdr.cmake) and the Python build frontend. CPU execution
uses standard C++11 and host memory. The optional EGL backend creates a headless
GPU device. Applications with an existing GL context can supply their own resolver.

| Component | Requirement |
| --- | --- |
| Runtime | A C++11 compiler |
| Native shader build | CMake 3.12+ and Python 3.8+ |
| GPU compute | OpenGL ES 3.1/3.2 or desktop OpenGL 4.3+ |
| Headless GPU backend | EGL development files and a compatible driver |

Start with the [documentation](https://vrbadev.github.io/shdr/) and its
[example walkthroughs](https://vrbadev.github.io/shdr/examples.html). Read the [compatibility guide](https://vrbadev.github.io/shdr/portability.html) for the supported GLSL subset,
the [API reference](https://vrbadev.github.io/shdr/api.html) for runtime functions, and the
[imaging guide](https://vrbadev.github.io/shdr/imaging.html) for reusable processing kernels.

## Build and try a pipeline

```sh
# Configure the example build directory.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# Compile the configured targets and embedded shaders.
cmake --build build -j
# Run this example on the CPU.
./build/example_pipeline
```

This runs an affine transform, a clamp, and a reduction on the CPU. The same
executable can run the same shaders on a GPU:

```sh
# Run this example on the requested headless GPU profile.
./build/example_pipeline --gpu es31
```

GPU examples accept `es31`, `es32`, `gl43`, `gl45`, and `gl46`. Choose a profile
supported by the driver. CMake builds the CPU examples and enables GPU examples
when EGL development files are available. `-DSHDR_BUILD_GPU=OFF` selects CPU execution.
`SHDR_BUILD_EXAMPLES` and `SHDR_BUILD_TESTS` control those respective targets.

## Your first linked shaders

Create a `quickstart` directory beside `shdr.h` and add these three files.
The application will run two instances of one shader, passing the first result
to the second through a shared resource.

`quickstart/scale.cpp` is a native compute shader:

```cpp
#version 310 es // Declare the original shader language version for standalone GLSL compilation.
#ifdef __cplusplus // Expose the runtime header only to the generated C++ adapter.
#include "shdr.h" // Provide portable math types and kernel registration to C++.
#endif // Keep the C++ header inactive for the GLSL compiler.
precision highp float; // Use high precision for floating-point arithmetic on OpenGL ES.
precision highp int; // Use high precision for integer indices on OpenGL ES.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // Launch 64 independent invocations in each workgroup.
layout(std430, binding = 0) readonly buffer Inputs { float input_values[]; }; // Read the input array from storage-buffer binding zero.
layout(std430, binding = 1) writeonly buffer Outputs { float output_values[]; }; // Write the result array to storage-buffer binding one.
uniform uint count; // Limit processing to the allocated element count.
uniform float gain; // Set the multiplier independently for each kernel instance.

void main() { // Compute the output belonging to this invocation.
    uint i = gl_GlobalInvocationID.x; // Find this invocation's element index.
    if (i < count) // Ignore padded invocations beyond the input length.
        output_values[i] = input_values[i] * gain; // Scale one input value into its matching output position.
} // Finish this independent element computation.
```

`quickstart/main.cpp` creates the resources and connects the stages:

```cpp
#include "shdr.h" // Use resources, factories and dispatches from the single runtime header.
#include <iostream> // Print the final host-visible values.

int main() { // Build and run two linked compute stages.
    shdr::buffer<float> input = {1.f, 2.f, 3.f, 4.f}; // Start with four host-resident input values.
    shdr::buffer<float> temporary(4), output(4); // Allocate a buffer between stages and one for the final result.
    auto first = shdr::make_kernel("scale_kernel"); // Create the first instance of the embedded scale shader.
    auto second = shdr::make_kernel("scale_kernel"); // Create an independent instance of the same shader.

    first->bind("input_values", input); // Connect the original input to the first stage.
    first->bind("output_values", temporary); // Connect the first stage's output to the intermediate buffer.
    first->set("count", unsigned(input.size())); // Set the number of valid elements for the first stage.
    first->set("gain", 2.f); // Multiply every input value by two.

    second->bind("input_values", temporary); // Read the first stage's results in the second stage.
    second->bind("output_values", output); // Write the second stage's results into the final buffer.
    second->set("count", unsigned(input.size())); // Set the valid element count for the second stage.
    second->set("gain", 3.f); // Multiply the intermediate values by three.

    const shdr::extent groups(1, 1, 1); // One workgroup covers all four values.
    auto pipeline = shdr::dispatch(*first, groups) // Dispatch the first kernel before its consumer.
                 >> shdr::dispatch(*second, groups); // Append the second kernel to the ordered pipeline.
    pipeline.run(); // Execute both stages on the CPU.

    const shdr::buffer<float>& result = output; // Use a const view when reading the result.
    for (std::size_t i = 0; i < result.size(); ++i) // Visit every valid output value.
        std::cout << result.data()[i] << ' '; // Print each completed pipeline result.
    std::cout << '\n'; // 6 12 18 24
} // Release kernels and buffers after their final use.
```

`quickstart/CMakeLists.txt` attaches the shader to the executable:

```cmake
cmake_minimum_required(VERSION 3.12) # Require the CMake version supported by the build helper.
project(shdr_quickstart LANGUAGES CXX) # Configure a project containing C++ translation units.
find_package(Threads REQUIRED) # Find the platform thread library used by the runtime.
add_executable(quickstart main.cpp) # Build the small host application.
target_compile_features(quickstart PRIVATE cxx_std_11) # Compile the host and generated adapter as standard C++11.
target_link_libraries(quickstart PRIVATE Threads::Threads) # Link the runtime thread support on every supported platform.
target_include_directories(quickstart PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..") # Find shdr.h beside this quickstart directory.
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/shdr.cmake") # Load the automatic CPU-adapter and GPU-embedding helper.
shdr_target_shaders(quickstart PRIVATE SOURCES scale.cpp) # Compile and register scale.cpp as part of the executable.
```

Build and run from the project directory:

```sh
# Configure the example build directory.
cmake -S quickstart -B quickstart/build
# Compile the configured targets and embedded shaders.
cmake --build quickstart/build -j
# Run this example on the CPU.
./quickstart/build/quickstart
```

The filename `scale.cpp` registers the factory `scale_kernel`. Each factory call
returns an independent kernel object. `bind` connects typed buffer storage.
`set` assigns a uniform of its declared type. In C++, use `unsigned` for shader
`uint` values and `f` suffixes for `float` uniform values.

A dispatch extent counts **workgroups**. With 64 local invocations and
`extent(1)`, the shader executes 64 invocations. Its bounds check selects the
four invocations that access the four-element buffer.
For `count` elements, dispatch `extent((count+63)/64)`.

Native GLSL interface declarations and swizzles require translation into C++
grammar. GCC compiles the generated C++11 adapter, while the GL driver compiles
the embedded GLSL. The executable and its linked libraries contain the shader
programs needed at runtime.

## Run on a GPU or combine backends

Enable the optional EGL backend, then create an offscreen compute device. For an installed package, link the `shdr::egl` CMake target. It enables
`SHDR_ENABLE_EGL=1` and supplies the EGL headers and link library.

```cmake
target_link_libraries(app PRIVATE shdr::egl) # Enable the optional EGL backend and link its dependencies.
```

The application uses the same kernels, buffers and pipeline:

```cpp
#include "shdr.h" // SHDR_ENABLE_EGL=1 is supplied by the build target.

// Inside main(), after constructing the pipeline:
shdr::headless_device gpu(shdr::profile::es31); // Create a headless context using the ES 3.1 profile.
pipeline.run(gpu); // Execute the same ordered pipeline on that GPU device.
```

For a manual build, define `SHDR_ENABLE_EGL=1` before including the header and
link EGL. The backend uses a 1-by-1 pbuffer, prefers a surfaceless EGL display,
and otherwise tries EGL's default display. `backend_unavailable` reports an
unavailable driver or requested context. Shader and dispatch failures use
`shdr::error` and retain their diagnostics.

An application that already owns a GL context can make it current and construct
`shdr::device(resolve_gl, profile)` with its own function resolver. This path uses
the application's context backend. CPU execution uses host resources.

Automatic stages use the CPU with `pipeline.run()` and the GPU with
`pipeline.run(gpu)`. A stage may select its own backend:

```cpp
auto mixed = shdr::dispatch(*first, groups, shdr::backend::gpu) // Choose GPU execution explicitly for the first stage.
           >> shdr::dispatch(*second, groups, shdr::backend::cpu); // Choose CPU execution explicitly for its consumer.
mixed.run(gpu); // Run the mixed pipeline and transfer intermediate data when needed.
```

A shared resource links the stages and transfers its latest contents when a
CPU/GPU transition needs them. GPU stages insert memory barriers. Host access to
a GPU-produced resource waits for its data. `device.synchronize()` waits for
completion while retaining results on the GPU.

A pipeline retains references to its kernels. Keep those objects alive until
execution finishes. Rebind resources or update uniforms before another run to
reuse the pipeline. `pipeline += dispatch(...)` appends a stage, and
`pipeline += another_pipeline` appends an existing sequence. Two pipelines can
also be linked with `first_pipeline >> second_pipeline`. Their kernels retain
the same lifetime requirements.

## Submit work and poll completion

Compile and upload a GPU pipeline before a processing loop, then enqueue work
and poll its fence:

```cpp
pipeline.prepare(gpu); // Compile programs and upload resources before the submission loop.
auto completion = pipeline.submit(gpu); // Enqueue GPU work and retain its completion handle.
// Perform other application work on this thread.
if (completion.ready()) { // Check the GPU fence with a zero timeout.
    // GPU work has completed. This resource slot can be read or reused.
} // Continue the application loop after the poll.
```

`ready()` queries the fence with a zero timeout. `wait(duration)` permits a
bounded driver wait. `wait()` waits until completion. GPU submission accepts
GPU stages. Host readback transfers data and may wait, so schedule it after
completion. Use separate resource slots for work in flight.

For background CPU processing, `shdr::executor workers(2, 8)` starts two workers
with up to eight pending jobs. `workers.submit(pipeline)` returns a
`std::future<void>` and snapshots shader uniforms and bindings. A full queue
throws immediately. Inputs must already be available on the host, and concurrent
jobs must have distinct writable resources. Call `workers.shutdown()` at an
explicit shutdown point. It finishes accepted jobs and joins the workers.

The [asynchronous execution guide](https://vrbadev.github.io/shdr/async.html) explains resource lifetimes,
polling, queue limits and latency boundaries. Submission returns a completion
handle, and the application chooses when to poll or wait. Driver scheduling and
resource preparation also contribute to submission latency.

## Understand resource storage

`buffer<T>(count)` and `image2d<T, Components>(width, height)` own host storage and
allocate GPU storage when it is needed. Resource copies and bound ports share
the same storage. `buffer<T>(pointer, count)` borrows existing host memory. That
memory must remain alive while a resource or port can use it.

Mutable `data()` first synchronizes newer GPU contents, then marks the host copy
modified. Readonly `data()` synchronizes and preserves the modification state. Prefer
a const resource reference when inspecting a result. Obtain a fresh `data()`
pointer after GPU work before accessing host storage. `upload(device)` and
`download(device)` provide explicit transfer points.

Storage buffers support 32-bit `float`, `int`, and `uint`, and matching
`vec2`/`vec4` types. These element layouts match `std430` arrays. Three-component
vectors and matrices support shader math and uniforms.

Images support one or four interleaved 32-bit components with matching
`r32f/r32i/r32ui` or `rgba32f/rgba32i/rgba32ui` formats. Use separate input and
output resources for neighborhood filters. Portable RGBA image ports declare
`readonly` or `writeonly` access.

For GL interoperation, `gl_handle(device)` returns a synchronized object handle.
`from_gl` borrows an application-owned buffer or texture. Notify
the resource with `mark_gpu_modified(device)` after application-issued GL writes,
or `mark_host_modified()` after writes through a borrowed host pointer. Supply
barriers appropriate to application-issued GL consumers.

Resources belong to one device. Perform GPU operations and final resource
destruction on its context thread. A headless device retains its context with
GPU objects and makes it current when used. An application-owned context must
remain alive and current. The runtime changes program, buffer, and image bindings. Restore the
application's own GL state where needed. Give concurrent jobs independent
mutable kernels and writable resources.

## Use shader modules in a larger project

An installed CMake package provides the header target and shader build helper:

```cmake
find_package(shdr 1.0.0 CONFIG REQUIRED) # Find the installed header target and shader build helper.
add_executable(app main.cpp) # Build the application that binds and dispatches kernels.
target_link_libraries(app PRIVATE shdr::shdr) # Link the CPU-compatible runtime target.
shdr_target_shaders(app PRIVATE SOURCES scale.cpp another_shader.cpp) # Compile and embed both native shader sources into this target.
```

Link `shdr::egl` when the application creates a `headless_device`. EGL must be
available when configuring that optional target.

Attach adapters directly to an executable or a loaded shared-library target for
automatic factory discovery. Static-library builds need explicit linker
retention of the factory registration objects.

The dynamic-library example exports creation and destruction callbacks with C
linkage. Keep the library loaded while its kernel objects and pipelines are in
use, and destroy those objects through the library's callback. Host and plugin
must use compatible C++ runtimes and shdr versions.

## Import an existing compute shader

Add a self-contained compute source to `shdr_target_shaders`, then bind its
resource and uniform names through the returned kernel. GLSL source extensions
can be retained. Factory names come from the file stem. A file stem used as a
kernel name must be a valid C++ identifier.

For an existing shader, keep its version declaration first and add this guarded
include immediately afterwards:

```cpp
#version 450 // Keep an imported shader's original version directive first.
#ifdef __cplusplus // Add the runtime header only when compiling generated C++.
#include "shdr.h" // Expose shdr declarations to the C++ adapter.
#endif // Leave this include inactive in standalone GPU compilation.
// Existing compute declarations and main() follow.
```

The guard exposes the header to C++ compilation. The build frontend supplies
the chosen device profile's version for GLSL compilation. The shader's language
features must be supported by that profile. Add the source to the build helper
to generate its CPU adapter and embed its GPU program automatically.

The frontend accepts fixed and runtime-sized single-array storage blocks,
constant or macro layout expressions, assigned or explicit bindings, native
fixed-size local arrays, array parameters, swizzles, and supported constructor
conversions. The embedded portable GPU variant can normalize interface layouts,
numeric casts and local `const` qualifiers while retaining expression order.
The compatibility guide describes supported features and source transformations.

A [pinned public shader collection](https://vrbadev.github.io/shdr/corpus.html) checks
664 compute inputs from five projects. Of 443 inputs accepted by a reference
compiler, 258 build strict C++11 adapters and all 258 compile for all five GPU
profiles. The report records source configuration, provenance, failures and
per-profile results. These are compiler checks. The bundled pipeline tests
separately verify CPU/GPU execution and output parity.

Begin by checking the [source compatibility guide](https://vrbadev.github.io/shdr/portability.html). Compute
shaders with independent invocations, supported storage-buffer/image interfaces,
and ordinary scalar/vector/matrix expressions are the common portable case.
For shaders that use samplers, graphics stages, synchronization, or custom
resource layouts, use the guide to plan the interface and execution model.
Compiler acceptance and successful GPU execution are separate validation steps.

`kernel::glsl(profile)` exposes the embedded GPU program for debugging and
standard GLSL compiler checks. Shader compile/link exceptions include driver
logs. Missing ports and incompatible uniform types produce named diagnostics.
Check workgroup sizes, buffer lengths, image formats, and the selected profile
when diagnosing a failed import. Floating-point results should be compared with
an appropriate tolerance.

## Explore the examples

| Executable | Pipeline |
| --- | --- |
| `example_pipeline` | Affine transform → clamp → sum |
| `example_images` | Buffer → image → blur → buffer, borrowed texture variant |
| `example_hybrid` | GPU affine → CPU clamp → GPU sum |
| `example_dynamic` | Shared-library shader → clamp → sum, Unix platforms |
| `example_async` | Three resource slots with GPU fences or bounded CPU workers |
| `example_imaging` | Bayer unpack → demosaic → luminance, tiled lossless RLE |
| `example_spectral` | Dynamic 2D/3D convolutions and complex separable Fourier transforms |
| `example_vision` | Gaussian → Sobel → tensor → Harris → nonmaximum suppression |
| `example_evt3` | Streaming EVT3 decode → compact CPU events or GPU/CPU normal optical flow |
| `example_neural` | Fully trainable two-feature ANN with eight ReLU neurons |
| `example_cnn` | Two learned convolution filters, global pooling and a sigmoid classifier |
| `example_transformer` | Learned attention, normalization, feed-forward layers and a two-class head |
| `example_signal` | Batched FFTs, frequency masks, FIR filters and streaming biquads |
| `example_real_data` | Public images, ECG, event recordings, real-digit training and deeper LeNet-style inference |
| `example_benchmark` | Seeded CPU/GPU comparisons across the processing pipelines |

Run an executable with its default arguments for CPU execution or with
`--gpu es31` for a supported GPU profile. Examples check their outputs against independent
references. The vision example also checks four known square corners. The
[imaging guide](https://vrbadev.github.io/shdr/imaging.html) lists formats, ports, dimensions, and parameters
for the reusable native shader modules.
The [EVT3 guide](https://vrbadev.github.io/shdr/evt3.html) shows ordered event-camera decoding across batches,
including the input packing, event representation and stream-state policies.
The [neural guide](https://vrbadev.github.io/shdr/neural.html) shows complete training and separate inference
for the small ANN, CNN and transformer models, including their parameter layouts
and gradient checks.
The [signal guide](https://vrbadev.github.io/shdr/signals.html) shows one-dimensional FFTs, low-pass,
high-pass and band filters, windows and persistent stream state.
The [public-data guide](https://vrbadev.github.io/shdr/datasets.html) runs those processing families on
recorded inputs with checked provenance and transformations. It includes a
seven-stage LeNet-style classifier with 61,706 prepared parameters, original
MNIST test labels and independent framework logits. The default evaluation uses
512 held-out images. An optional full run uses all 10,000 test images.

```sh
# Verify the shipped data and model files against their recorded hashes.
python3 scripts/prepare_datasets.py --verify
# Run the public-data cases on the CPU.
./build/example_real_data
# Run the same cases on a headless ES 3.1 device.
./build/example_real_data --gpu es31
```

The Fourier example uses a direct separable DFT with cost
`O(width*height*(width+height))`. It demonstrates complex data and linked stages
for small images. The signal example provides radix-2 FFT pipelines.

## Check your build

```sh
# Run the example and runtime correctness checks.
ctest --test-dir build --output-on-failure
# Check the generated shader sources with glslangValidator.
cmake --build build --target validate_glsl
```

`validate_glsl` is available when `glslangValidator` and Python are installed.
It checks every bundled shader against all five output profiles. GPU tests skip
when the requested context is unavailable.

## Measured pipeline performance

These results were measured on 2026-10-15 with CMake Release builds, C++11,
seed `12345`, scale `1` and twenty timed repetitions per backend:

```sh
# Measure completed ES 3.1 workloads with the recorded benchmark settings.
SHDR_BENCH_ITERATIONS=20 SHDR_BENCH_SCALE=1 ./build/example_benchmark --gpu es31
```

Times are mean completed workload milliseconds, measured after compilation,
initial uploads and warmup, and before final downloads. The timed interval
includes transfers within mixed pipelines. Ordinary CPU pipelines execute serially. The asynchronous row uses
**two CPU workers** and measures a completed batch of 32 dispatches through three
slots. Each neural training row measures one mini-batch gradient/SGD update.
Inference is a separate forward dispatch. The seven-stage LeNet row uses a
resident batch of 32 images and seeded random parameters. Full trained-model
evaluation is reported separately below.

The Ryzen 5900X graphics measurement uses **llvmpipe software rendering on the
CPU**. Iris Xe and V3D measurements use hardware graphics. All measurements use
the `es31` shader profile. The reports record compiler, driver and validation
details for each configuration.

| Pipeline | Workload | Ryzen 5900X CPU ms | llvmpipe ms | i7-1260P CPU ms | Iris Xe ms | Pi 5 CPU ms | V3D ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buffer affine/clamp/sum | 16384 floats | 0.300 | 1.374 | 0.139 | 3.928 | 0.516 | 1.956 |
| Buffer/image blur | 128 × 128 | 0.589 | 1.639 | 0.422 | 0.054 | 1.033 | 0.471 |
| Borrowed image blur | 128 × 128 | 0.637 | 1.582 | 0.299 | 0.052 | 0.951 | 0.469 |
| Hybrid affine/clamp/sum | 16384 floats | 0.288 | 1.217 | 0.145 | 2.825 | 0.597 | 2.188 |
| Dynamic library stages | 16384 floats | 0.303 | 1.463 | 0.460 | 2.100 | 0.838 | 1.968 |
| RAW10/debayer/luminance | 128 × 128 | 1.078 | 1.643 | 1.080 | 0.068 | 2.466 | 2.522 |
| 2D convolution 5 x 5 | 128 × 128 vec2 | 1.521 | 0.748 | 1.140 | 0.059 | 3.283 | 2.444 |
| 3D convolution 3 x 3 x 3 | 24 × 24 × 24 | 1.503 | 0.706 | 1.266 | 0.057 | 3.156 | 3.020 |
| Forward/inverse direct DFT | 32 × 32 complex | 1.232 | 1.643 | 0.892 | 0.080 | 2.589 | 0.579 |
| Gaussian/Sobel/Harris/NMS | 128 × 128 | 2.379 | 3.138 | 1.841 | 0.089 | 4.571 | 2.345 |
| Lossless tiled RLE round trip | 65536 uints | 0.625 | 1.367 | 0.462 | 0.248 | 1.256 | 1.253 |
| EVT3 event decoding | 65536 sixteen-bit words | 2.990 | 2.222 | 2.993 | 0.734 | 4.873 | 6.988 |
| Async ring | 32 × 257 floats, 3 slots | 1.207 | 7.759 | 0.094 | 0.730 | 0.143 | 1.976 |
| EVT3 + optical flow | 32 × 24 sampled plane, history 64 | 0.393 | 1.731 | 0.290 | 0.215 | 0.640 | 0.656 |
| ANN training | 32 samples, 33 parameters | 0.085 | 0.520 | 0.072 | 0.042 | 0.206 | 0.114 |
| ANN inference | 32 samples, 33 parameters | 0.004 | 0.156 | 0.002 | 0.015 | 0.006 | 0.037 |
| CNN training | 32 samples, 23 parameters | 1.615 | 0.644 | 1.359 | 0.086 | 3.254 | 0.490 |
| CNN inference | 32 samples, 23 parameters | 0.064 | 0.161 | 0.054 | 0.034 | 0.138 | 0.057 |
| Transformer training | 8 sequences, 4 tokens × 4, 134 parameters | 1.324 | 0.770 | 0.972 | 0.284 | 2.404 | 0.729 |
| Transformer inference | 8 sequences, 4 tokens × 4, 134 parameters | 0.008 | 0.171 | 0.006 | 0.057 | 0.018 | 0.055 |
| FFT forward/inverse | 8 × 1024 complex | 3.050 | 10.656 | 2.226 | 0.226 | 5.936 | 4.174 |
| FIR low-pass | 8 × 2048 floats, 65 taps | 2.718 | 0.772 | 1.956 | 0.124 | 6.570 | 5.499 |
| Biquad low-pass | 8 × 2048 floats, persistent state | 0.097 | 0.233 | 0.040 | 0.515 | 0.156 | 0.691 |
| FFT band-pass | 8 × 1024 complex, 4-10 kHz | 3.219 | 11.268 | 2.250 | 0.222 | 6.153 | 5.001 |
| LeNet classifier | 32 × 28 × 28, 61,706 parameters | 46.521 | 7.622 | 40.790 | 1.756 | 116.491 | 136.617 |

Reports: [Ryzen 5900X / llvmpipe](https://vrbadev.github.io/shdr/benchmarks/amd-ryzen-5900x.html),
[Core i7-1260P / Iris Xe](https://vrbadev.github.io/shdr/benchmarks/intel-core-i7-1260p.html), and
[Raspberry Pi 5 / V3D](https://vrbadev.github.io/shdr/benchmarks/raspberry-pi-5.html).

EVT3 decoding processes 65536 sixteen-bit words and emits 223772 ordered events.
Its five stages include state reset. The normal-flow case processes 610 seeded
samples on a temporal plane with history limited to 64. The RLE payload and
offset metadata reduce 65536 input words by 4.74 times in this run. The encoder
still reserves worst-case capacity. Single-invocation sum and prefix stages can
dominate GPU latency. Performance varies with workload size and driver.

Validation results: Ryzen/llvmpipe **92 passed, 17 context skips** out of 109 tests.
Raspberry Pi 5 **41 passed, 68 context skips** out of 109. Core i7/Iris Xe
**108 passed** out of 108. Iris Xe exercised all five
GPU profiles. V3D supports ES 3.1 on the tested driver. All **250** bundled
shader/profile compiler checks passed on the Ryzen and Pi configurations. The
Iris Xe configuration ran live driver checks for every GPU profile. The
Ryzen and Pi configurations also ran the optional `glslangValidator` checks.
The same runtime header, shader frontend, native shaders and data bytes were
verified for all three configurations.

### Complete held-out digit evaluation

The prepared LeNet-style model classified **9778 of 10,000** official MNIST test
images correctly, or **97.78%**, on every CPU and GPU path below. All 100,000
native class scores were compared with independent framework logits. The error
column is the maximum of `abs(actual-reference)/(1+abs(reference))`.

| Configuration | CPU seconds | GPU seconds | CPU normalized error | GPU normalized error |
| --- | ---: | ---: | ---: | ---: |
| Ryzen 5900X / llvmpipe | 17.2063 | 1.76238 | 4.57699e-06 | 4.57699e-06 |
| Core i7-1260P / Iris Xe | 13.8871 | 1.95504 | 4.57699e-06 | 4.00418e-06 |
| Pi 5 / V3D | 35.0628 | 41.5901 | 4.00418e-06 | 4.57699e-06 |

These evaluation times include batch construction, input normalization, program
setup, inference, result reads and score comparisons. They are separate from the
resident pipeline timings above. The llvmpipe path uses software rendering on
the Ryzen CPU. Every public-data functional case passed in all six runs.
The [data guide](https://vrbadev.github.io/shdr/datasets.html) explains the input transformations and metric
limits. The [complete results](https://vrbadev.github.io/shdr/results.html#real-data) record exact model,
dataset and implementation fingerprints.
