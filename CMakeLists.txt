cmake_minimum_required(VERSION 3.12) # Require CMake features used by the shader generation helper.
project(shdr VERSION 1.0.0 LANGUAGES CXX) # Set the package version and enable C++ targets.
include(cmake/shdr.cmake) # Load the native shader build helper or CMake argument parser.

option(SHDR_BUILD_EXAMPLES "Build the dual CPU/GPU pipeline examples" ON) # Expose this optional group of examples or validation targets.
option(SHDR_BUILD_TESTS "Build runtime and language compatibility tests" ON) # Expose this optional group of examples or validation targets.
option(SHDR_BUILD_GPU "Enable optional EGL contexts for GPU examples/tests" ON) # Expose this optional group of examples or validation targets.

set(CMAKE_CXX_STANDARD 11) # Set CMAKE_CXX_STANDARD for the following build configuration.
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Set CMAKE_CXX_STANDARD_REQUIRED for the following build configuration.
set(CMAKE_CXX_EXTENSIONS OFF) # Set CMAKE_CXX_EXTENSIONS for the following build configuration.

# The core header provides CPU execution, externally managed GL devices and an executor.
find_package(Threads REQUIRED) # Locate the dependency needed by this build feature.
add_library(shdr INTERFACE) # Declare the header library, optional backend or dynamic example module.
add_library(shdr::shdr ALIAS shdr) # Declare the header library, optional backend or dynamic example module.
target_compile_features(shdr INTERFACE cxx_std_11) # Require the C++11 language features used by the public header.
target_link_libraries(shdr INTERFACE Threads::Threads) # Propagate the selected header, thread, EGL or module dependencies.
target_include_directories(shdr INTERFACE # Expose the header location to build and installed consumers.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # Select the correct dependency path for build or installed consumers.
    $<INSTALL_INTERFACE:include>) # Select the correct dependency path for build or installed consumers.

if(SHDR_BUILD_TESTS) # Enable this block only when its build condition holds.
    enable_testing() # Allow the configured executable checks to run through CTest.
endif() # Finish this conditional build block.

# Enable EGL specifically for applications that use the headless backend.
if(SHDR_BUILD_GPU) # Enable this block only when its build condition holds.
    find_path(SHDR_EGL_INCLUDE_DIR EGL/egl.h) # Locate the optional EGL development headers.
    find_library(SHDR_EGL_LIBRARY NAMES EGL) # Locate the optional EGL runtime library.
    if(SHDR_EGL_INCLUDE_DIR AND SHDR_EGL_LIBRARY) # Enable this block only when its build condition holds.
        set(SHDR_HAVE_EGL TRUE) # Set SHDR_HAVE_EGL for the following build configuration.
        add_library(shdr_egl INTERFACE) # Declare the header library, optional backend or dynamic example module.
        add_library(shdr::egl ALIAS shdr_egl) # Declare the header library, optional backend or dynamic example module.
        set_target_properties(shdr_egl PROPERTIES EXPORT_NAME egl) # Set the target export name, visibility or generated source behavior.
        target_compile_definitions(shdr_egl INTERFACE SHDR_ENABLE_EGL=1) # Provide the optional backend, module path or fixture path to this application.
        target_include_directories(shdr_egl INTERFACE # Expose the header location to build and installed consumers.
            $<BUILD_INTERFACE:${SHDR_EGL_INCLUDE_DIR}>) # Select the correct dependency path for build or installed consumers.
        target_link_libraries(shdr_egl INTERFACE shdr::shdr # Propagate the selected header, thread, EGL or module dependencies.
            $<BUILD_INTERFACE:${SHDR_EGL_LIBRARY}> $<INSTALL_INTERFACE:EGL>) # Select the correct dependency path for build or installed consumers.
        message(STATUS "EGL found: GPU examples and tests enabled") # Explain the dependency choice or reject an inconsistent target configuration.
    else() # Handle the alternative installation or dependency case.
        message(STATUS "Building CPU examples and tests. GPU targets require EGL development files") # Explain the dependency choice or reject an inconsistent target configuration.
    endif() # Finish this conditional build block.
endif() # Finish this conditional build block.

function(shdr_example target source) # Define this reusable target configuration operation.
    add_executable(${target} ${source}) # Build this example, test or shader export executable.
    target_link_libraries(${target} PRIVATE shdr::shdr) # Propagate the selected header, thread, EGL or module dependencies.
    if(SHDR_HAVE_EGL) # Enable this block only when its build condition holds.
        target_compile_definitions(${target} PRIVATE SHDR_HAVE_EGL=1) # Provide the optional backend, module path or fixture path to this application.
        target_link_libraries(${target} PRIVATE shdr::egl) # Propagate the selected header, thread, EGL or module dependencies.
    endif() # Finish this conditional build block.
    if(SHDR_BUILD_TESTS) # Enable this block only when its build condition holds.
        add_test(NAME ${target}_cpu COMMAND ${target}) # Register this executable or validation script with CTest.
        if(SHDR_HAVE_EGL) # Enable this block only when its build condition holds.
            foreach(profile es31 es32 gl43 gl45 gl46) # Apply these settings to each selected profile, shader or target.
                add_test(NAME ${target}_${profile} COMMAND ${target} --gpu ${profile}) # Register this executable or validation script with CTest.
                set_tests_properties(${target}_${profile} PROPERTIES SKIP_RETURN_CODE 77) # Apply this target or validation configuration.
            endforeach() # Finish configuring the selected items.
        endif() # Finish this conditional build block.
    endif() # Finish this conditional build block.
endfunction() # Finish this target configuration operation.

function(shdr_check_standard target) # Define this reusable target configuration operation.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang") # Enable this block only when its build condition holds.
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -pedantic-errors) # Check portable C++ syntax and useful compiler warnings.
    endif() # Finish this conditional build block.
endfunction() # Finish this target configuration operation.

if(SHDR_BUILD_EXAMPLES) # Enable this block only when its build condition holds.
    shdr_example(example_pipeline examples/pipeline.cpp) # Apply this target or validation configuration.
    shdr_example(example_images examples/images.cpp) # Apply this target or validation configuration.
    shdr_example(example_hybrid examples/hybrid.cpp) # Apply this target or validation configuration.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/async.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_async examples/async.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_async PRIVATE SOURCES examples/shaders/affine.cpp) # Apply this target or validation configuration.
    endif() # Finish this conditional build block.
    foreach(target example_pipeline example_hybrid) # Apply these settings to each selected profile, shader or target.
        shdr_target_shaders(${target} PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/affine.cpp examples/shaders/clamp.cpp examples/shaders/sum.cpp) # Compile these native shader stages into this target.
    endforeach() # Finish configuring the selected items.
    shdr_target_shaders(example_images PRIVATE SOURCES # Apply this target or validation configuration.
        examples/shaders/buffer_to_image.cpp examples/shaders/blur.cpp examples/shaders/image_to_buffer.cpp) # Compile these native shader stages into this target.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/spectral.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_spectral examples/spectral.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_spectral PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/convolution2d.cpp examples/shaders/convolution3d.cpp # Compile these native shader stages into this target.
            examples/shaders/dft_rows.cpp examples/shaders/dft_columns.cpp) # Compile these native shader stages into this target.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/vision.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_vision examples/vision.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_vision PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/gaussian_horizontal.cpp examples/shaders/gaussian_vertical.cpp # Compile these native shader stages into this target.
            examples/shaders/sobel.cpp examples/shaders/harris_tensor.cpp # Compile these native shader stages into this target.
            examples/shaders/harris_response.cpp examples/shaders/nonmax.cpp) # Compile these native shader stages into this target.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/imaging.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_imaging examples/imaging.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_imaging PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/bayer_unpack.cpp examples/shaders/debayer.cpp # Compile these native shader stages into this target.
            examples/shaders/luminance.cpp examples/shaders/rle_count.cpp # Compile these native shader stages into this target.
            examples/shaders/rle_offsets.cpp examples/shaders/rle_encode.cpp # Compile these native shader stages into this target.
            examples/shaders/rle_decode.cpp) # Compile these native shader stages into this target.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/evt3.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_evt3 examples/evt3.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_evt3 PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/evt3_summary.cpp examples/shaders/evt3_seed.cpp # Compile these native shader stages into this target.
            examples/shaders/evt3_emit.cpp examples/shaders/evt3_reset.cpp # Compile these native shader stages into this target.
            examples/shaders/evt3_flow.cpp # Compile these native shader stages into this target.
            examples/shaders/rle_offsets.cpp) # Compile these native shader stages into this target.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/neural.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_neural examples/neural.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_neural PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/ann_infer.cpp examples/shaders/ann_gradient.cpp # Compile these native shader stages into this target.
            examples/shaders/nn_sgd.cpp examples/shaders/nn_binary_loss.cpp) # Compile these native shader stages into this target.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cnn.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_cnn examples/cnn.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_cnn PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/cnn_infer.cpp examples/shaders/cnn_gradient.cpp # Compile these native shader stages into this target.
            examples/shaders/nn_sgd.cpp examples/shaders/nn_binary_loss.cpp) # Compile these native shader stages into this target.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/transformer.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_transformer examples/transformer.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_transformer PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/transformer_infer.cpp examples/shaders/transformer_gradient.cpp # Compile these native shader stages into this target.
            examples/shaders/nn_sgd.cpp) # Compile these native shader stages into this target.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/signal.cpp") # Enable this block only when its build condition holds.
        shdr_example(example_signal examples/signal.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_signal PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/fft_bit_reverse.cpp examples/shaders/fft_butterfly.cpp # Compile these native shader stages into this target.
            examples/shaders/signal_window.cpp examples/shaders/signal_frequency_mask.cpp # Compile these native shader stages into this target.
            examples/shaders/signal_fir.cpp examples/shaders/signal_biquad.cpp) # Compile these native shader stages into this target.
    endif() # Finish this conditional build block.

    # The dynamically loaded plugin embeds the very same shader source compiled as C++.
    if(UNIX) # Enable this block only when its build condition holds.
        add_library(shdr_example_plugin SHARED examples/plugin.cpp) # Declare the header library, optional backend or dynamic example module.
        target_link_libraries(shdr_example_plugin PRIVATE shdr::shdr) # Propagate the selected header, thread, EGL or module dependencies.
        shdr_target_shaders(shdr_example_plugin PRIVATE SOURCES examples/shaders/affine.cpp) # Apply this target or validation configuration.
        set_target_properties(shdr_example_plugin PROPERTIES CXX_VISIBILITY_PRESET hidden) # Set the target export name, visibility or generated source behavior.
        shdr_example(example_dynamic examples/dynamic.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(example_dynamic PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/clamp.cpp examples/shaders/sum.cpp) # Compile these native shader stages into this target.
        target_link_libraries(example_dynamic PRIVATE ${CMAKE_DL_LIBS}) # Propagate the selected header, thread, EGL or module dependencies.
        target_compile_definitions(example_dynamic PRIVATE # Provide the optional backend, module path or fixture path to this application.
            SHDR_PLUGIN_PATH="$<TARGET_FILE:shdr_example_plugin>") # Provide the built module path for the dynamic loader example.
        add_dependencies(example_dynamic shdr_example_plugin) # Build the dynamically loaded module before its application.
    endif() # Finish this conditional build block.
endif() # Finish this conditional build block.

if(SHDR_BUILD_TESTS) # Enable this block only when its build condition holds.
    add_executable(test_runtime tests/test_runtime.cpp) # Build this example, test or shader export executable.
    target_link_libraries(test_runtime PRIVATE shdr::shdr) # Propagate the selected header, thread, EGL or module dependencies.
    shdr_target_shaders(test_runtime PRIVATE SOURCES examples/shaders/affine.cpp # Apply this target or validation configuration.
        examples/shaders/buffer_to_image.cpp examples/shaders/blur.cpp examples/shaders/image_to_buffer.cpp) # Compile these native shader stages into this target.
    add_test(NAME runtime_cpu COMMAND test_runtime) # Register this executable or validation script with CTest.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_async.cpp") # Enable this block only when its build condition holds.
        add_executable(test_async tests/test_async.cpp) # Build this example, test or shader export executable.
        target_link_libraries(test_async PRIVATE shdr::shdr) # Propagate the selected header, thread, EGL or module dependencies.
        shdr_target_shaders(test_async PRIVATE SOURCES examples/shaders/affine.cpp) # Apply this target or validation configuration.
        add_test(NAME async_cpu COMMAND test_async) # Register this executable or validation script with CTest.
    endif() # Finish this conditional build block.
    if(SHDR_HAVE_EGL) # Enable this block only when its build condition holds.
        add_executable(test_gpu tests/test_gpu.cpp) # Build this example, test or shader export executable.
        target_link_libraries(test_gpu PRIVATE shdr::egl) # Propagate the selected header, thread, EGL or module dependencies.
        target_compile_definitions(test_gpu PRIVATE SHDR_HAVE_EGL=1) # Provide the optional backend, module path or fixture path to this application.
        shdr_target_shaders(test_gpu PRIVATE SOURCES # Apply this target or validation configuration.
            examples/shaders/affine.cpp examples/shaders/scalar_images.cpp # Compile these native shader stages into this target.
            examples/shaders/rgba_float_image.cpp examples/shaders/rgba_int_image.cpp # Compile these native shader stages into this target.
            examples/shaders/rgba_uint_image.cpp) # Compile these native shader stages into this target.
        foreach(profile es31 es32 gl43 gl45 gl46) # Apply these settings to each selected profile, shader or target.
            add_test(NAME runtime_${profile} COMMAND test_gpu --gpu ${profile}) # Register this executable or validation script with CTest.
            set_tests_properties(runtime_${profile} PROPERTIES SKIP_RETURN_CODE 77) # Apply this target or validation configuration.
        endforeach() # Finish configuring the selected items.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_math_gpu.cpp") # Enable this block only when its build condition holds.
        shdr_example(test_math_gpu tests/test_math_gpu.cpp) # Apply this target or validation configuration.
        shdr_target_shaders(test_math_gpu PRIVATE SOURCES examples/shaders/math.cpp) # Apply this target or validation configuration.
    endif() # Finish this conditional build block.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_math.cpp") # Enable this block only when its build condition holds.
        add_executable(test_math tests/test_math.cpp) # Build this example, test or shader export executable.
        target_link_libraries(test_math PRIVATE shdr::shdr) # Propagate the selected header, thread, EGL or module dependencies.
        add_test(NAME math_cpu COMMAND test_math) # Register this executable or validation script with CTest.
    endif() # Finish this conditional build block.
endif() # Finish this conditional build block.

# Offline validation covers every example shader and every GLSL output profile.
add_executable(export_shader examples/export_shader.cpp) # Build this example, test or shader export executable.
target_link_libraries(export_shader PRIVATE shdr::shdr) # Propagate the selected header, thread, EGL or module dependencies.
file(GLOB SHDR_EXAMPLE_SHADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/shaders/*.cpp") # Collect shader inputs or create the installed package configuration.
shdr_target_shaders(export_shader PRIVATE SOURCES ${SHDR_EXAMPLE_SHADERS}) # Apply this target or validation configuration.
if(SHDR_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/benchmark.cpp") # Enable this block only when its build condition holds.
    shdr_example(example_benchmark examples/benchmark.cpp) # Apply this target or validation configuration.
    shdr_target_shaders(example_benchmark PRIVATE SOURCES ${SHDR_EXAMPLE_SHADERS}) # Apply this target or validation configuration.
    if(TARGET shdr_example_plugin) # Enable this block only when its build condition holds.
        target_link_libraries(example_benchmark PRIVATE ${CMAKE_DL_LIBS}) # Propagate the selected header, thread, EGL or module dependencies.
        target_compile_definitions(example_benchmark PRIVATE # Provide the optional backend, module path or fixture path to this application.
            SHDR_PLUGIN_PATH="$<TARGET_FILE:shdr_example_plugin>") # Provide the built module path for the dynamic loader example.
        add_dependencies(example_benchmark shdr_example_plugin) # Build the dynamically loaded module before its application.
    endif() # Finish this conditional build block.
endif() # Finish this conditional build block.
if(SHDR_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/real_data.cpp") # Enable this block only when its build condition holds.
    shdr_example(example_real_data examples/real_data.cpp) # Apply this target or validation configuration.
    shdr_target_shaders(example_real_data PRIVATE SOURCES ${SHDR_EXAMPLE_SHADERS}) # Apply this target or validation configuration.
    target_compile_definitions(example_real_data PRIVATE # Provide the optional backend, module path or fixture path to this application.
        SHDR_EXAMPLE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples/data") # Locate the public example fixtures from the generated executable.
    if(TARGET shdr_example_plugin) # Enable this block only when its build condition holds.
        target_link_libraries(example_real_data PRIVATE ${CMAKE_DL_LIBS}) # Propagate the selected header, thread, EGL or module dependencies.
        target_compile_definitions(example_real_data PRIVATE # Provide the optional backend, module path or fixture path to this application.
            SHDR_PLUGIN_PATH="$<TARGET_FILE:shdr_example_plugin>") # Provide the built module path for the dynamic loader example.
        add_dependencies(example_real_data shdr_example_plugin) # Build the dynamically loaded module before its application.
    endif() # Finish this conditional build block.
endif() # Finish this conditional build block.
find_package(Python3 COMPONENTS Interpreter QUIET) # Locate the dependency needed by this build feature.
find_program(SHDR_GLSLANG_VALIDATOR glslangValidator) # Locate the standalone GLSL compiler when it is available.
if(Python3_Interpreter_FOUND AND SHDR_GLSLANG_VALIDATOR) # Enable this block only when its build condition holds.
    set(SHDR_GLSL_VALIDATION_COMMAND ${Python3_EXECUTABLE} # Set SHDR_GLSL_VALIDATION_COMMAND for the following build configuration.
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_glsl.py # Supply these files, target properties or command arguments.
        --exporter $<TARGET_FILE:export_shader> # Supply this explicit tool option.
        --validator ${SHDR_GLSLANG_VALIDATOR} # Supply this explicit tool option.
        --source-dir ${CMAKE_CURRENT_SOURCE_DIR}) # Supply this explicit tool option.
    add_custom_target(validate_glsl COMMAND ${SHDR_GLSL_VALIDATION_COMMAND} # Expose this validation or documentation command as a build target.
        DEPENDS export_shader VERBATIM) # Regenerate when one of these source dependencies changes.
    add_custom_target(verify_public_shaders # Expose this validation or documentation command as a build target.
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_public_shaders.py # Run this tool with the arguments continued below.
            --cache ${CMAKE_CURRENT_BINARY_DIR}/public-shaders # Supply this explicit tool option.
            --output ${CMAKE_CURRENT_BINARY_DIR}/public-shader-results.html # Write a complete report that opens in a browser.
            --cxx ${CMAKE_CXX_COMPILER} --glslang ${SHDR_GLSLANG_VALIDATOR} # Supply this explicit tool option.
        VERBATIM) # Preserve each command argument exactly when invoking the tool.
    if(SHDR_BUILD_TESTS) # Enable this block only when its build condition holds.
        add_test(NAME glsl_compilation COMMAND ${SHDR_GLSL_VALIDATION_COMMAND}) # Register this executable or validation script with CTest.
    endif() # Finish this conditional build block.
endif() # Finish this conditional build block.

if(Python3_Interpreter_FOUND) # Enable this block only when its build condition holds.
    if(SHDR_BUILD_TESTS) # Enable this block only when its build condition holds.
        add_test(NAME header_is_current COMMAND ${Python3_EXECUTABLE} # Register this executable or validation script with CTest.
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/amalgamate.py --check) # Supply these files, target properties or command arguments.
        add_test(NAME fixture_integrity COMMAND ${Python3_EXECUTABLE} # Verify the bundled public fixtures against their recorded hashes.
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_datasets.py --verify) # Verify each recorded fixture size and SHA-256 digest.
        set(SHDR_FRONTEND_TEST_ARGS --cxx ${CMAKE_CXX_COMPILER}) # Set SHDR_FRONTEND_TEST_ARGS for the following build configuration.
        if(SHDR_GLSLANG_VALIDATOR) # Enable this block only when its build condition holds.
            list(APPEND SHDR_FRONTEND_TEST_ARGS --glslang ${SHDR_GLSLANG_VALIDATOR}) # Add these optional arguments or targets to the configured list.
        endif() # Finish this conditional build block.
        add_test(NAME shader_frontend COMMAND ${Python3_EXECUTABLE} # Register this executable or validation script with CTest.
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_shader_compiler.py ${SHDR_FRONTEND_TEST_ARGS}) # Supply these files, target properties or command arguments.
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_transformer.py") # Enable this block only when its build condition holds.
            add_test(NAME transformer_shaders_are_current COMMAND ${Python3_EXECUTABLE} # Register this executable or validation script with CTest.
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_transformer.py --check) # Supply these files, target properties or command arguments.
        endif() # Finish this conditional build block.
    endif() # Finish this conditional build block.
    add_custom_target(generate_docs # Expose this validation or documentation command as a build target.
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_api.py # Refresh the header's browsable API reference.
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_sources.py # Refresh the example and build-tool source browsers.
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/highlight_docs.py --site ${CMAKE_CURRENT_SOURCE_DIR}/docs # Color maintained code excerpts with local static HTML and CSS.
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_docs.py # Check the generated site's local links and document structure.
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} VERBATIM) # Supply these files, target properties or command arguments.
    add_custom_target(check_docs # Check local pages, links and assets throughout the static site.
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_docs.py # Validate local links, fragment targets and document structure.
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} VERBATIM) # Resolve the published assets from this source checkout.
endif() # Finish this conditional build block.

foreach(target example_pipeline example_images example_hybrid example_dynamic # Apply these settings to each selected profile, shader or target.
               example_spectral example_vision example_imaging example_benchmark example_async example_evt3 # Supply these files, target properties or command arguments.
               example_neural example_cnn example_transformer # Supply these files, target properties or command arguments.
               example_signal example_real_data # Check both synthetic signal and public dataset examples with strict C++11 flags.
               shdr_example_plugin test_runtime test_gpu test_math test_math_gpu test_async export_shader) # Supply these files, target properties or command arguments.
    if(TARGET ${target}) # Enable this block only when its build condition holds.
        shdr_check_standard(${target}) # Apply this target or validation configuration.
    endif() # Finish this conditional build block.
endforeach() # Finish configuring the selected items.

include(GNUInstallDirs) # Load the native shader build helper or CMake argument parser.
include(CMakePackageConfigHelpers) # Generate a version file understood by installed find_package consumers.
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/shdrConfigVersion.cmake" # Put package version checks beside the generated configuration.
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion) # Accept this release and compatible requested versions within major version one.
install(FILES shdr.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}) # Install the public header, helper or exported CMake package files.
install(TARGETS shdr EXPORT shdrTargets) # Install the public header, helper or exported CMake package files.
if(TARGET shdr_egl) # Enable this block only when its build condition holds.
    install(TARGETS shdr_egl EXPORT shdrTargets) # Install the public header, helper or exported CMake package files.
endif() # Finish this conditional build block.
install(EXPORT shdrTargets NAMESPACE shdr:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shdr) # Install the public header, helper or exported CMake package files.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/shdrConfig.cmake" # Collect shader inputs or create the installed package configuration.
    "include(CMakeFindDependencyMacro)\nfind_dependency(Threads)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/shdrTargets.cmake\")\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/shdr.cmake\")\nif(TARGET shdr::egl)\n  find_path(SHDR_PACKAGE_EGL_INCLUDE_DIR EGL/egl.h)\n  find_library(SHDR_PACKAGE_EGL_LIBRARY NAMES EGL)\n  if(SHDR_PACKAGE_EGL_INCLUDE_DIR AND SHDR_PACKAGE_EGL_LIBRARY)\n    set_property(TARGET shdr::egl APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES \"\${SHDR_PACKAGE_EGL_INCLUDE_DIR}\")\n    set_property(TARGET shdr::egl PROPERTY INTERFACE_LINK_LIBRARIES \"shdr::shdr;\${SHDR_PACKAGE_EGL_LIBRARY}\")\n  endif()\nendif()\n") # Supply these files, target properties or command arguments.
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shdrConfig.cmake" # Install the public header, helper or exported CMake package files.
    "${CMAKE_CURRENT_BINARY_DIR}/shdrConfigVersion.cmake" # Let find_package require a compatible version of the installed library.
    cmake/shdr.cmake scripts/compile_shader.py # Supply these files, target properties or command arguments.
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shdr) # Supply these files, target properties or command arguments.
