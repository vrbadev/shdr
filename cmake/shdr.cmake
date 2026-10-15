# Compile native GLSL .cpp sources into C++11 adapters with embedded GPU source.
include(CMakeParseArguments) # Load CMake's keyword-argument parser for the shader source list.
find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED) # Locate Python 3.8 or newer to run the native shader adapter generator.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../scripts/compile_shader.py") # Select the source-tree adapter generator when it is available.
    set(_SHDR_COMPILE_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/../scripts/compile_shader.py") # Use the frontend script shipped in the source tree.
else() # Select the frontend location used by installed packages.
    set(_SHDR_COMPILE_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/compile_shader.py") # Find the installed frontend beside this CMake helper.
endif() # Finish choosing the frontend location.

# shdr_target_shaders(app PRIVATE SOURCES shader.cpp other.cpp)
# Attach adapters directly to an executable or a loaded shared-library/plugin target.
# Final targets retain the shader registration objects used by make_kernel.
function(shdr_target_shaders target visibility) # Attach each native source and generated adapter to the requested final target.
    if(NOT TARGET ${target}) # Validate that the destination target has already been declared.
        message(FATAL_ERROR "shdr_target_shaders: target ${target} does not exist") # Explain that the application target must exist before attaching shaders.
    endif() # Finish validating the destination target.
    cmake_parse_arguments(SHDR "" "" "SOURCES" ${ARGN}) # Read the explicitly supplied native shader source list.
    if(SHDR_UNPARSED_ARGUMENTS OR NOT SHDR_SOURCES) # Require a nonempty SOURCES list with recognized helper arguments.
        message(FATAL_ERROR "usage: shdr_target_shaders(target PRIVATE SOURCES shader.cpp ...)") # Show the expected source-list syntax for an invalid call.
    endif() # Finish validating the shader source arguments.
    foreach(shader IN LISTS SHDR_SOURCES) # Attach one generated adapter for each supplied native shader.
        get_filename_component(shader_path "${shader}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}") # Resolve this shader path or derive its factory basename.
        get_filename_component(shader_name "${shader}" NAME_WE) # Resolve this shader path or derive its factory basename.
        get_target_property(registered_names ${target} SHDR_SHADER_NAMES) # Read the shader basenames already attached to this target.
        if(shader_name IN_LIST registered_names) # Reject factory-name collisions from repeated shader basenames.
            message(FATAL_ERROR "shdr_target_shaders: duplicate shader basename ${shader_name}") # Identify the repeated basename that would register the same factory twice.
        endif() # Finish checking this target for a duplicate shader name.
        set_property(TARGET ${target} APPEND PROPERTY SHDR_SHADER_NAMES "${shader_name}") # Remember this target metadata for duplicate detection.
        set(generated "${CMAKE_CURRENT_BINARY_DIR}/shdr-generated/${target}/${shader_name}.cpp") # Choose a separate generated C++ file for this target and shader basename.
        add_custom_command(OUTPUT "${generated}" # Generate a C++11 adapter whenever its shader or compiler changes.
            COMMAND ${Python3_EXECUTABLE} "${_SHDR_COMPILE_SCRIPT}" # Run this tool with the arguments continued below.
                --input "${shader_path}" --output "${generated}" --name "${shader_name}_kernel" # Supply this explicit tool option.
            DEPENDS "${shader_path}" "${_SHDR_COMPILE_SCRIPT}" # Regenerate when one of these source dependencies changes.
            COMMENT "Compiling and embedding shader ${shader_name}.cpp" # Describe the generated shader step in the build output.
            VERBATIM) # Preserve each command argument exactly when invoking the tool.
        # The native shader is shown in the IDE, GCC compiles its generated C++11 adapter.
        set_source_files_properties("${shader_path}" PROPERTIES HEADER_FILE_ONLY TRUE) # Show the native shader in the IDE while compiling its generated adapter.
        target_sources(${target} ${visibility} "${generated}" "${shader_path}") # Attach the generated adapter and its native shader to the application target.
    endforeach() # Finish attaching every requested shader.
endfunction() # Finish defining the native shader attachment helper.
