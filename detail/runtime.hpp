#ifndef SHDR_RUNTIME_HPP // Skip this section if it was already included.
#define SHDR_RUNTIME_HPP /* Mark this section as included. */

/** @brief Major version of shdr. */
#define SHDR_VERSION_MAJOR 1 // Identify the public API's major release series.
/** @brief Minor version within the current major release. */
#define SHDR_VERSION_MINOR 0 // Identify additions within the public API's release series.
/** @brief Patch version within the current minor release. */
#define SHDR_VERSION_PATCH 0 // Identify corrections within this minor release.
/** @brief The library release version as a string. */
#define SHDR_VERSION "1.0.0" // Expose the library release as a compile-time string.

#include <algorithm> // Use standard searches, bounds, and resource copying.
#include <cstddef> // Use portable pointer sizes and offsets.
#include <cstdint> // Use integer widths that match GLSL values.
#include <cctype> // Recognize characters in embedded shader declarations.
#include <cstring> // Copy bytes and function addresses with std::memcpy.
#include <functional> // Store GL resolvers and deferred cleanup callbacks.
#include <initializer_list> // Accept concise resource initialization from application values.
#include <limits> // Check arithmetic against the type limits.
#include <map> // Find registered shader factories and shared kernel snapshots.
#include <memory> // Retain resources while pipelines and queued jobs use them.
#include <mutex> // Protect shared work queues and deferred deletion.
#include <thread> // Track the context thread and start CPU workers.
#include <sstream> // Build readable shader and driver diagnostics.
#include <stdexcept> // Report invalid resources and failed execution.
#include <string> // Keep embedded shader text and named interfaces.
#include <type_traits> // Check shader storage and port types at compile time.
#include <vector> // Store ordered stages, ports, and resource values.

/** @brief Portable, header-only CPU/OpenGL compute pipelines. */
namespace shdr { // Group these definitions in the shdr namespace.
/** @brief A shader, resource, context or dispatch validation failure. */
struct error : std::runtime_error { // Keep the error values and operations together.
    explicit error(const std::string& s) : std::runtime_error(s) { // Preserve the supplied validation or driver diagnostic.
    } // End this scope in runtime.
}; // End this scope in runtime.

/** @brief Three unsigned dimensions. dispatch dimensions count work groups. */
struct extent { // Keep the extent values and operations together.
    unsigned x, y, z; // Keep the three unsigned dispatch or local dimensions.
    /** @brief Construct three workgroup counts, defaulting each dimension to one. */
    explicit extent(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) { // Initialize all three positive-or-zero dimensions supplied by the caller.
    } // End this scope in runtime.
}; // End this scope in runtime.
/** @brief Minimum shader language selected independently of the context's actual version. */
enum class profile { es31, es32, gl43, gl45, gl46 }; // Name the supported shader language profiles.
/** @brief Automatic uses CPU execution for run() and GPU execution for run(device). */
enum class backend { automatic, cpu, gpu }; // Select automatic, explicit CPU or explicit GPU execution.
/** @brief Resource access qualifiers. readonly also prevents CPU writes at compile time. */
enum class access { readonly, writeonly, readwrite }; // Express how a shader may access a bound resource.

namespace detail { // Group these definitions in the detail namespace.
inline bool identifier_char(char c) { // Recognize the letters, digits, and underscore allowed in an identifier.
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; // Recognize the letters, digits, and underscore allowed in an identifier.
} // End this scope in identifier char.
inline std::string trim(const std::string& text) { // Remove surrounding whitespace from one resource argument.
    const std::size_t first = text.find_first_not_of(" \t\r\n"); // Set the first non-space character.
    return first == std::string::npos // Remove surrounding whitespace from one resource argument.
               ? std::string() // Select the value for this branch of the condition.
               : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1); // Copy only the non-whitespace portion of this argument.
} // End this scope in trim.
 // ESSL forbids token-pasting macros. Expand only resource declarations in the host,
 // preserving comments, string literals, shader statements and original line numbers.
inline std::string expand_declarations(const std::string& source, bool es) { // Expand host resource declarations into native GLSL interfaces.
    std::string out; // Keep the assembled shader text.
    for (std::size_t i = 0; i < source.size();) { // Walk the embedded shader text while respecting its token boundaries.
        const std::size_t start = i; // Set the beginning of the current token or range.
        if (source.compare(i, 2, "//") == 0) { // Copy the line comment verbatim.
            const std::size_t end = source.find('\n', i); // Set the end of the current range.
            i = end == std::string::npos ? source.size() : end + 1; // Advance beyond the preserved comment or declaration.
        } else if (source.compare(i, 2, "/*") == 0) { // Copy the block comment verbatim.
            const std::size_t end = source.find("*/", i + 2); // Set the end of the current range.
            i = end == std::string::npos ? source.size() : end + 2; // Advance beyond the preserved comment or declaration.
        } else if (source[i] == '\"' || source[i] == '\'') { // Copy quoted shader text as one lexical unit.
            const char quote = source[i++]; // Remember which quote terminates this shader literal.
            while (i < source.size()) { // Walk the embedded shader text while respecting its token boundaries.
                if (source[i] == '\\') { // Treat an escaped character as part of the quoted shader text.
                    i = (std::min)(i + 2, source.size()); // Advance past the escaped character within the source bounds.
                    continue; // Continue with the next token or work item.
                } // End this scope in expand declarations.
                if (source[i++] == quote) // Choose this branch when source[i++] equals quote.
                    break; // Stop this search or loop once its current item is complete.
            } // End this scope in expand declarations.
        } else if (identifier_char(source[i])) { // Recognize the letters, digits, and underscore allowed in an identifier.
            while (i < source.size() && identifier_char(source[i])) // Walk the embedded shader text while respecting its token boundaries.
                ++i; // Expand host resource declarations into native GLSL interfaces.
            const std::string token = source.substr(start, i - start); // Set the current shader identifier.
            if (token == "SHDR_BUFFER" || token == "SHDR_IMAGE2D") { // Choose this branch when token equals "SHDR BUFFER" or token equals "SHDR IMAGE2D".
                std::size_t open = i; // Set the opening parenthesis of a resource declaration.
                while (open < source.size() && // Walk the embedded shader text while respecting its token boundaries.
                       std::isspace(static_cast<unsigned char>(source[open]))) // Keep the opening parenthesis of a resource declaration.
                    ++open; // Keep the opening parenthesis of a resource declaration.
                if (open >= source.size() || source[open] != '(') // Check resource size or driver limits before accessing storage.
                    throw error("resource declaration requires parentheses: " + token); // Report resource declaration requires parentheses: .
                std::size_t end = open + 1, argument = end; // Set the end of the current range.
                unsigned depth = 1; // Set the current parenthesis nesting.
                std::vector<std::string> args; // Keep the parsed declaration arguments.
                for (; end < source.size(); ++end) { // Walk the embedded shader text while respecting its token boundaries.
                    if (source[end] == '(') // Choose this branch when source[end] equals '('.
                        ++depth; // Keep the current parenthesis nesting.
                    if (source[end] == ')') // Choose this branch when source[end] equals ')'.
                        --depth; // Keep the current parenthesis nesting.
                    if ((source[end] == ',' && depth == 1) || depth == 0) { // Choose this branch when (source[end] equals ',' and depth equals 1) or depth equals 0.
                        args.push_back(trim(source.substr(argument, end - argument))); // Retain this entry in the ordered resource or work collection.
                        argument = end + 1; // Set the beginning of the next resource argument.
                    } // End this scope in expand declarations.
                    if (!depth) // Finish the argument list at its matching closing parenthesis.
                        break; // Stop this search or loop once its current item is complete.
                } // End this scope in expand declarations.
                const bool image = token == "SHDR_IMAGE2D"; // Choose image or storage-buffer declaration syntax.
                if (depth || args.size() != (image ? 5u : 4u)) // Check resource size or driver limits before accessing storage.
                    throw error("invalid resource declaration: " + token); // Report invalid resource declaration: .
                const std::string mode = args.back(); // Set the declared shader resource access.
                if (mode != "readonly" && mode != "writeonly" && mode != "readwrite") // Choose the branch matching this interface or resource access.
                    throw error("invalid resource access: " + mode); // Report invalid resource access: .
                const std::string qualifier = mode == "readwrite" ? "" : mode + " "; // Set the access qualifier emitted into GLSL.
                if (image) { // Choose this branch when image.
                    if (es && mode == "readwrite" && args[1].compare(0, 4, "rgba") == 0) // Choose the branch matching this interface or resource access.
                        throw error( // Report the invalid operation.
                            "GLES RGBA images require readonly or writeonly declarations: " + // Append this literal GLSL source fragment.
                            args[3]); // Expand host resource declarations into native GLSL interfaces.
                    out += "layout(" + args[1] + ",binding=" + args[0] + ") " + qualifier + // Set the assembled shader text.
                           "uniform highp " + args[2] + " " + args[3] + ";"; // Append this literal GLSL source fragment.
                } else // Handle the alternative resource or completion case.
                    out += "layout(std430,binding=" + args[0] + ") " + qualifier + // Set the assembled shader text.
                           "buffer shdr_block_" + args[2] + " { " + args[1] + " " + args[2] + // Append this literal GLSL source fragment.
                           "[]; };"; // Append this literal GLSL source fragment.
                out.append(static_cast<std::size_t>( // Restore the original declaration's line count after expansion.
                               std::count(source.begin() + static_cast<std::ptrdiff_t>(start), // Keep the beginning of the current token or range.
                                          source.begin() + static_cast<std::ptrdiff_t>(end + 1), // Stop counting at the end of the replaced resource declaration.
                                          '\n')), // Count the newlines originally present in this declaration.
                           '\n'); // Append one newline for each removed original newline.
                i = end + 1; // Advance beyond the preserved comment or declaration.
                continue; // Continue with the next token or work item.
            } // End this scope in expand declarations.
        } else // Handle the alternative resource or completion case.
            ++i; // Expand host resource declarations into native GLSL interfaces.
        out.append(source, start, i - start); // Preserve this untouched token or comment in the exported GLSL.
    } // End this scope in expand declarations.
    return out; // Return the assembled shader text.
} // End this scope in expand declarations.
} // namespace detail

/** @internal Export an embedded shader as standard GLSL, with version and declaration macros.
 * This function supplies the version directive. Guard the body's C++ include
 * with #ifdef __cplusplus. The exported source contains the selected version
 * and expanded declarations.
 */
inline std::string glsl_source(const std::string& source, profile p) { // Add the selected GLSL version and shared-declaration definitions.
    std::istringstream lines(source); // Read the embedded shader one line at a time.
    std::string line; // Read each embedded source line while checking version directives.
    while (std::getline(lines, line)) { // Inspect each source line for an unsupported version directive.
        std::size_t start = line.find_first_not_of(" \t\r"); // Set the beginning of the current token or range.
        if (start != std::string::npos && line[start] == '#') { // Choose this branch when start differs from std::string::npos and line[start] equals '#'.
            start = line.find_first_not_of(" \t", start + 1); // Set the beginning of the current token or range.
            if (start != std::string::npos && line.compare(start, 7, "version") == 0) // Choose this branch when start differs from std::string::npos and line.compare(start, 7, "version") equals 0.
                throw error("shared shaders must omit #version; glsl_source supplies it"); // Report shared shaders must omit #version. glsl_source supplies it.
        } // End this scope in glsl source.
    } // End this scope in glsl source.
    const char* version = p == profile::es31   ? "310 es" // Set the current GL version text.
                          : p == profile::es32 ? "320 es" // Select the version associated with this shader profile.
                          : p == profile::gl43 ? "430 core" // Select the version associated with this shader profile.
                          : p == profile::gl45 ? "450 core" // Select the version associated with this shader profile.
                                               : "460 core"; // Select desktop GLSL 4.60 for the remaining profile.
    std::string out = std::string("#version ") + version + "\n"; // Set the assembled shader text.
    if (p == profile::es31 || p == profile::es32) // Choose this branch when p equals profile::es31 or p equals profile::es32.
        out += "precision highp float;\nprecision highp int;\n"; // Set the assembled shader text.
    out += // Set the assembled shader text.
        "#define SHDR_BEGIN(name,x,y,z) layout(local_size_x=x,local_size_y=y,local_size_z=z) in;\n" // Emit a native workgroup declaration for this shader.
        "#define SHDR_END()\n" // Remove the host-only shader wrapper terminator.
        "#define SHDR_MAIN() void main()\n" // Map the shared entry-point spelling to native GLSL main.
        "#define SHDR_UNIFORM(t,n) uniform t n;\n" // Emit a native typed uniform declaration.
        "#define SHDR_SWIZZLE(v,s) (v).s\n" // Map shared swizzle syntax to native GLSL component selection.
        "#define SHDR_NOT(v) not(v)\n" // Map shared boolean negation to the native GLSL builtin.
        "#define SHDR_IN(t,n) in t n\n" // Emit a native input parameter qualifier.
        "#define SHDR_OUT(t,n) out t n\n" // Emit a native output parameter qualifier.
        "#define SHDR_INOUT(t,n) inout t n\n" // Emit a native input-output parameter qualifier.
        "#line 1\n"; // Restore shader line numbering before appending the original source.
 // Desktop GLSL accepts highp precision qualifiers from GLSL 4.10 onward.
    return out + detail::expand_declarations(source, p == profile::es31 || p == profile::es32); // Add the selected GLSL version and shared-declaration definitions.
} // End this scope in glsl source.

namespace detail { // Group these definitions in the detail namespace.
 // Private GL types and entry points keep the default header compatible with standard C++11.
typedef unsigned int GLenum; // Name the representation used by this GL or shader interface.
typedef unsigned int GLuint; // Name the representation used by this GL or shader interface.
typedef int GLint; // Name the representation used by this GL or shader interface.
typedef int GLsizei; // Name the representation used by this GL or shader interface.
typedef unsigned int GLbitfield; // Name the representation used by this GL or shader interface.
typedef unsigned char GLboolean; // Name the representation used by this GL or shader interface.
typedef std::ptrdiff_t GLsizeiptr; // Name the representation used by this GL or shader interface.
typedef std::ptrdiff_t GLintptr; // Name the representation used by this GL or shader interface.
typedef std::uint64_t GLuint64; // Name the representation used by this GL or shader interface.
typedef void* GLsync; // Name the representation used by this GL or shader interface.
#if defined(_WIN32) // Compile this branch only when its platform or backend condition holds.
#define SHDR_GL_CALL __stdcall /* Match the calling convention of the GL entry points. */
#else // Use the alternative platform or backend definition.
#define SHDR_GL_CALL /* Match the calling convention of the GL entry points. */
#endif // Finish this conditional section.
enum : unsigned { // Name the profile, access, or completion values accepted by this section.
    VERSION = 0x1F02, // Use the GL numeric code for version.
    MAJOR_VERSION = 0x821B, // Use the GL numeric code for major version.
    MINOR_VERSION = 0x821C, // Use the GL numeric code for minor version.
    COMPUTE_SHADER = 0x91B9, // Use the GL numeric code for compute shader.
    COMPILE_STATUS = 0x8B81, // Use the GL numeric code for compile status.
    LINK_STATUS = 0x8B82, // Use the GL numeric code for link status.
    INFO_LOG_LENGTH = 0x8B84, // Use the GL numeric code for info log length.
    COMPUTE_WORK_GROUP_SIZE = 0x8267, // Use the GL numeric code for compute work group size.
    MAX_COMPUTE_WORK_GROUP_COUNT = 0x91BE, // Use the GL numeric code for max compute work group count.
    MAX_COMPUTE_WORK_GROUP_SIZE = 0x91BF, // Use the GL numeric code for max compute work group size.
    MAX_COMPUTE_WORK_GROUP_INVOCATIONS = 0x90EB, // Use the GL numeric code for max compute work group invocations.
    MAX_SHADER_STORAGE_BUFFER_BINDINGS = 0x90DD, // Use the GL numeric code for max shader storage buffer bindings.
    MAX_IMAGE_UNITS = 0x8F38, // Use the GL numeric code for max image units.
    SHADER_STORAGE_BUFFER = 0x90D2, // Use the GL numeric code for shader storage buffer.
    DYNAMIC_DRAW = 0x88E8, // Use the GL numeric code for dynamic draw.
    MAP_READ_BIT = 1, // Use the GL numeric code for map read bit.
    ALL_BARRIER_BITS = 0xFFFFFFFFu, // Use the GL numeric code for all barrier bits.
    TEXTURE_2D = 0x0DE1, // Identify a two-dimensional texture target.
    FLOAT = 0x1406, // Use the GL numeric code for float.
    INT = 0x1404, // Use the GL numeric code for int.
    UNSIGNED_INT = 0x1405, // Use the GL numeric code for unsigned int.
    RED = 0x1903, // Use the GL numeric code for red.
    RGBA = 0x1908, // Use the GL numeric code for rgba.
    RED_INTEGER = 0x8D94, // Use the GL numeric code for red integer.
    RGBA_INTEGER = 0x8D99, // Use the GL numeric code for rgba integer.
    R32F = 0x822E, // Store one 32-bit floating-point component per pixel.
    R32I = 0x8235, // Store one signed 32-bit component per pixel.
    R32UI = 0x8236, // Store one unsigned 32-bit component per pixel.
    RGBA32F = 0x8814, // Store four 32-bit floating-point components per pixel.
    RGBA32I = 0x8D82, // Store four signed 32-bit components per pixel.
    RGBA32UI = 0x8D70, // Store four unsigned 32-bit components per pixel.
    READ_ONLY = 0x88B8, // Use the GL numeric code for read only.
    WRITE_ONLY = 0x88B9, // Use the GL numeric code for write only.
    READ_WRITE = 0x88BA, // Use the GL numeric code for read write.
    UNPACK_ALIGNMENT = 0x0CF5, // Use the GL numeric code for unpack alignment.
    UNPACK_ROW_LENGTH = 0x0CF2, // Use the GL numeric code for unpack row length.
    UNPACK_SKIP_ROWS = 0x0CF3, // Use the GL numeric code for unpack skip rows.
    UNPACK_SKIP_PIXELS = 0x0CF4, // Use the GL numeric code for unpack skip pixels.
    PIXEL_UNPACK_BUFFER = 0x88EC, // Use the GL numeric code for pixel unpack buffer.
    SYNC_GPU_COMMANDS_COMPLETE = 0x9117, // Use the GL numeric code for sync gpu commands complete.
    ALREADY_SIGNALED = 0x911A, // Use the GL numeric code for already signaled.
    TIMEOUT_EXPIRED = 0x911B, // Use the GL numeric code for timeout expired.
    CONDITION_SATISFIED = 0x911C, // Use the GL numeric code for condition satisfied.
    WAIT_FAILED = 0x911D, // Use the GL numeric code for wait failed.
    SYNC_FLUSH_COMMANDS_BIT = 1 // Use the GL numeric code for sync flush commands bit.
}; // End this scope in runtime.

struct gl_api { // Keep the gl api values and operations together.
#define SHDR_GL_PROC(ret, n, args) /* Declare a GL function pointer with its exact argument types. */ \
    typedef ret(SHDR_GL_CALL* n##_type) args; /* Name the representation used by this GL or shader interface. */ \
    n##_type n; /* Store the resolved function pointer for this GL entry point. */
    SHDR_GL_PROC(const unsigned char*, GetString, (GLenum)) // Declare a function to read the driver version or renderer text.
    SHDR_GL_PROC(void, GetIntegerv, (GLenum, GLint*)) // Declare a function to read a scalar device limit or current binding.
    SHDR_GL_PROC(void, GetIntegeri_v, (GLenum, GLuint, GLint*)) // Declare a function to read the limit for a selected compute axis.
    SHDR_GL_PROC(GLenum, GetError, ()) // Declare a function to read the pending GL error code.
    SHDR_GL_PROC(GLuint, CreateShader, (GLenum)) // Declare a function to allocate a compute shader object.
    SHDR_GL_PROC(void, ShaderSource, (GLuint, GLsizei, const char* const*, const GLint*)) // Declare a function to supply the embedded shader text to the driver.
    SHDR_GL_PROC(void, CompileShader, (GLuint)) // Declare a function to ask the driver to compile the compute shader.
    SHDR_GL_PROC(void, GetShaderiv, (GLuint, GLenum, GLint*)) // Declare a function to read shader compilation status or log length.
    SHDR_GL_PROC(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei*, char*)) // Declare a function to read the driver explanation of a shader compilation failure.
    SHDR_GL_PROC(void, DeleteShader, (GLuint)) // Declare a function to release the temporary shader object.
    SHDR_GL_PROC(GLuint, CreateProgram, ()) // Declare a function to allocate the linked compute program.
    SHDR_GL_PROC(void, AttachShader, (GLuint, GLuint)) // Declare a function to attach the compute shader before linking.
    SHDR_GL_PROC(void, LinkProgram, (GLuint)) // Declare a function to link the compiled compute program.
    SHDR_GL_PROC(void, GetProgramiv, (GLuint, GLenum, GLint*)) // Declare a function to read linked-program status, log length, or local dimensions.
    SHDR_GL_PROC(void, GetProgramInfoLog, (GLuint, GLsizei, GLsizei*, char*)) // Declare a function to read the driver explanation of a program-link failure.
    SHDR_GL_PROC(void, DeleteProgram, (GLuint)) // Declare a function to release the GL program after its owners finish using it.
    SHDR_GL_PROC(void, UseProgram, (GLuint)) // Declare a function to select the compute program for the next dispatch.
    SHDR_GL_PROC(GLint, GetUniformLocation, (GLuint, const char*)) // Declare a function to find a named uniform in the linked program.
    SHDR_GL_PROC(void, Uniform1f, (GLint, float)) // Declare a function to assign a float uniform.
    SHDR_GL_PROC(void, Uniform1i, (GLint, GLint)) // Declare a function to assign an integer or boolean uniform.
    SHDR_GL_PROC(void, Uniform1ui, (GLint, GLuint)) // Declare a function to assign an unsigned integer uniform.
#define SHDR_GL_UNIFORM(n) /* Define the SHDR_GL_UNIFORM helper for this group of declarations. */ \
    SHDR_GL_PROC(void, Uniform##n##fv, (GLint, GLsizei, const float*)) /* Declare a function to send this typed uniform to GL. */ \
    SHDR_GL_PROC(void, Uniform##n##iv, (GLint, GLsizei, const GLint*)) /* Declare a function to send this typed uniform to GL. */ \
    SHDR_GL_PROC(void, Uniform##n##uiv, (GLint, GLsizei, const GLuint*)) /* Declare a function to send this typed uniform to GL. */
    SHDR_GL_UNIFORM(2) // Declare the GL functions for vectors with this component count.
    SHDR_GL_UNIFORM(3) // Declare the GL functions for vectors with this component count.
    SHDR_GL_UNIFORM(4) // Declare the GL functions for vectors with this component count.
#undef SHDR_GL_UNIFORM // Remove this temporary macro after its declarations are complete.
#define SHDR_GL_MATRIX(n) /* Define the SHDR_GL_MATRIX helper for this group of declarations. */ \
    SHDR_GL_PROC(void, UniformMatrix##n##fv, (GLint, GLsizei, GLboolean, const float*)) /* Declare a function to send this typed uniform to GL. */
    SHDR_GL_MATRIX(2) // Declare the GL function for this column-major matrix shape.
    SHDR_GL_MATRIX(3) // Declare the GL function for this column-major matrix shape.
    SHDR_GL_MATRIX(4) // Declare the GL function for this column-major matrix shape.
    SHDR_GL_MATRIX(2x3) // Declare the GL function for this column-major matrix shape.
    SHDR_GL_MATRIX(2x4) // Declare the GL function for this column-major matrix shape.
    SHDR_GL_MATRIX(3x2) // Declare the GL function for this column-major matrix shape.
    SHDR_GL_MATRIX(3x4) // Declare the GL function for this column-major matrix shape.
    SHDR_GL_MATRIX(4x2) // Declare the GL function for this column-major matrix shape.
    SHDR_GL_MATRIX(4x3) // Declare the GL function for this column-major matrix shape.
#undef SHDR_GL_MATRIX // Remove this temporary macro after its declarations are complete.
    SHDR_GL_PROC(void, GenBuffers, (GLsizei, GLuint*)) // Declare a function to allocate names for GL storage buffers.
    SHDR_GL_PROC(void, DeleteBuffers, (GLsizei, const GLuint*)) // Declare a function to release owned GL buffer names.
    SHDR_GL_PROC(void, BindBuffer, (GLenum, GLuint)) // Declare a function to select a buffer for allocation or transfer.
    SHDR_GL_PROC(void, BufferData, (GLenum, GLsizeiptr, const void*, GLenum)) // Declare a function to allocate buffer storage and supply its initial host values.
    SHDR_GL_PROC(void, BufferSubData, (GLenum, GLintptr, GLsizeiptr, const void*)) // Declare a function to upload the current host buffer values.
    SHDR_GL_PROC(void, BindBufferBase, (GLenum, GLuint, GLuint)) // Declare a function to bind storage to the shader interface index.
    SHDR_GL_PROC(void*, MapBufferRange, (GLenum, GLintptr, GLsizeiptr, GLbitfield)) // Declare a function to map completed GPU buffer values for a host readback.
    SHDR_GL_PROC(GLboolean, UnmapBuffer, (GLenum)) // Declare a function to finish reading the mapped GPU buffer.
    SHDR_GL_PROC(void, DispatchCompute, (GLuint, GLuint, GLuint)) // Declare a function to queue the shader workgroups.
    SHDR_GL_PROC(void, MemoryBarrier, (GLbitfield)) // Declare a function to make earlier resource writes visible to following GPU commands.
    SHDR_GL_PROC(void, Finish, ()) // Declare a function to explicitly wait until queued GPU commands complete.
    SHDR_GL_PROC(GLsync, FenceSync, (GLenum, GLbitfield)) // Declare a function to insert a completion fence after queued compute work.
    SHDR_GL_PROC(GLenum, ClientWaitSync, (GLsync, GLbitfield, GLuint64)) // Declare a function to check a GPU fence with the caller-selected timeout.
    SHDR_GL_PROC(void, DeleteSync, (GLsync)) // Declare the GL completion-fence deletion entry point.
    SHDR_GL_PROC(void, Flush, ()) // Declare the GL command-flush entry point.
    SHDR_GL_PROC(void, GenTextures, (GLsizei, GLuint*)) // Declare a function to allocate names for owned GL images.
    SHDR_GL_PROC(void, DeleteTextures, (GLsizei, const GLuint*)) // Declare a function to release owned GL image names.
    SHDR_GL_PROC(void, BindTexture, (GLenum, GLuint)) // Declare a function to select the image for allocation or upload.
    SHDR_GL_PROC(void, TexStorage2D, (GLenum, GLsizei, GLenum, GLsizei, GLsizei)) // Declare a function to allocate immutable two-dimensional image storage.
    SHDR_GL_PROC(void, // Declare the next dynamically resolved GL function signature.
                 TexSubImage2D, // Name the function that uploads pixels into an existing texture.
                 (GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*)) // Match the argument order required by this GL entry point.
    SHDR_GL_PROC(void, BindImageTexture, (GLuint, GLuint, GLint, GLboolean, GLint, GLenum, GLenum)) // Declare a function to bind a typed image to its shader interface index.
    SHDR_GL_PROC(void, PixelStorei, (GLenum, GLint)) // Declare a function to set or restore the application pixel-transfer layout.
#undef SHDR_GL_PROC // Remove this temporary macro after its declarations are complete.
    explicit gl_api(const std::function<void*(const char*)>& resolver) { // Resolve the GL calls used by resource transfers and dispatches.
#define SHDR_LOAD(n) /* Resolve one GL entry point and reject missing functions. */ \
    { /* Begin the protected scope for this operation. */ \
        void* p = resolver("gl" #n); /* Resolve the address of this named GL entry point. */ \
        if (!p) /* Reject a missing dynamically resolved GL function. */ \
            throw error("missing GL entry point: gl" #n); /* Report missing gl entry point: gl. */ \
        static_assert(sizeof(p) == sizeof(n), "function pointer size unsupported"); /* Require a representation that matches the shader interface. */ \
        std::memcpy(&n, &p, sizeof(n)); /* Copy the representation with the alias-safe byte-copy operation. */ \
    } /* End this scope in gl api. */
        SHDR_LOAD(GetString) // Load the function to read the driver version or renderer text.
        SHDR_LOAD(GetIntegerv) // Load the function to read a scalar device limit or current binding.
        SHDR_LOAD(GetIntegeri_v) // Load the function to read the limit for a selected compute axis.
        SHDR_LOAD(GetError) // Load the function to read the pending GL error code.
        SHDR_LOAD(CreateShader) // Load the function to allocate a compute shader object.
        SHDR_LOAD(ShaderSource) // Load the function to supply the embedded shader text to the driver.
        SHDR_LOAD(CompileShader) // Load the function to ask the driver to compile the compute shader.
        SHDR_LOAD(GetShaderiv) // Load the function to read shader compilation status or log length.
        SHDR_LOAD(GetShaderInfoLog) // Load the function to read the driver explanation of a shader compilation failure.
        SHDR_LOAD(DeleteShader) // Load the function to release the temporary shader object.
        SHDR_LOAD(CreateProgram) // Load the function to allocate the linked compute program.
        SHDR_LOAD(AttachShader) // Load the function to attach the compute shader before linking.
        SHDR_LOAD(LinkProgram) // Load the function to link the compiled compute program.
        SHDR_LOAD(GetProgramiv) // Load the function to read linked-program status, log length, or local dimensions.
        SHDR_LOAD(GetProgramInfoLog) // Load the function to read the driver explanation of a program-link failure.
        SHDR_LOAD(DeleteProgram) // Load the function to release the GL program after its owners finish using it.
        SHDR_LOAD(UseProgram) // Load the function to select the compute program for the next dispatch.
        SHDR_LOAD(GetUniformLocation) // Load the function to find a named uniform in the linked program.
        SHDR_LOAD(Uniform1f) // Load the function to assign a float uniform.
        SHDR_LOAD(Uniform1i) // Load the function to assign an integer or boolean uniform.
        SHDR_LOAD(Uniform1ui) // Load the function to assign an unsigned integer uniform.
#define SHDR_LOAD_UNIFORM(n) /* Load the float, signed and unsigned uniform functions for this vector width. */ \
    SHDR_LOAD(Uniform##n##fv) /* Load the function to send this typed uniform to GL. */ \
    SHDR_LOAD(Uniform##n##iv) /* Load the function to send this typed uniform to GL. */ \
    SHDR_LOAD(Uniform##n##uiv) /* Load the function to send this typed uniform to GL. */
        SHDR_LOAD_UNIFORM(2) // Load the float, signed and unsigned uniform functions for this vector width.
        SHDR_LOAD_UNIFORM(3) // Load the float, signed and unsigned uniform functions for this vector width.
        SHDR_LOAD_UNIFORM(4) // Load the float, signed and unsigned uniform functions for this vector width.
#undef SHDR_LOAD_UNIFORM // Remove this temporary macro after its declarations are complete.
        SHDR_LOAD(UniformMatrix2fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(UniformMatrix3fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(UniformMatrix4fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(UniformMatrix2x3fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(UniformMatrix2x4fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(UniformMatrix3x2fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(UniformMatrix3x4fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(UniformMatrix4x2fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(UniformMatrix4x3fv) // Load the function to send this typed uniform to GL.
        SHDR_LOAD(GenBuffers) // Load the function to allocate names for GL storage buffers.
        SHDR_LOAD(DeleteBuffers) // Load the function to release owned GL buffer names.
        SHDR_LOAD(BindBuffer) // Load the function to select a buffer for allocation or transfer.
        SHDR_LOAD(BufferData) // Load the function to allocate buffer storage and supply its initial host values.
        SHDR_LOAD(BufferSubData) // Load the function to upload the current host buffer values.
        SHDR_LOAD(BindBufferBase) // Load the function to bind storage to the shader interface index.
        SHDR_LOAD(MapBufferRange) // Load the function to map completed GPU buffer values for a host readback.
        SHDR_LOAD(UnmapBuffer) // Load the function to finish reading the mapped GPU buffer.
        SHDR_LOAD(DispatchCompute) // Load the function to queue the shader workgroups.
        SHDR_LOAD(MemoryBarrier) // Load the function to make earlier resource writes visible to following GPU commands.
        SHDR_LOAD(Finish) // Load the function to explicitly wait until queued GPU commands complete.
        SHDR_LOAD(GenTextures) // Load the function to allocate names for owned GL images.
        SHDR_LOAD(DeleteTextures) // Load the function to release owned GL image names.
        SHDR_LOAD(FenceSync) // Load the function to insert a completion fence after queued compute work.
        SHDR_LOAD(ClientWaitSync) // Load the function to check a GPU fence with the caller-selected timeout.
        SHDR_LOAD(DeleteSync) // Resolve the GL completion-fence deletion entry point.
        SHDR_LOAD(Flush) // Resolve the GL command-flush entry point.
        SHDR_LOAD(BindTexture) // Load the function to select the image for allocation or upload.
        SHDR_LOAD(TexStorage2D) // Load the function to allocate immutable two-dimensional image storage.
        SHDR_LOAD(TexSubImage2D) // Load the function to upload the current host pixel values.
        SHDR_LOAD(BindImageTexture) // Load the function to bind a typed image to its shader interface index.
        SHDR_LOAD(PixelStorei) // Load the function to set or restore the application pixel-transfer layout.
#undef SHDR_LOAD // Remove this temporary macro after its declarations are complete.
    } // End this scope in gl api.
    void check(const std::string& operation) const { // Check for a driver error from the preceding GL operation.
        const unsigned code = GetError(); // Read the pending GL error code.
        if (code) { // Choose this branch when code.
            std::ostringstream s; // Build a readable diagnostic containing the operation and GL error.
            s << operation << ": GL error 0x" << std::hex << code; // Combine the operation name and hexadecimal driver error.
            throw error(s.str()); // Report the invalid operation.
        } // End this scope in runtime.
    } // End this scope in runtime.
}; // End this scope in runtime.
#undef SHDR_GL_CALL // Remove this temporary macro after its declarations are complete.

struct device_state { // Retain GL functions, context ownership, limits, and pending cleanup.
    gl_api gl; // Keep the retained GL function table.
    profile language; // Keep the selected shader profile.
    unsigned max_groups[3], max_local[3], max_invocations, max_buffers, max_images; // Keep work group dimensions, invocation and resource binding limits.
    std::shared_ptr<void> context_keepalive; // Keep the retained headless context.
    std::function<void()> activate_context; // Keep the callback that makes this device context current.
    std::thread::id context_thread; // Keep the thread allowed to issue GL commands.
    std::mutex retirement_mutex; // Keep the lock for deferred resource cleanup.
    std::vector<std::function<void(gl_api&)>> retirements; // Keep GL deletions requested by another thread.
    struct deferred { // Keep the deferred values and operations together.
        std::function<bool(gl_api&)> poll; // Keep the zero-time completion callback.
        std::function<void(gl_api&)> cancel; // Keep the fence-deletion callback used at shutdown.
    }; // End this scope in runtime.
    std::vector<deferred> pending_work; // Keep discarded submissions awaiting completion.
    bool retirements_closed; // Keep whether device teardown has stopped retaining submissions.
    device_state(const std::function<void*(const char*)>& resolver, profile p) // Load GL entry points and validate the selected compute profile.
        : gl(resolver), language(p), context_thread(std::this_thread::get_id()), // Resolve GL functions and record this context's profile and owning thread.
          retirements_closed(false) { // Allow pending completion leases until the public device is destroyed.
        const unsigned char* version = gl.GetString(VERSION); // Read the current context version before checking the requested profile.
        if (!version) // Require a version string from the current context.
            throw error("device requires a current GL context"); // Report device requires a current gl context.
        const bool es = std::string(reinterpret_cast<const char*>(version)).find("OpenGL ES") != // Set whether the current context uses OpenGL ES.
                        std::string::npos; // Detect the GLES marker anywhere in the driver version string.
        const bool requested_es = p == profile::es31 || p == profile::es32; // Set whether the requested profile uses GLES.
        if (es != requested_es) // Choose this branch when es differs from requested es.
            throw error("shader profile does not match current GL/ES context"); // Report a mismatch between the selected profile and the current context.
        GLint major = 0, minor = 0; // Prepare both parts of the current GL version query.
        gl.GetIntegerv(MAJOR_VERSION, &major); // Read a scalar device limit or current binding.
        gl.GetIntegerv(MINOR_VERSION, &minor); // Read a scalar device limit or current binding.
        const int required_major = es ? 3 : 4; // Set the minimum GL major version for this profile.
        const int required_minor = p == profile::es31   ? 1 // Set the minimum GL minor version for this profile.
                                   : p == profile::es32 ? 2 // Select the version associated with this shader profile.
                                   : p == profile::gl43 ? 3 // Select the version associated with this shader profile.
                                   : p == profile::gl45 ? 5 // Select the version associated with this shader profile.
                                                        : 6; // Select minor version six for the desktop GL 4.6 profile.
        if (major < required_major || (major == required_major && minor < required_minor)) // Choose this branch when major < required major or (major equals required major and minor < required minor).
            throw error("current context is older than requested compute profile"); // Report current context is older than requested compute profile.
        for (unsigned i = 0; i < 3; ++i) { // Validate or query each of the three compute dimensions.
            GLint n = 0; // Set the queried GL limit or element count.
            gl.GetIntegeri_v(MAX_COMPUTE_WORK_GROUP_COUNT, i, &n); // Read the limit for a selected compute axis.
            max_groups[i] = static_cast<unsigned>(n); // Set the device workgroup-count limits.
            gl.GetIntegeri_v(MAX_COMPUTE_WORK_GROUP_SIZE, i, &n); // Read the limit for a selected compute axis.
            max_local[i] = static_cast<unsigned>(n); // Set the device local-size limits.
        } // End this scope in device state.
        GLint n = 0; // Set the queried GL limit or element count.
        gl.GetIntegerv(MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &n); // Read a scalar device limit or current binding.
        max_invocations = static_cast<unsigned>(n); // Set the maximum invocations in one workgroup.
        gl.GetIntegerv(MAX_SHADER_STORAGE_BUFFER_BINDINGS, &n); // Read a scalar device limit or current binding.
        max_buffers = static_cast<unsigned>(n); // Set the storage-buffer binding limit.
        gl.GetIntegerv(MAX_IMAGE_UNITS, &n); // Read a scalar device limit or current binding.
        max_images = static_cast<unsigned>(n); // Set the image binding limit.
        gl.check("query compute capabilities"); // Check for a driver error from the preceding GL operation.
    } // End this scope in device state.
    void require_thread() { // Require the device context thread and activate its headless context.
        if (std::this_thread::get_id() != context_thread) // Check that the operation uses the context or lifecycle thread.
            throw error("GL operations require the device's context thread"); // Report gl operations require the device's context thread.
        if (activate_context) // Choose this branch when activate context.
            activate_context(); // Keep the callback that makes this device context current.
    } // End this scope in require thread.
    void collect() { // Run queued deletions and zero-time completion checks.
        require_thread(); // Require the device context thread and activate its headless context.
        std::vector<std::function<void(gl_api&)>> pending; // Keep the queued work or cleanup callbacks awaiting processing.
        std::vector<deferred> polls, unfinished; // Keep completion polls that must be checked again.
        { // Begin the protected scope for this operation.
            std::lock_guard<std::mutex> lock(retirement_mutex); // Hold the mutex while this shared state is accessed.
            pending.swap(retirements); // Take the shared entries into a local collection for unlocked processing.
            polls.swap(pending_work); // Take the shared entries into a local collection for unlocked processing.
        } // End this scope in collect.
        for (std::size_t i = 0; i < pending.size(); ++i) // Process the cleanup or completion callbacks taken from the queue.
            pending[i](gl); // Keep the retained GL function table.
        for (std::size_t i = 0; i < polls.size(); ++i) // Process the cleanup or completion callbacks taken from the queue.
            if (!polls[i].poll(gl)) // Retain a completion callback while its work is pending.
                unfinished.push_back(std::move(polls[i])); // Retain this entry in the ordered resource or work collection.
        if (!unfinished.empty()) { // Keep the remaining callbacks for the next collection pass.
            std::lock_guard<std::mutex> lock(retirement_mutex); // Hold the mutex while this shared state is accessed.
            pending_work.insert(pending_work.end(), // Retain this entry in the ordered resource or work collection.
                                std::make_move_iterator(unfinished.begin()), // Supply the next value for the current call or declaration.
                                std::make_move_iterator(unfinished.end())); // Keep the end of the current range.
        } // End this scope in collect.
    } // End this scope in collect.
    void defer_poll(const std::function<bool(gl_api&)>& poll, // Retain a completion check until its fence signals.
                    const std::function<void(gl_api&)>& cancel = std::function<void(gl_api&)>()) { // Set the fence-deletion callback used at shutdown.
        { // Begin the protected scope for this operation.
            std::lock_guard<std::mutex> lock(retirement_mutex); // Hold the mutex while this shared state is accessed.
            if (!retirements_closed) { // Check whether teardown still allows retained completion callbacks.
                pending_work.push_back(deferred{poll, cancel}); // Retain this entry in the ordered resource or work collection.
                return; // Finish this operation.
            } // End this scope in defer poll.
        } // End this scope in defer poll.
 // Handles may outlive the public device wrapper. After wrapper teardown,
 // schedule their fence deletion directly on the context thread.
        if (cancel) // Choose this branch when cancel.
            retire(cancel); // Delete on the context thread or defer deletion to that thread.
    } // End this scope in defer poll.
    void close_retirements() { // Stop retaining new submission bundles during device teardown.
        { // Begin the protected scope for this operation.
            std::lock_guard<std::mutex> lock(retirement_mutex); // Hold the mutex while this shared state is accessed.
            retirements_closed = true; // Set whether device teardown has stopped retaining submissions.
        } // End this scope in close retirements.
        cancel_pending(); // Schedule deferred-fence deletion on the context thread.
    } // End this scope in close retirements.
    void cancel_pending() { // Schedule deferred-fence deletion on the context thread.
        require_thread(); // Require the device context thread and activate its headless context.
        std::vector<deferred> pending; // Keep the queued work or cleanup callbacks awaiting processing.
        { // Begin the protected scope for this operation.
            std::lock_guard<std::mutex> lock(retirement_mutex); // Hold the mutex while this shared state is accessed.
            pending.swap(pending_work); // Take the shared entries into a local collection for unlocked processing.
        } // End this scope in cancel pending.
        for (std::size_t i = 0; i < pending.size(); ++i) // Process the cleanup or completion callbacks taken from the queue.
            if (pending[i].cancel) // Check the queue state before accepting or executing work.
                pending[i].cancel(gl); // Keep the retained GL function table.
        pending.clear(); // Release the retained entries after their work or cleanup is complete.
        collect(); // Run queued deletions and zero-time completion checks.
    } // End this scope in cancel pending.
    void retire(const std::function<void(gl_api&)>& deletion) { // Delete on the context thread or defer deletion to that thread.
        if (std::this_thread::get_id() == context_thread) { // Check that the operation uses the context or lifecycle thread.
            require_thread(); // Require the device context thread and activate its headless context.
            deletion(gl); // Keep the retained GL function table.
        } else { // Handle the alternative resource or completion case.
            std::lock_guard<std::mutex> lock(retirement_mutex); // Hold the mutex while this shared state is accessed.
            retirements.push_back(deletion); // Retain this entry in the ordered resource or work collection.
        } // End this scope in retire.
    } // End this scope in retire.
    ~device_state() noexcept { // Close pending retirement leases on the context thread.
        if (std::this_thread::get_id() == context_thread) { // Check that the operation uses the context or lifecycle thread.
            try { // Keep cleanup available if an allocation, shader, or driver operation fails.
                close_retirements(); // Stop retaining new submission bundles during device teardown.
            } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
            } // End this scope in device state.
        } // End this scope in device state.
    } // End this scope in device state.
    GLuint compile(const std::string& source, const std::string& name) { // Compile and link the embedded compute source with driver diagnostics.
        if (source.size() > static_cast<std::size_t>((std::numeric_limits<GLint>::max)())) // Check resource size or driver limits before accessing storage.
            throw error("shader source too large"); // Report shader source too large.
        GLuint shader = gl.CreateShader(COMPUTE_SHADER), program = 0; // Allocate a compute shader object.
        if (!shader) // Check that the driver allocated a shader object.
            throw error("cannot create compute shader"); // Report compute-shader allocation failure.
        try { // Keep cleanup available if an allocation, shader, or driver operation fails.
            const char* text = source.c_str(); // Give GL a pointer to the retained shader characters.
            GLint len = static_cast<GLint>(source.size()); // Pass the validated shader character count to GL.
            gl.ShaderSource(shader, 1, &text, &len); // Supply the embedded shader text to the driver.
            gl.CompileShader(shader); // Ask the driver to compile the compute shader.
            GLint success = 0; // Prepare the compile or link status returned by GL.
            gl.GetShaderiv(shader, COMPILE_STATUS, &success); // Read shader compilation status or log length.
            if (!success) { // Read the driver diagnostic for this failed operation.
                GLint size = 0; // Prepare the driver diagnostic log length.
                gl.GetShaderiv(shader, INFO_LOG_LENGTH, &size); // Read shader compilation status or log length.
                std::vector<char> log(static_cast<std::size_t>((std::max)(size, 1)), 0); // Allocate enough characters for the driver diagnostic and terminator.
                gl.GetShaderInfoLog(shader, size, 0, log.data()); // Read the driver explanation of a shader compilation failure.
                throw error(name + ": shader compilation failed\n" + log.data()); // Report : shader compilation failed\n.
            } // End this scope in compile.
            program = gl.CreateProgram(); // Allocate the linked compute program.
            if (!program) // Check that the driver allocated a program object.
                throw error("cannot create compute program"); // Report compute-program allocation failure.
            gl.AttachShader(program, shader); // Attach the compute shader before linking.
            gl.LinkProgram(program); // Link the compiled compute program.
            gl.GetProgramiv(program, LINK_STATUS, &success); // Read linked-program status, log length, or local dimensions.
            if (!success) { // Read the driver diagnostic for this failed operation.
                GLint size = 0; // Prepare the driver diagnostic log length.
                gl.GetProgramiv(program, INFO_LOG_LENGTH, &size); // Read linked-program status, log length, or local dimensions.
                std::vector<char> log(static_cast<std::size_t>((std::max)(size, 1)), 0); // Allocate enough characters for the driver diagnostic and terminator.
                gl.GetProgramInfoLog(program, size, 0, log.data()); // Read the driver explanation of a program-link failure.
                throw error(name + ": program link failed\n" + log.data()); // Report : program link failed\n.
            } // End this scope in compile.
            gl.DeleteShader(shader); // Release the temporary shader object.
            shader = 0; // Set the shader used by this dispatch.
            gl.check("link " + name); // Keep the resource or entry-point name.
            return program; // Compile and link the embedded compute source with driver diagnostics.
        } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
            if (shader) // Choose this branch when shader.
                gl.DeleteShader(shader); // Release the temporary shader object.
            if (program) // Choose this branch when program.
                gl.DeleteProgram(program); // Release the GL program after its owners finish using it.
            throw; // Rethrow the original failure after arranging the necessary cleanup.
        } // End this scope in compile.
    } // End this scope in compile.
}; // End this scope in runtime.
struct program_state { // Keep the program state values and operations together.
    std::shared_ptr<device_state> owner; // Keep the device that owns the GL resource.
    GLuint id; // Keep the compiled GL program name.
    program_state(const std::shared_ptr<device_state>& d, GLuint id_) : owner(d), id(id_) { // Expose the retained compiled program for asynchronous ownership.
    } // End this scope in program state.
    ~program_state() { // Release this object's owned resources at teardown.
        if (id) { // Choose this branch when id.
            const GLuint name = id; // Set the resource or entry-point name.
            owner->retire([name](gl_api& gl) { // Delete on the context thread or defer deletion to that thread.
                gl.DeleteProgram(name); // Release the GL program after its owners finish using it.
            }); // End this scope in program state.
        } // End this scope in program state.
    } // End this scope in program state.
}; // End this scope in runtime.

inline std::size_t checked_product(std::size_t a, std::size_t b) { // Reject resource-size multiplication that would overflow.
    if (b && a > (std::numeric_limits<std::size_t>::max)() / b) // Check resource size or driver limits before accessing storage.
        throw error("resource size overflow"); // Report resource size overflow.
    return a * b; // Reject resource-size multiplication that would overflow.
} // End this scope in checked product.
inline GLsizeiptr gl_size(std::size_t size) { // Require a byte count representable by the GL buffer API.
    if (size > static_cast<std::size_t>((std::numeric_limits<GLsizeiptr>::max)())) // Check resource size or driver limits before accessing storage.
        throw error("resource exceeds GLsizeiptr"); // Report resource exceeds glsizeiptr.
    return static_cast<GLsizeiptr>(size); // Require a byte count representable by the GL buffer API.
} // End this scope in gl size.
enum class authority { host, gpu, both }; // Name the profile, access, or completion values accepted by this section.
enum class resource_kind { buffer, image }; // Name the profile, access, or completion values accepted by this section.
struct resource_state { // Track shared storage and which backend has the newest values.
    resource_kind kind; // Keep the buffer, image, or uniform interface kind.
    authority latest; // Keep which copy contains the latest resource values.
    std::shared_ptr<device_state> owner; // Keep the device that owns the GL resource.
    GLuint handle; // Keep the GL buffer or texture name.
    bool borrowed; // Keep whether the application owns the supplied GL object.
    unsigned width, height, components, format, scalar; // Keep the GL scalar representation.
    resource_state(resource_kind k) // Initialize ownership and synchronization for one resource.
        : kind(k), latest(authority::host), handle(0), borrowed(false), width(0), height(0), // Start with authoritative host data and allocate GL storage on demand.
          components(0), format(0), scalar(0) { // Initialize image layout metadata to zero.
    } // End this scope in resource state.
    virtual ~resource_state() { // Release this object's owned resources at teardown.
        if (handle && !borrowed) { // Delete the resource when its GL handle is owned by this object.
            const GLuint name = handle; // Set the resource or entry-point name.
            if (kind == resource_kind::buffer) // Choose the branch matching this interface or resource access.
                owner->retire([name](gl_api& gl) { // Delete on the context thread or defer deletion to that thread.
                    gl.DeleteBuffers(1, &name); // Release owned GL buffer names.
                }); // End this scope in resource state.
            else // Handle the other resource, platform, or completion case.
                owner->retire([name](gl_api& gl) { // Delete on the context thread or defer deletion to that thread.
                    gl.DeleteTextures(1, &name); // Release owned GL image names.
                }); // End this scope in resource state.
        } // End this scope in resource state.
    } // End this scope in resource state.
    virtual void* host_pointer() = 0; // Require each resource type to expose its host storage.
    virtual std::size_t bytes() const = 0; // Require each resource type to report its byte count.
    void ensure_host(); // Download newer GPU values only when host access needs them.
    void ensure_gpu(const std::shared_ptr<device_state>& d) { // Allocate or upload GPU storage when its host copy is newer.
        d->collect(); // Run queued deletions and zero-time completion checks.
        if (owner && owner != d) // Check that this storage belongs to the requested device.
            throw error("resource belongs to a different device"); // Report resource belongs to a different device.
        owner = d; // Set the device that owns the GL resource.
        const gl_api& gl = d->gl; // Set the retained GL function table.
        if (!handle) { // Allocate GPU storage on the first device access.
            if (kind == resource_kind::buffer) { // Choose the branch matching this interface or resource access.
                gl.GenBuffers(1, &handle); // Allocate names for GL storage buffers.
                gl.BindBuffer(SHADER_STORAGE_BUFFER, handle); // Select a buffer for allocation or transfer.
                gl.BufferData(SHADER_STORAGE_BUFFER, // Allocate buffer storage and supply its initial host values.
                              gl_size(bytes()), // Require a byte count representable by the GL buffer API.
                              bytes() ? host_pointer() : 0, // Supply the next value for the current call or declaration.
                              DYNAMIC_DRAW); // Require a byte count representable by the GL buffer API.
            } else { // Handle the alternative resource or completion case.
                gl.GenTextures(1, &handle); // Allocate names for owned GL images.
                gl.BindTexture(TEXTURE_2D, handle); // Select the image for allocation or upload.
                gl.TexStorage2D(TEXTURE_2D, // Allocate immutable two-dimensional image storage.
                                1, // Supply the next value for the current call or declaration.
                                format, // Visit the values or interfaces needed by ensure gpu.
                                static_cast<GLsizei>(width), // Keep the image width in pixels.
                                static_cast<GLsizei>(height)); // Keep the image height in pixels.
            } // End this scope in ensure gpu.
            gl.check("allocate compute resource"); // Check for a driver error from the preceding GL operation.
            if (kind == resource_kind::buffer) // Choose the branch matching this interface or resource access.
                latest = authority::both; // Record which backend now owns the newest resource values.
        } // End this scope in ensure gpu.
        if (latest == authority::host) { // Check which resource copy must be synchronized.
            if (kind == resource_kind::buffer) { // Choose the branch matching this interface or resource access.
                gl.BindBuffer(SHADER_STORAGE_BUFFER, handle); // Select a buffer for allocation or transfer.
                if (bytes()) // Check resource size or driver limits before accessing storage.
                    gl.BufferSubData(SHADER_STORAGE_BUFFER, 0, gl_size(bytes()), host_pointer()); // Upload the current host buffer values.
            } else { // Handle the alternative resource or completion case.
 // Make upload independent of caller pixel unpack state and restore that state.
                GLint alignment = 0, row = 0, skip_rows = 0, skip_pixels = 0, pbo = 0; // Save the complete pixel-unpack layout and bound transfer buffer.
                gl.GetIntegerv(UNPACK_ALIGNMENT, &alignment); // Read a scalar device limit or current binding.
                gl.GetIntegerv(UNPACK_ROW_LENGTH, &row); // Read a scalar device limit or current binding.
                gl.GetIntegerv(UNPACK_SKIP_ROWS, &skip_rows); // Read a scalar device limit or current binding.
                gl.GetIntegerv(UNPACK_SKIP_PIXELS, &skip_pixels); // Read a scalar device limit or current binding.
                gl.GetIntegerv(0x88EF, &pbo); // GL_PIXEL_UNPACK_BUFFER_BINDING
                gl.BindBuffer(PIXEL_UNPACK_BUFFER, 0); // Select a buffer for allocation or transfer.
                gl.PixelStorei(UNPACK_ALIGNMENT, 1); // Set or restore the application pixel-transfer layout.
                gl.PixelStorei(UNPACK_ROW_LENGTH, 0); // Set or restore the application pixel-transfer layout.
                gl.PixelStorei(UNPACK_SKIP_ROWS, 0); // Set or restore the application pixel-transfer layout.
                gl.PixelStorei(UNPACK_SKIP_PIXELS, 0); // Set or restore the application pixel-transfer layout.
                gl.BindTexture(TEXTURE_2D, handle); // Select the image for allocation or upload.
                const GLenum channels = scalar == FLOAT // Choose the float or integer pixel-transfer channel representation.
                                            ? (components == 1 ? RED : RGBA) // Set the number of scalar values per pixel.
                                            : (components == 1 ? RED_INTEGER : RGBA_INTEGER); // Choose the upload pixel format matching integer channels.
                gl.TexSubImage2D(TEXTURE_2D, // Upload the current host pixel values.
                                 0, // Supply the next value for the current call or declaration.
                                 0, // Supply the next value for the current call or declaration.
                                 0, // Supply the next value for the current call or declaration.
                                 static_cast<GLsizei>(width), // Keep the image width in pixels.
                                 static_cast<GLsizei>(height), // Keep the image height in pixels.
                                 channels, // Supply the next value for the current call or declaration.
                                 scalar, // Keep the GL scalar representation.
                                 host_pointer()); // Expose this resource's borrowed or owned host storage.
                gl.PixelStorei(UNPACK_ALIGNMENT, alignment); // Set or restore the application pixel-transfer layout.
                gl.PixelStorei(UNPACK_ROW_LENGTH, row); // Set or restore the application pixel-transfer layout.
                gl.PixelStorei(UNPACK_SKIP_ROWS, skip_rows); // Set or restore the application pixel-transfer layout.
                gl.PixelStorei(UNPACK_SKIP_PIXELS, skip_pixels); // Set or restore the application pixel-transfer layout.
                gl.BindBuffer(PIXEL_UNPACK_BUFFER, static_cast<GLuint>(pbo)); // Select a buffer for allocation or transfer.
            } // End this scope in ensure gpu.
            gl.check("upload compute resource"); // Check for a driver error from the preceding GL operation.
            latest = authority::both; // Record which backend now owns the newest resource values.
        } // End this scope in ensure gpu.
    } // End this scope in ensure gpu.
}; // End this scope in runtime.

 // Image readback uses a compute copy on both desktop GL and GLES.
inline void resource_state::ensure_host() { // Download newer GPU values only when host access needs them.
    if (latest != authority::gpu) // Check which resource copy must be synchronized.
        return; // Finish this operation.
    owner->collect(); // Run queued deletions and zero-time completion checks.
    const gl_api& gl = owner->gl; // Set the retained GL function table.
    if (!bytes()) { // Check resource size or driver limits before accessing storage.
        latest = authority::both; // Record which backend now owns the newest resource values.
        return; // Finish this operation.
    } // End this scope in ensure host.
    GLuint staging = 0, program = 0; // Track the temporary readback buffer and copy program for cleanup.
    try { // Keep cleanup available if an allocation, shader, or driver operation fails.
        gl.MemoryBarrier(ALL_BARRIER_BITS); // Make earlier resource writes visible to following GPU commands.
        if (kind == resource_kind::buffer) // Choose the branch matching this interface or resource access.
            gl.BindBuffer(SHADER_STORAGE_BUFFER, handle); // Select a buffer for allocation or transfer.
        else { // Handle the other resource, platform, or completion case.
            const char* scalar_name = scalar == FLOAT ? "float" : scalar == INT ? "int" : "uint"; // Choose the GLSL scalar spelling matching these image pixels.
            const char* vector_name = scalar == FLOAT ? "vec4" : scalar == INT ? "ivec4" : "uvec4"; // Choose the four-component GLSL value used by imageLoad.
            const char* image_name = scalar == FLOAT ? "image2D" // Choose the matching float, signed or unsigned storage-image type.
                                     : scalar == INT ? "iimage2D" // Choose the shader image type matching the resource's scalar representation.
                                                     : "uimage2D"; // Choose the shader image type matching the resource's scalar representation.
            const char* format_name = components == 1 ? (scalar == FLOAT ? "r32f" // Choose the declared storage-image format for the copy shader.
                                                         : scalar == INT ? "r32i" // Choose the shader format matching scalar type and channel count.
                                                                         : "r32ui") // Choose the shader format matching scalar type and channel count.
                                                      : (scalar == FLOAT ? "rgba32f" // Choose the shader format matching scalar type and channel count.
                                                         : scalar == INT ? "rgba32i" // Choose the shader format matching scalar type and channel count.
                                                                         : "rgba32ui"); // Choose the shader format matching scalar type and channel count.
            std::ostringstream body; // Download newer GPU values only when host access needs them.
            body << "SHDR_BEGIN(shdr_readback,8,8,1)\nSHDR_IMAGE2D(0," << format_name << ',' // Download newer GPU values only when host access needs them.
                 << image_name << ",src,readonly)\n" // Download newer GPU values only when host access needs them.
                 << "SHDR_BUFFER(0," << scalar_name // Download newer GPU values only when host access needs them.
                 << ",dst,writeonly)\nSHDR_MAIN(){uvec2 p=gl_GlobalInvocationID.xy;" // Download newer GPU values only when host access needs them.
                 << "ivec2 s=imageSize(src);if(p.x>=uint(s.x)||p.y>=uint(s.y))return;" // Download newer GPU values only when host access needs them.
                 << vector_name << " v=imageLoad(src,ivec2(p));uint i=(p.y*uint(s.x)+p.x)*" // Download newer GPU values only when host access needs them.
                 << components << "u;"; // Download newer GPU values only when host access needs them.
            for (unsigned c = 0; c < components; ++c) // Visit the values or interfaces needed by ensure host.
                body << "dst[i+" << c << "u]=v[" << c << "];"; // Download newer GPU values only when host access needs them.
            body << "} SHDR_END()\n"; // Download newer GPU values only when host access needs them.
            program = owner->compile(glsl_source(body.str(), owner->language), "image readback"); // Compile the portable image-to-buffer readback shader.
            gl.GenBuffers(1, &staging); // Allocate names for GL storage buffers.
            gl.BindBuffer(SHADER_STORAGE_BUFFER, staging); // Select a buffer for allocation or transfer.
            gl.BufferData(SHADER_STORAGE_BUFFER, gl_size(bytes()), 0, DYNAMIC_DRAW); // Allocate buffer storage and supply its initial host values.
            gl.UseProgram(program); // Select the compute program for the next dispatch.
            gl.BindBufferBase(SHADER_STORAGE_BUFFER, 0, staging); // Bind storage to the shader interface index.
            gl.BindImageTexture(0, handle, 0, 0, 0, READ_ONLY, format); // Bind a typed image to its shader interface index.
            gl.DispatchCompute((width + 7) / 8, (height + 7) / 8, 1); // Queue the shader workgroups.
            gl.MemoryBarrier(ALL_BARRIER_BITS); // Make earlier resource writes visible to following GPU commands.
            gl.check("copy image to host staging buffer"); // Check for a driver error from the preceding GL operation.
            gl.BindBuffer(SHADER_STORAGE_BUFFER, staging); // Select a buffer for allocation or transfer.
        } // End this scope in ensure host.
        void* mapped = gl.MapBufferRange(SHADER_STORAGE_BUFFER, 0, gl_size(bytes()), MAP_READ_BIT); // Map completed GPU buffer values for a host readback.
        if (!mapped) { // Check that GL returned a readable mapped range.
            gl.check("map compute readback"); // Check for a driver error from the preceding GL operation.
            throw error("cannot map compute readback"); // Report readback mapping failure.
        } // End this scope in ensure host.
        std::memcpy(host_pointer(), mapped, bytes()); // Copy the representation with the alias-safe byte-copy operation.
        if (!gl.UnmapBuffer(SHADER_STORAGE_BUFFER)) // Finish reading the mapped GPU buffer.
            throw error("compute readback buffer contents corrupted"); // Report compute readback buffer contents corrupted.
        gl.check("read back compute resource"); // Check for a driver error from the preceding GL operation.
        latest = authority::both; // Record which backend now owns the newest resource values.
        if (staging) // Choose this branch when staging.
            gl.DeleteBuffers(1, &staging); // Release owned GL buffer names.
        if (program) // Choose this branch when program.
            gl.DeleteProgram(program); // Release the GL program after its owners finish using it.
    } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
        if (staging) // Choose this branch when staging.
            gl.DeleteBuffers(1, &staging); // Release owned GL buffer names.
        if (program) // Choose this branch when program.
            gl.DeleteProgram(program); // Release the GL program after its owners finish using it.
        throw; // Rethrow the original failure after arranging the necessary cleanup.
    } // End this scope in ensure host.
} // End this scope in ensure host.

 // Only layouts identical to std430 arrays are accepted. vec3, bool and arbitrary structs need explicit packing.
template <class T> // Select the scalar, dimension, or access type at compile time.
struct storage_type // Keep the storage type values and operations together.
    : std::integral_constant<bool, // Expose the following storage-layout checks as a compile-time boolean.
                             std::is_same<T, float>::value || std::is_same<T, int>::value || // Accept the supported 32-bit float and signed integer storage types.
                                 std::is_same<T, unsigned>::value> {}; // Also accept the supported 32-bit unsigned storage type.
template <class T, std::size_t N> // Select the scalar, dimension, or access type at compile time.
struct storage_type<glsl::vec<T, N>> // Keep the storage type values and operations together.
    : std::integral_constant<bool, // Expose the following storage-layout checks as a compile-time boolean.
                             storage_type<T>::value && (N == 2 || N == 4) && // Require a supported scalar and a two- or four-component vector.
                                 sizeof(glsl::vec<T, N>) == sizeof(T) * N> {}; // Require a tightly packed vector that matches its std430 array stride.
template <class T> struct buffer_state : resource_state { // Select the scalar, dimension, or access type at compile time.
    std::vector<T> owned; // Keep host storage allocated by this resource.
    T* external; // Point to application-owned host memory.
    std::size_t count; // Keep the element or configuration count.
    explicit buffer_state(std::size_t n) // Allocate owned host storage for this number of elements.
        : resource_state(resource_kind::buffer), owned(n), external(0), count(n) { // Initialize ownership and synchronization for one resource.
    } // End this scope in runtime.
    buffer_state(T* p, std::size_t n) // Retain a reference to an application-owned host range.
        : resource_state(resource_kind::buffer), external(p), count(n) { // Initialize ownership and synchronization for one resource.
        if (!p && n) // Require a valid pointer for a nonempty borrowed range.
            throw error("null borrowed CPU buffer"); // Report null borrowed cpu buffer.
    } // End this scope in runtime.
    void* host_pointer() override { // Expose this resource's borrowed or owned host storage.
        return external ? external : owned.data(); // Use borrowed host memory when present and owned memory otherwise.
    } // End this scope in runtime.
    std::size_t bytes() const override { // Report this resource's byte count with overflow checking.
        return checked_product(count, sizeof(T)); // Calculate the byte count with checked size arithmetic.
    } // End this scope in runtime.
}; // End this scope in runtime.
template <class T> struct image_state : resource_state { // Select the scalar, dimension, or access type at compile time.
    std::vector<T> owned; // Keep host storage allocated by this resource.
    image_state(unsigned w, unsigned h, unsigned c) : resource_state(resource_kind::image) { // Initialize ownership and synchronization for one resource.
        if (!w || !h || w > static_cast<unsigned>((std::numeric_limits<GLsizei>::max)()) || // Check resource size or driver limits before accessing storage.
            h > static_cast<unsigned>((std::numeric_limits<GLsizei>::max)())) // Require the image height to fit the GL signed dimension type.
            throw error("image dimensions must be positive GLsizei values"); // Report image dimensions must be positive glsizei values.
        width = w; // Set the image width in pixels.
        height = h; // Set the image height in pixels.
        components = c; // Set the number of scalar values per pixel.
        scalar = std::is_same<T, float>::value ? FLOAT // Set the GL scalar representation.
                 : std::is_same<T, int>::value ? INT // Choose the GL scalar type matching the host pixel representation.
                                               : UNSIGNED_INT; // Choose the GL scalar type matching the host pixel representation.
        format = c == 1 ? (scalar == FLOAT ? R32F // Visit the values or interfaces needed by runtime.
                           : scalar == INT ? R32I // Choose the GL internal image format matching scalar type and channel count.
                                           : R32UI) // Choose the GL internal image format matching scalar type and channel count.
                        : (scalar == FLOAT ? RGBA32F // Choose the GL internal image format matching scalar type and channel count.
                           : scalar == INT ? RGBA32I // Choose the GL internal image format matching scalar type and channel count.
                                           : RGBA32UI); // Choose the GL internal image format matching scalar type and channel count.
        owned.resize(checked_product(checked_product(w, h), c)); // Reject resource-size multiplication that would overflow.
    } // End this scope in runtime.
    void* host_pointer() override { // Expose this resource's borrowed or owned host storage.
        return owned.data(); // Expose the contiguous host storage owned by this resource.
    } // End this scope in runtime.
    std::size_t bytes() const override { // Report this resource's byte count with overflow checking.
        return checked_product(owned.size(), sizeof(T)); // Calculate the byte count with checked size arithmetic.
    } // End this scope in runtime.
}; // End this scope in runtime.
} // namespace detail

/** @brief An application's current GL context, accessed through its function resolver.
 * The context must remain current on this thread during operations and destruction of GL objects.
 * The default header uses standard C++11 declarations. The optional
 * headless_device supplies EGL context creation when SHDR_ENABLE_EGL is enabled.
 */
class device { // Keep the device values and operations together.
    std::shared_ptr<detail::device_state> state_; // Keep the state shared by this wrapper and its retained owners.

  public: // Expose the operations callers can use.
    /** @brief Load GL entry points and validate the current context against a profile.
     * @param resolver Callable returning the address of a named GL function, or null.
     * @param p Minimum GLSL and GL/ES profile. defaults to ES 3.1.
     * @throws error If function resolution or current-context validation fails.
     */
    explicit device(const std::function<void*(const char*)>& resolver, profile p = profile::es31) // Create a compute device using the caller's current context and function resolver.
        : state_(std::make_shared<detail::device_state>(resolver, p)) { // Retain the resolved GL functions, profile and device limits.
    } // End this scope in device.
    /** @brief Release deferred completion handles through context-thread cleanup.
     * Destroy devices on their context thread after CPU workers have stopped.
     */
    virtual ~device() noexcept { // Close pending retirement leases on the context thread.
        if (std::this_thread::get_id() == state_->context_thread) { // Check that the operation uses the context or lifecycle thread.
            try { // Keep cleanup available if an allocation, shader, or driver operation fails.
                state_->close_retirements(); // Stop retaining new submission bundles during device teardown.
            } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
            } // End this scope in device.
        } // End this scope in device.
    } // End this scope in device.
    device(const device&) = delete; // Prevent copying this object and duplicating its retained identity.
    device& operator=(const device&) = delete; // Prevent copying this object and duplicating its retained identity.
    /** @brief Return the shader language profile selected for this device. */
    profile shader_profile() const { // Return the device profile used to generate shader source.
        return state_->language; // Return the device profile used to generate shader source.
    } // End this scope in shader profile.
    /** @brief Wait for submitted GPU work to complete, including for accurate wall-clock benchmarks. */
    void synchronize() { // Explicitly wait for completed GPU commands.
        state_->collect(); // Run queued deletions and zero-time completion checks.
        state_->gl.Finish(); // Explicitly wait until queued GPU commands complete.
        state_->gl.check("wait for GPU completion"); // Check for a driver error from the preceding GL operation.
    } // End this scope in synchronize.
    /** @brief Delete GL objects retired by CPU workers and poll pending completions. */
    void collect_garbage() { // Collect deferred GL deletions and poll completion fences.
        state_->collect(); // Run queued deletions and zero-time completion checks.
    } // End this scope in collect garbage.
    /** @brief Current GL renderer name, for diagnostics and reproducible benchmark reports. */
    std::string renderer() const { // Read the current renderer name for reproducible diagnostics.
        state_->require_thread(); // Require the device context thread and activate its headless context.
        const unsigned char* s = state_->gl.GetString(0x1F01); // Read the driver version or renderer text.
        if (!s) // Validate the renderer query from the current context.
            throw error("no current GL context"); // Report that the renderer query requires a current GL context.
        return reinterpret_cast<const char*>(s); // Read the current renderer name for reproducible diagnostics.
    } // End this scope in renderer.
    /** @internal Runtime access to the retained GL function table and device limits. */
    const std::shared_ptr<detail::device_state>& state() const { // Expose retained runtime state to the resource and backend helpers.
        return state_; // Return the storage or execution state retained by this wrapper.
    } // End this scope in state.
}; // End this scope in runtime.

/** @brief Shared typed std430 storage, owned or borrowed, with automatic CPU/GPU synchronization.
 * Copying a buffer shares its storage. Mutable data() marks host data authoritative.
 * Obtain a fresh data() pointer after a GPU dispatch before accessing host values.
 * GPU handles belong to one device.
 */
template <class T> class buffer { // Select the scalar, dimension, or access type at compile time.
    static_assert(detail::storage_type<T>::value, // Require a representation that matches the shader interface.
                  "buffer requires float/int/uint or matching vec2/vec4 std430 layout"); // Explain which element types match the supported storage-buffer layout.
    static_assert(sizeof(float) == 4 && sizeof(int) == 4 && sizeof(unsigned) == 4, // Require a representation that matches the shader interface.
                  "shdr requires 32-bit GLSL scalars"); // Require the host scalar representation to match GLSL storage.
    std::shared_ptr<detail::buffer_state<T>> state_; // Keep the state shared by this wrapper and its retained owners.

  public: // Expose the operations callers can use.
    /** @brief Allocate a zero-initialized host buffer containing n elements.
     * GPU storage is allocated lazily. Zero-length buffers are valid on the CPU.
     * GPU storage-buffer bindings require at least one element.
     */
    explicit buffer(std::size_t n = 0) : state_(std::make_shared<detail::buffer_state<T>>(n)) { // Create a typed buffer with owned host storage.
        if (n > static_cast<std::size_t>((std::numeric_limits<unsigned>::max)())) // Check resource size or driver limits before accessing storage.
            throw error("buffer length exceeds GLSL uint"); // Report buffer length exceeds glsl uint.
        detail::gl_size(state_->bytes()); // Require a byte count representable by the GL buffer API.
    } // End this scope in buffer.
    /** @brief Borrow n elements stored in application-owned host memory.
     * Keep the memory alive while any resource copy or shader port retains it.
     * Call mark_host_modified() after writing through the original pointer.
     */
    buffer(T* p, std::size_t n) : state_(std::make_shared<detail::buffer_state<T>>(p, n)) { // Initialize the typed resource storage and validate its size.
        if (n > static_cast<std::size_t>((std::numeric_limits<unsigned>::max)())) // Check resource size or driver limits before accessing storage.
            throw error("buffer length exceeds GLSL uint"); // Report buffer length exceeds glsl uint.
        detail::gl_size(state_->bytes()); // Require a byte count representable by the GL buffer API.
    } // End this scope in buffer.
    /** @brief Allocate owned host storage and copy an initializer list into it. */
    buffer(std::initializer_list<T> values) : buffer(values.size()) { // Initialize the typed resource storage and validate its size.
        std::copy(values.begin(), values.end(), data()); // Access host values after bringing their resource copy up to date.
    } // End this scope in buffer.
    /** @brief Return the stored element count. */
    std::size_t size() const { // Read the stored element or stage count.
        return state_->count; // Return the result of this check or operation.
    } // End this scope in runtime.
    /** @brief Synchronize newer GPU contents and return mutable host elements.
     * Marks host data modified. Reacquire this pointer after GPU work before using it.
     */
    T* data() { // Access host values after bringing their resource copy up to date.
        state_->ensure_host(); // Download newer GPU values only when host access needs them.
        state_->latest = detail::authority::host; // Record which backend now owns the newest resource values.
        return static_cast<T*>(state_->host_pointer()); // Access host values after bringing their resource copy up to date.
    } // End this scope in data.
    /** @brief Synchronize newer GPU contents and return readonly host elements. */
    const T* data() const { // Access host values after bringing their resource copy up to date.
        state_->ensure_host(); // Download newer GPU values only when host access needs them.
        return static_cast<const T*>(state_->host_pointer()); // Access host values after bringing their resource copy up to date.
    } // End this scope in data.
    /** @brief Access one host element after synchronization. reject an out-of-range index. */
    T& operator[](std::size_t i) { // Bounds-check this host element before returning its typed reference.
        if (i >= size()) // Check resource size or driver limits before accessing storage.
            throw error("host buffer index out of range"); // Report host buffer index out of range.
        return data()[i]; // Return the result of this check or operation.
    } // End this scope in runtime.
    /** @brief Read one host element after synchronization. reject an out-of-range index. */
    const T& operator[](std::size_t i) const { // Bounds-check this host element before returning its typed reference.
        if (i >= size()) // Check resource size or driver limits before accessing storage.
            throw error("host buffer index out of range"); // Report host buffer index out of range.
        return data()[i]; // Return the result of this check or operation.
    } // End this scope in runtime.
    /** @brief Ensure this device has the latest contents, allocating or uploading as needed. */
    void upload(device& d) { // Explicitly make current host values available to the GPU.
        state_->ensure_gpu(d.state()); // Allocate or upload GPU storage when its host copy is newer.
    } // End this scope in upload.
    /** @brief Download newer GPU contents. reject a different owning device. */
    void download(device& d) { // Explicitly bring newer GPU values back to host storage.
        if (state_->owner && state_->owner != d.state()) // Check that this storage belongs to the requested device.
            throw error("resource belongs to a different device"); // Report resource belongs to a different device.
        state_->ensure_host(); // Download newer GPU values only when host access needs them.
    } // End this scope in download.
    /** @brief Obtain the synchronized GL SSBO handle for application interop. */
    unsigned gl_handle(device& d) { // Expose a synchronized GL resource handle for interoperation.
        upload(d); // Explicitly make current host values available to the GPU.
        return state_->handle; // Expose a synchronized GL resource handle for interoperation.
    } // End this scope in gl handle.
    /** @brief Notify shdr after the application writes the GL buffer. its context must be current. */
    void mark_gpu_modified(device& d) { // Record a write performed by application GL commands.
        if (state_->owner != d.state() || !state_->handle) // Check that this storage belongs to the requested device.
            throw error("resource has no handle on this device"); // Report a mismatch between the resource handle and the device.
        state_->latest = detail::authority::gpu; // Record which backend now owns the newest resource values.
    } // End this scope in mark gpu modified.
    /** @brief Notify shdr after changing previously obtained or borrowed host memory.
     * Call data() before editing if GPU data must first be preserved.
     */
    void mark_host_modified() { // Record a write through retained or borrowed host memory.
        state_->latest = detail::authority::host; // Record which backend now owns the newest resource values.
    } // End this scope in mark host modified.
    /** @brief Borrow an existing SSBO. Its contents initially belong to the GPU. caller retains ownership. */
    static buffer from_gl(device& d, unsigned handle, std::size_t count) { // Borrow an application-owned GL object.
        if (!handle) // Validate the resource handle before using it.
            throw error("cannot borrow GL buffer zero"); // Require a nonzero GL buffer handle.
        buffer result(count); // Keep the element or configuration count.
        result.state_->owner = d.state(); // Set the device that owns the GL resource.
        result.state_->handle = handle; // Set the GL buffer or texture name.
        result.state_->borrowed = true; // Set whether the application owns the supplied GL object.
        result.state_->latest = detail::authority::gpu; // Record which backend now owns the newest resource values.
        return result; // Return the value returned by this operation.
    } // End this scope in from gl.
    /** @internal Runtime access to shared resource storage and synchronization state. */
    const std::shared_ptr<detail::buffer_state<T>>& state() const { // Expose retained runtime state to the resource and backend helpers.
        return state_; // Return the storage or execution state retained by this wrapper.
    } // End this scope in state.
}; // End this scope in runtime.

/** @brief Interleaved 32-bit scalar or RGBA image shared between CPU and GPU.
 * C is 1 or 4. T is float, int, or unsigned. Use matching r32f/r32i/r32ui or
 * rgba32f/rgba32i/rgba32ui GLSL declarations. Copies and bound ports alias storage.
 */
template <class T, unsigned C = 1> class image2d { // Select the scalar, dimension, or access type at compile time.
    static_assert(detail::storage_type<T>::value && std::is_arithmetic<T>::value, // Require a representation that matches the shader interface.
                  "image requires float/int/uint"); // Require an image scalar type supported by the runtime.
    static_assert(C == 1 || C == 4, "portable image formats have one or four components"); // Require a representation that matches the shader interface.
    std::shared_ptr<detail::image_state<T>> state_; // Keep the state shared by this wrapper and its retained owners.

  public: // Expose the operations callers can use.
    /** @brief Allocate a zero-initialized image with positive width and height in pixels. */
    image2d(unsigned w, unsigned h) : state_(std::make_shared<detail::image_state<T>>(w, h, C)) { // Initialize the image geometry and typed pixel storage.
    } // End this scope in image2d.
    /** @brief Return the stored number of pixels in each row. */
    unsigned width() const { // Return the image width in pixels.
        return state_->width; // Return the result of this check or operation.
    } // End this scope in runtime.
    /** @brief Return the stored number of pixel rows. */
    unsigned height() const { // Return the image height in pixels.
        return state_->height; // Return the result of this check or operation.
    } // End this scope in runtime.
    /** @brief Synchronize newer GPU contents and return mutable interleaved host pixels.
     * Index component c at (x,y) as data()[(y*width()+x)*C+c]. Marks host data
     * modified. reacquire the pointer after GPU work before using it.
     */
    T* data() { // Access host values after bringing their resource copy up to date.
        state_->ensure_host(); // Download newer GPU values only when host access needs them.
        state_->latest = detail::authority::host; // Record which backend now owns the newest resource values.
        return static_cast<T*>(state_->host_pointer()); // Access host values after bringing their resource copy up to date.
    } // End this scope in data.
    /** @brief Synchronize newer GPU contents and return readonly interleaved host pixels. */
    const T* data() const { // Access host values after bringing their resource copy up to date.
        state_->ensure_host(); // Download newer GPU values only when host access needs them.
        return static_cast<const T*>(state_->host_pointer()); // Access host values after bringing their resource copy up to date.
    } // End this scope in data.
    /** @brief Ensure this device has the latest pixels, allocating or uploading as needed. */
    void upload(device& d) { // Explicitly make current host values available to the GPU.
        state_->ensure_gpu(d.state()); // Allocate or upload GPU storage when its host copy is newer.
    } // End this scope in upload.
    /** @brief Download newer GPU pixels through a portable compute-to-buffer copy. */
    void download(device& d) { // Explicitly bring newer GPU values back to host storage.
        if (state_->owner && state_->owner != d.state()) // Check that this storage belongs to the requested device.
            throw error("resource belongs to a different device"); // Report resource belongs to a different device.
        state_->ensure_host(); // Download newer GPU values only when host access needs them.
    } // End this scope in download.
    /** @brief Obtain the synchronized level-zero GL texture handle for application interop. */
    unsigned gl_handle(device& d) { // Expose a synchronized GL resource handle for interoperation.
        upload(d); // Explicitly make current host values available to the GPU.
        return state_->handle; // Expose a synchronized GL resource handle for interoperation.
    } // End this scope in gl handle.
    /** @brief Notify shdr after the application writes the GL texture. its context must be current. */
    void mark_gpu_modified(device& d) { // Record a write performed by application GL commands.
        if (state_->owner != d.state() || !state_->handle) // Check that this storage belongs to the requested device.
            throw error("resource has no handle on this device"); // Report a mismatch between the resource handle and the device.
        state_->latest = detail::authority::gpu; // Record which backend now owns the newest resource values.
    } // End this scope in mark gpu modified.
    /** @brief Notify shdr after changing a retained host pixel pointer. */
    void mark_host_modified() { // Record a write through retained or borrowed host memory.
        state_->latest = detail::authority::host; // Record which backend now owns the newest resource values.
    } // End this scope in mark host modified.
    /** @brief Borrow a matching immutable-format, level-zero 2D GL texture. caller retains ownership. */
    static image2d from_gl(device& d, unsigned handle, unsigned w, unsigned h) { // Borrow an application-owned GL object.
        if (!handle) // Validate the resource handle before using it.
            throw error("cannot borrow GL texture zero"); // Require a nonzero GL texture handle.
        image2d result(w, h); // Allocate host pixel storage before attaching the borrowed texture.
        result.state_->owner = d.state(); // Set the device that owns the GL resource.
        result.state_->handle = handle; // Set the GL buffer or texture name.
        result.state_->borrowed = true; // Set whether the application owns the supplied GL object.
        result.state_->latest = detail::authority::gpu; // Record which backend now owns the newest resource values.
        return result; // Return the value returned by this operation.
    } // End this scope in from gl.
    /** @internal Runtime access to shared image storage and synchronization state. */
    const std::shared_ptr<detail::image_state<T>>& state() const { // Expose retained runtime state to the resource and backend helpers.
        return state_; // Return the storage or execution state retained by this wrapper.
    } // End this scope in state.
}; // End this scope in runtime.

class kernel; // Keep the kernel values and operations together.
namespace detail { // Group these definitions in the detail namespace.
enum class port_kind { buffer, image, uniform }; // Name the profile, access, or completion values accepted by this section.
struct port_base { // Keep the port base values and operations together.
    kernel& parent; // Keep the shader instance that owns this interface.
    std::string name; // Keep the resource or entry-point name.
    unsigned binding; // Keep the numeric shader resource binding.
    access mode; // Keep the declared shader resource access.
    port_kind kind; // Keep the buffer, image, or uniform interface kind.
    port_base(kernel&, const char*, unsigned, access, port_kind); // Register the named interface with its owning shader.
    virtual ~port_base() { // Allow typed interfaces to be destroyed through this base interface.
    } // End this scope in runtime.
    virtual std::shared_ptr<resource_state> resource() const { // Expose the retained storage used by this port.
        return std::shared_ptr<resource_state>(); // Represent a uniform-only interface with an empty resource reference.
    } // End this scope in runtime.
    virtual void bind_gpu(device_state&, GLuint) { // Bind this interface or assign its uniform before dispatch.
    } // End this scope in bind gpu.
    virtual void validate() const { // Check that this resource binding satisfies the shader declaration.
    } // End this scope in validate.
    virtual void copy_to(port_base&) const { // Copy the typed port into a matching snapshot interface.
        throw error("shader port does not support snapshots"); // Report that this port requires snapshot support.
    } // End this scope in copy to.
}; // End this scope in runtime.
} // namespace detail

/** @brief Base for compiled shader objects, owning resource declarations and invocation builtins.
 * Shader objects are neither copyable nor movable because their ports refer to their instance.
 * CMake-generated adapters embed their native GLSL source automatically.
 */
class kernel { // Keep the kernel values and operations together.
    std::string name_, source_; // Keep the diagnostic shader name and its embedded GLSL text.
    extent local_; // Keep the fixed workgroup dimensions.
    std::vector<detail::port_base*> ports_; // Keep the registered resource and uniform interfaces.
    std::shared_ptr<detail::program_state> program_; // Keep the retained compiled compute program.

  protected: // Allow derived kernels to initialize their embedded source.
 // Generated shader constructors install their build-embedded GLSL through this method.
    void set_embedded_source(const char* text) { // Retain the build-generated GLSL text inside the shader.
        source_ = text; // Set the automatically embedded shader text.
    } // End this scope in runtime.

  public: // Expose the operations callers can use.
    glsl::uvec3 gl_GlobalInvocationID, gl_LocalInvocationID, gl_WorkGroupID, gl_NumWorkGroups, // Keep CPU global, local, group and dispatch coordinates for GLSL builtins.
        gl_WorkGroupSize; // Keep the fixed local dimensions exposed to CPU shader code.
    unsigned gl_LocalInvocationIndex; // Keep the flattened index within the current CPU work group.
    /** @brief Construct a kernel identity and positive local workgroup dimensions.
     * SHDR_BEGIN emits this constructor for shared shader classes.
     */
    kernel(const std::string& name, extent local) // Initialize the shader name, local dimensions, and invocation values.
        : name_(name), local_(local), gl_LocalInvocationIndex(0) { // Remember the shader name and local size with initially zero CPU invocation state.
        if (!local.x || !local.y || !local.z) // Require positive local dimensions on every axis.
            throw error("local work group dimensions must be positive"); // Report local work group dimensions must be positive.
        gl_WorkGroupSize = glsl::uvec3(local.x, local.y, local.z); // Publish the fixed local dimensions to CPU shader builtins.
    } // End this scope in kernel.
    virtual ~kernel() { // Allow derived shader instances to be destroyed through the base interface.
    } // End this scope in kernel.
    kernel(const kernel&) = delete; // Prevent copying this object and duplicating its retained identity.
    kernel& operator=(const kernel&) = delete; // Prevent copying this object and duplicating its retained identity.
    /** @brief Execute one CPU invocation. SHDR_MAIN supplies the override.
     * Dispatch through a stage or pipeline to initialize builtins and synchronize resources.
     */
    virtual void main() = 0; // Require a compiled shader to supply its CPU invocation method.
    /** @brief Make an independent shader instance with a snapshot of its resource bindings and uniforms. */
    virtual std::unique_ptr<kernel> clone() const { // Create an independent shader instance for deferred CPU execution.
        throw error("shader does not support snapshots"); // Report that this shader requires snapshot support.
    } // End this scope in runtime.
    /** @internal Copy registered interfaces into a newly constructed instance. */
    void copy_interfaces(kernel& destination) const { // Copy resource bindings and uniforms into a fresh shader clone.
        if (ports_.size() != destination.ports_.size()) // Check resource size or driver limits before accessing storage.
            throw error("shader snapshot interface mismatch"); // Report shader snapshot interface mismatch.
        for (std::size_t i = 0; i < ports_.size(); ++i) // Visit every registered resource and uniform interface.
            ports_[i]->copy_to(*destination.ports_[i]); // Copy the typed port into a matching snapshot interface.
    } // End this scope in copy interfaces.
    /** @internal Retain all resources used by an enqueued dispatch. */
    std::vector<std::shared_ptr<detail::resource_state>> resources() const { // Collect the shared storage used by this shader.
        std::vector<std::shared_ptr<detail::resource_state>> result; // Keep the value returned by this operation.
        for (std::size_t i = 0; i < ports_.size(); ++i) { // Visit every registered resource and uniform interface.
            auto r = ports_[i]->resource(); // Get the shared storage behind this resource interface.
            if (r) // Choose this branch when r.
                result.push_back(r); // Retain this entry in the ordered resource or work collection.
        } // End this scope in resources.
        return result; // Return the value returned by this operation.
    } // End this scope in resources.
    /** @internal Retain a compiled program during asynchronous execution. */
    const std::shared_ptr<detail::program_state>& program_state() const { // Expose the retained compiled program for asynchronous ownership.
        return program_; // Return the retained compiled compute program.
    } // End this scope in program state.
    /** @brief Return the kernel's diagnostic name. */
    const std::string& name() const { // Return the diagnostic shader name.
        return name_; // Return the kernel or registration name.
    } // End this scope in runtime.
    /** @brief Return the fixed local workgroup dimensions declared by SHDR_BEGIN. */
    extent local_size() const { // Return the fixed local work group dimensions.
        return local_; // Return the fixed workgroup dimensions.
    } // End this scope in runtime.
    /** @brief Export this shader's automatically embedded source for compiler diagnostics or tooling. */
    std::string glsl(profile p) const { // Export the embedded shader for the requested language profile.
        if (source_.empty()) // Choose this branch when source .empty().
            throw error("shader has no embedded GPU source: " + name_); // Report that this shader requires an embedded GPU program.
        return glsl_source(source_, p); // Return the result of this check or operation.
    } // End this scope in runtime.
    /** @internal Register declarations emitted by the shared shader macros. */
    void register_port(detail::port_base& port) { // Register one named resource or uniform with duplicate checks.
        for (std::size_t i = 0; i < ports_.size(); ++i) { // Visit every registered resource and uniform interface.
            if (ports_[i]->name == port.name) // Choose this branch when ports [i]->name equals port.name.
                throw error("duplicate shader port: " + port.name); // Report duplicate shader port: .
            if (port.kind != detail::port_kind::uniform && ports_[i]->kind == port.kind && // Choose the branch matching this interface or resource access.
                ports_[i]->binding == port.binding) // Set the numeric shader resource binding.
                throw error("duplicate shader resource binding: " + port.name); // Report duplicate shader resource binding: .
        } // End this scope in register port.
        ports_.push_back(&port); // Retain this entry in the ordered resource or work collection.
    } // End this scope in register port.
    /** @brief Bind by name, for runtime-selected shader modules. Type mismatches throw error. */
    template <class T> kernel& bind(const std::string& name, const buffer<T>& resource); // Select the scalar, dimension, or access type at compile time.
    /** @brief Bind an image by name. scalar type and component count must match its port. */
    template <class T, unsigned C> // Select the scalar, dimension, or access type at compile time.
    kernel& bind(const std::string& name, const image2d<T, C>& resource); // Rebind a named resource port while preserving its declared type.
    /** @brief Assign a uniform by name. T must exactly match its declared shader type. */
    template <class T> kernel& set(const std::string& name, const T& value); // Select the scalar, dimension, or access type at compile time.
    /** @internal Find a registered port for runtime binding and uniform assignment. */
    detail::port_base& find_port(const std::string& name) { // Resolve the named shader interface.
        for (std::size_t i = 0; i < ports_.size(); ++i) // Visit every registered resource and uniform interface.
            if (ports_[i]->name == name) // Choose this branch when ports [i]->name equals name.
                return *ports_[i]; // Resolve the named shader interface.
        throw error("unknown shader port: " + name); // Report unknown shader port: .
    } // End this scope in find port.
    /** @brief Execute one dispatch with workgroup counts on a device, or on the CPU if null.
     * Normally called by pipeline.run(). GPU work uses the embedded shader and a current context.
     */
    void run(extent groups, device* d); // Keep the target device.
    /** @brief Compile and allocate GPU resources before entering a latency-sensitive dispatch loop. */
    void prepare(device& d); // Compile the program and prepare its bound GPU resources.
}; // End this scope in runtime.

namespace detail { // Group these definitions in the detail namespace.
inline port_base::port_base(kernel& p, const char* n, unsigned b, access a, port_kind k) // Register the named interface with its owning shader.
    : parent(p), name(n), binding(b), mode(a), kind(k) { // Record the owning kernel, interface name, binding, access and resource kind.
    parent.register_port(*this); // Register one named resource or uniform with duplicate checks.
} // End this scope in runtime.
} // namespace detail

/** @brief Typed shader SSBO declaration. assign a buffer to bind or rebind it. */
template <class T, access A = access::readwrite> class buffer_port : public detail::port_base { // Select the scalar, dimension, or access type at compile time.
    std::shared_ptr<detail::buffer_state<T>> state_; // Keep the state shared by this wrapper and its retained owners.
    unsigned fixed_length_; // Keep the fixed array length declared by the shader.

  public: // Expose the operations callers can use.
    /** @brief Declare a typed buffer port on a kernel, usually through SHDR_BUFFER. */
    buffer_port(kernel& p, unsigned b, const char* n, unsigned fixed = 0) // Register a typed buffer interface and any fixed shader array length.
        : port_base(p, n, b, A, detail::port_kind::buffer), fixed_length_(fixed) { // Record the buffer interface and its optional fixed shader length.
    } // End this scope in runtime.
    /** @brief Bind or rebind this port to the buffer's retained shared storage. */
    buffer_port& operator=(const buffer<T>& b) { // Retain the newly bound typed resource in this shader interface.
        state_ = b.state(); // Set the state shared by this wrapper and its retained owners.
        return *this; // Allow another operation to follow in the same expression.
    } // End this scope in runtime.
    void copy_to(detail::port_base& destination) const override { // Copy the typed port into a matching snapshot interface.
        buffer_port* p = dynamic_cast<buffer_port*>(&destination); // Require a snapshot port with the same buffer type and access qualifier.
        if (!p) // Validate the typed interface pointer before accessing it.
            throw error("buffer snapshot type mismatch: " + name); // Report buffer snapshot type mismatch: .
        p->state_ = state_; // Set the state shared by this wrapper and its retained owners.
    } // End this scope in copy to.
    /** @brief Return the bound buffer's element count, matching GLSL array length(). */
    unsigned length() const { // Return the GLSL-visible array length after validating its binding.
        validate(); // Check that this resource binding satisfies the shader declaration.
        return fixed_length_ ? fixed_length_ : static_cast<unsigned>(state_->count); // Return the GLSL-visible array length after validating its binding.
    } // End this scope in length.
    /** @brief Access an element during CPU execution, checking bounds and declared access.
     * Readonly ports return const references. Pipeline execution synchronizes host contents.
     */
    typename std::conditional<A == access::readonly, const T&, T&>::type operator[](std::size_t i) { // Expose a const component for readonly ports and a mutable one otherwise.
        validate(); // Check that this resource binding satisfies the shader declaration.
        const std::size_t limit = fixed_length_ ? fixed_length_ : state_->count; // Use the declared array size or the full bound buffer length.
        if (i >= limit) // Choose this branch when i >= limit.
            throw error("CPU SSBO index out of bounds: " + name); // Report cpu ssbo index out of bounds: .
        return static_cast<T*>(state_->host_pointer())[i]; // Return the result of this check or operation.
    } // End this scope in runtime.
    void validate() const override { // Check that this resource binding satisfies the shader declaration.
        if (!state_) // Require retained storage for this interface.
            throw error("unbound shader buffer: " + name); // Report unbound shader buffer: .
        if (fixed_length_ && state_->count < fixed_length_) // Check resource size or driver limits before accessing storage.
            throw error("buffer is smaller than the shader's fixed array: " + name); // Report buffer is smaller than the shader's fixed array: .
    } // End this scope in validate.
    std::shared_ptr<detail::resource_state> resource() const override { // Expose the retained storage used by this port.
        return state_; // Return the storage or execution state retained by this wrapper.
    } // End this scope in runtime.
    void bind_gpu(detail::device_state& d, unsigned) override { // Bind this interface or assign its uniform before dispatch.
        if (binding >= d.max_buffers) // Choose this branch when binding >= d.max buffers.
            throw error("SSBO binding exceeds device limit: " + name); // Report ssbo binding exceeds device limit: .
        if (!state_->bytes()) // Check resource size or driver limits before accessing storage.
            throw error("GPU SSBO bindings require a nonempty buffer: " + name); // Report gpu ssbo bindings require a nonempty buffer: .
        d.gl.BindBufferBase(detail::SHADER_STORAGE_BUFFER, binding, state_->handle); // Bind storage to the shader interface index.
    } // End this scope in bind gpu.
}; // End this scope in runtime.

namespace detail { // Group these definitions in the detail namespace.
inline void uniform_value(gl_api& gl, GLint loc, float v) { // Send the typed uniform value through its matching GL call.
    gl.Uniform1f(loc, v); // Assign a float uniform.
} // End this scope in uniform value.
inline void uniform_value(gl_api& gl, GLint loc, int v) { // Send the typed uniform value through its matching GL call.
    gl.Uniform1i(loc, v); // Assign an integer or boolean uniform.
} // End this scope in uniform value.
inline void uniform_value(gl_api& gl, GLint loc, unsigned v) { // Send the typed uniform value through its matching GL call.
    gl.Uniform1ui(loc, v); // Assign an unsigned integer uniform.
} // End this scope in uniform value.
inline void uniform_value(gl_api& gl, GLint loc, bool v) { // Send the typed uniform value through its matching GL call.
    gl.Uniform1i(loc, v ? 1 : 0); // Assign an integer or boolean uniform.
} // End this scope in uniform value.
#define SHDR_UNIFORM_VEC(n, t, suffix) /* Define the SHDR_UNIFORM_VEC helper for this group of declarations. */ \
    inline void uniform_value(gl_api& gl, GLint loc, const glsl::vec<t, n>& v) { /* Send the typed uniform value through its matching GL call. */ \
        t packed[n]; /* Pack the vector components contiguously for the GL uniform call. */ \
        for (unsigned i = 0; i < n; ++i) /* Visit the values or interfaces needed by uniform value. */ \
            packed[i] = v[i]; /* Copy the component into the contiguous GL uniform payload. */ \
        gl.Uniform##n##suffix(loc, 1, packed); /* Send the packed vector components through their matching GL function. */ \
    } /* End this scope in uniform value. */
SHDR_UNIFORM_VEC(2, float, fv) // Define the listed float, signed or unsigned vector uniform overloads.
SHDR_UNIFORM_VEC(3, float, fv) SHDR_UNIFORM_VEC(4, float, fv) SHDR_UNIFORM_VEC(2, int, iv) // Define the listed float, signed or unsigned vector uniform overloads.
    SHDR_UNIFORM_VEC(3, int, iv) SHDR_UNIFORM_VEC(4, int, iv) SHDR_UNIFORM_VEC(2, unsigned, uiv) // Define the listed float, signed or unsigned vector uniform overloads.
        SHDR_UNIFORM_VEC(3, unsigned, uiv) SHDR_UNIFORM_VEC(4, unsigned, uiv) // Define the listed float, signed or unsigned vector uniform overloads.
#undef SHDR_UNIFORM_VEC // Remove this temporary macro after its declarations are complete.
            template <std::size_t N> // Select the scalar, dimension, or access type at compile time.
            inline void uniform_value(gl_api& gl, GLint loc, const glsl::vec<bool, N>& v) { // Send the typed uniform value through its matching GL call.
    GLint packed[N]; // Pack boolean components as the integer values expected by GL.
    for (unsigned i = 0; i < N; ++i) // Visit the values or interfaces needed by uniform value.
        packed[i] = v[i] ? 1 : 0; // Represent a boolean uniform component as a GL integer.
    if (N == 2) // Choose this branch when N equals 2.
        gl.Uniform2iv(loc, 1, packed); // Send two integer-represented boolean components to GL.
    else if (N == 3) // Choose this branch when N equals 3.
        gl.Uniform3iv(loc, 1, packed); // Send three integer-represented boolean components to GL.
    else // Handle the other resource, platform, or completion case.
        gl.Uniform4iv(loc, 1, packed); // Send four integer-represented boolean components to GL.
} // End this scope in uniform value.
#define SHDR_UNIFORM_MATRIX(c, r, n) /* Define the SHDR_UNIFORM_MATRIX helper for this group of declarations. */ \
    inline void uniform_value(gl_api& gl, GLint loc, const glsl::mat<float, c, r>& m) { /* Send the typed uniform value through its matching GL call. */ \
        float packed[c * r]; /* Send the typed uniform value through its matching GL call. */ \
        for (unsigned i = 0; i < c; ++i) /* Visit the values or interfaces needed by uniform value. */ \
            for (unsigned j = 0; j < r; ++j) /* Visit the values or interfaces needed by uniform value. */ \
                packed[i * r + j] = m[i][j]; /* Flatten the matrix in GLSL column order. */ \
        gl.UniformMatrix##n##fv(loc, 1, 0, packed); /* Upload the column-major matrix with the transpose flag set to false. */ \
    } /* End this scope in uniform value. */
SHDR_UNIFORM_MATRIX(2, 2, 2) // Define the listed column-major matrix uniform overloads.
SHDR_UNIFORM_MATRIX(3, 3, 3) SHDR_UNIFORM_MATRIX(4, 4, 4) SHDR_UNIFORM_MATRIX(2, 3, 2x3) // Define the listed column-major matrix uniform overloads.
    SHDR_UNIFORM_MATRIX(2, 4, 2x4) SHDR_UNIFORM_MATRIX(3, 2, 3x2) SHDR_UNIFORM_MATRIX(3, 4, 3x4) // Define the listed column-major matrix uniform overloads.
        SHDR_UNIFORM_MATRIX(4, 2, 4x2) SHDR_UNIFORM_MATRIX(4, 3, 4x3) // Define the listed column-major matrix uniform overloads.
#undef SHDR_UNIFORM_MATRIX // Remove this temporary macro after its declarations are complete.
} // namespace detail

/** @brief A zero-initialized typed shader uniform, assigned with ordinary C++ syntax. */
template <class T> class uniform_port : public detail::port_base { // Select the scalar, dimension, or access type at compile time.
    T owned_; // Keep the uniform value stored by the port.
    T* value_; // Keep the current uniform field.

  public: // Expose the operations callers can use.
    /** @brief Declare a zero-initialized uniform backed by this port's own value. */
    uniform_port(kernel& p, const char* n) // Register a typed uniform backed by this port's own zero-initialized value.
        : port_base(p, n, 0, access::readonly, detail::port_kind::uniform), owned_(), // Keep the uniform value stored by the port.
          value_(&owned_) { // Keep the uniform value stored by the port.
    } // End this scope in runtime.
    /** @brief Register an existing typed kernel field as a uniform. emitted by SHDR_UNIFORM. */
    uniform_port(kernel& p, const char* n, T& value) // Register a uniform and refer to its shader member value.
        : port_base(p, n, 0, access::readonly, detail::port_kind::uniform), owned_(), // Keep the uniform value stored by the port.
          value_(&value) { // Refer to the uniform member owned by this shader instance.
    } // End this scope in runtime.
    /** @brief Set the uniform value uploaded at the next dispatch. */
    uniform_port& operator=(const T& value) { // Assign the shader member referred to by this uniform interface.
        *value_ = value; // Set the current uniform field.
        return *this; // Allow another operation to follow in the same expression.
    } // End this scope in runtime.
    /** @brief Read the uniform as its declared shader type during CPU execution. */
    operator T() const { // Read the uniform member as an ordinary value of its declared type.
        return *value_; // Return the result of this check or operation.
    } // End this scope in runtime.
    /** @brief Return the stored uniform value. */
    const T& value() const { // Read the shader member referred to by this uniform port.
        return *value_; // Return the result of this check or operation.
    } // End this scope in runtime.
    void copy_to(detail::port_base& destination) const override { // Copy the typed port into a matching snapshot interface.
        uniform_port* p = dynamic_cast<uniform_port*>(&destination); // Require a snapshot port with the same uniform type.
        if (!p) // Validate the typed interface pointer before accessing it.
            throw error("uniform snapshot type mismatch: " + name); // Report uniform snapshot type mismatch: .
        *p->value_ = *value_; // Set the current uniform field.
    } // End this scope in copy to.
    void bind_gpu(detail::device_state& d, unsigned program) override { // Bind this interface or assign its uniform before dispatch.
        const int loc = d.gl.GetUniformLocation(program, name.c_str()); // Find a named uniform in the linked program.
        if (loc >= 0) // Choose this branch when loc >= 0.
            detail::uniform_value(d.gl, loc, *value_); // Send the typed uniform value through its matching GL call.
    } // End this scope in bind gpu.
}; // End this scope in runtime.

/** @brief A typed image declaration with format/access matching its shared shader. */
template <class T, unsigned C, access A> class image_port : public detail::port_base { // Select the scalar, dimension, or access type at compile time.
    std::shared_ptr<detail::image_state<T>> state_; // Keep the state shared by this wrapper and its retained owners.

  public: // Expose the operations callers can use.
    /** @brief Declare an image port, usually through SHDR_IMAGE2D. */
    image_port(kernel& p, unsigned b, const char* n) // Register a typed image interface at the specified binding.
        : port_base(p, n, b, A, detail::port_kind::image) { // Record the image interface and its declared access qualifier.
    } // End this scope in runtime.
    /** @brief Bind or rebind this port to the image's retained shared storage. */
    image_port& operator=(const image2d<T, C>& image) { // Retain the newly bound typed resource in this shader interface.
        state_ = image.state(); // Set the state shared by this wrapper and its retained owners.
        return *this; // Allow another operation to follow in the same expression.
    } // End this scope in runtime.
    void copy_to(detail::port_base& destination) const override { // Copy the typed port into a matching snapshot interface.
        image_port* p = dynamic_cast<image_port*>(&destination); // Require a snapshot port with the same image type and access qualifier.
        if (!p) // Validate the typed interface pointer before accessing it.
            throw error("image snapshot type mismatch: " + name); // Report image snapshot type mismatch: .
        p->state_ = state_; // Set the state shared by this wrapper and its retained owners.
    } // End this scope in copy to.
    void validate() const override { // Check that this resource binding satisfies the shader declaration.
        if (!state_) // Require retained storage for this interface.
            throw error("unbound shader image: " + name); // Report unbound shader image: .
    } // End this scope in validate.
    std::shared_ptr<detail::resource_state> resource() const override { // Expose the retained storage used by this port.
        return state_; // Return the storage or execution state retained by this wrapper.
    } // End this scope in runtime.
    /** @internal Read image state for the CPU imageLoad/imageStore/imageSize helpers. */
    const std::shared_ptr<detail::image_state<T>>& image_state() const { // Expose the validated image storage to CPU image helpers.
        validate(); // Check that this resource binding satisfies the shader declaration.
        return state_; // Return the storage or execution state retained by this wrapper.
    } // End this scope in runtime.
    void bind_gpu(detail::device_state& d, unsigned) override { // Bind this interface or assign its uniform before dispatch.
        if (binding >= d.max_images) // Choose this branch when binding >= d.max images.
            throw error("image binding exceeds device limit: " + name); // Report image binding exceeds device limit: .
        d.gl.BindImageTexture(binding, // Bind a typed image to its shader interface index.
                              state_->handle, // Keep the GL buffer or texture name.
                              0, // Supply the next value for the current call or declaration.
                              0, // Supply the next value for the current call or declaration.
                              0, // Supply the next value for the current call or declaration.
                              A == access::readonly    ? detail::READ_ONLY // Select the GL read or write qualifier matching this typed image port.
                              : A == access::writeonly ? detail::WRITE_ONLY // Choose the GL image access matching the shader qualifier.
                                                       : detail::READ_WRITE, // Choose the GL image access matching the shader qualifier.
                              state_->format); // Keep the image storage format.
    } // End this scope in bind gpu.
}; // End this scope in runtime.

/** @brief GLSL image dimensions in pixels. */
template <class T, unsigned C, access A> // Select the scalar, dimension, or access type at compile time.
inline glsl::ivec2 imageSize(const image_port<T, C, A>& image) { // Return the bound image dimensions as GLSL integers.
    return glsl::ivec2(image.image_state()->width, image.image_state()->height); // Return the bound image dimensions as GLSL integers.
} // End this scope in imageSize.
/** @brief Load one texel. Missing RGB components are zero and missing alpha is one.
 * Out-of-range coordinates return zero. loading a writeonly image is a compile error.
 */
template <class T, unsigned C, access A> // Select the scalar, dimension, or access type at compile time.
inline glsl::vec<T, 4> imageLoad(const image_port<T, C, A>& image, glsl::ivec2 pos) { // Load one pixel while enforcing readonly access and image bounds.
    static_assert(A != access::writeonly, "cannot imageLoad a writeonly image"); // Require a representation that matches the shader interface.
    const std::shared_ptr<detail::image_state<T>>& s = image.image_state(); // Access the retained image geometry and host pixel storage.
    if (pos.x < 0 || pos.y < 0 || static_cast<unsigned>(pos.x) >= s->width || // Choose this branch when pos.x < 0 or pos.y < 0 or static cast<unsigned>(pos.x) >= s->width or .
        static_cast<unsigned>(pos.y) >= s->height) // Reject an image row outside the allocated pixel storage.
        return glsl::vec<T, 4>(T(0)); // Load one pixel while enforcing readonly access and image bounds.
    glsl::vec<T, 4> result(T(0)); // Start the loaded texel with zero components before filling its channels.
    result[3] = T(1); // Set the value returned by this operation.
    const std::size_t start = // Set the beginning of the current token or range.
        (static_cast<std::size_t>(pos.y) * s->width + static_cast<unsigned>(pos.x)) * C; // Load one pixel while enforcing readonly access and image bounds.
    for (unsigned c = 0; c < C; ++c) // Visit the values or interfaces needed by imageLoad.
        result[c] = s->owned[start + c]; // Set the value returned by this operation.
    return result; // Return the value returned by this operation.
} // End this scope in imageLoad.
/** @brief Store a texel, ignoring out-of-range coordinates. Readonly stores are a compile error. */
template <class T, unsigned C, access A> // Select the scalar, dimension, or access type at compile time.
inline void imageStore(image_port<T, C, A>& image, glsl::ivec2 pos, const glsl::vec<T, 4>& value) { // Store one pixel while enforcing writable access and image bounds.
    static_assert(A != access::readonly, "cannot imageStore a readonly image"); // Require a representation that matches the shader interface.
    const std::shared_ptr<detail::image_state<T>>& s = image.image_state(); // Access the retained image geometry and host pixel storage.
    if (pos.x < 0 || pos.y < 0 || static_cast<unsigned>(pos.x) >= s->width || // Choose this branch when pos.x < 0 or pos.y < 0 or static cast<unsigned>(pos.x) >= s->width or .
        static_cast<unsigned>(pos.y) >= s->height) // Reject an image row outside the allocated pixel storage.
        return; // Finish this operation.
    const std::size_t start = // Set the beginning of the current token or range.
        (static_cast<std::size_t>(pos.y) * s->width + static_cast<unsigned>(pos.x)) * C; // Store one pixel while enforcing writable access and image bounds.
    for (unsigned c = 0; c < C; ++c) // Visit the values or interfaces needed by imageStore.
        s->owned[start + c] = value[c]; // Set host storage allocated by this resource.
} // End this scope in imageStore.

template <class T> inline kernel& kernel::bind(const std::string& name, const buffer<T>& b) { // Select the scalar, dimension, or access type at compile time.
    detail::port_base& p = find_port(name); // Find the declared shader interface with this resource name.
    if (buffer_port<T, access::readonly>* v = dynamic_cast<buffer_port<T, access::readonly>*>(&p)) // Choose this branch when buffer port<T, access::readonly>* v = dynamic cast<buffer port<T, access::readonly>*>(&p).
        *v = b; // Store the current v value for this operation.
    else if (buffer_port<T, access::writeonly>* v = // Choose this branch when buffer port<T, access::writeonly>* v =.
                 dynamic_cast<buffer_port<T, access::writeonly>*>(&p)) // Try the matching typed write-only buffer interface.
        *v = b; // Store the current v value for this operation.
    else if (buffer_port<T, access::readwrite>* v = // Choose this branch when buffer port<T, access::readwrite>* v =.
                 dynamic_cast<buffer_port<T, access::readwrite>*>(&p)) // Try the matching typed read-write buffer interface.
        *v = b; // Store the current v value for this operation.
    else // Handle the other resource, platform, or completion case.
        throw error("buffer type mismatch: " + name); // Report buffer type mismatch: .
    return *this; // Allow another operation to follow in the same expression.
} // End this scope in runtime.
template <class T, unsigned C> // Select the scalar, dimension, or access type at compile time.
inline kernel& kernel::bind(const std::string& name, const image2d<T, C>& b) { // Rebind a named resource port while preserving its declared type.
    detail::port_base& p = find_port(name); // Find the declared shader interface with this resource name.
    if (image_port<T, C, access::readonly>* v = // Choose this branch when image port<T, C, access::readonly>* v =.
            dynamic_cast<image_port<T, C, access::readonly>*>(&p)) // Try the matching typed read-only image interface.
        *v = b; // Store the current v value for this operation.
    else if (image_port<T, C, access::writeonly>* v = // Choose this branch when image port<T, C, access::writeonly>* v =.
                 dynamic_cast<image_port<T, C, access::writeonly>*>(&p)) // Try the matching typed write-only image interface.
        *v = b; // Store the current v value for this operation.
    else if (image_port<T, C, access::readwrite>* v = // Choose this branch when image port<T, C, access::readwrite>* v =.
                 dynamic_cast<image_port<T, C, access::readwrite>*>(&p)) // Try the matching typed read-write image interface.
        *v = b; // Store the current v value for this operation.
    else // Handle the other resource, platform, or completion case.
        throw error("image type mismatch: " + name); // Report image type mismatch: .
    return *this; // Allow another operation to follow in the same expression.
} // End this scope in runtime.
template <class T> inline kernel& kernel::set(const std::string& name, const T& value) { // Select the scalar, dimension, or access type at compile time.
    uniform_port<T>* p = dynamic_cast<uniform_port<T>*>(&find_port(name)); // Require the named uniform to have the supplied value's type.
    if (!p) // Validate the typed interface pointer before accessing it.
        throw error("uniform type mismatch: " + name); // Report uniform type mismatch: .
    *p = value; // Update the typed uniform value for subsequent dispatches.
    return *this; // Allow another operation to follow in the same expression.
} // End this scope in runtime.

inline void kernel::prepare(device& d) { // Compile the program and prepare its bound GPU resources.
    const auto& state = d.state(); // Set the shared device or worker state.
    state->collect(); // Run queued deletions and zero-time completion checks.
    for (std::size_t i = 0; i < ports_.size(); ++i) // Visit every registered resource and uniform interface.
        ports_[i]->validate(); // Check that this resource binding satisfies the shader declaration.
    if (!program_) { // Compile the shader when its cached program is empty.
        if (source_.empty()) // Choose this branch when source .empty().
            throw error("shader has no embedded GPU source: " + name_); // Report that this shader requires an embedded GPU program.
        program_ = std::make_shared<detail::program_state>( // Set the retained compiled compute program.
            state, state->compile(glsl_source(source_, state->language), name_)); // Add the selected GLSL version and shared-declaration definitions.
        detail::GLint actual[3] = {0, 0, 0}; // Prepare all three local dimensions reported by the linked program.
        state->gl.GetProgramiv(program_->id, detail::COMPUTE_WORK_GROUP_SIZE, actual); // Read linked-program status, log length, or local dimensions.
        if (actual[0] != static_cast<int>(local_.x) || actual[1] != static_cast<int>(local_.y) || // Choose this branch when actual[0] differs from static cast<int>(local .x) or actual[1] differs from static cast<int>(local .y) or .
            actual[2] != static_cast<int>(local_.z)) { // Require the linked program's third local dimension to match the CPU shader.
            program_.reset(); // Release or replace the owned object at this lifecycle boundary.
            throw error("GPU source and C++ local sizes differ: " + name_); // Report gpu source and c++ local sizes differ: .
        } // End this scope in prepare.
    } // End this scope in prepare.
    if (program_->owner != state) // Check that this storage belongs to the requested device.
        throw error("shader program belongs to a different device; create a new shader instance"); // Report shader program belongs to a different device. create a new shader instance.
    for (std::size_t i = 0; i < ports_.size(); ++i) { // Visit every registered resource and uniform interface.
        auto r = ports_[i]->resource(); // Get the shared storage behind this resource interface.
        if (r) // Choose this branch when r.
            r->ensure_gpu(state); // Allocate or upload GPU storage when its host copy is newer.
    } // End this scope in prepare.
} // End this scope in prepare.

inline void kernel::run(extent groups, device* d) { // Keep the target device.
    if (!groups.x || !groups.y || !groups.z) // Finish an empty dispatch before submitting GL commands.
        return; // Finish this operation.
    const unsigned g[3] = {groups.x, groups.y, groups.z}, l[3] = {local_.x, local_.y, local_.z}; // Keep dispatch counts and local dimensions for validation and CPU indexing.
    for (unsigned i = 0; i < 3; ++i) // Validate or query each of the three compute dimensions.
        if (g[i] > (std::numeric_limits<unsigned>::max)() / l[i]) // Choose this branch when g[i] > (std::numeric limits<unsigned>::max)() / l[i].
            throw error("global invocation index overflow"); // Report global invocation index overflow.
    std::vector<std::shared_ptr<detail::resource_state>> outputs; // Retain the writable resources whose authority changes after this dispatch.
    for (std::size_t i = 0; i < ports_.size(); ++i) { // Visit every registered resource and uniform interface.
        ports_[i]->validate(); // Check that this resource binding satisfies the shader declaration.
        std::shared_ptr<detail::resource_state> r = ports_[i]->resource(); // Get the shared storage behind this resource interface.
        if (r && ports_[i]->mode != access::readonly) // Choose the branch matching this interface or resource access.
            outputs.push_back(r); // Retain this entry in the ordered resource or work collection.
    } // End this scope in runtime.
    if (d) { // Choose this branch when d.
        const std::shared_ptr<detail::device_state>& state = d->state(); // Set the shared device or worker state.
        for (unsigned i = 0; i < 3; ++i) { // Validate or query each of the three compute dimensions.
            if (g[i] > state->max_groups[i]) // Choose this branch when g[i] > state->max groups[i].
                throw error("dispatch exceeds device work group count"); // Report dispatch exceeds device work group count.
            if (l[i] > state->max_local[i]) // Choose this branch when l[i] > state->max local[i].
                throw error("local work group dimension exceeds device limit"); // Report local work group dimension exceeds device limit.
        } // End this scope in runtime.
        const std::size_t invocations = // Count the shader invocations in one local work group.
            detail::checked_product(detail::checked_product(l[0], l[1]), l[2]); // Reject resource-size multiplication that would overflow.
        if (invocations > state->max_invocations) // Choose this branch when invocations > state->max invocations.
            throw error("local invocation count exceeds device limit"); // Report local invocation count exceeds device limit.
        prepare(*d); // Compile the program and prepare its bound GPU resources.
        state->gl.UseProgram(program_->id); // Select the compute program for the next dispatch.
        for (std::size_t i = 0; i < ports_.size(); ++i) // Visit every registered resource and uniform interface.
            ports_[i]->bind_gpu(*state, program_->id); // Bind this interface or assign its uniform before dispatch.
        state->gl.check("bind shader " + name_); // Keep the kernel or registration name.
        state->gl.DispatchCompute(groups.x, groups.y, groups.z); // Queue the shader workgroups.
        state->gl.MemoryBarrier(detail::ALL_BARRIER_BITS); // Make earlier resource writes visible to following GPU commands.
 // Conservatively record GPU authority even if an error follows a submitted dispatch.
        for (std::size_t i = 0; i < outputs.size(); ++i) // Visit this axis of the dispatch or current workgroup.
            outputs[i]->latest = detail::authority::gpu; // Record which backend now owns the newest resource values.
        state->gl.check("dispatch shader " + name_); // Keep the kernel or registration name.
    } else { // Handle the alternative resource or completion case.
        for (std::size_t i = 0; i < ports_.size(); ++i) { // Visit every registered resource and uniform interface.
            std::shared_ptr<detail::resource_state> r = ports_[i]->resource(); // Get the shared storage behind this resource interface.
            if (r) // Choose this branch when r.
                r->ensure_host(); // Download newer GPU values only when host access needs them.
        } // End this scope in runtime.
        gl_NumWorkGroups = glsl::uvec3(groups.x, groups.y, groups.z); // Publish the dispatch dimensions to CPU shader builtins.
        try { // Keep cleanup available if an allocation, shader, or driver operation fails.
            for (unsigned gz = 0; gz < groups.z; ++gz) // Visit this axis of the dispatch or current workgroup.
                for (unsigned gy = 0; gy < groups.y; ++gy) // Visit this axis of the dispatch or current workgroup.
                    for (unsigned gx = 0; gx < groups.x; ++gx) { // Visit this axis of the dispatch or current workgroup.
                        gl_WorkGroupID = glsl::uvec3(gx, gy, gz); // Publish the CPU work group currently being executed.
                        for (unsigned z = 0; z < local_.z; ++z) // Visit this axis of the dispatch or current workgroup.
                            for (unsigned y = 0; y < local_.y; ++y) // Visit this axis of the dispatch or current workgroup.
                                for (unsigned x = 0; x < local_.x; ++x) { // Visit this axis of the dispatch or current workgroup.
                                    gl_LocalInvocationID = glsl::uvec3(x, y, z); // Publish the invocation coordinates inside this work group.
                                    gl_GlobalInvocationID = glsl::uvec3( // Combine work group and local coordinates into global indices.
                                        gx * local_.x + x, gy * local_.y + y, gz * local_.z + z); // Use the local size to place this invocation in the complete dispatch.
                                    gl_LocalInvocationIndex = (z * local_.y + y) * local_.x + x; // Flatten the local coordinates with X changing fastest.
                                    main(); // Execute the shader with all CPU invocation builtins initialized.
                                } // End this scope in runtime.
                    } // End this scope in runtime.
        } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
            for (std::size_t i = 0; i < outputs.size(); ++i) // Visit this axis of the dispatch or current workgroup.
                outputs[i]->latest = detail::authority::host; // Record which backend now owns the newest resource values.
            throw; // Rethrow the original failure after arranging the necessary cleanup.
        } // End this scope in runtime.
        for (std::size_t i = 0; i < outputs.size(); ++i) // Visit this axis of the dispatch or current workgroup.
            outputs[i]->latest = detail::authority::host; // Record which backend now owns the newest resource values.
    } // End this scope in runtime.
} // End this scope in runtime.

/** @brief One dispatch retaining a reference to its shader. Keep that shader alive during execution. */
struct stage { // Keep the stage values and operations together.
    kernel* shader; // Keep the shader used by this dispatch.
    extent groups; // Keep the number of workgroups to dispatch.
    backend target; // Keep the chosen CPU or GPU backend.
    std::shared_ptr<kernel> owner; // Retain an owned kernel while this stage uses it.
    /** @brief Retain a kernel reference with workgroup counts and a backend override. */
    stage(kernel& k, extent g, backend b) : shader(&k), groups(g), target(b) { // Keep the shader reference, work group counts and backend choice.
    } // End this scope in runtime.
    stage(std::shared_ptr<kernel> k, extent g, backend b) // Keep the shader reference, work group counts and backend choice.
        : shader(k.get()), groups(g), target(b), owner(k) { // Retain the kernel owner together with dispatch dimensions and backend choice.
        if (!shader) // Require a retained kernel for this dispatch.
            throw error("cannot dispatch a null shader"); // Require a valid kernel for this dispatch.
    } // End this scope in runtime.
}; // End this scope in runtime.
/** @brief Build a dispatch using work group counts, with an optional per-stage backend override. */
inline stage dispatch(kernel& shader, extent groups, backend b = backend::automatic) { // Build one ordered dispatch with the requested backend choice.
    return stage(shader, groups, b); // Return the result of this check or operation.
} // End this scope in runtime.
/** @brief Build a dispatch that also retains ownership of its shader. */
inline stage // Return a dispatch value that retains its shader owner.
dispatch(std::shared_ptr<kernel> shader, extent groups, backend b = backend::automatic) { // Build one ordered dispatch with the requested backend choice.
    return stage(shader, groups, b); // Return the result of this check or operation.
} // End this scope in runtime.

class submission; // Keep the submission values and operations together.

/** @brief An ordered pipeline, composed with >> or +=. Resources link stages by shared storage.
 * GPU stages issue memory barriers. CPU/GPU transitions transfer authoritative resource contents.
 * Running again observes updated uniforms and resource bindings. CPU execution
 * completes each stage before returning. GPU execution queues stages in order.
 * Request host data or use an explicit wait to observe GPU completion.
 */
class pipeline { // Keep the pipeline values and operations together.
    std::vector<stage> stages_; // Keep the dispatches in execution order.
    void execute(device* d) const { // Run stages in order while honoring their backend choices.
        for (std::size_t i = 0; i < stages_.size(); ++i) { // Visit every dispatch in processing order.
            const stage& s = stages_[i]; // Inspect the next dispatch, its dimensions and backend choice.
            if (s.target == backend::gpu && !d) // Require a device for an explicitly GPU stage.
                throw error("GPU stage requires a device"); // Report gpu stage requires a device.
            s.shader->run(s.groups, s.target == backend::cpu ? 0 : d); // Dispatch this stage on its selected CPU or GPU backend.
        } // End this scope in execute.
    } // End this scope in execute.

  public: // Expose the operations callers can use.
    /** @brief Construct an empty pipeline. append dispatches with operator+=. */
    pipeline() { // Create the ordered dispatch container.
    } // End this scope in runtime.
    /** @brief Construct a pipeline containing one dispatch. */
    pipeline(stage s) { // Create a pipeline containing this initial dispatch.
        stages_.push_back(s); // Keep this dispatch at the end of the processing order.
    } // End this scope in runtime.
    /** @brief Append a dispatch in execution order and return this pipeline. */
    pipeline& operator+=(stage s) { // Append one dispatch while retaining its optional kernel owner.
        stages_.push_back(s); // Keep this dispatch at the end of the processing order.
        return *this; // Allow another operation to follow in the same expression.
    } // End this scope in runtime.
    /** @brief Append every dispatch from another pipeline, including a safe copy of this pipeline. */
    pipeline& operator+=(const pipeline& other) { // Compose complete processing blocks in their original stage order.
        const std::vector<stage> additions(other.stages_); // Preserve the source range before self-append can reallocate it.
        stages_.insert(stages_.end(), additions.begin(), additions.end()); // Retain each dispatch's kernel ownership and backend choice.
        return *this; // Allow another stage or pipeline to follow in the same expression.
    } // Appending copies dispatch references and preserves the stages for later execution.
    /** @brief Return the number of stages. */
    std::size_t size() const { // Read the stored element or stage count.
        return stages_.size(); // Return the result of this check or operation.
    } // End this scope in runtime.
    /** @brief Run automatic and CPU stages on the CPU. explicit GPU stages need a device. */
    void run() const { // Execute the pipeline using the selected CPU or GPU backend.
        execute(0); // Run stages in order while honoring their backend choices.
    } // End this scope in runtime.
    /** @brief Run automatic and GPU stages on the device, respecting explicit CPU stages.
     * Host readback completes queued GPU writes when resource data is requested.
     */
    void run(device& d) const { // Execute the pipeline using the selected CPU or GPU backend.
        execute(&d); // Run stages in order while honoring their backend choices.
    } // End this scope in runtime.
    /** @brief Compile GPU programs and upload resources before steady-state submission. */
    void prepare(device& d) const { // Compile the program and prepare its bound GPU resources.
        for (std::size_t i = 0; i < stages_.size(); ++i) // Visit every dispatch in processing order.
            if (stages_[i].target != backend::cpu) // Precompile and upload stages that can execute on the GPU.
                stages_[i].shader->prepare(d); // Compile the program and prepare its bound GPU resources.
    } // End this scope in prepare.
    /** @brief Enqueue GPU stages and return a fence for zero-timeout completion polls. */
    submission submit(device& d) const; // Queue compute work and return its completion handle.
    /** @internal Access dispatches for submission and snapshot handling. */
    const std::vector<stage>& stages() const { // Expose the ordered dispatches for submission and snapshot handling.
        return stages_; // Return the dispatches in execution order.
    } // End this scope in runtime.
    /** @brief Snapshot shader instances, uniforms and bindings for deferred CPU execution. */
    pipeline snapshot() const { // Clone kernels and retain the current binding and uniform values.
        pipeline result; // Keep the value returned by this operation.
        std::map<kernel*, std::shared_ptr<kernel>> copies; // Keep the shader clones reused by repeated stages.
        for (std::size_t i = 0; i < stages_.size(); ++i) { // Visit every dispatch in processing order.
            const stage& s = stages_[i]; // Inspect the next dispatch, its dimensions and backend choice.
            auto& copy = copies[s.shader]; // Reuse one clone for every occurrence of the same source kernel.
            if (!copy) // Create a clone only on the source kernel's first occurrence.
                copy = std::shared_ptr<kernel>(s.shader->clone().release()); // Own an isolated clone of this kernel's current interfaces.
            result += stage(copy, s.groups, s.target); // Retain the cloned kernel with its original dimensions and backend choice.
        } // End this scope in snapshot.
        return result; // Return the value returned by this operation.
    } // End this scope in snapshot.
    /** @brief Validate CPU execution support and current host data for a deferred job. */
    void require_host_ready() const { // Validate CPU stages and host-ready resources.
        for (std::size_t i = 0; i < stages_.size(); ++i) { // Visit every dispatch in processing order.
            if (stages_[i].target == backend::gpu) // Reject GPU-only stages before submitting work to a CPU worker.
                throw error("CPU executor cannot execute a GPU-only stage"); // Require CPU-compatible stages for the CPU executor.
            auto resources = stages_[i].shader->resources(); // Inspect the resources bound to this stage.
            for (std::size_t r = 0; r < resources.size(); ++r) // Check every bound resource for data requiring GPU readback.
                if (resources[r]->latest == detail::authority::gpu) // Require the caller to download newer GPU contents on the context thread.
                    throw error( // Report the invalid operation.
                        "download GPU data on its context thread before submitting CPU work"); // Explain how to make this resource ready for CPU execution.
        } // End this scope in require host ready.
    } // End this scope in require host ready.
}; // End this scope in runtime.
/** @brief Compose ordered stages into a reusable pipeline. */
inline pipeline operator>>(stage a, stage b) { // Compose these dispatches in their execution order.
    pipeline p(a); // Start the pipeline with the first dispatch.
    p += b; // Append the following dispatch to the ordered pipeline.
    return p; // Return the composed pipeline for later execution.
} // End this scope in runtime.
/** @brief Append a stage to a pipeline value, retaining references to the same kernels. */
inline pipeline operator>>(pipeline p, stage b) { // Compose these dispatches in their execution order.
    p += b; // Append the following dispatch to the ordered pipeline.
    return p; // Return the composed pipeline for later execution.
} // End this scope in runtime.
/** @brief Prepend a stage to a pipeline while retaining the same linked shader resources. */
inline pipeline operator>>(stage a, const pipeline& b) { // Place the independent dispatch before all existing processing stages.
    pipeline result(a); // Start with the supplied dispatch and its retained kernel owner.
    result += b; // Append the complete processing block in its original order.
    return result; // Return the reusable pipeline for later execution.
} // The source pipeline remains available for further composition or execution.
/** @brief Join two complete processing pipelines in execution order. */
inline pipeline operator>>(pipeline a, const pipeline& b) { // Compose reusable algorithm blocks with ordinary shader dispatches.
    a += b; // Append a copy of the second block's dispatch references.
    return a; // Return the combined ordered processing plan.
} // Resource sharing follows the same rules as individual stage composition.

namespace detail { // Group these definitions in the detail namespace.
typedef kernel* (*kernel_factory)(); // Name the representation used by this GL or shader interface.
inline std::map<std::string, std::vector<kernel_factory>>& shader_registry() { // Share the compiled shader factories discovered during startup.
    static std::map<std::string, std::vector<kernel_factory>> registry; // Keep the shared display lifetime registry.
    return registry; // Return the shared display lifetime registry.
} // End this scope in shader registry.
 // Generated shader translation units register factories directly in their target.
 // Unregistering makes plugin unload safe for future lookups. existing shader objects
 // still require the plugin to stay loaded through destruction.
class shader_registration { // Keep the shader registration values and operations together.
    std::string name_; // Keep the kernel or registration name.
    kernel_factory factory_; // Keep the compiled shader construction callback.

  public: // Expose the operations callers can use.
    shader_registration(const char* name, kernel_factory factory) : name_(name), factory_(factory) { // Keep the callback used to create a registered shader.
        shader_registry()[name_].push_back(factory_); // Retain this entry in the ordered resource or work collection.
    } // End this scope in runtime.
    ~shader_registration() { // Unregister this factory before its dynamic module is unloaded.
        std::vector<kernel_factory>& entries = shader_registry()[name_]; // Find the registered factories using this shader name.
        const std::vector<kernel_factory>::iterator it = // Find this module's factory among the registrations.
            std::find(entries.begin(), entries.end(), factory_); // Keep the compiled shader construction callback.
        if (it != entries.end()) // Choose this branch when it differs from entries.end().
            entries.erase(it); // Remove this module's registration and preserve the other factories.
        if (entries.empty()) // Choose this branch when entries.empty().
            shader_registry().erase(name_); // Share the compiled shader factories discovered during startup.
    } // End this scope in runtime.
    shader_registration(const shader_registration&) = delete; // Prevent copying this object and duplicating its retained identity.
    shader_registration& operator=(const shader_registration&) = delete; // Prevent copying this object and duplicating its retained identity.
}; // End this scope in runtime.
} // namespace detail

/** @brief Create a shader compiled and embedded by shdr_target_shaders during the CMake build.
 * A file named scale.cpp registers scale_kernel. The returned instance owns its CPU/GPU program
 * and exposes resource binding through bind() and set(). Both programs are part of the binary.
 * Unknown or ambiguous names throw error. Call from one application thread after startup.
 */
inline std::unique_ptr<kernel> make_kernel(const std::string& name) { // Create one automatically registered shader by its build-time name.
    const std::map<std::string, std::vector<detail::kernel_factory>>& registry = // Set the shared display lifetime registry.
        detail::shader_registry(); // Share the compiled shader factories discovered during startup.
    const std::map<std::string, std::vector<detail::kernel_factory>>::const_iterator it = // Find the unique compiled factory requested by the application.
        registry.find(name); // Keep the resource or entry-point name.
    if (it == registry.end()) // Choose this branch when it equals registry.end().
        throw error("unknown compiled shader: " + name); // Report unknown compiled shader: .
    if (it->second.size() != 1) // Check resource size or driver limits before accessing storage.
        throw error("ambiguous compiled shader name: " + name); // Report ambiguous compiled shader name: .
    return std::unique_ptr<kernel>(it->second.front()()); // Create one automatically registered shader by its build-time name.
} // End this scope in make kernel.
} // namespace shdr

 // A short namespace alias keeps host-side mathematical expressions readable.
namespace glsl = shdr::glsl; // Group these definitions in the glsl namespace.
#endif // SHDR_RUNTIME_HPP
