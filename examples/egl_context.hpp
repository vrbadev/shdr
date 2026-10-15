#ifndef SHDR_EXAMPLES_EGL_CONTEXT_HPP // Open this include guard or optional feature branch.
#define SHDR_EXAMPLES_EGL_CONTEXT_HPP // Define SHDR_EXAMPLES_EGL_CONTEXT_HPP for the declarations that follow.

#include "shdr.h" // Include shdr.h declarations.

#if defined(SHDR_ENABLE_EGL) && SHDR_ENABLE_EGL // Select the platform or build feature required by this branch.
namespace shdr_examples { // Group these helpers within their namespace.

/// Compatibility names shared by the example command-line runners and GL tests.
using egl_context = shdr::headless_context; // Give this type or declaration a convenient local name.
using context_unavailable = shdr::backend_unavailable; // Give this type or declaration a convenient local name.

} // namespace shdr_examples
#endif // Finish this conditional compilation branch.
#endif // Finish this conditional compilation branch.
