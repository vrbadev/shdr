#ifndef SHDR_EGL_HPP // Skip this section if it was already included.
#define SHDR_EGL_HPP /* Mark this section as included. */

 // Enable this optional backend before including shdr.h and link an EGL library.
#if defined(SHDR_ENABLE_EGL) && SHDR_ENABLE_EGL // Compile this branch only when its platform or backend condition holds.
#include <EGL/egl.h> // Load the EGL display, context, and surface API.
#include <EGL/eglext.h> // Use the available EGL context and platform extensions.

namespace shdr { // Group these definitions in the shdr namespace.

/** @brief The requested graphics backend or context is unavailable.
 * This exception identifies display, configuration, context and GL entry-point
 * failures. Shader compilation and dispatch errors remain ordinary shdr::error.
 */
class backend_unavailable : public error { // Keep the backend unavailable values and operations together.
  public: // Expose the operations callers can use.
    /** @brief Describe the unavailable graphics infrastructure. */
    explicit backend_unavailable(const std::string& message) : error(message) { // Keep the graphics initialization failure as an ordinary exception.
    } // End this scope in egl.
}; // End this scope in egl.

namespace detail { // Group these definitions in the detail namespace.

inline backend_unavailable egl_failure(const char* operation) { // Build a readable diagnostic from the EGL error code.
    const EGLint code = eglGetError(); // Read the EGL error belonging to the failed context operation.
    std::ostringstream message; // Build a readable diagnostic from the EGL error code.
    message << operation << " unavailable (EGL error 0x" << std::hex << code << ')'; // Build a readable diagnostic from the EGL error code.
    return backend_unavailable(message.str()); // Build a readable diagnostic from the EGL error code.
} // End this scope in egl failure.

 // Initialization and the last termination share a lock, including during weak-pointer expiry.
struct egl_display_registry { // Keep the egl display registry values and operations together.
    std::mutex mutex; // Keep the lock protecting shared state.
    EGLDisplay display; // Keep the initialized EGL display.
    std::size_t owners; // Keep the number of contexts retaining this display.
    egl_display_registry() : display(EGL_NO_DISPLAY), owners(0) { // Start the initialized-display owner count at zero.
    } // End this scope in egl.
}; // End this scope in egl.
inline egl_display_registry& egl_displays() { // Share the display-lifetime registry among headless devices.
    static egl_display_registry registry; // Keep the shared display lifetime registry.
    return registry; // Return the shared display lifetime registry.
} // End this scope in egl displays.

 // Contexts share one initialized display. the last context releases it.
struct egl_display_state { // Keep the egl display state values and operations together.
    EGLDisplay display; // Keep the initialized EGL display.
    egl_display_state() : display(EGL_NO_DISPLAY) { // Retain one initialized EGL display for living contexts.
        egl_display_registry& registry = egl_displays(); // Set the shared display lifetime registry.
        std::lock_guard<std::mutex> lock(registry.mutex); // Hold the mutex while this shared state is accessed.
        if (registry.owners) { // Reuse this display when another context already retains its initialization.
            display = registry.display; // Set the initialized EGL display.
            ++registry.owners; // Retain the existing initialized display for this new owner.
            return; // Finish this operation.
        } // End this scope in egl display state.
        typedef EGLDisplay(EGLAPIENTRYP platform_display_fn)(EGLenum, void*, const EGLint*); // Name the representation used by this GL or shader interface.
        const platform_display_fn platform_display = // Resolve the optional surfaceless display entry point.
            reinterpret_cast<platform_display_fn>(eglGetProcAddress("eglGetPlatformDisplayEXT")); // Look up the requested GL entry point through EGL.
        if (platform_display) { // Choose this branch when platform display.
 // Mesa's surfaceless platform creates an offscreen display on supported drivers.
            display = platform_display(0x31DD, EGL_DEFAULT_DISPLAY, 0); // Set the initialized EGL display.
            if (display != EGL_NO_DISPLAY && !eglInitialize(display, 0, 0)) // Prepare the default-display fallback if surfaceless initialization fails.
                display = EGL_NO_DISPLAY; // Set the initialized EGL display.
        } // End this scope in egl display state.
        if (display == EGL_NO_DISPLAY) { // Try the default display after the surfaceless path.
            display = eglGetDisplay(EGL_DEFAULT_DISPLAY); // Set the initialized EGL display.
            if (display == EGL_NO_DISPLAY || !eglInitialize(display, 0, 0)) { // Report failure to acquire and initialize the default display.
                display = EGL_NO_DISPLAY; // Set the initialized EGL display.
                throw egl_failure("EGL display"); // Report egl display.
            } // End this scope in egl display state.
        } // End this scope in egl display state.
        registry.display = display; // Set the initialized EGL display.
        registry.owners = 1; // Set the number of contexts retaining this display.
    } // End this scope in egl display state.
    ~egl_display_state() { // Release this object's owned resources at teardown.
        egl_display_registry& registry = egl_displays(); // Set the shared display lifetime registry.
        std::lock_guard<std::mutex> lock(registry.mutex); // Hold the mutex while this shared state is accessed.
        if (display != EGL_NO_DISPLAY && --registry.owners == 0) { // Terminate an initialized display after its final context owner releases it.
            eglTerminate(display); // Terminate the display after its last context owner is released.
            registry.display = EGL_NO_DISPLAY; // Set the initialized EGL display.
        } // End this scope in egl display state.
    } // End this scope in egl display state.
    egl_display_state(const egl_display_state&) = delete; // Prevent copying this object and duplicating its retained identity.
    egl_display_state& operator=(const egl_display_state&) = delete; // Prevent copying this object and duplicating its retained identity.
}; // End this scope in egl.

inline std::shared_ptr<egl_display_state> acquire_egl_display() { // Reuse the initialized display while any context retains it.
    static std::mutex mutex; // Keep the lock protecting shared state.
    static std::weak_ptr<egl_display_state> cached; // Keep the weak reference to the shared EGL display.
    std::lock_guard<std::mutex> lock(mutex); // Hold the mutex while this shared state is accessed.
    std::shared_ptr<egl_display_state> display = cached.lock(); // Set the initialized EGL display.
    if (!display) { // Create the shared display state for its first owner.
        display = std::make_shared<egl_display_state>(); // Set the initialized EGL display.
        cached = display; // Set the weak reference to the shared EGL display.
    } // End this scope in acquire egl display.
    return display; // Return the initialized EGL display.
} // End this scope in acquire egl display.

struct egl_context_state { // Retain one headless EGL display, context, and pbuffer.
    std::shared_ptr<egl_display_state> display; // Keep the initialized EGL display.
    EGLContext context; // Keep the EGL context used for compute commands.
    EGLSurface surface; // Keep the one-pixel EGL pbuffer.
    EGLenum api; // Keep the selected desktop GL or GLES API.
    std::thread::id thread; // Keep the thread that created this EGL context.

    explicit egl_context_state(profile p) // Create a pbuffer context for the requested shader profile.
        : display(acquire_egl_display()), context(EGL_NO_CONTEXT), surface(EGL_NO_SURFACE), // Reuse the initialized display while any context retains it.
          api(p == profile::es31 || p == profile::es32 ? EGL_OPENGL_ES_API : EGL_OPENGL_API), // Choose the EGL client API matching the requested shader profile.
          thread(std::this_thread::get_id()) { // Remember the only thread allowed to use this EGL context.
        try { // Keep cleanup available if an allocation, shader, or driver operation fails.
            const bool desktop = p != profile::es31 && p != profile::es32; // Select the API family for the requested profile.
            if (!eglBindAPI(api)) // Check that EGL accepts the selected graphics API.
                throw egl_failure("EGL graphics API"); // Report egl graphics api.
            const EGLint config_attributes[] = {EGL_SURFACE_TYPE, // Request a pbuffer configuration with the chosen client API.
                                                EGL_PBUFFER_BIT, // Request a configuration capable of creating a small offscreen surface.
                                                EGL_RENDERABLE_TYPE, // Select the client API supported by the framebuffer configuration.
                                                desktop ? EGL_OPENGL_BIT : 0x0040, // Request desktop GL or an OpenGL ES 3-capable configuration.
                                                EGL_RED_SIZE, // Specify the number of red bits in the offscreen surface.
                                                8, // Request eight bits for this color component.
                                                EGL_GREEN_SIZE, // Specify the number of green bits in the offscreen surface.
                                                8, // Request eight bits for this color component.
                                                EGL_BLUE_SIZE, // Specify the number of blue bits in the offscreen surface.
                                                8, // Request eight bits for this color component.
                                                EGL_NONE}; // Create a pbuffer context for the requested shader profile.
            EGLConfig config = 0; // Set the selected EGL framebuffer configuration.
            EGLint count = 0; // Set the element or configuration count.
            if (!eglChooseConfig(display->display, config_attributes, &config, 1, &count) || !count) // Check resource size or driver limits before accessing storage.
                throw egl_failure("EGL framebuffer configuration"); // Report egl framebuffer configuration.
            const EGLint major = desktop ? 4 : 3; // Set the GL major version.
            const EGLint minor = p == profile::es31   ? 1 // Set the GL minor version.
                                 : p == profile::es32 ? 2 // Select the version associated with this shader profile.
                                 : p == profile::gl43 ? 3 // Select the version associated with this shader profile.
                                 : p == profile::gl45 ? 5 // Select the version associated with this shader profile.
                                                      : 6; // Select minor version six for the desktop GL 4.6 profile.
 // These KHR_create_context attributes are also core in EGL 1.5.
            const EGLint desktop_attributes[] = { // Request the selected desktop version and core profile.
                0x3098, major, 0x30FB, minor, 0x30FD, 0x00000001, EGL_NONE}; // Create a pbuffer context for the requested shader profile.
            const EGLint es_attributes[] = {0x3098, major, 0x30FB, minor, EGL_NONE}; // Request the selected GLES major and minor versions.
            context = eglCreateContext(display->display, // Create a compute context for the requested GL or GLES version.
                                       config, // Keep the selected EGL framebuffer configuration.
                                       EGL_NO_CONTEXT, // Create a context with its own GL object namespace.
                                       desktop ? desktop_attributes : es_attributes); // Create a pbuffer context for the requested shader profile.
            if (context == EGL_NO_CONTEXT) // Check that the driver created the requested context.
                throw egl_failure("Requested GL context"); // Report requested gl context.
            const EGLint surface_attributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE}; // Create a one-pixel offscreen pbuffer for compute commands.
            surface = eglCreatePbufferSurface(display->display, config, surface_attributes); // Set the one-pixel EGL pbuffer.
            if (surface == EGL_NO_SURFACE) // Check that the driver created the offscreen pbuffer.
                throw egl_failure("EGL pbuffer"); // Report egl pbuffer.
            make_current(); // Select this context on its creating thread.
        } catch (...) { // Preserve cleanup or exception propagation for this failed operation.
            release(); // Release this surface and context while preserving other active contexts.
            throw; // Rethrow the original failure after arranging the necessary cleanup.
        } // End this scope in egl context state.
    } // End this scope in egl context state.
    void make_current() const { // Select this context on its creating thread.
        if (std::this_thread::get_id() != thread) // Check that the operation uses the context or lifecycle thread.
            throw error("headless EGL operations require the context's creating thread"); // Report headless egl operations require the context's creating thread.
        if (!eglBindAPI(api)) // Check that EGL accepts the selected graphics API.
            throw egl_failure("EGL graphics API"); // Report egl graphics api.
        if (eglGetCurrentContext() == context && eglGetCurrentDisplay() == display->display) // Choose this branch when eglGetCurrentContext() equals context and eglGetCurrentDisplay() equals display->display.
            return; // Finish this operation.
        if (!eglMakeCurrent(display->display, surface, surface, context)) // Check that EGL activates this context and its pbuffer.
            throw egl_failure("Current EGL context"); // Report current egl context.
    } // End this scope in make current.
    void release() { // Release this surface and context while preserving other active contexts.
 // Unbind this context only when it is current on this thread.
        const EGLenum previous_api = eglQueryAPI(); // Set the EGL API current before context release.
        eglBindAPI(api); // Select the desktop GL or GLES API for this EGL operation.
        if (context != EGL_NO_CONTEXT && eglGetCurrentContext() == context) // Unbind this context when it is current.
            eglMakeCurrent(display->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // Make this pbuffer context current on its owning thread.
        if (surface != EGL_NO_SURFACE) // Destroy the pbuffer allocated by this state.
            eglDestroySurface(display->display, surface); // Release the EGL object after its context-thread owners finish.
        if (context != EGL_NO_CONTEXT) // Destroy the context allocated by this state.
            eglDestroyContext(display->display, context); // Release the EGL object after its context-thread owners finish.
        if (previous_api != api) // Choose this branch when previous api differs from api.
            eglBindAPI(previous_api); // Restore the API binding that belonged to the other live context.
        surface = EGL_NO_SURFACE; // Set the one-pixel EGL pbuffer.
        context = EGL_NO_CONTEXT; // Set the EGL context used for compute commands.
    } // End this scope in release.
    ~egl_context_state() { // Release this object's owned resources at teardown.
        release(); // Release this surface and context while preserving other active contexts.
    } // End this scope in egl context state.
    egl_context_state(const egl_context_state&) = delete; // Prevent copying this object and duplicating its retained identity.
    egl_context_state& operator=(const egl_context_state&) = delete; // Prevent copying this object and duplicating its retained identity.
}; // End this scope in egl.

inline std::shared_ptr<egl_context_state> make_egl_context(profile p) { // Create and retain the headless EGL context.
    return std::make_shared<egl_context_state>(p); // Create and retain the headless EGL context.
} // End this scope in make egl context.
} // namespace detail

/** @brief An optional, headless EGL pbuffer context for a selected compute profile.
 * Define SHDR_ENABLE_EGL=1 before including shdr.h and link EGL to use this class.
 * It prefers a surfaceless display and otherwise uses EGL's default display.
 * Create, use and finally release the context on its creating thread. Displays
 * are shared among living shdr contexts. Let shdr manage their display lifetime.
 */
class headless_context { // Keep the headless context values and operations together.
    std::shared_ptr<detail::egl_context_state> context_; // Keep the retained headless context state.

  public: // Expose the operations callers can use.
    /** @brief Create a 1-by-1 pbuffer and make its context current.
     * @param p Minimum GL or GLES version. defaults to OpenGL ES 3.1.
     * @throws backend_unavailable If backend initialization fails.
     */
    explicit headless_context(profile p = profile::es31) : context_(detail::make_egl_context(p)) { // Create and retain an EGL context for the requested compute profile.
    } // End this scope in egl.
    /** @brief Make this context current on its creating thread. */
    void make_current() const { // Select this context on its creating thread.
        context_->make_current(); // Select this context on its creating thread.
    } // End this scope in make current.
    /** @brief Resolve a GL entry point through EGL, for shdr::device or external GL calls. */
    static void* resolve(const char* name) { // Resolve a named GL function through EGL.
        const __eglMustCastToProperFunctionPointerType function = eglGetProcAddress(name); // Look up the requested GL entry point through EGL.
        static_assert(sizeof(function) == sizeof(void*), // Require a representation that matches the shader interface.
                      "EGL function addresses must fit in void*"); // Require EGL function addresses to fit the portable resolver representation.
        void* address = 0; // Set the resolved GL function address.
        std::memcpy(&address, &function, sizeof(address)); // Copy the representation with the alias-safe byte-copy operation.
        return address; // Return the resolved GL function address.
    } // End this scope in resolve.
    headless_context(const headless_context&) = delete; // Prevent copying this object and duplicating its retained identity.
    headless_context& operator=(const headless_context&) = delete; // Prevent copying this object and duplicating its retained identity.
}; // End this scope in egl.

/** @brief A compute device that creates and retains its own headless EGL context.
 * Define SHDR_ENABLE_EGL=1 and link EGL. Construct this device on the thread that
 * performs its GPU operations. Its context remains alive while GPU resources,
 * programs and submissions retain the device state. release those objects on
 * that thread before shutdown. CPU execution uses host resources.
 */
class headless_device : public device { // Keep the headless device values and operations together.
    std::shared_ptr<detail::egl_context_state> context_; // Keep the retained headless context state.
    headless_device(const std::shared_ptr<detail::egl_context_state>& context, profile p) try // Connect the retained EGL context to the GL runtime.
        : device(headless_context::resolve, p), context_(context) { // Retain a validated GL device state.
        state()->context_keepalive = context; // Set the retained headless context.
        state()->activate_context = [context] { // Set the callback that makes this device context current.
            context->make_current(); // Select this context on its creating thread.
        }; // End this scope in state.
    } catch (const backend_unavailable&) { // Clean up objects allocated before the operation failed.
        throw; // Rethrow the original failure after arranging the necessary cleanup.
    } catch (const error& failure) { // Translate device-initialization failures into an unavailable-backend result.
        throw backend_unavailable(failure.what()); // Preserve the initialization diagnostic in the backend exception.
    } // End this scope in egl.

  public: // Expose the operations callers can use.
    /** @brief Create a headless context and initialize its compute device.
     * @param p Minimum shader and context profile. defaults to OpenGL ES 3.1.
     * @throws backend_unavailable If context creation or entry-point loading fails.
     */
    explicit headless_device(profile p = profile::es31) // Create a compute device with its own retained EGL context.
        : headless_device(detail::make_egl_context(p), p) { // Connect the retained EGL context to the GL runtime.
    } // End this scope in headless device.
    /** @brief Make the owned context current for external GL interoperability. */
    void make_current() const { // Select this context on its creating thread.
        context_->make_current(); // Select this context on its creating thread.
    } // End this scope in make current.
    headless_device(const headless_device&) = delete; // Prevent copying this object and duplicating its retained identity.
    headless_device& operator=(const headless_device&) = delete; // Prevent copying this object and duplicating its retained identity.
}; // End this scope in egl.

} // namespace shdr
#endif // SHDR_ENABLE_EGL
#endif // SHDR_EGL_HPP
