#ifndef SHDR_MACROS_HPP // Include these helper macros only once.
#define SHDR_MACROS_HPP // Remember that the helper macros are available.

/** @brief Begin one shared shader declaration with a fixed local work group size. */
#define SHDR_BEGIN(name, x, y, z) /* Start the shared shader type and its CPU snapshot method. */ \
    using namespace ::shdr::glsl; /* Make the GLSL math names available inside this shader. */ \
    struct name : ::shdr::kernel { /* Give the shared shader the CPU kernel interface. */ \
        name() : ::shdr::kernel(#name, ::shdr::extent(x, y, z)) { /* Store the shader name and fixed local work group size. */ \
        } /* Close the generated method or shader declaration. */ \
        std::unique_ptr<::shdr::kernel> clone() const override { /* Build an independent CPU instance for deferred execution. */ \
            std::unique_ptr<name> copy(new name); /* Allocate the shader instance that will hold the snapshot. */ \
            copy_interfaces(*copy); /* Copy uniform values and resource bindings into that instance. */ \
            return std::unique_ptr<::shdr::kernel>(copy.release()); /* Return ownership through the common kernel interface. */ \
        } /* Close the generated method or shader declaration. */
/** @brief End a shared shader declaration. */
#define SHDR_END() /* Close the shared shader type. */ \
    } /* Close the generated method or shader declaration. */ \
    ; /* Finish the generated shader type declaration. */
/** @brief Declare the shared entry point, executed once per invocation on CPU or GPU. */
#define SHDR_MAIN() void main() override // Make the shader entry point override the CPU kernel method.
/** @brief Declare a typed std430 SSBO with readonly/writeonly/readwrite access. */
#define SHDR_BUFFER(binding, type, name, mode) /* Create a named storage buffer port with the chosen access mode. */ \
    ::shdr::buffer_port<type, ::shdr::access::mode> name{*this, binding, #name}; /* Register this typed buffer port with its owning shader. */
/** @brief Declare a typed scalar, vector or float matrix uniform. */
#define SHDR_UNIFORM(type, name) /* Create a uniform value and its binding descriptor. */ \
    type name{}; /* Initialize the uniform value to its default value. */ \
    ::shdr::uniform_port<type> shdr_uniform_##name{*this, #name, name}; /* Register the uniform so host code can set it by name. */
/** @brief Declare a 2D storage image. Supported formats are r32f/i/ui and rgba32f/i/ui. */
#define SHDR_IMAGE2D(binding, format, type, name, mode) /* Choose the matching image scalar type and component count. */ \
    SHDR_DETAIL_IMAGE_##type(::shdr::access::mode, SHDR_DETAIL_FORMAT_##format, name, binding) /* Expand the selected image helper with its access and binding. */
#define SHDR_DETAIL_IMAGE_image2D(mode, format, name, binding) /* Define the CPU image port for this GLSL image type. */ \
    ::shdr::image_port<float, format, mode> name{*this, binding, #name}; /* Register the image port with its owning shader. */
#define SHDR_DETAIL_IMAGE_iimage2D(mode, format, name, binding) /* Define the CPU image port for this GLSL image type. */ \
    ::shdr::image_port<int, format, mode> name{*this, binding, #name}; /* Register the image port with its owning shader. */
#define SHDR_DETAIL_IMAGE_uimage2D(mode, format, name, binding) /* Define the CPU image port for this GLSL image type. */ \
    ::shdr::image_port<unsigned, format, mode> name{*this, binding, #name}; /* Register the image port with its owning shader. */
#define SHDR_DETAIL_FORMAT_r32f 1 // Use one scalar component for this image format.
#define SHDR_DETAIL_FORMAT_r32i 1 // Use one scalar component for this image format.
#define SHDR_DETAIL_FORMAT_r32ui 1 // Use one scalar component for this image format.
#define SHDR_DETAIL_FORMAT_rgba32f 4 // Use four scalar components for this image format.
#define SHDR_DETAIL_FORMAT_rgba32i 4 // Use four scalar components for this image format.
#define SHDR_DETAIL_FORMAT_rgba32ui 4 // Use four scalar components for this image format.
/** @brief GLSL-style component selection, with safe simultaneous CPU swizzle assignment.
 * Use xyzw, rgba or stpq consistently. Duplicate components are read-only.
 */
#define SHDR_SWIZZLE(value, pattern) (value).template swizzle<SHDR_DETAIL_SWIZZLE_##pattern>() // Translate component letters into a safe CPU component selection.
/** @brief Portable spelling of GLSL's boolean vector not builtin. */
#define SHDR_NOT(value) ::shdr::glsl::logicalNot(value) // Invert every component of a boolean vector.
/** @brief Declare a shared helper function input parameter. */
#define SHDR_IN(type, name) const type& name // Pass a helper input through a const reference.
/** @brief Declare a shared helper function output parameter. */
#define SHDR_OUT(type, name) type& name // Let the helper write its output argument.
/** @brief Declare a shared helper function input/output parameter. */
#define SHDR_INOUT(type, name) type& name // Let the helper read and update its argument.
#define SHDR_DETAIL_SWIZZLE_x 0 // Select the x component.
#define SHDR_DETAIL_SWIZZLE_y 1 // Select the y component.
#define SHDR_DETAIL_SWIZZLE_z 2 // Select the z component.
#define SHDR_DETAIL_SWIZZLE_w 3 // Select the w component.
#define SHDR_DETAIL_SWIZZLE_xx 0, 0 // Select x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xy 0, 1 // Select x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xz 0, 2 // Select x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xw 0, 3 // Select x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yx 1, 0 // Select y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yy 1, 1 // Select y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yz 1, 2 // Select y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yw 1, 3 // Select y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zx 2, 0 // Select z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zy 2, 1 // Select z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zz 2, 2 // Select z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zw 2, 3 // Select z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wx 3, 0 // Select w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wy 3, 1 // Select w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wz 3, 2 // Select w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_ww 3, 3 // Select w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xxx 0, 0, 0 // Select x, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xxy 0, 0, 1 // Select x, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xxz 0, 0, 2 // Select x, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xxw 0, 0, 3 // Select x, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xyx 0, 1, 0 // Select x, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xyy 0, 1, 1 // Select x, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xyz 0, 1, 2 // Select x, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xyw 0, 1, 3 // Select x, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xzx 0, 2, 0 // Select x, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xzy 0, 2, 1 // Select x, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xzz 0, 2, 2 // Select x, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xzw 0, 2, 3 // Select x, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xwx 0, 3, 0 // Select x, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xwy 0, 3, 1 // Select x, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xwz 0, 3, 2 // Select x, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xww 0, 3, 3 // Select x, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yxx 1, 0, 0 // Select y, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yxy 1, 0, 1 // Select y, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yxz 1, 0, 2 // Select y, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yxw 1, 0, 3 // Select y, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yyx 1, 1, 0 // Select y, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yyy 1, 1, 1 // Select y, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yyz 1, 1, 2 // Select y, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yyw 1, 1, 3 // Select y, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yzx 1, 2, 0 // Select y, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yzy 1, 2, 1 // Select y, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yzz 1, 2, 2 // Select y, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yzw 1, 2, 3 // Select y, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_ywx 1, 3, 0 // Select y, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_ywy 1, 3, 1 // Select y, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_ywz 1, 3, 2 // Select y, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yww 1, 3, 3 // Select y, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zxx 2, 0, 0 // Select z, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zxy 2, 0, 1 // Select z, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zxz 2, 0, 2 // Select z, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zxw 2, 0, 3 // Select z, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zyx 2, 1, 0 // Select z, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zyy 2, 1, 1 // Select z, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zyz 2, 1, 2 // Select z, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zyw 2, 1, 3 // Select z, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zzx 2, 2, 0 // Select z, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zzy 2, 2, 1 // Select z, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zzz 2, 2, 2 // Select z, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zzw 2, 2, 3 // Select z, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zwx 2, 3, 0 // Select z, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zwy 2, 3, 1 // Select z, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zwz 2, 3, 2 // Select z, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zww 2, 3, 3 // Select z, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wxx 3, 0, 0 // Select w, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wxy 3, 0, 1 // Select w, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wxz 3, 0, 2 // Select w, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wxw 3, 0, 3 // Select w, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wyx 3, 1, 0 // Select w, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wyy 3, 1, 1 // Select w, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wyz 3, 1, 2 // Select w, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wyw 3, 1, 3 // Select w, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wzx 3, 2, 0 // Select w, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wzy 3, 2, 1 // Select w, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wzz 3, 2, 2 // Select w, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wzw 3, 2, 3 // Select w, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wwx 3, 3, 0 // Select w, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wwy 3, 3, 1 // Select w, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wwz 3, 3, 2 // Select w, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_www 3, 3, 3 // Select w, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xxxx 0, 0, 0, 0 // Select x, then x, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xxxy 0, 0, 0, 1 // Select x, then x, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xxxz 0, 0, 0, 2 // Select x, then x, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xxxw 0, 0, 0, 3 // Select x, then x, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xxyx 0, 0, 1, 0 // Select x, then x, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xxyy 0, 0, 1, 1 // Select x, then x, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xxyz 0, 0, 1, 2 // Select x, then x, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xxyw 0, 0, 1, 3 // Select x, then x, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xxzx 0, 0, 2, 0 // Select x, then x, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xxzy 0, 0, 2, 1 // Select x, then x, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xxzz 0, 0, 2, 2 // Select x, then x, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xxzw 0, 0, 2, 3 // Select x, then x, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xxwx 0, 0, 3, 0 // Select x, then x, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xxwy 0, 0, 3, 1 // Select x, then x, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xxwz 0, 0, 3, 2 // Select x, then x, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xxww 0, 0, 3, 3 // Select x, then x, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xyxx 0, 1, 0, 0 // Select x, then y, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xyxy 0, 1, 0, 1 // Select x, then y, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xyxz 0, 1, 0, 2 // Select x, then y, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xyxw 0, 1, 0, 3 // Select x, then y, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xyyx 0, 1, 1, 0 // Select x, then y, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xyyy 0, 1, 1, 1 // Select x, then y, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xyyz 0, 1, 1, 2 // Select x, then y, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xyyw 0, 1, 1, 3 // Select x, then y, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xyzx 0, 1, 2, 0 // Select x, then y, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xyzy 0, 1, 2, 1 // Select x, then y, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xyzz 0, 1, 2, 2 // Select x, then y, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xyzw 0, 1, 2, 3 // Select x, then y, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xywx 0, 1, 3, 0 // Select x, then y, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xywy 0, 1, 3, 1 // Select x, then y, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xywz 0, 1, 3, 2 // Select x, then y, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xyww 0, 1, 3, 3 // Select x, then y, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xzxx 0, 2, 0, 0 // Select x, then z, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xzxy 0, 2, 0, 1 // Select x, then z, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xzxz 0, 2, 0, 2 // Select x, then z, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xzxw 0, 2, 0, 3 // Select x, then z, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xzyx 0, 2, 1, 0 // Select x, then z, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xzyy 0, 2, 1, 1 // Select x, then z, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xzyz 0, 2, 1, 2 // Select x, then z, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xzyw 0, 2, 1, 3 // Select x, then z, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xzzx 0, 2, 2, 0 // Select x, then z, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xzzy 0, 2, 2, 1 // Select x, then z, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xzzz 0, 2, 2, 2 // Select x, then z, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xzzw 0, 2, 2, 3 // Select x, then z, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xzwx 0, 2, 3, 0 // Select x, then z, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xzwy 0, 2, 3, 1 // Select x, then z, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xzwz 0, 2, 3, 2 // Select x, then z, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xzww 0, 2, 3, 3 // Select x, then z, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xwxx 0, 3, 0, 0 // Select x, then w, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xwxy 0, 3, 0, 1 // Select x, then w, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xwxz 0, 3, 0, 2 // Select x, then w, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xwxw 0, 3, 0, 3 // Select x, then w, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xwyx 0, 3, 1, 0 // Select x, then w, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xwyy 0, 3, 1, 1 // Select x, then w, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xwyz 0, 3, 1, 2 // Select x, then w, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xwyw 0, 3, 1, 3 // Select x, then w, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xwzx 0, 3, 2, 0 // Select x, then w, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xwzy 0, 3, 2, 1 // Select x, then w, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xwzz 0, 3, 2, 2 // Select x, then w, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xwzw 0, 3, 2, 3 // Select x, then w, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_xwwx 0, 3, 3, 0 // Select x, then w, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_xwwy 0, 3, 3, 1 // Select x, then w, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_xwwz 0, 3, 3, 2 // Select x, then w, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_xwww 0, 3, 3, 3 // Select x, then w, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yxxx 1, 0, 0, 0 // Select y, then x, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yxxy 1, 0, 0, 1 // Select y, then x, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yxxz 1, 0, 0, 2 // Select y, then x, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yxxw 1, 0, 0, 3 // Select y, then x, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yxyx 1, 0, 1, 0 // Select y, then x, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yxyy 1, 0, 1, 1 // Select y, then x, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yxyz 1, 0, 1, 2 // Select y, then x, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yxyw 1, 0, 1, 3 // Select y, then x, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yxzx 1, 0, 2, 0 // Select y, then x, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yxzy 1, 0, 2, 1 // Select y, then x, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yxzz 1, 0, 2, 2 // Select y, then x, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yxzw 1, 0, 2, 3 // Select y, then x, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yxwx 1, 0, 3, 0 // Select y, then x, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yxwy 1, 0, 3, 1 // Select y, then x, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yxwz 1, 0, 3, 2 // Select y, then x, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yxww 1, 0, 3, 3 // Select y, then x, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yyxx 1, 1, 0, 0 // Select y, then y, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yyxy 1, 1, 0, 1 // Select y, then y, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yyxz 1, 1, 0, 2 // Select y, then y, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yyxw 1, 1, 0, 3 // Select y, then y, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yyyx 1, 1, 1, 0 // Select y, then y, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yyyy 1, 1, 1, 1 // Select y, then y, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yyyz 1, 1, 1, 2 // Select y, then y, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yyyw 1, 1, 1, 3 // Select y, then y, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yyzx 1, 1, 2, 0 // Select y, then y, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yyzy 1, 1, 2, 1 // Select y, then y, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yyzz 1, 1, 2, 2 // Select y, then y, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yyzw 1, 1, 2, 3 // Select y, then y, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yywx 1, 1, 3, 0 // Select y, then y, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yywy 1, 1, 3, 1 // Select y, then y, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yywz 1, 1, 3, 2 // Select y, then y, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yyww 1, 1, 3, 3 // Select y, then y, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yzxx 1, 2, 0, 0 // Select y, then z, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yzxy 1, 2, 0, 1 // Select y, then z, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yzxz 1, 2, 0, 2 // Select y, then z, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yzxw 1, 2, 0, 3 // Select y, then z, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yzyx 1, 2, 1, 0 // Select y, then z, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yzyy 1, 2, 1, 1 // Select y, then z, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yzyz 1, 2, 1, 2 // Select y, then z, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yzyw 1, 2, 1, 3 // Select y, then z, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yzzx 1, 2, 2, 0 // Select y, then z, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yzzy 1, 2, 2, 1 // Select y, then z, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yzzz 1, 2, 2, 2 // Select y, then z, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yzzw 1, 2, 2, 3 // Select y, then z, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_yzwx 1, 2, 3, 0 // Select y, then z, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_yzwy 1, 2, 3, 1 // Select y, then z, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_yzwz 1, 2, 3, 2 // Select y, then z, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_yzww 1, 2, 3, 3 // Select y, then z, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_ywxx 1, 3, 0, 0 // Select y, then w, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_ywxy 1, 3, 0, 1 // Select y, then w, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_ywxz 1, 3, 0, 2 // Select y, then w, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_ywxw 1, 3, 0, 3 // Select y, then w, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_ywyx 1, 3, 1, 0 // Select y, then w, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_ywyy 1, 3, 1, 1 // Select y, then w, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_ywyz 1, 3, 1, 2 // Select y, then w, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_ywyw 1, 3, 1, 3 // Select y, then w, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_ywzx 1, 3, 2, 0 // Select y, then w, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_ywzy 1, 3, 2, 1 // Select y, then w, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_ywzz 1, 3, 2, 2 // Select y, then w, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_ywzw 1, 3, 2, 3 // Select y, then w, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_ywwx 1, 3, 3, 0 // Select y, then w, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_ywwy 1, 3, 3, 1 // Select y, then w, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_ywwz 1, 3, 3, 2 // Select y, then w, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_ywww 1, 3, 3, 3 // Select y, then w, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zxxx 2, 0, 0, 0 // Select z, then x, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zxxy 2, 0, 0, 1 // Select z, then x, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zxxz 2, 0, 0, 2 // Select z, then x, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zxxw 2, 0, 0, 3 // Select z, then x, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zxyx 2, 0, 1, 0 // Select z, then x, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zxyy 2, 0, 1, 1 // Select z, then x, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zxyz 2, 0, 1, 2 // Select z, then x, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zxyw 2, 0, 1, 3 // Select z, then x, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zxzx 2, 0, 2, 0 // Select z, then x, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zxzy 2, 0, 2, 1 // Select z, then x, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zxzz 2, 0, 2, 2 // Select z, then x, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zxzw 2, 0, 2, 3 // Select z, then x, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zxwx 2, 0, 3, 0 // Select z, then x, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zxwy 2, 0, 3, 1 // Select z, then x, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zxwz 2, 0, 3, 2 // Select z, then x, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zxww 2, 0, 3, 3 // Select z, then x, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zyxx 2, 1, 0, 0 // Select z, then y, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zyxy 2, 1, 0, 1 // Select z, then y, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zyxz 2, 1, 0, 2 // Select z, then y, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zyxw 2, 1, 0, 3 // Select z, then y, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zyyx 2, 1, 1, 0 // Select z, then y, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zyyy 2, 1, 1, 1 // Select z, then y, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zyyz 2, 1, 1, 2 // Select z, then y, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zyyw 2, 1, 1, 3 // Select z, then y, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zyzx 2, 1, 2, 0 // Select z, then y, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zyzy 2, 1, 2, 1 // Select z, then y, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zyzz 2, 1, 2, 2 // Select z, then y, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zyzw 2, 1, 2, 3 // Select z, then y, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zywx 2, 1, 3, 0 // Select z, then y, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zywy 2, 1, 3, 1 // Select z, then y, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zywz 2, 1, 3, 2 // Select z, then y, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zyww 2, 1, 3, 3 // Select z, then y, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zzxx 2, 2, 0, 0 // Select z, then z, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zzxy 2, 2, 0, 1 // Select z, then z, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zzxz 2, 2, 0, 2 // Select z, then z, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zzxw 2, 2, 0, 3 // Select z, then z, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zzyx 2, 2, 1, 0 // Select z, then z, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zzyy 2, 2, 1, 1 // Select z, then z, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zzyz 2, 2, 1, 2 // Select z, then z, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zzyw 2, 2, 1, 3 // Select z, then z, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zzzx 2, 2, 2, 0 // Select z, then z, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zzzy 2, 2, 2, 1 // Select z, then z, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zzzz 2, 2, 2, 2 // Select z, then z, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zzzw 2, 2, 2, 3 // Select z, then z, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zzwx 2, 2, 3, 0 // Select z, then z, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zzwy 2, 2, 3, 1 // Select z, then z, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zzwz 2, 2, 3, 2 // Select z, then z, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zzww 2, 2, 3, 3 // Select z, then z, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zwxx 2, 3, 0, 0 // Select z, then w, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zwxy 2, 3, 0, 1 // Select z, then w, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zwxz 2, 3, 0, 2 // Select z, then w, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zwxw 2, 3, 0, 3 // Select z, then w, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zwyx 2, 3, 1, 0 // Select z, then w, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zwyy 2, 3, 1, 1 // Select z, then w, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zwyz 2, 3, 1, 2 // Select z, then w, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zwyw 2, 3, 1, 3 // Select z, then w, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zwzx 2, 3, 2, 0 // Select z, then w, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zwzy 2, 3, 2, 1 // Select z, then w, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zwzz 2, 3, 2, 2 // Select z, then w, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zwzw 2, 3, 2, 3 // Select z, then w, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_zwwx 2, 3, 3, 0 // Select z, then w, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_zwwy 2, 3, 3, 1 // Select z, then w, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_zwwz 2, 3, 3, 2 // Select z, then w, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_zwww 2, 3, 3, 3 // Select z, then w, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wxxx 3, 0, 0, 0 // Select w, then x, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wxxy 3, 0, 0, 1 // Select w, then x, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wxxz 3, 0, 0, 2 // Select w, then x, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wxxw 3, 0, 0, 3 // Select w, then x, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wxyx 3, 0, 1, 0 // Select w, then x, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wxyy 3, 0, 1, 1 // Select w, then x, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wxyz 3, 0, 1, 2 // Select w, then x, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wxyw 3, 0, 1, 3 // Select w, then x, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wxzx 3, 0, 2, 0 // Select w, then x, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wxzy 3, 0, 2, 1 // Select w, then x, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wxzz 3, 0, 2, 2 // Select w, then x, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wxzw 3, 0, 2, 3 // Select w, then x, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wxwx 3, 0, 3, 0 // Select w, then x, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wxwy 3, 0, 3, 1 // Select w, then x, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wxwz 3, 0, 3, 2 // Select w, then x, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wxww 3, 0, 3, 3 // Select w, then x, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wyxx 3, 1, 0, 0 // Select w, then y, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wyxy 3, 1, 0, 1 // Select w, then y, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wyxz 3, 1, 0, 2 // Select w, then y, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wyxw 3, 1, 0, 3 // Select w, then y, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wyyx 3, 1, 1, 0 // Select w, then y, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wyyy 3, 1, 1, 1 // Select w, then y, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wyyz 3, 1, 1, 2 // Select w, then y, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wyyw 3, 1, 1, 3 // Select w, then y, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wyzx 3, 1, 2, 0 // Select w, then y, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wyzy 3, 1, 2, 1 // Select w, then y, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wyzz 3, 1, 2, 2 // Select w, then y, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wyzw 3, 1, 2, 3 // Select w, then y, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wywx 3, 1, 3, 0 // Select w, then y, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wywy 3, 1, 3, 1 // Select w, then y, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wywz 3, 1, 3, 2 // Select w, then y, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wyww 3, 1, 3, 3 // Select w, then y, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wzxx 3, 2, 0, 0 // Select w, then z, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wzxy 3, 2, 0, 1 // Select w, then z, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wzxz 3, 2, 0, 2 // Select w, then z, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wzxw 3, 2, 0, 3 // Select w, then z, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wzyx 3, 2, 1, 0 // Select w, then z, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wzyy 3, 2, 1, 1 // Select w, then z, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wzyz 3, 2, 1, 2 // Select w, then z, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wzyw 3, 2, 1, 3 // Select w, then z, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wzzx 3, 2, 2, 0 // Select w, then z, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wzzy 3, 2, 2, 1 // Select w, then z, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wzzz 3, 2, 2, 2 // Select w, then z, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wzzw 3, 2, 2, 3 // Select w, then z, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wzwx 3, 2, 3, 0 // Select w, then z, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wzwy 3, 2, 3, 1 // Select w, then z, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wzwz 3, 2, 3, 2 // Select w, then z, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wzww 3, 2, 3, 3 // Select w, then z, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wwxx 3, 3, 0, 0 // Select w, then w, then x, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wwxy 3, 3, 0, 1 // Select w, then w, then x, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wwxz 3, 3, 0, 2 // Select w, then w, then x, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wwxw 3, 3, 0, 3 // Select w, then w, then x, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wwyx 3, 3, 1, 0 // Select w, then w, then y, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wwyy 3, 3, 1, 1 // Select w, then w, then y, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wwyz 3, 3, 1, 2 // Select w, then w, then y, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wwyw 3, 3, 1, 3 // Select w, then w, then y, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wwzx 3, 3, 2, 0 // Select w, then w, then z, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wwzy 3, 3, 2, 1 // Select w, then w, then z, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wwzz 3, 3, 2, 2 // Select w, then w, then z, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wwzw 3, 3, 2, 3 // Select w, then w, then z, then w in that order.
#define SHDR_DETAIL_SWIZZLE_wwwx 3, 3, 3, 0 // Select w, then w, then w, then x in that order.
#define SHDR_DETAIL_SWIZZLE_wwwy 3, 3, 3, 1 // Select w, then w, then w, then y in that order.
#define SHDR_DETAIL_SWIZZLE_wwwz 3, 3, 3, 2 // Select w, then w, then w, then z in that order.
#define SHDR_DETAIL_SWIZZLE_wwww 3, 3, 3, 3 // Select w, then w, then w, then w in that order.
#define SHDR_DETAIL_SWIZZLE_r 0 // Select the r component.
#define SHDR_DETAIL_SWIZZLE_g 1 // Select the g component.
#define SHDR_DETAIL_SWIZZLE_b 2 // Select the b component.
#define SHDR_DETAIL_SWIZZLE_a 3 // Select the a component.
#define SHDR_DETAIL_SWIZZLE_rr 0, 0 // Select r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rg 0, 1 // Select r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rb 0, 2 // Select r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_ra 0, 3 // Select r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gr 1, 0 // Select g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gg 1, 1 // Select g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gb 1, 2 // Select g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_ga 1, 3 // Select g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_br 2, 0 // Select b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bg 2, 1 // Select b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bb 2, 2 // Select b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_ba 2, 3 // Select b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_ar 3, 0 // Select a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_ag 3, 1 // Select a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_ab 3, 2 // Select a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_aa 3, 3 // Select a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rrr 0, 0, 0 // Select r, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rrg 0, 0, 1 // Select r, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rrb 0, 0, 2 // Select r, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rra 0, 0, 3 // Select r, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rgr 0, 1, 0 // Select r, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rgg 0, 1, 1 // Select r, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rgb 0, 1, 2 // Select r, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rga 0, 1, 3 // Select r, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rbr 0, 2, 0 // Select r, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rbg 0, 2, 1 // Select r, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rbb 0, 2, 2 // Select r, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rba 0, 2, 3 // Select r, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rar 0, 3, 0 // Select r, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rag 0, 3, 1 // Select r, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rab 0, 3, 2 // Select r, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_raa 0, 3, 3 // Select r, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_grr 1, 0, 0 // Select g, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_grg 1, 0, 1 // Select g, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_grb 1, 0, 2 // Select g, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gra 1, 0, 3 // Select g, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_ggr 1, 1, 0 // Select g, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_ggg 1, 1, 1 // Select g, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_ggb 1, 1, 2 // Select g, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gga 1, 1, 3 // Select g, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gbr 1, 2, 0 // Select g, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gbg 1, 2, 1 // Select g, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gbb 1, 2, 2 // Select g, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gba 1, 2, 3 // Select g, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gar 1, 3, 0 // Select g, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gag 1, 3, 1 // Select g, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gab 1, 3, 2 // Select g, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gaa 1, 3, 3 // Select g, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_brr 2, 0, 0 // Select b, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_brg 2, 0, 1 // Select b, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_brb 2, 0, 2 // Select b, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bra 2, 0, 3 // Select b, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bgr 2, 1, 0 // Select b, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bgg 2, 1, 1 // Select b, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bgb 2, 1, 2 // Select b, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bga 2, 1, 3 // Select b, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bbr 2, 2, 0 // Select b, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bbg 2, 2, 1 // Select b, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bbb 2, 2, 2 // Select b, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bba 2, 2, 3 // Select b, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bar 2, 3, 0 // Select b, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bag 2, 3, 1 // Select b, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bab 2, 3, 2 // Select b, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_baa 2, 3, 3 // Select b, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_arr 3, 0, 0 // Select a, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_arg 3, 0, 1 // Select a, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_arb 3, 0, 2 // Select a, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_ara 3, 0, 3 // Select a, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_agr 3, 1, 0 // Select a, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_agg 3, 1, 1 // Select a, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_agb 3, 1, 2 // Select a, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_aga 3, 1, 3 // Select a, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_abr 3, 2, 0 // Select a, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_abg 3, 2, 1 // Select a, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_abb 3, 2, 2 // Select a, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_aba 3, 2, 3 // Select a, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_aar 3, 3, 0 // Select a, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_aag 3, 3, 1 // Select a, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_aab 3, 3, 2 // Select a, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_aaa 3, 3, 3 // Select a, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rrrr 0, 0, 0, 0 // Select r, then r, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rrrg 0, 0, 0, 1 // Select r, then r, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rrrb 0, 0, 0, 2 // Select r, then r, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rrra 0, 0, 0, 3 // Select r, then r, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rrgr 0, 0, 1, 0 // Select r, then r, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rrgg 0, 0, 1, 1 // Select r, then r, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rrgb 0, 0, 1, 2 // Select r, then r, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rrga 0, 0, 1, 3 // Select r, then r, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rrbr 0, 0, 2, 0 // Select r, then r, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rrbg 0, 0, 2, 1 // Select r, then r, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rrbb 0, 0, 2, 2 // Select r, then r, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rrba 0, 0, 2, 3 // Select r, then r, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rrar 0, 0, 3, 0 // Select r, then r, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rrag 0, 0, 3, 1 // Select r, then r, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rrab 0, 0, 3, 2 // Select r, then r, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rraa 0, 0, 3, 3 // Select r, then r, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rgrr 0, 1, 0, 0 // Select r, then g, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rgrg 0, 1, 0, 1 // Select r, then g, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rgrb 0, 1, 0, 2 // Select r, then g, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rgra 0, 1, 0, 3 // Select r, then g, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rggr 0, 1, 1, 0 // Select r, then g, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rggg 0, 1, 1, 1 // Select r, then g, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rggb 0, 1, 1, 2 // Select r, then g, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rgga 0, 1, 1, 3 // Select r, then g, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rgbr 0, 1, 2, 0 // Select r, then g, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rgbg 0, 1, 2, 1 // Select r, then g, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rgbb 0, 1, 2, 2 // Select r, then g, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rgba 0, 1, 2, 3 // Select r, then g, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rgar 0, 1, 3, 0 // Select r, then g, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rgag 0, 1, 3, 1 // Select r, then g, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rgab 0, 1, 3, 2 // Select r, then g, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rgaa 0, 1, 3, 3 // Select r, then g, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rbrr 0, 2, 0, 0 // Select r, then b, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rbrg 0, 2, 0, 1 // Select r, then b, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rbrb 0, 2, 0, 2 // Select r, then b, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rbra 0, 2, 0, 3 // Select r, then b, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rbgr 0, 2, 1, 0 // Select r, then b, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rbgg 0, 2, 1, 1 // Select r, then b, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rbgb 0, 2, 1, 2 // Select r, then b, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rbga 0, 2, 1, 3 // Select r, then b, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rbbr 0, 2, 2, 0 // Select r, then b, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rbbg 0, 2, 2, 1 // Select r, then b, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rbbb 0, 2, 2, 2 // Select r, then b, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rbba 0, 2, 2, 3 // Select r, then b, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rbar 0, 2, 3, 0 // Select r, then b, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rbag 0, 2, 3, 1 // Select r, then b, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rbab 0, 2, 3, 2 // Select r, then b, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rbaa 0, 2, 3, 3 // Select r, then b, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rarr 0, 3, 0, 0 // Select r, then a, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rarg 0, 3, 0, 1 // Select r, then a, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rarb 0, 3, 0, 2 // Select r, then a, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_rara 0, 3, 0, 3 // Select r, then a, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_ragr 0, 3, 1, 0 // Select r, then a, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_ragg 0, 3, 1, 1 // Select r, then a, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_ragb 0, 3, 1, 2 // Select r, then a, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_raga 0, 3, 1, 3 // Select r, then a, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_rabr 0, 3, 2, 0 // Select r, then a, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_rabg 0, 3, 2, 1 // Select r, then a, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_rabb 0, 3, 2, 2 // Select r, then a, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_raba 0, 3, 2, 3 // Select r, then a, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_raar 0, 3, 3, 0 // Select r, then a, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_raag 0, 3, 3, 1 // Select r, then a, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_raab 0, 3, 3, 2 // Select r, then a, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_raaa 0, 3, 3, 3 // Select r, then a, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_grrr 1, 0, 0, 0 // Select g, then r, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_grrg 1, 0, 0, 1 // Select g, then r, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_grrb 1, 0, 0, 2 // Select g, then r, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_grra 1, 0, 0, 3 // Select g, then r, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_grgr 1, 0, 1, 0 // Select g, then r, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_grgg 1, 0, 1, 1 // Select g, then r, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_grgb 1, 0, 1, 2 // Select g, then r, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_grga 1, 0, 1, 3 // Select g, then r, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_grbr 1, 0, 2, 0 // Select g, then r, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_grbg 1, 0, 2, 1 // Select g, then r, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_grbb 1, 0, 2, 2 // Select g, then r, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_grba 1, 0, 2, 3 // Select g, then r, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_grar 1, 0, 3, 0 // Select g, then r, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_grag 1, 0, 3, 1 // Select g, then r, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_grab 1, 0, 3, 2 // Select g, then r, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_graa 1, 0, 3, 3 // Select g, then r, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_ggrr 1, 1, 0, 0 // Select g, then g, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_ggrg 1, 1, 0, 1 // Select g, then g, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_ggrb 1, 1, 0, 2 // Select g, then g, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_ggra 1, 1, 0, 3 // Select g, then g, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gggr 1, 1, 1, 0 // Select g, then g, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gggg 1, 1, 1, 1 // Select g, then g, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gggb 1, 1, 1, 2 // Select g, then g, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_ggga 1, 1, 1, 3 // Select g, then g, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_ggbr 1, 1, 2, 0 // Select g, then g, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_ggbg 1, 1, 2, 1 // Select g, then g, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_ggbb 1, 1, 2, 2 // Select g, then g, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_ggba 1, 1, 2, 3 // Select g, then g, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_ggar 1, 1, 3, 0 // Select g, then g, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_ggag 1, 1, 3, 1 // Select g, then g, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_ggab 1, 1, 3, 2 // Select g, then g, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_ggaa 1, 1, 3, 3 // Select g, then g, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gbrr 1, 2, 0, 0 // Select g, then b, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gbrg 1, 2, 0, 1 // Select g, then b, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gbrb 1, 2, 0, 2 // Select g, then b, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gbra 1, 2, 0, 3 // Select g, then b, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gbgr 1, 2, 1, 0 // Select g, then b, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gbgg 1, 2, 1, 1 // Select g, then b, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gbgb 1, 2, 1, 2 // Select g, then b, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gbga 1, 2, 1, 3 // Select g, then b, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gbbr 1, 2, 2, 0 // Select g, then b, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gbbg 1, 2, 2, 1 // Select g, then b, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gbbb 1, 2, 2, 2 // Select g, then b, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gbba 1, 2, 2, 3 // Select g, then b, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gbar 1, 2, 3, 0 // Select g, then b, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gbag 1, 2, 3, 1 // Select g, then b, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gbab 1, 2, 3, 2 // Select g, then b, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gbaa 1, 2, 3, 3 // Select g, then b, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_garr 1, 3, 0, 0 // Select g, then a, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_garg 1, 3, 0, 1 // Select g, then a, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_garb 1, 3, 0, 2 // Select g, then a, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gara 1, 3, 0, 3 // Select g, then a, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gagr 1, 3, 1, 0 // Select g, then a, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gagg 1, 3, 1, 1 // Select g, then a, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gagb 1, 3, 1, 2 // Select g, then a, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gaga 1, 3, 1, 3 // Select g, then a, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gabr 1, 3, 2, 0 // Select g, then a, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gabg 1, 3, 2, 1 // Select g, then a, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gabb 1, 3, 2, 2 // Select g, then a, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gaba 1, 3, 2, 3 // Select g, then a, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_gaar 1, 3, 3, 0 // Select g, then a, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_gaag 1, 3, 3, 1 // Select g, then a, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_gaab 1, 3, 3, 2 // Select g, then a, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_gaaa 1, 3, 3, 3 // Select g, then a, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_brrr 2, 0, 0, 0 // Select b, then r, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_brrg 2, 0, 0, 1 // Select b, then r, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_brrb 2, 0, 0, 2 // Select b, then r, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_brra 2, 0, 0, 3 // Select b, then r, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_brgr 2, 0, 1, 0 // Select b, then r, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_brgg 2, 0, 1, 1 // Select b, then r, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_brgb 2, 0, 1, 2 // Select b, then r, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_brga 2, 0, 1, 3 // Select b, then r, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_brbr 2, 0, 2, 0 // Select b, then r, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_brbg 2, 0, 2, 1 // Select b, then r, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_brbb 2, 0, 2, 2 // Select b, then r, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_brba 2, 0, 2, 3 // Select b, then r, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_brar 2, 0, 3, 0 // Select b, then r, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_brag 2, 0, 3, 1 // Select b, then r, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_brab 2, 0, 3, 2 // Select b, then r, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_braa 2, 0, 3, 3 // Select b, then r, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bgrr 2, 1, 0, 0 // Select b, then g, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bgrg 2, 1, 0, 1 // Select b, then g, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bgrb 2, 1, 0, 2 // Select b, then g, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bgra 2, 1, 0, 3 // Select b, then g, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bggr 2, 1, 1, 0 // Select b, then g, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bggg 2, 1, 1, 1 // Select b, then g, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bggb 2, 1, 1, 2 // Select b, then g, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bgga 2, 1, 1, 3 // Select b, then g, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bgbr 2, 1, 2, 0 // Select b, then g, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bgbg 2, 1, 2, 1 // Select b, then g, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bgbb 2, 1, 2, 2 // Select b, then g, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bgba 2, 1, 2, 3 // Select b, then g, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bgar 2, 1, 3, 0 // Select b, then g, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bgag 2, 1, 3, 1 // Select b, then g, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bgab 2, 1, 3, 2 // Select b, then g, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bgaa 2, 1, 3, 3 // Select b, then g, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bbrr 2, 2, 0, 0 // Select b, then b, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bbrg 2, 2, 0, 1 // Select b, then b, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bbrb 2, 2, 0, 2 // Select b, then b, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bbra 2, 2, 0, 3 // Select b, then b, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bbgr 2, 2, 1, 0 // Select b, then b, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bbgg 2, 2, 1, 1 // Select b, then b, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bbgb 2, 2, 1, 2 // Select b, then b, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bbga 2, 2, 1, 3 // Select b, then b, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bbbr 2, 2, 2, 0 // Select b, then b, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bbbg 2, 2, 2, 1 // Select b, then b, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bbbb 2, 2, 2, 2 // Select b, then b, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bbba 2, 2, 2, 3 // Select b, then b, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bbar 2, 2, 3, 0 // Select b, then b, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bbag 2, 2, 3, 1 // Select b, then b, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bbab 2, 2, 3, 2 // Select b, then b, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bbaa 2, 2, 3, 3 // Select b, then b, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_barr 2, 3, 0, 0 // Select b, then a, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_barg 2, 3, 0, 1 // Select b, then a, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_barb 2, 3, 0, 2 // Select b, then a, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_bara 2, 3, 0, 3 // Select b, then a, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_bagr 2, 3, 1, 0 // Select b, then a, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_bagg 2, 3, 1, 1 // Select b, then a, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_bagb 2, 3, 1, 2 // Select b, then a, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_baga 2, 3, 1, 3 // Select b, then a, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_babr 2, 3, 2, 0 // Select b, then a, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_babg 2, 3, 2, 1 // Select b, then a, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_babb 2, 3, 2, 2 // Select b, then a, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_baba 2, 3, 2, 3 // Select b, then a, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_baar 2, 3, 3, 0 // Select b, then a, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_baag 2, 3, 3, 1 // Select b, then a, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_baab 2, 3, 3, 2 // Select b, then a, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_baaa 2, 3, 3, 3 // Select b, then a, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_arrr 3, 0, 0, 0 // Select a, then r, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_arrg 3, 0, 0, 1 // Select a, then r, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_arrb 3, 0, 0, 2 // Select a, then r, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_arra 3, 0, 0, 3 // Select a, then r, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_argr 3, 0, 1, 0 // Select a, then r, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_argg 3, 0, 1, 1 // Select a, then r, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_argb 3, 0, 1, 2 // Select a, then r, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_arga 3, 0, 1, 3 // Select a, then r, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_arbr 3, 0, 2, 0 // Select a, then r, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_arbg 3, 0, 2, 1 // Select a, then r, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_arbb 3, 0, 2, 2 // Select a, then r, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_arba 3, 0, 2, 3 // Select a, then r, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_arar 3, 0, 3, 0 // Select a, then r, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_arag 3, 0, 3, 1 // Select a, then r, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_arab 3, 0, 3, 2 // Select a, then r, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_araa 3, 0, 3, 3 // Select a, then r, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_agrr 3, 1, 0, 0 // Select a, then g, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_agrg 3, 1, 0, 1 // Select a, then g, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_agrb 3, 1, 0, 2 // Select a, then g, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_agra 3, 1, 0, 3 // Select a, then g, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_aggr 3, 1, 1, 0 // Select a, then g, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_aggg 3, 1, 1, 1 // Select a, then g, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_aggb 3, 1, 1, 2 // Select a, then g, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_agga 3, 1, 1, 3 // Select a, then g, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_agbr 3, 1, 2, 0 // Select a, then g, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_agbg 3, 1, 2, 1 // Select a, then g, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_agbb 3, 1, 2, 2 // Select a, then g, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_agba 3, 1, 2, 3 // Select a, then g, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_agar 3, 1, 3, 0 // Select a, then g, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_agag 3, 1, 3, 1 // Select a, then g, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_agab 3, 1, 3, 2 // Select a, then g, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_agaa 3, 1, 3, 3 // Select a, then g, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_abrr 3, 2, 0, 0 // Select a, then b, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_abrg 3, 2, 0, 1 // Select a, then b, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_abrb 3, 2, 0, 2 // Select a, then b, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_abra 3, 2, 0, 3 // Select a, then b, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_abgr 3, 2, 1, 0 // Select a, then b, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_abgg 3, 2, 1, 1 // Select a, then b, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_abgb 3, 2, 1, 2 // Select a, then b, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_abga 3, 2, 1, 3 // Select a, then b, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_abbr 3, 2, 2, 0 // Select a, then b, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_abbg 3, 2, 2, 1 // Select a, then b, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_abbb 3, 2, 2, 2 // Select a, then b, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_abba 3, 2, 2, 3 // Select a, then b, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_abar 3, 2, 3, 0 // Select a, then b, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_abag 3, 2, 3, 1 // Select a, then b, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_abab 3, 2, 3, 2 // Select a, then b, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_abaa 3, 2, 3, 3 // Select a, then b, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_aarr 3, 3, 0, 0 // Select a, then a, then r, then r in that order.
#define SHDR_DETAIL_SWIZZLE_aarg 3, 3, 0, 1 // Select a, then a, then r, then g in that order.
#define SHDR_DETAIL_SWIZZLE_aarb 3, 3, 0, 2 // Select a, then a, then r, then b in that order.
#define SHDR_DETAIL_SWIZZLE_aara 3, 3, 0, 3 // Select a, then a, then r, then a in that order.
#define SHDR_DETAIL_SWIZZLE_aagr 3, 3, 1, 0 // Select a, then a, then g, then r in that order.
#define SHDR_DETAIL_SWIZZLE_aagg 3, 3, 1, 1 // Select a, then a, then g, then g in that order.
#define SHDR_DETAIL_SWIZZLE_aagb 3, 3, 1, 2 // Select a, then a, then g, then b in that order.
#define SHDR_DETAIL_SWIZZLE_aaga 3, 3, 1, 3 // Select a, then a, then g, then a in that order.
#define SHDR_DETAIL_SWIZZLE_aabr 3, 3, 2, 0 // Select a, then a, then b, then r in that order.
#define SHDR_DETAIL_SWIZZLE_aabg 3, 3, 2, 1 // Select a, then a, then b, then g in that order.
#define SHDR_DETAIL_SWIZZLE_aabb 3, 3, 2, 2 // Select a, then a, then b, then b in that order.
#define SHDR_DETAIL_SWIZZLE_aaba 3, 3, 2, 3 // Select a, then a, then b, then a in that order.
#define SHDR_DETAIL_SWIZZLE_aaar 3, 3, 3, 0 // Select a, then a, then a, then r in that order.
#define SHDR_DETAIL_SWIZZLE_aaag 3, 3, 3, 1 // Select a, then a, then a, then g in that order.
#define SHDR_DETAIL_SWIZZLE_aaab 3, 3, 3, 2 // Select a, then a, then a, then b in that order.
#define SHDR_DETAIL_SWIZZLE_aaaa 3, 3, 3, 3 // Select a, then a, then a, then a in that order.
#define SHDR_DETAIL_SWIZZLE_s 0 // Select the s component.
#define SHDR_DETAIL_SWIZZLE_t 1 // Select the t component.
#define SHDR_DETAIL_SWIZZLE_p 2 // Select the p component.
#define SHDR_DETAIL_SWIZZLE_q 3 // Select the q component.
#define SHDR_DETAIL_SWIZZLE_ss 0, 0 // Select s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_st 0, 1 // Select s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sp 0, 2 // Select s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sq 0, 3 // Select s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ts 1, 0 // Select t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tt 1, 1 // Select t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tp 1, 2 // Select t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tq 1, 3 // Select t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ps 2, 0 // Select p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pt 2, 1 // Select p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pp 2, 2 // Select p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pq 2, 3 // Select p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qs 3, 0 // Select q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qt 3, 1 // Select q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qp 3, 2 // Select q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qq 3, 3 // Select q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_sss 0, 0, 0 // Select s, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sst 0, 0, 1 // Select s, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ssp 0, 0, 2 // Select s, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ssq 0, 0, 3 // Select s, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_sts 0, 1, 0 // Select s, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_stt 0, 1, 1 // Select s, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_stp 0, 1, 2 // Select s, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_stq 0, 1, 3 // Select s, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_sps 0, 2, 0 // Select s, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_spt 0, 2, 1 // Select s, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_spp 0, 2, 2 // Select s, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_spq 0, 2, 3 // Select s, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_sqs 0, 3, 0 // Select s, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sqt 0, 3, 1 // Select s, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sqp 0, 3, 2 // Select s, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sqq 0, 3, 3 // Select s, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tss 1, 0, 0 // Select t, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tst 1, 0, 1 // Select t, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tsp 1, 0, 2 // Select t, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tsq 1, 0, 3 // Select t, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tts 1, 1, 0 // Select t, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ttt 1, 1, 1 // Select t, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ttp 1, 1, 2 // Select t, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ttq 1, 1, 3 // Select t, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tps 1, 2, 0 // Select t, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tpt 1, 2, 1 // Select t, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tpp 1, 2, 2 // Select t, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tpq 1, 2, 3 // Select t, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tqs 1, 3, 0 // Select t, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tqt 1, 3, 1 // Select t, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tqp 1, 3, 2 // Select t, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tqq 1, 3, 3 // Select t, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_pss 2, 0, 0 // Select p, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pst 2, 0, 1 // Select p, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_psp 2, 0, 2 // Select p, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_psq 2, 0, 3 // Select p, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_pts 2, 1, 0 // Select p, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ptt 2, 1, 1 // Select p, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ptp 2, 1, 2 // Select p, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ptq 2, 1, 3 // Select p, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_pps 2, 2, 0 // Select p, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ppt 2, 2, 1 // Select p, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ppp 2, 2, 2 // Select p, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ppq 2, 2, 3 // Select p, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_pqs 2, 3, 0 // Select p, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pqt 2, 3, 1 // Select p, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pqp 2, 3, 2 // Select p, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pqq 2, 3, 3 // Select p, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qss 3, 0, 0 // Select q, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qst 3, 0, 1 // Select q, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qsp 3, 0, 2 // Select q, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qsq 3, 0, 3 // Select q, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qts 3, 1, 0 // Select q, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qtt 3, 1, 1 // Select q, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qtp 3, 1, 2 // Select q, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qtq 3, 1, 3 // Select q, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qps 3, 2, 0 // Select q, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qpt 3, 2, 1 // Select q, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qpp 3, 2, 2 // Select q, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qpq 3, 2, 3 // Select q, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qqs 3, 3, 0 // Select q, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qqt 3, 3, 1 // Select q, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qqp 3, 3, 2 // Select q, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qqq 3, 3, 3 // Select q, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ssss 0, 0, 0, 0 // Select s, then s, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ssst 0, 0, 0, 1 // Select s, then s, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sssp 0, 0, 0, 2 // Select s, then s, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sssq 0, 0, 0, 3 // Select s, then s, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ssts 0, 0, 1, 0 // Select s, then s, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sstt 0, 0, 1, 1 // Select s, then s, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sstp 0, 0, 1, 2 // Select s, then s, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sstq 0, 0, 1, 3 // Select s, then s, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ssps 0, 0, 2, 0 // Select s, then s, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sspt 0, 0, 2, 1 // Select s, then s, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sspp 0, 0, 2, 2 // Select s, then s, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sspq 0, 0, 2, 3 // Select s, then s, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ssqs 0, 0, 3, 0 // Select s, then s, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ssqt 0, 0, 3, 1 // Select s, then s, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ssqp 0, 0, 3, 2 // Select s, then s, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ssqq 0, 0, 3, 3 // Select s, then s, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_stss 0, 1, 0, 0 // Select s, then t, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_stst 0, 1, 0, 1 // Select s, then t, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_stsp 0, 1, 0, 2 // Select s, then t, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_stsq 0, 1, 0, 3 // Select s, then t, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_stts 0, 1, 1, 0 // Select s, then t, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sttt 0, 1, 1, 1 // Select s, then t, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sttp 0, 1, 1, 2 // Select s, then t, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sttq 0, 1, 1, 3 // Select s, then t, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_stps 0, 1, 2, 0 // Select s, then t, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_stpt 0, 1, 2, 1 // Select s, then t, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_stpp 0, 1, 2, 2 // Select s, then t, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_stpq 0, 1, 2, 3 // Select s, then t, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_stqs 0, 1, 3, 0 // Select s, then t, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_stqt 0, 1, 3, 1 // Select s, then t, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_stqp 0, 1, 3, 2 // Select s, then t, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_stqq 0, 1, 3, 3 // Select s, then t, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_spss 0, 2, 0, 0 // Select s, then p, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_spst 0, 2, 0, 1 // Select s, then p, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_spsp 0, 2, 0, 2 // Select s, then p, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_spsq 0, 2, 0, 3 // Select s, then p, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_spts 0, 2, 1, 0 // Select s, then p, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sptt 0, 2, 1, 1 // Select s, then p, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sptp 0, 2, 1, 2 // Select s, then p, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sptq 0, 2, 1, 3 // Select s, then p, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_spps 0, 2, 2, 0 // Select s, then p, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sppt 0, 2, 2, 1 // Select s, then p, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sppp 0, 2, 2, 2 // Select s, then p, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sppq 0, 2, 2, 3 // Select s, then p, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_spqs 0, 2, 3, 0 // Select s, then p, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_spqt 0, 2, 3, 1 // Select s, then p, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_spqp 0, 2, 3, 2 // Select s, then p, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_spqq 0, 2, 3, 3 // Select s, then p, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_sqss 0, 3, 0, 0 // Select s, then q, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sqst 0, 3, 0, 1 // Select s, then q, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sqsp 0, 3, 0, 2 // Select s, then q, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sqsq 0, 3, 0, 3 // Select s, then q, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_sqts 0, 3, 1, 0 // Select s, then q, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sqtt 0, 3, 1, 1 // Select s, then q, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sqtp 0, 3, 1, 2 // Select s, then q, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sqtq 0, 3, 1, 3 // Select s, then q, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_sqps 0, 3, 2, 0 // Select s, then q, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sqpt 0, 3, 2, 1 // Select s, then q, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sqpp 0, 3, 2, 2 // Select s, then q, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sqpq 0, 3, 2, 3 // Select s, then q, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_sqqs 0, 3, 3, 0 // Select s, then q, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_sqqt 0, 3, 3, 1 // Select s, then q, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_sqqp 0, 3, 3, 2 // Select s, then q, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_sqqq 0, 3, 3, 3 // Select s, then q, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tsss 1, 0, 0, 0 // Select t, then s, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tsst 1, 0, 0, 1 // Select t, then s, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tssp 1, 0, 0, 2 // Select t, then s, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tssq 1, 0, 0, 3 // Select t, then s, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tsts 1, 0, 1, 0 // Select t, then s, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tstt 1, 0, 1, 1 // Select t, then s, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tstp 1, 0, 1, 2 // Select t, then s, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tstq 1, 0, 1, 3 // Select t, then s, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tsps 1, 0, 2, 0 // Select t, then s, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tspt 1, 0, 2, 1 // Select t, then s, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tspp 1, 0, 2, 2 // Select t, then s, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tspq 1, 0, 2, 3 // Select t, then s, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tsqs 1, 0, 3, 0 // Select t, then s, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tsqt 1, 0, 3, 1 // Select t, then s, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tsqp 1, 0, 3, 2 // Select t, then s, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tsqq 1, 0, 3, 3 // Select t, then s, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ttss 1, 1, 0, 0 // Select t, then t, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ttst 1, 1, 0, 1 // Select t, then t, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ttsp 1, 1, 0, 2 // Select t, then t, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ttsq 1, 1, 0, 3 // Select t, then t, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ttts 1, 1, 1, 0 // Select t, then t, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tttt 1, 1, 1, 1 // Select t, then t, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tttp 1, 1, 1, 2 // Select t, then t, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tttq 1, 1, 1, 3 // Select t, then t, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ttps 1, 1, 2, 0 // Select t, then t, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ttpt 1, 1, 2, 1 // Select t, then t, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ttpp 1, 1, 2, 2 // Select t, then t, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ttpq 1, 1, 2, 3 // Select t, then t, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ttqs 1, 1, 3, 0 // Select t, then t, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ttqt 1, 1, 3, 1 // Select t, then t, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ttqp 1, 1, 3, 2 // Select t, then t, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ttqq 1, 1, 3, 3 // Select t, then t, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tpss 1, 2, 0, 0 // Select t, then p, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tpst 1, 2, 0, 1 // Select t, then p, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tpsp 1, 2, 0, 2 // Select t, then p, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tpsq 1, 2, 0, 3 // Select t, then p, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tpts 1, 2, 1, 0 // Select t, then p, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tptt 1, 2, 1, 1 // Select t, then p, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tptp 1, 2, 1, 2 // Select t, then p, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tptq 1, 2, 1, 3 // Select t, then p, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tpps 1, 2, 2, 0 // Select t, then p, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tppt 1, 2, 2, 1 // Select t, then p, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tppp 1, 2, 2, 2 // Select t, then p, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tppq 1, 2, 2, 3 // Select t, then p, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tpqs 1, 2, 3, 0 // Select t, then p, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tpqt 1, 2, 3, 1 // Select t, then p, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tpqp 1, 2, 3, 2 // Select t, then p, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tpqq 1, 2, 3, 3 // Select t, then p, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tqss 1, 3, 0, 0 // Select t, then q, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tqst 1, 3, 0, 1 // Select t, then q, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tqsp 1, 3, 0, 2 // Select t, then q, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tqsq 1, 3, 0, 3 // Select t, then q, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tqts 1, 3, 1, 0 // Select t, then q, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tqtt 1, 3, 1, 1 // Select t, then q, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tqtp 1, 3, 1, 2 // Select t, then q, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tqtq 1, 3, 1, 3 // Select t, then q, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tqps 1, 3, 2, 0 // Select t, then q, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tqpt 1, 3, 2, 1 // Select t, then q, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tqpp 1, 3, 2, 2 // Select t, then q, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tqpq 1, 3, 2, 3 // Select t, then q, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_tqqs 1, 3, 3, 0 // Select t, then q, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_tqqt 1, 3, 3, 1 // Select t, then q, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_tqqp 1, 3, 3, 2 // Select t, then q, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_tqqq 1, 3, 3, 3 // Select t, then q, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_psss 2, 0, 0, 0 // Select p, then s, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_psst 2, 0, 0, 1 // Select p, then s, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pssp 2, 0, 0, 2 // Select p, then s, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pssq 2, 0, 0, 3 // Select p, then s, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_psts 2, 0, 1, 0 // Select p, then s, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pstt 2, 0, 1, 1 // Select p, then s, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pstp 2, 0, 1, 2 // Select p, then s, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pstq 2, 0, 1, 3 // Select p, then s, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_psps 2, 0, 2, 0 // Select p, then s, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pspt 2, 0, 2, 1 // Select p, then s, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pspp 2, 0, 2, 2 // Select p, then s, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pspq 2, 0, 2, 3 // Select p, then s, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_psqs 2, 0, 3, 0 // Select p, then s, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_psqt 2, 0, 3, 1 // Select p, then s, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_psqp 2, 0, 3, 2 // Select p, then s, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_psqq 2, 0, 3, 3 // Select p, then s, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ptss 2, 1, 0, 0 // Select p, then t, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ptst 2, 1, 0, 1 // Select p, then t, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ptsp 2, 1, 0, 2 // Select p, then t, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ptsq 2, 1, 0, 3 // Select p, then t, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ptts 2, 1, 1, 0 // Select p, then t, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pttt 2, 1, 1, 1 // Select p, then t, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pttp 2, 1, 1, 2 // Select p, then t, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pttq 2, 1, 1, 3 // Select p, then t, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ptps 2, 1, 2, 0 // Select p, then t, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ptpt 2, 1, 2, 1 // Select p, then t, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ptpp 2, 1, 2, 2 // Select p, then t, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ptpq 2, 1, 2, 3 // Select p, then t, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ptqs 2, 1, 3, 0 // Select p, then t, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ptqt 2, 1, 3, 1 // Select p, then t, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ptqp 2, 1, 3, 2 // Select p, then t, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ptqq 2, 1, 3, 3 // Select p, then t, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ppss 2, 2, 0, 0 // Select p, then p, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ppst 2, 2, 0, 1 // Select p, then p, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ppsp 2, 2, 0, 2 // Select p, then p, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ppsq 2, 2, 0, 3 // Select p, then p, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ppts 2, 2, 1, 0 // Select p, then p, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pptt 2, 2, 1, 1 // Select p, then p, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pptp 2, 2, 1, 2 // Select p, then p, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pptq 2, 2, 1, 3 // Select p, then p, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ppps 2, 2, 2, 0 // Select p, then p, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pppt 2, 2, 2, 1 // Select p, then p, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pppp 2, 2, 2, 2 // Select p, then p, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pppq 2, 2, 2, 3 // Select p, then p, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_ppqs 2, 2, 3, 0 // Select p, then p, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_ppqt 2, 2, 3, 1 // Select p, then p, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_ppqp 2, 2, 3, 2 // Select p, then p, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_ppqq 2, 2, 3, 3 // Select p, then p, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_pqss 2, 3, 0, 0 // Select p, then q, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pqst 2, 3, 0, 1 // Select p, then q, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pqsp 2, 3, 0, 2 // Select p, then q, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pqsq 2, 3, 0, 3 // Select p, then q, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_pqts 2, 3, 1, 0 // Select p, then q, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pqtt 2, 3, 1, 1 // Select p, then q, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pqtp 2, 3, 1, 2 // Select p, then q, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pqtq 2, 3, 1, 3 // Select p, then q, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_pqps 2, 3, 2, 0 // Select p, then q, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pqpt 2, 3, 2, 1 // Select p, then q, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pqpp 2, 3, 2, 2 // Select p, then q, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pqpq 2, 3, 2, 3 // Select p, then q, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_pqqs 2, 3, 3, 0 // Select p, then q, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_pqqt 2, 3, 3, 1 // Select p, then q, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_pqqp 2, 3, 3, 2 // Select p, then q, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_pqqq 2, 3, 3, 3 // Select p, then q, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qsss 3, 0, 0, 0 // Select q, then s, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qsst 3, 0, 0, 1 // Select q, then s, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qssp 3, 0, 0, 2 // Select q, then s, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qssq 3, 0, 0, 3 // Select q, then s, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qsts 3, 0, 1, 0 // Select q, then s, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qstt 3, 0, 1, 1 // Select q, then s, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qstp 3, 0, 1, 2 // Select q, then s, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qstq 3, 0, 1, 3 // Select q, then s, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qsps 3, 0, 2, 0 // Select q, then s, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qspt 3, 0, 2, 1 // Select q, then s, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qspp 3, 0, 2, 2 // Select q, then s, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qspq 3, 0, 2, 3 // Select q, then s, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qsqs 3, 0, 3, 0 // Select q, then s, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qsqt 3, 0, 3, 1 // Select q, then s, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qsqp 3, 0, 3, 2 // Select q, then s, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qsqq 3, 0, 3, 3 // Select q, then s, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qtss 3, 1, 0, 0 // Select q, then t, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qtst 3, 1, 0, 1 // Select q, then t, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qtsp 3, 1, 0, 2 // Select q, then t, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qtsq 3, 1, 0, 3 // Select q, then t, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qtts 3, 1, 1, 0 // Select q, then t, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qttt 3, 1, 1, 1 // Select q, then t, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qttp 3, 1, 1, 2 // Select q, then t, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qttq 3, 1, 1, 3 // Select q, then t, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qtps 3, 1, 2, 0 // Select q, then t, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qtpt 3, 1, 2, 1 // Select q, then t, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qtpp 3, 1, 2, 2 // Select q, then t, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qtpq 3, 1, 2, 3 // Select q, then t, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qtqs 3, 1, 3, 0 // Select q, then t, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qtqt 3, 1, 3, 1 // Select q, then t, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qtqp 3, 1, 3, 2 // Select q, then t, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qtqq 3, 1, 3, 3 // Select q, then t, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qpss 3, 2, 0, 0 // Select q, then p, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qpst 3, 2, 0, 1 // Select q, then p, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qpsp 3, 2, 0, 2 // Select q, then p, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qpsq 3, 2, 0, 3 // Select q, then p, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qpts 3, 2, 1, 0 // Select q, then p, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qptt 3, 2, 1, 1 // Select q, then p, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qptp 3, 2, 1, 2 // Select q, then p, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qptq 3, 2, 1, 3 // Select q, then p, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qpps 3, 2, 2, 0 // Select q, then p, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qppt 3, 2, 2, 1 // Select q, then p, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qppp 3, 2, 2, 2 // Select q, then p, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qppq 3, 2, 2, 3 // Select q, then p, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qpqs 3, 2, 3, 0 // Select q, then p, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qpqt 3, 2, 3, 1 // Select q, then p, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qpqp 3, 2, 3, 2 // Select q, then p, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qpqq 3, 2, 3, 3 // Select q, then p, then q, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qqss 3, 3, 0, 0 // Select q, then q, then s, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qqst 3, 3, 0, 1 // Select q, then q, then s, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qqsp 3, 3, 0, 2 // Select q, then q, then s, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qqsq 3, 3, 0, 3 // Select q, then q, then s, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qqts 3, 3, 1, 0 // Select q, then q, then t, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qqtt 3, 3, 1, 1 // Select q, then q, then t, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qqtp 3, 3, 1, 2 // Select q, then q, then t, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qqtq 3, 3, 1, 3 // Select q, then q, then t, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qqps 3, 3, 2, 0 // Select q, then q, then p, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qqpt 3, 3, 2, 1 // Select q, then q, then p, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qqpp 3, 3, 2, 2 // Select q, then q, then p, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qqpq 3, 3, 2, 3 // Select q, then q, then p, then q in that order.
#define SHDR_DETAIL_SWIZZLE_qqqs 3, 3, 3, 0 // Select q, then q, then q, then s in that order.
#define SHDR_DETAIL_SWIZZLE_qqqt 3, 3, 3, 1 // Select q, then q, then q, then t in that order.
#define SHDR_DETAIL_SWIZZLE_qqqp 3, 3, 3, 2 // Select q, then q, then q, then p in that order.
#define SHDR_DETAIL_SWIZZLE_qqqq 3, 3, 3, 3 // Select q, then q, then q, then q in that order.

#endif // SHDR_MACROS_HPP
