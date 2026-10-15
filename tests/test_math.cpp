/** @file test_math.cpp
 * Regression checks for GLSL formulas, packed layout and alias-safe swizzles.
 * Exception-based checks execute in both Debug and Release builds.
 */
#include "../detail/math.hpp" // Include ../detail/math.hpp declarations.
#include <iostream> // Include iostream declarations.
#include <stdexcept> // Include stdexcept declarations.
#include <type_traits> // Include type_traits declarations.
#include <cfenv> // Include cfenv declarations.

namespace g = shdr::glsl; // Group these helpers within their namespace.
static void require(bool condition, const char* message) { // Check that this operation satisfies the expected behavior.
    if (!condition) // Check this condition before entering the following branch.
        throw std::runtime_error(message); // Report this failed operation.
} // End this block.
static bool near(double a, double b, double tolerance = 1e-5) { // Set tolerance from this expression.
    return std::abs(a - b) < tolerance; // Return std::abs(a - b) < tolerance.
} // End this block.
template <class T, std::size_t N> // Accept the value types or dimensions used by this template.
static bool near(const g::vec<T, N>& a, const g::vec<T, N>& b, double tolerance = 1e-5) { // Set tolerance from this expression.
    for (std::size_t i = 0; i < N; ++i) // Iterate over i for this calculation.
        if (!near(a[i], b[i], tolerance)) // Check this condition before entering the following branch.
            return false; // Return false.
    return true; // Return true.
} // End this block.
template <class T, std::size_t C, std::size_t R> // Accept the value types or dimensions used by this template.
static bool near(const g::mat<T, C, R>& a, const g::mat<T, C, R>& b, double tolerance = 1e-5) { // Set tolerance from this expression.
    for (std::size_t c = 0; c < C; ++c) // Iterate over c for this calculation.
        if (!near(a[c], b[c], tolerance)) // Check this condition before entering the following branch.
            return false; // Return false.
    return true; // Return true.
} // End this block.

static void vectors() { // Implement vectors for the operations below.
    static_assert(sizeof(g::vec2) == 8, "vec2 packed layout"); // Verify this invariant before proceeding.
    static_assert(sizeof(g::vec3) == 12, "vec3 packed host layout"); // Verify this invariant before proceeding.
    static_assert(sizeof(g::vec4) == 16, "vec4 packed layout"); // Verify this invariant before proceeding.
    static_assert(sizeof(g::uvec4) == 16, "uvec4 packed layout"); // Verify this invariant before proceeding.
    static_assert(sizeof(g::uint) == 4, "portable GLSL unsigned scalar"); // Verify this invariant before proceeding.
    static_assert(std::is_standard_layout<g::vec4>::value, "standard layout vectors"); // Verify this invariant before proceeding.
    static_assert(std::is_trivially_copyable<g::vec4>::value, "trivially copyable vectors"); // Verify this invariant before proceeding.
    require(g::vec4() == g::vec4(0), "value initialization"); // Check that this operation satisfies the expected behavior.
    require(g::vec4(2) == g::vec4(2, 2, 2, 2), "scalar broadcast"); // Check that this operation satisfies the expected behavior.
    require(g::vec4(g::vec2(1, 2), 3, 4) == g::vec4(1, 2, 3, 4), "flatten vector/scalars"); // Check that this operation satisfies the expected behavior.
    require(g::vec4(1, g::vec2(2, 3), 4) == g::vec4(1, 2, 3, 4), "flatten mixed components"); // Check that this operation satisfies the expected behavior.
    require(g::vec3(g::vec2(1, 2), g::vec2(3, 4)) == g::vec3(1, 2, 3), // Check that this operation satisfies the expected behavior.
            "truncate the final vector constructor argument"); // Provide this text argument for the operation above.
    require(g::vec4(1, 2, g::vec4(3, 4, 5, 6)) == g::vec4(1, 2, 3, 4), // Check that this operation satisfies the expected behavior.
            "flatten and truncate tail components"); // Provide this text argument for the operation above.
    require(g::vec3(g::ivec4(1, 2, 3, 4)) == g::vec3(1, 2, 3), "convert and truncate"); // Check that this operation satisfies the expected behavior.
    require(g::bvec4(g::vec2(0, -1), g::ivec2(3, 0)) == g::bvec4(false, true, true, false), // Check that this operation satisfies the expected behavior.
            "boolean conversion"); // Provide this text argument for the operation above.
    g::vec4 v(1, 2, 3, 4); // Construct v with these initial values.
    v.swizzle<0>() = 1; // Continue the expression using v, swizzle.
    require(v.swizzle<0>() == v.x, "scalar swizzle read/write"); // Check that this operation satisfies the expected behavior.
    require(v.r() == 1 && v.g() == 2 && v.b() == 3 && v.a() == 4, "color accessors"); // Check that this operation satisfies the expected behavior.
    require(v.s() == 1 && v.t() == 2 && v.p() == 3 && v.q() == 4, "texture accessors"); // Check that this operation satisfies the expected behavior.
    require(g::vec2(v.swizzle<3, 3>()) == g::vec2(4), "duplicate swizzle read"); // Check that this operation satisfies the expected behavior.
    v.swizzle<0, 1>() = v.swizzle<1, 0>(); // Continue the expression using v, swizzle, v.
    require(v == g::vec4(2, 1, 3, 4), "alias-safe swizzle assignment"); // Check that this operation satisfies the expected behavior.
    v.swizzle<0, 2>() = 8; // Continue the expression using v, swizzle.
    require(v == g::vec4(8, 1, 8, 4), "scalar swizzle assignment"); // Check that this operation satisfies the expected behavior.
    v.swizzle<0, 1>() += v.swizzle<1, 0>(); // Continue the expression using v, swizzle, v.
    require(v == g::vec4(9, 9, 8, 4), "alias-safe swizzle compound assignment"); // Check that this operation satisfies the expected behavior.
    g::vec2 q(1, 2); // Construct q with these initial values.
    q += q.swizzle<1, 0>(); // Update q with this contribution.
    require(q == g::vec2(3, 3), "vector compound assignment snapshots alias"); // Check that this operation satisfies the expected behavior.
    require(v.swizzle<0, 1>() * 2.0f == g::vec2(18), "swizzle expression arithmetic"); // Check that this operation satisfies the expected behavior.
    const g::vec4 constant(1, 2, 3, 4); // Construct constant with these initial values.
    require(constant.swizzle<3, 2, 1, 0>() == g::vec4(4, 3, 2, 1), "const swizzle copies values"); // Check that this operation satisfies the expected behavior.
    require(g::vec4(v.swizzle<0, 1>(), constant.swizzle<2, 3>()) == g::vec4(9, 9, 3, 4), // Check that this operation satisfies the expected behavior.
            "swizzle constructor flattening"); // Provide this text argument for the operation above.
    require(g::vec3(1, 2, 3) + g::vec3(4, 5, 6) == g::vec3(5, 7, 9), "vector addition"); // Check that this operation satisfies the expected behavior.
    require(2.0f * g::vec2(3, 4) == g::vec2(6, 8), "scalar-left multiply"); // Check that this operation satisfies the expected behavior.
    require(12.0f / g::vec2(3, 4) == g::vec2(4, 3), "scalar-left divide"); // Check that this operation satisfies the expected behavior.
    require(g::ivec2(5, 9) % 4 == g::ivec2(1), "integer remainder"); // Check that this operation satisfies the expected behavior.
    require((g::uvec2(1, 2) << 3) == g::uvec2(8, 16), "integer shifts"); // Check that this operation satisfies the expected behavior.
    require((g::ivec2(5, 6) & 3) == g::ivec2(1, 2), "integer and"); // Check that this operation satisfies the expected behavior.
    require((~g::uvec2(0)) == g::uvec2(UINT32_MAX), "integer complement"); // Check that this operation satisfies the expected behavior.
    require(g::all(g::equal(g::vec2(1), g::vec2(1))), "component equality"); // Check that this operation satisfies the expected behavior.
    require(g::any(g::lessThan(g::vec2(1, 5), g::vec2(2, 4))), "component ordering"); // Check that this operation satisfies the expected behavior.
    require(g::logicalNot(g::bvec2(true, false)) == g::bvec2(false, true), "logical not"); // Check that this operation satisfies the expected behavior.
} // End this block.

static void geometry_and_common() { // Implement geometry and common for the operations below.
    require(near(g::cross(g::vec3(1, 0, 0), g::vec3(0, 1, 0)), g::vec3(0, 0, 1)), "cross product"); // Check that this operation satisfies the expected behavior.
    require(near(g::cross(g::vec3(2, 3, 4), g::vec3(5, 6, 7)), g::vec3(-3, 6, -3)), // Check that this operation satisfies the expected behavior.
            "general cross product"); // Provide this text argument for the operation above.
    g::vec4 cross_lhs(2, 3, 4, 0), cross_rhs(5, 6, 7, 0); // Construct cross lhs with these initial values.
    require(g::cross(cross_lhs.swizzle<0, 1, 2>(), cross_rhs.swizzle<0, 1, 2>()) == // Check that this operation satisfies the expected behavior.
                g::vec3(-3, 6, -3), // Apply vec3 to these arguments.
            "cross swizzle expressions"); // Provide this text argument for the operation above.
    require(near(g::dot(g::vec3(1, 2, 3), g::vec3(4, 5, 6)), 32), "dot product"); // Check that this operation satisfies the expected behavior.
    require(near(g::length(g::vec2(3, 4)), 5), "length"); // Check that this operation satisfies the expected behavior.
    require(near(g::normalize(g::vec2(3, 4)), g::vec2(0.6f, 0.8f)), "normalize"); // Check that this operation satisfies the expected behavior.
    require(near(g::length(-3.0f), 3), "scalar length"); // Check that this operation satisfies the expected behavior.
    require(near(g::distance(g::vec2(1, 1), g::vec2(4, 5)), 5), "distance"); // Check that this operation satisfies the expected behavior.
    require(g::reflect(g::vec3(1, -1, 0), g::vec3(0, 1, 0)) == g::vec3(1, 1, 0), "reflect"); // Check that this operation satisfies the expected behavior.
    require(g::faceforward(g::vec3(0, 1, 0), g::vec3(0, 1, 0), g::vec3(0, 1, 0)) == // Check that this operation satisfies the expected behavior.
                g::vec3(0, -1, 0), // Apply vec3 to these arguments.
            "faceforward"); // Provide this text argument for the operation above.
    require(near(g::refract(g::vec2(0, -1), g::vec2(0, 1), 0.5f), g::vec2(0, -1)), // Check that this operation satisfies the expected behavior.
            "refract normal incidence"); // Provide this text argument for the operation above.
    require(g::refract(g::vec2(1, 0), g::vec2(0, 1), 2.0f) == g::vec2(0), // Check that this operation satisfies the expected behavior.
            "total internal reflection"); // Provide this text argument for the operation above.
    require(near(g::mod(-3.5f, 2.0f), 0.5), "GLSL mod negative dividend"); // Check that this operation satisfies the expected behavior.
    require(near(g::mod(3.5f, -2.0f), -0.5), "GLSL mod negative divisor"); // Check that this operation satisfies the expected behavior.
    require(near(g::mod(g::vec2(-3.5f, 4.5f), 2.0f), g::vec2(0.5f)), "vector mod"); // Check that this operation satisfies the expected behavior.
    require(near(g::fract(-1.25f), 0.75), "negative fract"); // Check that this operation satisfies the expected behavior.
    require(g::roundEven(g::vec4(0.5f, 1.5f, 2.5f, -1.5f)) == g::vec4(0, 2, 2, -2), "ties to even"); // Check that this operation satisfies the expected behavior.
    int rounding = std::fegetround(); // Set rounding from this expression.
    if (std::fesetround(FE_UPWARD) == 0) { // Check this condition before entering the following branch.
        bool same = g::roundEven(g::vec4(0.5f, 1.5f, 2.5f, -1.5f)) == g::vec4(0, 2, 2, -2); // Set same from this expression.
        std::fesetround(rounding); // Apply fesetround to these arguments.
        require(same, "roundEven independent of ambient rounding mode"); // Check that this operation satisfies the expected behavior.
    } // End this block.
    require(g::clamp(g::vec3(-1, 0.5f, 2), 0.0f, 1.0f) == g::vec3(0, 0.5f, 1), "clamp"); // Check that this operation satisfies the expected behavior.
    require(std::signbit(g::min(-0.0f, 0.0f)) && std::signbit(g::max(-0.0f, 0.0f)), // Check that this operation satisfies the expected behavior.
            "min/max equality preserve first argument"); // Provide this text argument for the operation above.
    require(near(g::mix(g::vec2(2), g::vec2(4), 0.25f), g::vec2(2.5f)), "mix numeric"); // Check that this operation satisfies the expected behavior.
    require(near(g::mix(g::vec2(2), g::vec2(4), 0.25), g::vec2(2.5f)), // Check that this operation satisfies the expected behavior.
            "mix C++ double literal remains numeric"); // Provide this text argument for the operation above.
    require(near(g::mix(2.0f, 4.0f, 0.25), 2.5), "scalar mix double literal remains numeric"); // Check that this operation satisfies the expected behavior.
    require(g::mix(g::ivec2(2, 3), g::ivec2(4, 5), g::bvec2(true, false)) == g::ivec2(4, 3), // Check that this operation satisfies the expected behavior.
            "mix boolean"); // Provide this text argument for the operation above.
    require(g::step(1.0f, g::vec2(0.5f, 1.0f)) == g::vec2(0, 1), "step scalar edge"); // Check that this operation satisfies the expected behavior.
    require(near(g::smoothstep(0.0f, 1.0f, g::vec3(0, 0.5f, 1)), g::vec3(0, 0.5f, 1)), // Check that this operation satisfies the expected behavior.
            "smoothstep"); // Provide this text argument for the operation above.
    require(near(g::degrees(g::radians(g::vec2(90, 180))), g::vec2(90, 180)), "angle conversions"); // Check that this operation satisfies the expected behavior.
    require(near(g::atan(g::vec2(1, -1), g::vec2(0)), g::vec2(1.57079633f, -1.57079633f)), // Check that this operation satisfies the expected behavior.
            "atan quadrant"); // Provide this text argument for the operation above.
    require(near(g::pow(g::vec2(2, 3), 2.0f), g::vec2(4, 9)), "vector power"); // Check that this operation satisfies the expected behavior.
    require(near(g::inversesqrt(g::vec2(4, 16)), g::vec2(0.5f, 0.25f)), "inverse square root"); // Check that this operation satisfies the expected behavior.
    g::vec2 integer; // Declare integer for the following calculation.
    require(g::modf(g::vec2(-1.25f, 2.5f), integer) == g::vec2(-0.25f, 0.5f) && // Check that this operation satisfies the expected behavior.
                integer == g::vec2(-1, 2), // Apply vec2 to these arguments.
            "modf outputs"); // Provide this text argument for the operation above.
    g::vec2 alias(-1.25f, 2.5f); // Construct alias with these initial values.
    g::vec2 fraction = g::modf(alias.swizzle<1, 0>(), alias.swizzle<0, 1>()); // Set fraction from this expression.
    require(fraction == g::vec2(0.5f, -0.25f) && alias == g::vec2(2, -1), // Check that this operation satisfies the expected behavior.
            "modf swizzle outputs snapshot aliased input"); // Provide this text argument for the operation above.
    g::ivec2 exponent; // Declare exponent for the following calculation.
    g::vec2 mantissa = g::frexp(g::vec2(4, 0.125f), exponent); // Set mantissa from this expression.
    require(g::ldexp(mantissa, exponent) == g::vec2(4, 0.125f), "frexp/ldexp"); // Check that this operation satisfies the expected behavior.
    g::ivec4 powers; // Declare powers for the following calculation.
    g::vec4 frexp_input(4, 0.125f, 8, 0.25f); // Construct frexp input with these initial values.
    g::vec2 significands = g::frexp(frexp_input.swizzle<0, 1>(), powers.swizzle<3, 2>()); // Set significands from this expression.
    require(g::ldexp(significands, powers.swizzle<3, 2>()) == g::vec2(4, 0.125f), // Check that this operation satisfies the expected behavior.
            "frexp and ldexp swizzle arguments"); // Provide this text argument for the operation above.
    require(g::isinf(g::vec2(std::numeric_limits<float>::infinity(), 1)) == g::bvec2(true, false), // Check that this operation satisfies the expected behavior.
            "isinf mask"); // Provide this text argument for the operation above.
    require(g::isnan(g::vec2(std::numeric_limits<float>::quiet_NaN(), 1)) == g::bvec2(true, false), // Check that this operation satisfies the expected behavior.
            "isnan mask"); // Provide this text argument for the operation above.
} // End this block.

static void matrices() { // Implement matrices for the operations below.
    g::mat2 a(1, 2, 3, 4); // Construct a with these initial values.
    require(g::vec4(a) == g::vec4(1, 2, 3, 4), "matrix to vector constructor column order"); // Check that this operation satisfies the expected behavior.
    require(g::vec4(9, a) == g::vec4(9, 1, 2, 3), "mixed matrix/vector component construction"); // Check that this operation satisfies the expected behavior.
    require(g::mat2(g::vec4(1, 2, 3, 4)) == a, "single vector matrix constructor"); // Check that this operation satisfies the expected behavior.
    require(g::mat2(g::vec3(1, 2, 3), g::vec2(4, 5)) == a, // Check that this operation satisfies the expected behavior.
            "matrix constructor discards final excess component"); // Provide this text argument for the operation above.
    require(g::scalar_cast<float>(g::vec3(5, 6, 7)) == 5, "vector to scalar constructor"); // Check that this operation satisfies the expected behavior.
    require(g::scalar_cast<int>(a) == 1, "matrix to scalar constructor"); // Check that this operation satisfies the expected behavior.
    require(g::scalar_cast<g::uint>(-1) == UINT32_MAX, "signed to unsigned scalar constructor"); // Check that this operation satisfies the expected behavior.
    const std::size_t& vector_length = g::vec4::length_value; // Set vector length from this expression.
    const std::size_t& column_count = g::mat2::columns; // Set column count from this expression.
    require(vector_length == 4 && column_count == 2, "public dimension constants have definitions"); // Check that this operation satisfies the expected behavior.
    require(a[0] == g::vec2(1, 2) && a[1] == g::vec2(3, 4), "matrix column order"); // Check that this operation satisfies the expected behavior.
    require(a * g::vec2(2, 3) == g::vec2(11, 16), "matrix-vector multiplication"); // Check that this operation satisfies the expected behavior.
    g::vec4 vector(2, 3, 2, 3); // Construct vector with these initial values.
    require(a * vector.swizzle<0, 1>() == g::vec2(11, 16), "matrix-swizzle multiplication"); // Check that this operation satisfies the expected behavior.
    require(vector.swizzle<2, 3>() * a == g::vec2(8, 18), "swizzle-matrix multiplication"); // Check that this operation satisfies the expected behavior.
    require(g::vec2(2, 3) * a == g::vec2(8, 18), "vector-matrix multiplication"); // Check that this operation satisfies the expected behavior.
    require(a * g::mat2(1) == a, "matrix multiplication identity"); // Check that this operation satisfies the expected behavior.
    require(a * a == g::mat2(7, 10, 15, 22), "matrix algebra multiplication"); // Check that this operation satisfies the expected behavior.
    require(near(g::determinant(a), -2), "determinant 2x2"); // Check that this operation satisfies the expected behavior.
    require(near(a * g::inverse(a), g::mat2(1)), "inverse 2x2"); // Check that this operation satisfies the expected behavior.
    require(g::transpose(a) == g::mat2(1, 3, 2, 4), "transpose"); // Check that this operation satisfies the expected behavior.
    require(g::matrixCompMult(a, a) == g::mat2(1, 4, 9, 16), "component matrix multiply"); // Check that this operation satisfies the expected behavior.
    require(g::outerProduct(g::vec3(1, 2, 3), g::vec2(4, 5)) == g::mat2x3(4, 8, 12, 5, 10, 15), // Check that this operation satisfies the expected behavior.
            "outer product columns"); // Provide this text argument for the operation above.
    g::mat3x2 rectangular(1, 2, 3, 4, 5, 6); // Construct rectangular with these initial values.
    require(rectangular * g::mat2x3(1, 0, 0, 0, 1, 0) == g::mat2(1, 2, 3, 4), // Check that this operation satisfies the expected behavior.
            "rectangular multiplication"); // Provide this text argument for the operation above.
    require(g::transpose(rectangular) == g::mat2x3(1, 3, 5, 2, 4, 6), "rectangular transpose"); // Check that this operation satisfies the expected behavior.
    require(g::mat3(a) == g::mat3(1, 2, 0, 3, 4, 0, 0, 0, 1), "matrix conversion identity fill"); // Check that this operation satisfies the expected behavior.
    g::mat3 b(3, 1, 3, 5, 5, 9, 2, 8, 2); // Construct b with these initial values.
    require(near(g::determinant(b), -88), "determinant 3x3"); // Check that this operation satisfies the expected behavior.
    require(near(b * g::inverse(b), g::mat3(1)), "inverse 3x3"); // Check that this operation satisfies the expected behavior.
    g::mat4 c(0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4); // Construct c with these initial values.
    require(near(g::determinant(c), -24), "determinant 4x4 pivot exchange"); // Check that this operation satisfies the expected behavior.
    require(near(c * g::inverse(c), g::mat4(1)), "inverse 4x4 pivot exchange"); // Check that this operation satisfies the expected behavior.
    require(g::all(g::isnan(g::inverse(g::mat2(0))[0])), "singular host inverse NaNs"); // Check that this operation satisfies the expected behavior.
    require(a + 2 == g::mat2(3, 4, 5, 6), "matrix scalar addition"); // Check that this operation satisfies the expected behavior.
    require(5 - a == g::mat2(4, 3, 2, 1), "scalar matrix subtraction"); // Check that this operation satisfies the expected behavior.
    require(12 / g::mat2(2, 3, 4, 6) == g::mat2(6, 4, 3, 2), "scalar matrix division"); // Check that this operation satisfies the expected behavior.
    a *= g::mat2(1); // Update a with this contribution.
    require(a == g::mat2(1, 2, 3, 4), "compound matrix multiplication"); // Check that this operation satisfies the expected behavior.
} // End this block.

static void arrays() { // Implement arrays for the operations below.
    g::array<float, 4> values(1, 2, 3, 4); // Construct values with these initial values.
    require(values.length() == 4 && values[2] == 3, "fixed array construction and length"); // Check that this operation satisfies the expected behavior.
    g::array<float, 4> zero; // Declare zero for the following calculation.
    require(zero[0] == 0 && zero[3] == 0, "fixed array zero initialization"); // Check that this operation satisfies the expected behavior.
    g::array<float, 4> copy = values; // Set copy from this expression.
    copy[0] = 7; // Set copy[0] from this expression.
    require(values[0] == 1 && copy[0] == 7, "array copy is independent"); // Check that this operation satisfies the expected behavior.
    zero = values; // Set zero from this expression.
    require(zero == values && zero != copy, "whole-array assignment and equality"); // Check that this operation satisfies the expected behavior.
    g::array<g::vec2, 2> vectors(g::vec2(1, 2), g::vec2(3, 4)); // Construct vectors with these initial values.
    require(vectors[1] == g::vec2(3, 4), "array vector elements are not flattened"); // Check that this operation satisfies the expected behavior.
    g::array<int, 3> braces = {1, 2, 3}; // Set braces from this expression.
    require(braces[2] == 3, "array brace initialization"); // Check that this operation satisfies the expected behavior.
    const std::size_t& count = g::array<int, 3>::size_value; // Set count from this expression.
    require(count == 3, "array dimension constant has a definition"); // Check that this operation satisfies the expected behavior.
    bool mismatch = false; // Set mismatch from this expression.
    try { // Run the operation while retaining the error handling below.
        g::array<int, 3> invalid = {1, 2}; // Set invalid from this expression.
        (void)invalid; // Continue the expression using void, invalid.
    } catch (const std::invalid_argument&) { // Report the exception from the preceding operation.
        mismatch = true; // Set mismatch from this expression.
    } // End this block.
    require(mismatch, "array initializer list rejects wrong element count"); // Check that this operation satisfies the expected behavior.
} // End this block.

static void bits_and_packing() { // Implement bits and packing for the operations below.
    require(g::floatBitsToUint(1.0f) == 0x3f800000u, "float bit encoding"); // Check that this operation satisfies the expected behavior.
    require(g::uintBitsToFloat(0xbf800000u) == -1.0f, "float bit decoding"); // Check that this operation satisfies the expected behavior.
    require(g::uintBitsToFloat(g::floatBitsToUint(g::vec2(1, -1))) == g::vec2(1, -1), // Check that this operation satisfies the expected behavior.
            "vector bit roundtrip"); // Provide this text argument for the operation above.
    require(g::bitCount(g::ivec2(-1, 5)) == g::ivec2(32, 2), "bit counts"); // Check that this operation satisfies the expected behavior.
    require(g::findLSB(g::uvec3(0, 8, 0x80000000u)) == g::ivec3(-1, 3, 31), // Check that this operation satisfies the expected behavior.
            "least significant set bits"); // Provide this text argument for the operation above.
    require(g::findMSB(g::ivec4(0, -1, -2, 16)) == g::ivec4(-1, -1, 0, 4), // Check that this operation satisfies the expected behavior.
            "signed most significant bit semantics"); // Provide this text argument for the operation above.
    require(g::bitfieldReverse(1u) == 0x80000000u, "bit reversal"); // Check that this operation satisfies the expected behavior.
    require(g::bitfieldExtract(std::int32_t(0xf0), 4, 4) == -1, "signed bitfield extension"); // Check that this operation satisfies the expected behavior.
    require(g::bitfieldExtract(std::uint32_t(0xf0), 4, 4) == 15u, "unsigned bitfield extraction"); // Check that this operation satisfies the expected behavior.
    require(g::bitfieldExtract(std::int32_t(-2), 0, 32) == -2, "full signed field"); // Check that this operation satisfies the expected behavior.
    require(g::bitfieldExtract(3u, 32, 0) == 0u, "empty bitfield"); // Check that this operation satisfies the expected behavior.
    require(g::bitfieldInsert(0xff00u, 3u, 0, 4) == 0xff03u, "bitfield insertion"); // Check that this operation satisfies the expected behavior.
    std::uint32_t carry, borrow, hi, lo; // Declare carry for the following calculation.
    require(g::uaddCarry(UINT32_MAX, 2u, carry) == 1u && carry == 1u, "unsigned carry"); // Check that this operation satisfies the expected behavior.
    g::uvec4 vector_carry(UINT32_MAX, 1, 0, 0); // Construct vector carry with these initial values.
    g::uvec2 vector_sum = // Set vector sum from this expression.
        g::uaddCarry(vector_carry.swizzle<0, 1>(), g::uvec2(2, 3), vector_carry.swizzle<1, 0>()); // Apply uadd Carry to these arguments.
    require(vector_sum == g::uvec2(1, 4) && vector_carry.swizzle<0, 1>() == g::uvec2(0, 1), // Check that this operation satisfies the expected behavior.
            "vector carry with aliased swizzle output"); // Provide this text argument for the operation above.
    require(g::usubBorrow(0u, 1u, borrow) == UINT32_MAX && borrow == 1u, "unsigned borrow"); // Check that this operation satisfies the expected behavior.
    g::umulExtended(UINT32_MAX, UINT32_MAX, hi, lo); // Apply umul Extended to these arguments.
    require(hi == UINT32_MAX - 1u && lo == 1u, "unsigned extended product"); // Check that this operation satisfies the expected behavior.
    g::uvec4 product; // Declare product for the following calculation.
    g::umulExtended(g::uvec2(UINT32_MAX), // Apply umul Extended to these arguments.
                    g::uvec2(UINT32_MAX), // Apply uvec2 to these arguments.
                    product.swizzle<0, 1>(), // Continue the expression using product, swizzle.
                    product.swizzle<2, 3>()); // Continue the expression using product, swizzle.
    require(product == g::uvec4(UINT32_MAX - 1u, UINT32_MAX - 1u, 1u, 1u), // Check that this operation satisfies the expected behavior.
            "extended vector product swizzle outputs"); // Provide this text argument for the operation above.
    std::int32_t signed_hi, signed_lo; // Declare signed hi for the following calculation.
    g::imulExtended(-1, 2, signed_hi, signed_lo); // Apply imul Extended to these arguments.
    require(signed_hi == -1 && signed_lo == -2, "signed extended product"); // Check that this operation satisfies the expected behavior.
    require(g::packUnorm2x16(g::vec2(0, 1)) == 0xffff0000u, "pack normalized halves"); // Check that this operation satisfies the expected behavior.
    require(g::unpackUnorm2x16(0xffff0000u) == g::vec2(0, 1), "unpack normalized halves"); // Check that this operation satisfies the expected behavior.
    require(near(g::unpackSnorm2x16(g::packSnorm2x16(g::vec2(-1, 1))), g::vec2(-1, 1)), // Check that this operation satisfies the expected behavior.
            "signed normalized halves"); // Provide this text argument for the operation above.
    require(g::packUnorm4x8(g::vec4(1, 0, 1, 0)) == 0x00ff00ffu, "packed byte order"); // Check that this operation satisfies the expected behavior.
    require(near(g::unpackSnorm4x8(g::packSnorm4x8(g::vec4(-1, 0, 1, 0))), g::vec4(-1, 0, 1, 0)), // Check that this operation satisfies the expected behavior.
            "signed normalized bytes"); // Provide this text argument for the operation above.
    require(g::packHalf2x16(g::vec2(1, -2)) == 0xc0003c00u, "half bit encoding"); // Check that this operation satisfies the expected behavior.
    require(g::unpackHalf2x16(0xc0003c00u) == g::vec2(1, -2), "half bit decoding"); // Check that this operation satisfies the expected behavior.
    require(g::unpackHalf2x16(g::packHalf2x16(g::vec2(65504, 0.00006103515625f))) == // Check that this operation satisfies the expected behavior.
                g::vec2(65504, 0.00006103515625f), // Apply vec2 to these arguments.
            "half finite limits"); // Provide this text argument for the operation above.
    require(g::unpackHalf2x16(g::packHalf2x16(g::vec2(0.000000059604644775390625f, 0))) == // Check that this operation satisfies the expected behavior.
                g::vec2(0.000000059604644775390625f, 0), // Apply vec2 to these arguments.
            "half subnormal"); // Provide this text argument for the operation above.
    require(g::packHalf2x16(g::vec2(1.00048828125f, 1.00146484375f)) == 0x3c023c00u, // Check that this operation satisfies the expected behavior.
            "half ties to even"); // Provide this text argument for the operation above.
    require(g::isinf(g::unpackHalf2x16(g::packHalf2x16( // Check that this operation satisfies the expected behavior.
                g::vec2(std::numeric_limits<float>::infinity())))) == g::bvec2(true), // Apply vec2 to these arguments.
            "half infinities"); // Provide this text argument for the operation above.
} // End this block.

int main() { // Run this example and validate its results.
    try { // Run the operation while retaining the error handling below.
        vectors(); // Apply vectors to these arguments.
        geometry_and_common(); // Apply geometry and common to these arguments.
        matrices(); // Apply matrices to these arguments.
        arrays(); // Apply arrays to these arguments.
        bits_and_packing(); // Apply bits and packing to these arguments.
        std::cout << "GLSL math checks passed\n"; // Write this part of the example result or diagnostic.
    } catch (const std::exception& e) { // Report the exception from the preceding operation.
        std::cerr << "GLSL math check failed: " << e.what() << '\n'; // Write this part of the example result or diagnostic.
        return 1; // Report a failed check or operation.
    } // End this block.
} // End this block.
