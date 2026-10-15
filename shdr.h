// shdr 1.0.0
// C++11 CPU and OpenGL compute pipelines.
#ifndef SHDR_SINGLE_HEADER_INCLUDED
#define SHDR_SINGLE_HEADER_INCLUDED

// ---- math.hpp ----
/** @file math.hpp
 * @brief GLSL-compatible scalar, vector and matrix operations for C++11.
 *
 * Vectors provide x/y/z/w component fields, indexed access, and writable
 * swizzles. Color and texture-coordinate aliases are accessor functions in
 * ordinary C++ code. Matrices use GLSL column-major order. The same types and
 * functions are available to generated CPU kernels and application code.
 *
 * Floating-point precision and exceptional values depend on the implementation.
 * Normalize nonzero vectors, invert nonsingular matrices and use valid shift
 * counts. Keep signed arithmetic within its representable range and use
 * nonnegative signed operands for left shifts to follow portable C++ rules.
 * Resource element layout is defined by the buffer and image APIs separately.
 */
#ifndef SHDR_DETAIL_MATH_HPP // Include this math layer only once.
#define SHDR_DETAIL_MATH_HPP // Remember that the math layer is available.
#include <cmath> // Use the standard scalar math functions.
#include <cstddef> // Use portable size types.
#include <cstdint> // Use exact-width GLSL integer types.
#include <cstring> // Copy bit patterns through byte storage.
#include <initializer_list> // Accept ordinary C++ brace lists.
#include <limits> // Query numeric limits and quiet NaNs.
#include <ostream> // Support readable vector and matrix output.
#include <stdexcept> // Report invalid array initializers.
#include <type_traits> // Choose only compatible template overloads.
namespace shdr { // Keep these names inside the shdr namespace.
namespace glsl { // Keep these names inside the glsl namespace.
/** @brief GLSL's portable 32-bit unsigned scalar spelling. */
typedef std::uint32_t uint; // Use a portable 32-bit unsigned scalar.
template <class T, std::size_t N> class vec; // Store a fixed-size vector with named component fields.
template <class T, std::size_t N> class array; // Store a fixed number of values of one element type.
template <class T, std::size_t N, std::size_t... I> class swizzle_ref; // Describe a component selection that refers to a vector.
template <class T, std::size_t C, std::size_t R> class mat; // Describe a matrix by its scalar type and dimensions.
namespace detail { // Keep these names inside the detail namespace.
template <class T, std::size_t N> struct storage; // Choose named storage for the vector dimension.
template <class T> struct storage<T, 2> { // Choose named storage for the vector dimension.
    T x, y; // Store the named vector components in order.
    T& at(std::size_t i) { // Select the named component matching this index.
        return i == 0 ? x : y; // Select the named field for this valid component index.
    } // Finish the at operation.
    const T& at(std::size_t i) const { // Select the named component matching this index.
        return i == 0 ? x : y; // Select the named field for this valid component index.
    } // Finish the at operation.
}; // Finish the storage type description.
template <class T> struct storage<T, 3> { // Choose named storage for the vector dimension.
    T x, y, z; // Store the named vector components in order.
    T& at(std::size_t i) { // Select the named component matching this index.
        return i == 0 ? x : (i == 1 ? y : z); // Select the named field for this valid component index.
    } // Finish the at operation.
    const T& at(std::size_t i) const { // Select the named component matching this index.
        return i == 0 ? x : (i == 1 ? y : z); // Select the named field for this valid component index.
    } // Finish the at operation.
}; // Finish the storage type description.
template <class T> struct storage<T, 4> { // Choose named storage for the vector dimension.
    T x, y, z, w; // Store the named vector components in order.
    T& at(std::size_t i) { // Select the named component matching this index.
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w)); // Select the named field for this valid component index.
    } // Finish the at operation.
    const T& at(std::size_t i) const { // Select the named component matching this index.
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w)); // Select the named field for this valid component index.
    } // Finish the at operation.
}; // Finish the storage type description.
template <class A> struct shape { // Describe an ordinary scalar input.
    typedef A scalar; // Keep the chosen component type for the calculation.
    static const std::size_t size = 1; // Treat a scalar as one component.
    static const bool vector = false; // Keep this input out of vector-only overloads.
    static const bool valid = std::is_arithmetic<A>::value; // Record whether componentwise arithmetic accepts this shape.
}; // Finish the shape type description.
template <class T, std::size_t N> struct shape<vec<T, N>> { // Describe the scalar type and size of a vector.
    typedef T scalar; // Keep the chosen component type for the calculation.
    static const std::size_t size = N; // Keep the fixed component count available at compile time.
    static const bool vector = true; // Mark this input as a vector-shaped value.
    static const bool valid = true; // Record whether componentwise arithmetic accepts this shape.
}; // Finish the shape type description.
template <class T, std::size_t N, std::size_t... I> struct shape<swizzle_ref<T, N, I...>> { // Describe the scalar type and size of a component selection.
    typedef T scalar; // Keep the chosen component type for the calculation.
    static const std::size_t size = sizeof...(I); // Count the selected vector components.
    static const bool vector = true; // Mark this input as a vector-shaped value.
    static const bool valid = true; // Record whether componentwise arithmetic accepts this shape.
}; // Finish the shape type description.
template <class A> struct traits : shape<typename std::decay<A>::type> {}; // Ignore references and const when inspecting an input type.
template <class... A> struct count; // Count components supplied by constructor arguments.
template <> struct count<> { // Count components supplied by constructor arguments.
    static const std::size_t value = 0; // Start the component count at zero.
}; // Finish the count type description.
template <class A, class... B> struct count<A, B...> { // Count components supplied by constructor arguments.
    static const std::size_t value = traits<A>::size + count<B...>::value; // Add this argument count to the remaining components.
}; // Finish the count type description.
template <class... A> struct prefix_count; // Count constructor components before the final argument.
template <class Last> struct prefix_count<Last> { // Count constructor components before the final argument.
    static const std::size_t value = 0; // Start the component count at zero.
}; // Finish the prefix_count type description.
template <class A, class B, class... Rest> struct prefix_count<A, B, Rest...> { // Count constructor components before the final argument.
    static const std::size_t value = traits<A>::size + prefix_count<B, Rest...>::value; // Add this argument count to the remaining components.
}; // Finish the prefix_count type description.
template <class A> struct is_matrix : std::false_type {}; // Identify matrix types at compile time.
template <class T, std::size_t C, std::size_t R> struct is_matrix<mat<T, C, R>> : std::true_type {}; // Identify matrix types at compile time.
template <class T, std::size_t C, std::size_t R> struct shape<mat<T, C, R>> { // Describe a matrix separately from vector arithmetic.
    typedef T scalar; // Keep the chosen component type for the calculation.
    static const std::size_t size = C * R; // Count scalar entries across every matrix column.
    static const bool vector = false; // Keep this input out of vector-only overloads.
    static const bool valid = false; // Record whether componentwise arithmetic accepts this shape.
}; // Finish the shape type description.
template <class... A> struct any_matrix; // Check whether a constructor argument is a matrix.
template <> struct any_matrix<> : std::false_type {}; // Check whether a constructor argument is a matrix.
template <class A, class... B> // Accept the matching input types and dimensions.
struct any_matrix<A, B...> // Describe this input or selection at compile time.
    : std::integral_constant<bool, // Turn the component rule into a compile-time boolean.
                             is_matrix<typename std::decay<A>::type>::value || // Complete the type and dimension rule for this overload.
                                 any_matrix<B...>::value> {}; // Complete the type and dimension rule for this overload.
template <std::size_t Needle, std::size_t... I> struct contains; // Check whether the remaining index list contains an index.
template <std::size_t Needle> struct contains<Needle> : std::false_type {}; // Check whether the remaining index list contains an index.
template <std::size_t Needle, std::size_t A, std::size_t... I> // Accept the matching input types and dimensions.
struct contains<Needle, A, I...> // Describe this input or selection at compile time.
    : std::integral_constant<bool, Needle == A || contains<Needle, I...>::value> {}; // Turn the component rule into a compile-time boolean.
template <std::size_t... I> struct unique; // Check that selected component indices are distinct.
template <> struct unique<> : std::true_type {}; // Check that selected component indices are distinct.
template <std::size_t A, std::size_t... I> // Accept the matching input types and dimensions.
struct unique<A, I...> // Describe this input or selection at compile time.
    : std::integral_constant<bool, !contains<A, I...>::value && unique<I...>::value> {}; // Turn the component rule into a compile-time boolean.
template <class A> struct writable_shape : std::false_type {}; // Identify vector selections that can receive values.
template <class T, std::size_t N> struct writable_shape<vec<T, N>> : std::true_type {}; // Describe the scalar type and size of a vector.
template <class T, std::size_t N, std::size_t... I> // Accept the matching input types and dimensions.
struct writable_shape<swizzle_ref<T, N, I...>> : unique<I...> {}; // Describe this input or selection at compile time.
template <class A> // Accept the matching input types and dimensions.
struct writable // Describe this input or selection at compile time.
    : std::integral_constant<bool, // Turn the component rule into a compile-time boolean.
                             writable_shape<typename std::decay<A>::type>::value && // Complete the type and dimension rule for this overload.
                                 !std::is_const<typename std::remove_reference<A>::type>::value> {}; // Complete the type and dimension rule for this overload.
template <std::size_t N, std::size_t... I> struct in_range; // Check that every selected index fits the vector.
template <std::size_t N> struct in_range<N> : std::true_type {}; // Check that every selected index fits the vector.
template <std::size_t N, std::size_t A, std::size_t... I> // Accept the matching input types and dimensions.
struct in_range<N, A, I...> : std::integral_constant<bool, // Check this index and the remaining selected indices.
    A < N && in_range<N, I...>::value> {}; // Require every selected component index to fit the vector.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<!traits<A>::vector, typename traits<A>::scalar>::type // Enable this overload only for compatible input types.
component(const A& a, std::size_t) { // Read the requested scalar or vector component.
    return a; // Return the scalar unchanged.
} // Finish the component operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<traits<A>::vector, typename traits<A>::scalar>::type // Enable this overload only for compatible input types.
component(const A& a, std::size_t i) { // Read the requested scalar or vector component.
    return a[i]; // Read the requested scalar or vector component.
} // Finish the component operation.
template <class A, class B> struct binary { // Choose compatible componentwise operand shapes.
    static const bool valid = // Record whether componentwise arithmetic accepts this shape.
        traits<A>::valid && traits<B>::valid && (traits<A>::vector || traits<B>::vector) && // Complete the type and dimension rule for this overload.
        (!traits<A>::vector || !traits<B>::vector || traits<A>::size == traits<B>::size); // Require equal lengths when both operands are vectors.
    typedef typename std::conditional<traits<A>::vector, // Name the type selected for this math result.
                                      typename traits<A>::scalar, // Preserve the selected component type and result shape.
                                      typename traits<B>::scalar>::type scalar; // Preserve the selected component type and result shape.
    static const std::size_t size = traits<A>::vector ? traits<A>::size : traits<B>::size; // Record the compile-time component rule.
    typedef vec<scalar, size> result; // Name the vector type produced by this operation.
}; // Finish the binary type description.
template <class A> struct result { // Choose the value type for this expression result.
    typedef typename std::conditional<traits<A>::vector, // Name the type selected for this math result.
                                      vec<typename traits<A>::scalar, traits<A>::size>, // Preserve the selected component type and result shape.
                                      typename traits<A>::scalar>::type type; // Preserve the selected component type and result shape.
}; // Finish the result type description.
template <class T, class U> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<U>::value>::type // Enable this overload only for compatible input types.
append(T* target, std::size_t& n, const U& a) { // Append the next argument to the flat component list.
    target[n++] = static_cast<T>(a); // Append this component and advance the destination position.
} // Finish the append operation.
template <class T, class U, std::size_t N> // Accept the matching input types and dimensions.
void append(T* target, std::size_t& n, const vec<U, N>& a) { // Append constructor components in their original order.
    for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
        target[n++] = static_cast<T>(a[i]); // Append this component and advance the destination position.
} // Finish the append operation.
template <class T, class U, std::size_t N, std::size_t... I> // Accept the matching input types and dimensions.
void append(T* target, std::size_t& n, const swizzle_ref<U, N, I...>& a) { // Append constructor components in their original order.
    for (std::size_t i = 0; i < sizeof...(I); ++i) // Visit each component needed by this calculation.
        target[n++] = static_cast<T>(a[i]); // Append this component and advance the destination position.
} // Finish the append operation.
template <class T, class U, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
void append(T* target, std::size_t& n, const mat<U, C, R>& a) { // Append constructor components in their original order.
    for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
        for (std::size_t r = 0; r < R; ++r) // Visit each matrix row.
            target[n++] = static_cast<T>(a[c][r]); // Append this component and advance the destination position.
} // Finish the append operation.
template <class T> void flatten(T*, std::size_t&) { // Accept the matching input types and dimensions.
} // Finish the flatten operation.
template <class T, class A, class... B> // Accept the matching input types and dimensions.
void flatten(T* target, std::size_t& n, const A& a, const B&... b) { // Collect components from the remaining constructor arguments.
    append(target, n, a); // Append the next argument to the flat component list.
    flatten(target, n, b...); // Append the remaining arguments in order.
} // Finish the flatten operation.
} // namespace detail

/** @brief A fixed-size array for GLSL local variables and function arguments.
 * The size must be positive. Default construction initializes every element to
 * its zero/default value.  component construction requires exactly N elements.
 * Copy construction and assignment copy all elements. Use length() to query
 * the element count and operator[] to select an element with a valid index.
 * Resource ports define buffer and uniform memory layouts separately.
 */
template <class T, std::size_t N> class array { // Store a fixed number of values of one element type.
    static_assert(N > 0, "GLSL fixed arrays require a positive size"); // Require at least one fixed-array element.
    T elements_[N]; // Keep exactly N fixed-array elements.

  public: // Expose the public math operations.
    typedef T value_type; // Expose the element type to host code and templates.
    static const std::size_t size_value = N; // Keep the fixed component count available at compile time.
    /** @brief Return the number of array elements. */
    static constexpr std::size_t length() { // Record the compile-time component rule.
        return N; // Return the fixed number of available components.
    } // Finish the length operation.
    /** @brief Initialize every element to its zero/default value. */
    array() : elements_() { // Initialize fixed-array storage from the supplied values.
    } // Finish initializing the fixed array.
    /** @brief Construct from exactly N scalar or aggregate element values. */
    template <class... A> // Accept the matching input types and dimensions.
    explicit array(const A&... values) : elements_{static_cast<T>(values)...} { // Initialize fixed-array storage from the supplied values.
        static_assert(sizeof...(A) == N, "array constructor element count mismatch"); // Require exactly one argument for each array element.
    } // Finish initializing the fixed array.
    /** @brief Initialize from a C++ brace list containing exactly N elements.
     * A length mismatch throws std::invalid_argument.
     */
    array(std::initializer_list<T> values) : elements_() { // Initialize fixed-array storage from the supplied values.
        if (values.size() != N) // Reject a brace list with the wrong element count.
            throw std::invalid_argument("array initializer element count mismatch"); // Report an initializer length mismatch.
        std::size_t i = 0; // Begin filling the first array element.
        for (const T& value : values) // Read initializer values in their original order.
            elements_[i++] = value; // Store the next brace-list element.
    } // Finish initializing the fixed array.
    /** @brief Read or modify element i.  i must be less than length(). */
    T& operator[](std::size_t i) { // Provide this arithmetic or component access overload.
        return elements_[i]; // Return the selected component with its original reference type.
    } // Finish this operator overload.
    const T& operator[](std::size_t i) const { // Provide this arithmetic or component access overload.
        return elements_[i]; // Return the selected component with its original reference type.
    } // Finish this operator overload.
}; // Finish the array type description.
template <class T, std::size_t N> const std::size_t array<T, N>::size_value; // Accept the matching input types and dimensions.
/** @brief Compare two arrays by comparing every element. */
template <class T, std::size_t N> bool operator==(const array<T, N>& a, const array<T, N>& b) { // Accept the matching input types and dimensions.
    for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
        if (!(a[i] == b[i])) // Stop when corresponding components differ.
            return false; // Report that at least one component differs.
    return true; // Report that every required component matched.
} // Finish this operator overload.
/** @brief Return true when any pair of corresponding elements differs. */
template <class T, std::size_t N> bool operator!=(const array<T, N>& a, const array<T, N>& b) { // Accept the matching input types and dimensions.
    return !(a == b); // Return the component count or geometric length.
} // Finish this operator overload.

/** @brief A vector with two, three or four components of one scalar type.
 * A scalar constructor fills every component. Component constructors consume
 * scalar, vector and matrix arguments from left to right.  matrices contribute
 * columns in order. Excess components in the final argument are discarded.
 * Vector conversion takes the first N components of a source with at least N components.
 * Default construction initializes every component to zero. Indexing requires
 * an index less than length().
 */
template <class T, std::size_t N> class vec : public detail::storage<T, N> { // Store a fixed-size vector with named component fields.
    static_assert(N >= 2 && N <= 4, "GLSL vectors require 2 to 4 components"); // Require two, three or four vector components.

  public: // Expose the public math operations.
    typedef T value_type; // Expose the element type to host code and templates.
    typedef T component_type; // Expose the element type to host code and templates.
    static const std::size_t length_value = N; // Keep the fixed component count available at compile time.
    /** @brief Return the number of vector components. */
    static constexpr std::size_t length() { // Record the compile-time component rule.
        return N; // Return the fixed number of available components.
    } // Finish the length operation.
    vec() { // Initialize vector components from the supplied values.
        for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
            (*this)[i] = T(); // Update the selected destination component.
    } // Finish initializing the vector.
    template <class U, typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0> // Accept the matching input types and dimensions.
    explicit vec(U value) { // Initialize vector components from the supplied values.
        for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
            (*this)[i] = static_cast<T>(value); // Update the selected destination component.
    } // Finish initializing the vector.
    template <class U, std::size_t M> explicit vec(const vec<U, M>& value) { // Accept the matching input types and dimensions.
        static_assert(M >= N, "vector conversion cannot invent components"); // Check this type or dimension requirement at compile time.
        for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
            (*this)[i] = static_cast<T>(value[i]); // Update the selected destination component.
    } // Finish initializing the vector.
    template <class U, std::size_t M, std::size_t... I> vec(const swizzle_ref<U, M, I...>& value) { // Accept the matching input types and dimensions.
        static_assert(sizeof...(I) >= N, "swizzle conversion cannot invent components"); // Check this type or dimension requirement at compile time.
        for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
            (*this)[i] = static_cast<T>(value[i]); // Update the selected destination component.
    } // Finish initializing the vector.
    /** @brief Read matrix columns consecutively to initialize the vector. */
    template <class U, std::size_t C, std::size_t R> explicit vec(const mat<U, C, R>& value) { // Accept the matching input types and dimensions.
        static_assert(C * R >= N, "matrix does not provide enough vector components"); // Check this type or dimension requirement at compile time.
        for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
            (*this)[i] = static_cast<T>(value[i / R][i % R]); // Update the selected destination component.
    } // Finish initializing the vector.
    template <class A, class B, class... Rest> // Accept the matching input types and dimensions.
    explicit vec(const A& a, const B& b, const Rest&... rest) { // Initialize vector components from the supplied values.
        static_assert(detail::count<A, B, Rest...>::value >= N, // Require enough components to fill the destination.
                      "vector constructor has too few components"); // Explain the rejected input: vector constructor has too few components.
        static_assert(detail::prefix_count<A, B, Rest...>::value < N, // Reject constructor arguments that would go unused.
                      "vector constructor has unused arguments"); // Explain the rejected input: vector constructor has unused arguments.
        T values[detail::count<A, B, Rest...>::value]; // Allocate space for the supplied constructor components.
        std::size_t n = 0; // Begin appending at the first temporary component.
        detail::flatten(values, n, a, b, rest...); // Append the remaining arguments in order.
        for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
            (*this)[i] = values[i]; // Update the selected destination component.
    } // Finish initializing the vector.
    /** @brief Read or modify component i. */
    T& operator[](std::size_t i) { // Provide this arithmetic or component access overload.
        return this->at(i); // Return the selected component with its original reference type.
    } // Finish this operator overload.
    const T& operator[](std::size_t i) const { // Provide this arithmetic or component access overload.
        return this->at(i); // Return the selected component with its original reference type.
    } // Finish this operator overload.
    /** @brief Color aliases for x/y/z/w.  b and a require three and four components. */
    T& r() { // Expose the first component through its color name.
        return (*this)[0]; // Return the selected component with its original reference type.
    } // Finish the r operation.
    const T& r() const { // Expose the first component through its color name.
        return (*this)[0]; // Return the selected component with its original reference type.
    } // Finish the r operation.
    T& g() { // Expose the second component through its color name.
        return (*this)[1]; // Return the selected component with its original reference type.
    } // Finish the g operation.
    const T& g() const { // Expose the second component through its color name.
        return (*this)[1]; // Return the selected component with its original reference type.
    } // Finish the g operation.
    T& b() { // Expose the third component through its color name.
        static_assert(N >= 3, "b needs three components"); // Check this type or dimension requirement at compile time.
        return (*this)[2]; // Return the selected component with its original reference type.
    } // Finish the b operation.
    const T& b() const { // Expose the third component through its color name.
        static_assert(N >= 3, "b needs three components"); // Check this type or dimension requirement at compile time.
        return (*this)[2]; // Return the selected component with its original reference type.
    } // Finish the b operation.
    T& a() { // Expose the fourth component through its color name.
        static_assert(N >= 4, "a needs four components"); // Check this type or dimension requirement at compile time.
        return (*this)[3]; // Return the selected component with its original reference type.
    } // Finish the a operation.
    const T& a() const { // Expose the fourth component through its color name.
        static_assert(N >= 4, "a needs four components"); // Check this type or dimension requirement at compile time.
        return (*this)[3]; // Return the selected component with its original reference type.
    } // Finish the a operation.
    /** @brief Texture-coordinate aliases for x/y/z/w. */
    T& s() { // Expose the first texture coordinate.
        return (*this)[0]; // Return the selected component with its original reference type.
    } // Finish the s operation.
    const T& s() const { // Expose the first texture coordinate.
        return (*this)[0]; // Return the selected component with its original reference type.
    } // Finish the s operation.
    T& t() { // Expose the second texture coordinate.
        return (*this)[1]; // Return the selected component with its original reference type.
    } // Finish the t operation.
    const T& t() const { // Expose the second texture coordinate.
        return (*this)[1]; // Return the selected component with its original reference type.
    } // Finish the t operation.
    T& p() { // Expose the third texture coordinate.
        static_assert(N >= 3, "p needs three components"); // Check this type or dimension requirement at compile time.
        return (*this)[2]; // Return the selected component with its original reference type.
    } // Finish the p operation.
    const T& p() const { // Expose the third texture coordinate.
        static_assert(N >= 3, "p needs three components"); // Check this type or dimension requirement at compile time.
        return (*this)[2]; // Return the selected component with its original reference type.
    } // Finish the p operation.
    T& q() { // Expose the fourth texture coordinate.
        static_assert(N >= 4, "q needs four components"); // Check this type or dimension requirement at compile time.
        return (*this)[3]; // Return the selected component with its original reference type.
    } // Finish the q operation.
    const T& q() const { // Expose the fourth texture coordinate.
        static_assert(N >= 4, "q needs four components"); // Check this type or dimension requirement at compile time.
        return (*this)[3]; // Return the selected component with its original reference type.
    } // Finish the q operation.
    /** @brief Select one component as a scalar reference. */
    template <std::size_t I> T& swizzle() { // Accept the matching input types and dimensions.
        static_assert(I < N, "swizzle index out of range"); // Reject a component index outside the vector.
        return (*this)[I]; // Return the selected component with its original reference type.
    } // Finish the swizzle operation.
    template <std::size_t I> const T& swizzle() const { // Accept the matching input types and dimensions.
        static_assert(I < N, "swizzle index out of range"); // Reject a component index outside the vector.
        return (*this)[I]; // Return the selected component with its original reference type.
    } // Finish the swizzle operation.
    /** @brief Select components in the requested order.  writable destinations
     * must contain unique indices. Const vectors produce an independent value.
     */
    template <std::size_t... I> // Accept the matching input types and dimensions.
    typename std::enable_if<(sizeof...(I) >= 2), swizzle_ref<T, N, I...>>::type swizzle() { // Enable this overload only for compatible input types.
        static_assert(sizeof...(I) >= 2 && sizeof...(I) <= 4, "swizzles require 2 to 4 components"); // Check this type or dimension requirement at compile time.
        static_assert(detail::in_range<N, I...>::value, "swizzle index out of range"); // Reject a component index outside the vector.
        return swizzle_ref<T, N, I...>(*this); // Select components in the requested order.
    } // Finish the swizzle operation.
    template <std::size_t... I> // Accept the matching input types and dimensions.
    typename std::enable_if<(sizeof...(I) >= 2), vec<T, sizeof...(I)>>::type swizzle() const { // Enable this overload only for compatible input types.
        static_assert(sizeof...(I) >= 2 && sizeof...(I) <= 4, "swizzles require 2 to 4 components"); // Check this type or dimension requirement at compile time.
        static_assert(detail::in_range<N, I...>::value, "swizzle index out of range"); // Reject a component index outside the vector.
        const std::size_t indices[] = {I...}; // Keep the requested component order available for indexing.
        vec<T, sizeof...(I)> out; // Keep an independent value for the result or aliased inputs.
        for (std::size_t j = 0; j < sizeof...(I); ++j) // Visit each component needed by this calculation.
            out[j] = (*this)[indices[j]]; // Store the computed component in the independent result.
        return out; // Return the completed result.
    } // Finish the swizzle operation.
#define SHDR_DETAIL_VEC_ASSIGN(OP) /* Generate the matching scalar and vector overloads. */ \
    template <class A> vec& operator OP(const A & a) { /* Accept the matching input types and dimensions. */ \
        static_assert(!detail::traits<A>::vector || detail::traits<A>::size == N, /* Require matching component counts. */ \
                      "vector dimension mismatch"); /* Explain the rejected input: vector dimension mismatch. */ \
        T values[N]; /* Allocate space for the supplied constructor components. */ \
        for (std::size_t i = 0; i < N; ++i) { /* Visit each component needed by this calculation. */ \
            values[i] = static_cast<T>(detail::component(a, i)); /* Copy source components before any destination is changed. */ \
        } /* Finish this component or matrix pass. */ \
        for (std::size_t i = 0; i < N; ++i) { /* Visit each component needed by this calculation. */ \
            (*this)[i] OP values[i]; /* Update the selected destination component. */ \
        } /* Finish this component or matrix pass. */ \
        return *this; /* Allow this update to be used in a chained expression. */ \
    } /* Finish this operator overload. */
    SHDR_DETAIL_VEC_ASSIGN(+=) // Provide componentwise addition.
    SHDR_DETAIL_VEC_ASSIGN(-=) // Provide componentwise subtraction.
    SHDR_DETAIL_VEC_ASSIGN(*=) // Provide componentwise multiplication.
    SHDR_DETAIL_VEC_ASSIGN(/=) // Provide componentwise division.
    SHDR_DETAIL_VEC_ASSIGN(%=) // Provide componentwise integer remainder.
    SHDR_DETAIL_VEC_ASSIGN(&=) // Provide componentwise bitwise AND.
    SHDR_DETAIL_VEC_ASSIGN(|=) // Provide componentwise bitwise OR.
    SHDR_DETAIL_VEC_ASSIGN(^=) // Provide componentwise bitwise XOR.
    SHDR_DETAIL_VEC_ASSIGN(<<=) // Provide componentwise left shifting.
    SHDR_DETAIL_VEC_ASSIGN(>>=) // Provide componentwise right shifting.
#undef SHDR_DETAIL_VEC_ASSIGN // Remove this local code generator macro.
    vec& operator++() { // Provide this arithmetic or component access overload.
        for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
            ++(*this)[i]; // Increment this vector component.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    vec operator++(int) { // Provide this arithmetic or component access overload.
        vec old(*this); // Save the vector value before incrementing or decrementing it.
        ++*this; // Increment every vector component.
        return old; // Return the value from before the update.
    } // Finish this operator overload.
    vec& operator--() { // Provide this arithmetic or component access overload.
        for (std::size_t i = 0; i < N; ++i) // Visit each component needed by this calculation.
            --(*this)[i]; // Decrement this vector component.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    vec operator--(int) { // Provide this arithmetic or component access overload.
        vec old(*this); // Save the vector value before incrementing or decrementing it.
        --*this; // Decrement every vector component.
        return old; // Return the value from before the update.
    } // Finish this operator overload.
}; // Finish the vec type description.
template <class T, std::size_t N> const std::size_t vec<T, N>::length_value; // Accept the matching input types and dimensions.

/** @brief A component selection referencing an existing vector.
 * Assignment evaluates all source components before writing any destination,
 * so exchanging components of the same vector is safe. Duplicate destination
 * indices are rejected at compile time. The vector must outlive the selection.
 * construct a vec value to keep an independent copy.
 */
template <class T, std::size_t N, std::size_t... I> class swizzle_ref { // Describe a component selection that refers to a vector.
    vec<T, N>* source_; // Refer to the vector that owns the selected components.

  public: // Expose the public math operations.
    static constexpr std::size_t length() { // Record the compile-time component rule.
        return sizeof...(I); // Return the fixed number of available components.
    } // Finish the length operation.
    explicit swizzle_ref(vec<T, N>& source) : source_(&source) { // Keep a reference to the source vector.
    } // Finish keeping the source vector reference.
    swizzle_ref(const swizzle_ref&) = default; // Copy the reference to the same source vector.
    T& operator[](std::size_t i) { // Provide this arithmetic or component access overload.
        const std::size_t indices[] = {I...}; // Keep the requested component order available for indexing.
        return (*source_)[indices[i]]; // Return the selected component with its original reference type.
    } // Finish this operator overload.
    const T& operator[](std::size_t i) const { // Provide this arithmetic or component access overload.
        const std::size_t indices[] = {I...}; // Keep the requested component order available for indexing.
        return (*source_)[indices[i]]; // Return the selected component with its original reference type.
    } // Finish this operator overload.
    operator vec<T, sizeof...(I)>() const { // Provide this arithmetic or component access overload.
        return vec<T, sizeof...(I)>(*this); // Return the component count or geometric length.
    } // Finish this operator overload.
    swizzle_ref& operator=(const swizzle_ref& rhs) { // Provide this arithmetic or component access overload.
        return assign(rhs); // Write through a temporary so overlapping selections remain safe.
    } // Finish this operator overload.
    template <class A> swizzle_ref& operator=(const A& rhs) { // Accept the matching input types and dimensions.
        return assign(rhs); // Write through a temporary so overlapping selections remain safe.
    } // Finish this operator overload.
    template <class A> swizzle_ref& assign(const A& rhs) { // Accept the matching input types and dimensions.
        static_assert(detail::unique<I...>::value, "a writable swizzle cannot repeat components"); // Prevent repeated destination components.
        static_assert(!detail::traits<A>::vector || detail::traits<A>::size == sizeof...(I), // Require matching component counts.
                      "swizzle dimension mismatch"); // Explain the rejected input: swizzle dimension mismatch.
        vec<T, sizeof...(I)> temp; // Keep an independent value for the result or aliased inputs.
        for (std::size_t j = 0; j < sizeof...(I); ++j) // Visit each component needed by this calculation.
            temp[j] = static_cast<T>(detail::component(rhs, j)); // Save this source component before writing the selection.
        for (std::size_t j = 0; j < sizeof...(I); ++j) // Visit each component needed by this calculation.
            (*this)[j] = temp[j]; // Update the selected destination component.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish the assign operation.
#define SHDR_DETAIL_SWIZZLE_ASSIGN(OP, BINARY) /* Generate the matching scalar and vector overloads. */ \
    template <class A> swizzle_ref& operator OP(const A & rhs) { /* Accept the matching input types and dimensions. */ \
        return assign(vec<T, sizeof...(I)>(*this) BINARY rhs); /* Write through a temporary so overlapping selections remain safe. */ \
    } /* Finish this operator overload. */
    SHDR_DETAIL_SWIZZLE_ASSIGN(+=, +) // Provide componentwise addition.
    SHDR_DETAIL_SWIZZLE_ASSIGN(-=, -) // Provide componentwise subtraction.
    SHDR_DETAIL_SWIZZLE_ASSIGN(*=, *) // Provide componentwise multiplication.
    SHDR_DETAIL_SWIZZLE_ASSIGN(/=, /) // Provide componentwise division.
    SHDR_DETAIL_SWIZZLE_ASSIGN(%=, %) // Provide componentwise integer remainder.
    SHDR_DETAIL_SWIZZLE_ASSIGN(&=, &) // Provide componentwise bitwise AND.
    SHDR_DETAIL_SWIZZLE_ASSIGN(|=, |) // Provide componentwise bitwise OR.
    SHDR_DETAIL_SWIZZLE_ASSIGN(^=, ^) // Provide componentwise bitwise XOR.
    SHDR_DETAIL_SWIZZLE_ASSIGN(<<=, <<) // Provide componentwise left shifting.
    SHDR_DETAIL_SWIZZLE_ASSIGN(>>=, >>) // Provide componentwise right shifting.
#undef SHDR_DETAIL_SWIZZLE_ASSIGN // Remove this local code generator macro.
}; // Finish the swizzle_ref type description.
/** @name Vector component arithmetic
 * Vector/scalar and scalar/vector forms broadcast the scalar.
 * Integer operators require integer components, as in GLSL.
 * @{ */
#define SHDR_DETAIL_VEC_BINARY(OP) /* Generate the matching scalar and vector overloads. */ \
    template <class A, class B> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<detail::binary<A, B>::valid, /* Enable this overload only for compatible input types. */ \
                            typename detail::binary<A, B>::result>::type /* Preserve the selected component type and result shape. */ \
    operator OP(const A& a, const B& b) { /* Provide this arithmetic or component access overload. */ \
        typedef typename detail::binary<A, B>::scalar T; /* Keep the chosen component type for the calculation. */ \
        typename detail::binary<A, B>::result out; /* Allocate an independent result with the correct shape. */ \
        for (std::size_t i = 0; i < detail::binary<A, B>::size; ++i) { /* Visit each component needed by this calculation. */ \
            out[i] = static_cast<T>(detail::component(a, i) OP detail::component(b, i)); /* Store the computed component in the independent result. */ \
        } /* Finish this component or matrix pass. */ \
        return out; /* Return the completed result. */ \
    } /* Finish this operator overload. */
SHDR_DETAIL_VEC_BINARY(+) // Provide componentwise addition.
SHDR_DETAIL_VEC_BINARY(-) // Provide componentwise subtraction.
SHDR_DETAIL_VEC_BINARY(*) // Provide componentwise multiplication.
SHDR_DETAIL_VEC_BINARY(/) // Provide componentwise division.
SHDR_DETAIL_VEC_BINARY(%) // Provide componentwise integer remainder.
SHDR_DETAIL_VEC_BINARY(&) // Provide componentwise bitwise AND.
SHDR_DETAIL_VEC_BINARY(|) // Provide componentwise bitwise OR.
SHDR_DETAIL_VEC_BINARY(^) // Provide componentwise bitwise XOR.
SHDR_DETAIL_VEC_BINARY(<<) // Provide componentwise left shifting.
SHDR_DETAIL_VEC_BINARY(>>) // Provide componentwise right shifting.
#undef SHDR_DETAIL_VEC_BINARY // Remove this local code generator macro.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
operator-(const A& a) { // Provide this arithmetic or component access overload.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = -a[i]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish this operator overload.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
operator+(const A& a) { // Provide this arithmetic or component access overload.
    return typename detail::result<A>::type(a); // Copy source components before changing the destination.
} // Finish this operator overload.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
operator~(const A& a) { // Provide this arithmetic or component access overload.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = ~a[i]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish this operator overload.
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector && // Enable this overload only for compatible input types.
                            detail::traits<A>::size == detail::traits<B>::size, // Complete the type and dimension rule for this overload.
                        bool>::type // Complete the type and dimension rule for this overload.
operator==(const A& a, const B& b) { // Provide this arithmetic or component access overload.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) { // Visit each component needed by this calculation.
        if (a[i] != b[i]) // Stop when corresponding components differ.
            return false; // Report that at least one component differs.
    } // Finish this component or matrix pass.
    return true; // Report that every required component matched.
} // Finish this operator overload.
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector && // Enable this overload only for compatible input types.
                            detail::traits<A>::size == detail::traits<B>::size, // Complete the type and dimension rule for this overload.
                        bool>::type // Complete the type and dimension rule for this overload.
operator!=(const A& a, const B& b) { // Provide this arithmetic or component access overload.
    return !(a == b); // Copy source components before changing the destination.
} // Finish this operator overload.
/** @} */
typedef vec<float, 2> vec2; // Name the 2-component float vector.
typedef vec<float, 3> vec3; // Name the 3-component float vector.
typedef vec<float, 4> vec4; // Name the 4-component float vector.
typedef vec<double, 2> dvec2; // Name the 2-component double vector.
typedef vec<double, 3> dvec3; // Name the 3-component double vector.
typedef vec<double, 4> dvec4; // Name the 4-component double vector.
typedef vec<std::int32_t, 2> ivec2; // Name the 2-component signed integer vector.
typedef vec<std::int32_t, 3> ivec3; // Name the 3-component signed integer vector.
typedef vec<std::int32_t, 4> ivec4; // Name the 4-component signed integer vector.
typedef vec<std::uint32_t, 2> uvec2; // Name the 2-component unsigned integer vector.
typedef vec<std::uint32_t, 3> uvec3; // Name the 3-component unsigned integer vector.
typedef vec<std::uint32_t, 4> uvec4; // Name the 4-component unsigned integer vector.
typedef vec<bool, 2> bvec2; // Name the 2-component boolean vector.
typedef vec<bool, 3> bvec3; // Name the 3-component boolean vector.
typedef vec<bool, 4> bvec4; // Name the 4-component boolean vector.
template <class T, std::size_t N> // Accept the matching input types and dimensions.
std::ostream& operator<<(std::ostream& out, const vec<T, N>& value) { // Provide this arithmetic or component access overload.
    out << '('; // Write the next component or separator for readable output.
    for (std::size_t i = 0; i < N; ++i) { // Visit each component needed by this calculation.
        if (i) // Write a separator only after the first component.
            out << ", "; // Write the next component or separator for readable output.
        out << value[i]; // Write the next component or separator for readable output.
    } // Finish this component or matrix pass.
    return out << ')'; // Finish the readable representation and return the stream.
} // Finish this operator overload.

/** @brief A floating-point matrix with C columns and R rows.
 * A scalar constructor initializes the diagonal and sets other entries to
 * zero. Component constructors consume scalars and vectors in column order,
 * discarding excess components in the final argument. Matrix conversion copies
 * the common region and fills the remaining entries with the identity matrix.
 * Default construction initializes every component to zero.
 */
template <class T, std::size_t C, std::size_t R> class mat { // Describe a matrix by its scalar type and dimensions.
    static_assert(C >= 2 && C <= 4 && R >= 2 && R <= 4, "matrix dimensions require 2 to 4"); // Require matching component counts.
    static_assert(std::is_floating_point<T>::value, // Require floating-point components for this operation.
                  "GLSL matrices require floating point components"); // Explain the rejected input: GLSL matrices require floating point components.
    vec<T, R> columns_[C]; // Store each matrix column as a vector.

  public: // Expose the public math operations.
    typedef T value_type; // Expose the element type to host code and templates.
    typedef vec<T, R> column_type; // Expose the vector type used for each matrix column.
    static const std::size_t columns = C; // Keep the matrix column count available at compile time.
    static const std::size_t rows = R; // Keep the matrix row count available at compile time.
    /** @brief Return the number of matrix columns. */
    static constexpr std::size_t length() { // Record the compile-time component rule.
        return C; // Return the fixed number of available components.
    } // Finish the length operation.
    mat() : columns_() { // Initialize matrix columns from the supplied values.
    } // Finish initializing the matrix.
    template <class U, typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0> // Accept the matching input types and dimensions.
    explicit mat(U diagonal) : columns_() { // Initialize matrix columns from the supplied values.
        for (std::size_t c = 0; c < C && c < R; ++c) // Visit each matrix column.
            columns_[c][c] = static_cast<T>(diagonal); // Set this diagonal entry to the requested scalar.
    } // Finish initializing the matrix.
    template <class U, std::size_t A, std::size_t B> // Accept the matching input types and dimensions.
    explicit mat(const mat<U, A, B>& other) : columns_() { // Initialize matrix columns from the supplied values.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            for (std::size_t r = 0; r < R; ++r) // Visit each matrix row.
                columns_[c][r] = // Copy or complete this matrix entry.
                    c < A && r < B ? static_cast<T>(other[c][r]) : (c == r ? T(1) : T(0)); // Copy the common region and use identity entries elsewhere.
    } // Finish initializing the matrix.
    /** @brief Construct a matrix from the components of one vector or swizzle. */
    template <class A, typename std::enable_if<detail::traits<A>::vector, int>::type = 0> // Accept the matching input types and dimensions.
    explicit mat(const A& value) { // Initialize matrix columns from the supplied values.
        static_assert(detail::traits<A>::size >= C * R, // Check this type or dimension requirement at compile time.
                      "matrix constructor has too few components"); // Explain the rejected input: matrix constructor has too few components.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            for (std::size_t r = 0; r < R; ++r) // Visit each matrix row.
                columns_[c][r] = static_cast<T>(value[c * R + r]); // Copy the next source component in column order.
    } // Finish initializing the matrix.
    template <class A, class B, class... Rest> // Accept the matching input types and dimensions.
    explicit mat(const A& a, const B& b, const Rest&... rest) { // Initialize matrix columns from the supplied values.
        static_assert(!detail::any_matrix<A, B, Rest...>::value, // Check this type or dimension requirement at compile time.
                      "matrix conversion cannot be mixed with other arguments"); // Require the matrix to be the sole conversion argument.
        static_assert(detail::count<A, B, Rest...>::value >= C * R, // Require enough components to fill the destination.
                      "matrix constructor has too few components"); // Explain the rejected input: matrix constructor has too few components.
        static_assert(detail::prefix_count<A, B, Rest...>::value < C * R, // Reject constructor arguments that would go unused.
                      "matrix constructor has unused arguments"); // Explain the rejected input: matrix constructor has unused arguments.
        T values[detail::count<A, B, Rest...>::value]; // Allocate space for the supplied constructor components.
        std::size_t n = 0; // Begin appending at the first temporary component.
        detail::flatten(values, n, a, b, rest...); // Append the remaining arguments in order.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            for (std::size_t r = 0; r < R; ++r) // Visit each matrix row.
                columns_[c][r] = values[c * R + r]; // Fill the column from the flattened constructor values.
    } // Finish initializing the matrix.
    /** @brief Read or modify column c.  m[c][r] selects row r within that column. */
    column_type& operator[](std::size_t c) { // Provide this arithmetic or component access overload.
        return columns_[c]; // Return the component count or geometric length.
    } // Finish this operator overload.
    const column_type& operator[](std::size_t c) const { // Provide this arithmetic or component access overload.
        return columns_[c]; // Return the component count or geometric length.
    } // Finish this operator overload.
    mat& operator+=(const mat& b) { // Provide this arithmetic or component access overload.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            columns_[c] += b[c]; // Add the corresponding matrix column.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    mat& operator-=(const mat& b) { // Provide this arithmetic or component access overload.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            columns_[c] -= b[c]; // Subtract the corresponding matrix column.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    mat& operator+=(T b) { // Provide this arithmetic or component access overload.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            columns_[c] += b; // Add the scalar to every entry in this column.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    mat& operator-=(T b) { // Provide this arithmetic or component access overload.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            columns_[c] -= b; // Subtract the scalar from every entry in this column.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    mat& operator*=(T b) { // Provide this arithmetic or component access overload.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            columns_[c] *= b; // Scale this column by the scalar.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    mat& operator/=(T b) { // Provide this arithmetic or component access overload.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            columns_[c] /= b; // Divide this column by the scalar.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    mat& operator/=(const mat& b) { // Provide this arithmetic or component access overload.
        for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
            columns_[c] /= b[c]; // Divide corresponding entries in this column.
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
    mat& operator*=(const mat<T, C, C>& b) { // Provide this arithmetic or component access overload.
        *this = *this * b;
        return *this; // Allow this update to be used in a chained expression.
    } // Finish this operator overload.
}; // Finish the mat type description.
template <class T, std::size_t C, std::size_t R> const std::size_t mat<T, C, R>::columns; // Accept the matching input types and dimensions.
template <class T, std::size_t C, std::size_t R> const std::size_t mat<T, C, R>::rows; // Accept the matching input types and dimensions.

/** @brief Convert a scalar, or the first component of a vector, to T.
 * This implements GLSL scalar constructors such as float(vec3) for generated
 * C++ kernels. Ordinary C++ callers may use scalar_cast<float>(value) directly.
 */
template <class T, class A> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<T>::value && detail::traits<A>::valid, T>::type // Enable this overload only for compatible input types.
scalar_cast(const A& value) { // Convert the first scalar component to the requested type.
    return static_cast<T>(detail::component(value, 0)); // Convert the first scalar component to the requested type.
} // Finish the scalar_cast operation.
/** @brief Convert the first column's first matrix component to scalar type T. */
template <class T, class U, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<T>::value, T>::type // Enable this overload only for compatible input types.
scalar_cast(const mat<U, C, R>& value) { // Convert the first scalar component to the requested type.
    return static_cast<T>(value[0][0]); // Convert the first scalar component to the requested type.
} // Finish the scalar_cast operation.
/** @brief GLSL square and rectangular matrix aliases.  dimensions are columns
 * followed by rows. Float aliases are mat2..4 and mat2x2..mat4x4.  double aliases
 * use dmat. Double arithmetic is a host extension and requires a supporting
 * desktop GLSL profile when used by shared shaders.
 */
#define SHDR_DETAIL_MATRIX_ALIASES(T, PREFIX) /* Generate the matching scalar and vector overloads. */ \
    typedef mat<T, 2, 2> PREFIX##2; /* Name the type selected for this math result. */ \
    typedef mat<T, 3, 3> PREFIX##3; /* Name the type selected for this math result. */ \
    typedef mat<T, 4, 4> PREFIX##4; /* Name the type selected for this math result. */ \
    typedef mat<T, 2, 2> PREFIX##2x2; /* Name the type selected for this math result. */ \
    typedef mat<T, 2, 3> PREFIX##2x3; /* Name the type selected for this math result. */ \
    typedef mat<T, 2, 4> PREFIX##2x4; /* Name the type selected for this math result. */ \
    typedef mat<T, 3, 2> PREFIX##3x2; /* Name the type selected for this math result. */ \
    typedef mat<T, 3, 3> PREFIX##3x3; /* Name the type selected for this math result. */ \
    typedef mat<T, 3, 4> PREFIX##3x4; /* Name the type selected for this math result. */ \
    typedef mat<T, 4, 2> PREFIX##4x2; /* Name the type selected for this math result. */ \
    typedef mat<T, 4, 3> PREFIX##4x3; /* Name the type selected for this math result. */ \
    typedef mat<T, 4, 4> PREFIX##4x4; /* Name the type selected for this math result. */
SHDR_DETAIL_MATRIX_ALIASES(float, mat) // Provide the single-precision matrix names.
SHDR_DETAIL_MATRIX_ALIASES(double, dmat) // Provide the double-precision matrix names.
#undef SHDR_DETAIL_MATRIX_ALIASES // Remove this local code generator macro.
template <class T, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
mat<T, C, R> operator+(mat<T, C, R> a, const mat<T, C, R>& b) { // Keep an independent value for the result or aliased inputs.
    return a += b; // Convert the first scalar component to the requested type.
} // Finish this operator overload.
template <class T, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
mat<T, C, R> operator-(mat<T, C, R> a, const mat<T, C, R>& b) { // Keep an independent value for the result or aliased inputs.
    return a -= b; // Convert the first scalar component to the requested type.
} // Finish this operator overload.
template <class T, std::size_t C, std::size_t R> mat<T, C, R> operator-(const mat<T, C, R>& a) { // Accept the matching input types and dimensions.
    mat<T, C, R> out; // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
        out[c] = -a[c]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish this operator overload.
template <class T, std::size_t C, std::size_t R> mat<T, C, R> operator+(const mat<T, C, R>& a) { // Accept the matching input types and dimensions.
    return a; // Return the scalar unchanged.
} // Finish this operator overload.
#define SHDR_DETAIL_MATRIX_SCALAR(OP, ASSIGN) /* Generate the matching scalar and vector overloads. */ \
    template <class T, std::size_t C, std::size_t R, class U> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<std::is_arithmetic<U>::value, mat<T, C, R>>::type operator OP( /* Enable this overload only for compatible input types. */ \
        mat<T, C, R> a, U b) { /* Keep an independent value for the result or aliased inputs. */ \
        return a ASSIGN static_cast<T>(b); /* Convert the first scalar component to the requested type. */ \
    } /* Finish this declaration. */ \
    template <class T, std::size_t C, std::size_t R, class U> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<std::is_arithmetic<U>::value, mat<T, C, R>>::type operator OP( /* Enable this overload only for compatible input types. */ \
        U a, const mat<T, C, R>& b) { /* Accept a scalar before the matrix operand. */ \
        mat<T, C, R> out; /* Keep an independent value for the result or aliased inputs. */ \
        for (std::size_t c = 0; c < C; ++c) { /* Visit each matrix column. */ \
            out[c] = static_cast<T>(a) OP b[c]; /* Store the computed component in the independent result. */ \
        } /* Finish this component or matrix pass. */ \
        return out; /* Return the completed result. */ \
    } /* Finish this declaration. */
SHDR_DETAIL_MATRIX_SCALAR(+, +=) // Provide componentwise addition.
SHDR_DETAIL_MATRIX_SCALAR(-, -=) // Provide componentwise subtraction.
#undef SHDR_DETAIL_MATRIX_SCALAR // Remove this local code generator macro.
template <class T, std::size_t C, std::size_t R, class U> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<U>::value, mat<T, C, R>>::type operator*(mat<T, C, R> a, // Enable this overload only for compatible input types.
                                                                                    U b) { // Accept a scalar after the matrix operand.
    return a *= static_cast<T>(b); // Convert the first scalar component to the requested type.
} // Finish this declaration.
template <class T, std::size_t C, std::size_t R, class U> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<U>::value, mat<T, C, R>>::type // Enable this overload only for compatible input types.
operator*(U a, mat<T, C, R> b) { // Provide this arithmetic or component access overload.
    return b *= static_cast<T>(a); // Convert the first scalar component to the requested type.
} // Finish this operator overload.
template <class T, std::size_t C, std::size_t R, class U> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<U>::value, mat<T, C, R>>::type operator/(mat<T, C, R> a, // Enable this overload only for compatible input types.
                                                                                    U b) { // Accept a scalar after the matrix operand.
    return a /= static_cast<T>(b); // Convert the first scalar component to the requested type.
} // Finish this declaration.
template <class T, std::size_t C, std::size_t R, class U> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<U>::value, mat<T, C, R>>::type // Enable this overload only for compatible input types.
operator/(U a, const mat<T, C, R>& b) { // Provide this arithmetic or component access overload.
    mat<T, C, R> out; // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
        out[c] = static_cast<T>(a) / b[c]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish this operator overload.
template <class T, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
mat<T, C, R> operator/(const mat<T, C, R>& a, const mat<T, C, R>& b) { // Provide this arithmetic or component access overload.
    mat<T, C, R> out; // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
        out[c] = a[c] / b[c]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish this operator overload.
template <class T, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
bool operator==(const mat<T, C, R>& a, const mat<T, C, R>& b) { // Provide this arithmetic or component access overload.
    for (std::size_t c = 0; c < C; ++c) { // Visit each matrix column.
        if (a[c] != b[c]) // Handle the exceptional or boundary case before continuing.
            return false; // Report that at least one component differs.
    } // Finish this component or matrix pass.
    return true; // Report that every required component matched.
} // Finish this operator overload.
template <class T, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
bool operator!=(const mat<T, C, R>& a, const mat<T, C, R>& b) { // Provide this arithmetic or component access overload.
    return !(a == b); // Convert the first scalar component to the requested type.
} // Finish this operator overload.
/** @brief Algebraic matrix-vector multiplication. */
template <class T, std::size_t C, std::size_t R, class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<A>::size == C, vec<T, R>>::type // Enable this overload only for compatible input types.
operator*(const mat<T, C, R>& a, const A& b) { // Provide this arithmetic or component access overload.
    vec<T, R> out(T(0)); // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
        out += a[c] * b[c]; // Add this component product to the running total.
    return out; // Return the completed result.
} // Finish this operator overload.
template <class T, std::size_t C, std::size_t R, class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<A>::size == R, vec<T, C>>::type // Enable this overload only for compatible input types.
operator*(const A& a, const mat<T, C, R>& b) { // Provide this arithmetic or component access overload.
    vec<T, C> out(T(0)); // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < C; ++c) { // Visit each matrix column.
        for (std::size_t r = 0; r < R; ++r) // Visit each matrix row.
            out[c] += a[r] * b[c][r]; // Store the computed component in the independent result.
    } // Finish this component or matrix pass.
    return out; // Return the completed result.
} // Finish this operator overload.
/** @brief Algebraic multiplication, including nonsquare matrices. */
template <class T, std::size_t C, std::size_t R, std::size_t K> // Accept the matching input types and dimensions.
mat<T, K, R> operator*(const mat<T, C, R>& a, const mat<T, K, C>& b) { // Provide this arithmetic or component access overload.
    mat<T, K, R> out; // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < K; ++c) // Visit each matrix column.
        out[c] = a * b[c]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish this operator overload.
/** @brief Exchange rows and columns, including rectangular matrices. */
template <class T, std::size_t C, std::size_t R> mat<T, R, C> transpose(const mat<T, C, R>& a) { // Accept the matching input types and dimensions.
    mat<T, R, C> out; // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
        for (std::size_t r = 0; r < R; ++r) // Visit each matrix row.
            out[r][c] = a[c][r]; // Copy the entry to its exchanged row and column.
    return out; // Return the completed result.
} // Finish the transpose operation.
/** @brief Multiply corresponding matrix components element by element. */
template <class T, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
mat<T, C, R> matrixCompMult(const mat<T, C, R>& a, const mat<T, C, R>& b) { // Multiply corresponding matrix entries.
    mat<T, C, R> out; // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
        out[c] = a[c] * b[c]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the matrixCompMult operation.
/** @brief Form a column-by-row outer product.  output column j is column*row[j]. */
template <class T, std::size_t R, std::size_t C> // Accept the matching input types and dimensions.
mat<T, C, R> outerProduct(const vec<T, R>& column, const vec<T, C>& row) { // Keep an independent value for the result or aliased inputs.
    mat<T, C, R> out; // Keep an independent value for the result or aliased inputs.
    for (std::size_t c = 0; c < C; ++c) // Visit each matrix column.
        out[c] = column * row[c]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the outerProduct operation.
/** @brief Determinant using partial-pivot elimination. */
template <class T, std::size_t N> T determinant(mat<T, N, N> a) { // Accept the matching input types and dimensions.
    T result(1); // Begin the determinant product with multiplicative identity.
    for (std::size_t i = 0; i < N; ++i) { // Visit each component needed by this calculation.
        std::size_t pivot = i; // Begin with the current row as the pivot candidate.
        for (std::size_t r = i + 1; r < N; ++r) // Visit each matrix row.
            if (std::abs(a[i][r]) > std::abs(a[i][pivot])) // Prefer the row with the larger pivot magnitude.
                pivot = r; // Use the stronger row for the next pivot.
        if (a[i][pivot] == T(0)) // Detect a singular column before division.
            return T(0); // Compute the determinant with row pivoting.
        if (pivot != i) { // Swap rows when a better pivot was found.
            for (std::size_t c = 0; c < N; ++c) { // Visit each matrix column.
                T temp = a[c][i]; // Save an entry while exchanging the two rows.
                a[c][i] = a[c][pivot]; // Update this matrix entry during row elimination or exchange.
                a[c][pivot] = temp; // Update this matrix entry during row elimination or exchange.
            } // Finish this component or matrix pass.
            result = -result; // Reverse the determinant sign after a row exchange.
        } // Finish handling this condition.
        T diagonal = a[i][i]; // Save the pivot value before normalizing the row.
        result *= diagonal; // Multiply the determinant by the next diagonal value.
        for (std::size_t r = i + 1; r < N; ++r) { // Visit each matrix row.
            T factor = a[i][r] / diagonal; // Find the amount of the pivot row to eliminate.
            for (std::size_t c = i + 1; c < N; ++c) // Visit each matrix column.
                a[c][r] -= factor * a[c][i]; // Update this matrix entry during row elimination or exchange.
        } // Finish this component or matrix pass.
    } // Finish this component or matrix pass.
    return result; // Return the completed result.
} // Finish the determinant operation.
/** @brief Inverse using partial-pivot Gauss-Jordan elimination.
 * Singular input has undefined GLSL semantics.  the host returns quiet NaNs.
 */
template <class T, std::size_t N> mat<T, N, N> inverse(mat<T, N, N> a) { // Accept the matching input types and dimensions.
    mat<T, N, N> out(T(1)); // Keep an independent value for the result or aliased inputs.
    for (std::size_t i = 0; i < N; ++i) { // Visit each component needed by this calculation.
        std::size_t pivot = i; // Begin with the current row as the pivot candidate.
        for (std::size_t r = i + 1; r < N; ++r) // Visit each matrix row.
            if (std::abs(a[i][r]) > std::abs(a[i][pivot])) // Prefer the row with the larger pivot magnitude.
                pivot = r; // Use the stronger row for the next pivot.
        if (a[i][pivot] == T(0)) { // Detect a singular column before division.
            for (std::size_t c = 0; c < N; ++c) // Visit each matrix column.
                out[c] = vec<T, N>(std::numeric_limits<T>::quiet_NaN()); // Store the computed component in the independent result.
            return out; // Return the completed result.
        } // Finish handling this condition.
        if (pivot != i) // Swap rows when a better pivot was found.
            for (std::size_t c = 0; c < N; ++c) { // Visit each matrix column.
                T temp = a[c][i]; // Save an entry while exchanging the two rows.
                a[c][i] = a[c][pivot]; // Update this matrix entry during row elimination or exchange.
                a[c][pivot] = temp; // Update this matrix entry during row elimination or exchange.
                temp = out[c][i]; // Save an entry while exchanging the two rows.
                out[c][i] = out[c][pivot]; // Store the computed component in the independent result.
                out[c][pivot] = temp; // Store the computed component in the independent result.
            } // Finish this component or matrix pass.
        T diagonal = a[i][i]; // Save the pivot value before normalizing the row.
        for (std::size_t c = 0; c < N; ++c) { // Visit each matrix column.
            a[c][i] /= diagonal; // Update this matrix entry during row elimination or exchange.
            out[c][i] /= diagonal; // Store the computed component in the independent result.
        } // Finish this component or matrix pass.
        for (std::size_t r = 0; r < N; ++r) // Visit each matrix row.
            if (r != i) { // Handle the exceptional or boundary case before continuing.
                T factor = a[i][r]; // Find the amount of the pivot row to eliminate.
                for (std::size_t c = 0; c < N; ++c) { // Visit each matrix column.
                    a[c][r] -= factor * a[c][i]; // Update this matrix entry during row elimination or exchange.
                    out[c][r] -= factor * out[c][i]; // Subtract the pivot-row contribution from the inverse.
                } // Finish this component or matrix pass.
            } // Finish handling this condition.
    } // Finish this component or matrix pass.
    return out; // Return the completed result.
} // Finish the inverse operation.
template <class T, std::size_t C, std::size_t R> // Accept the matching input types and dimensions.
std::ostream& operator<<(std::ostream& out, const mat<T, C, R>& value) { // Provide this arithmetic or component access overload.
    out << '['; // Write the next component or separator for readable output.
    for (std::size_t c = 0; c < C; ++c) { // Visit each matrix column.
        if (c) // Write a separator only after the first component.
            out << ", "; // Write the next component or separator for readable output.
        out << value[c]; // Write the next component or separator for readable output.
    } // Finish this component or matrix pass.
    return out << ']'; // Finish the readable representation and return the stream.
} // Finish this operator overload.
/** @name Angle, exponential and common scalar/vector math
 * Floating-point overloads preserve their component type. Vector overloads
 * apply each scalar operation independently, including swizzle expressions.
 * @{ */
#define SHDR_DETAIL_FLOAT_UNARY(NAME, EXPR) /* Generate the matching scalar and vector overloads. */ \
    template <class T> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<std::is_floating_point<T>::value, T>::type NAME(T x) { /* Enable this overload only for compatible input types. */ \
        return static_cast<T>(EXPR); /* Compute the inverse with row pivoting. */ \
    } /* Finish this generated overload. */ \
    template <class A> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type /* Enable this overload only for compatible input types. */ \
    NAME(const A& x) { /* Start the matching math overload. */ \
        typedef typename detail::traits<A>::scalar T; /* Keep the chosen component type for the calculation. */ \
        static_assert(std::is_floating_point<T>::value, "floating point components required"); /* Require floating-point components for this operation. */ \
        typename detail::result<A>::type out; /* Allocate an independent result with the correct shape. */ \
        for (std::size_t i = 0; i < detail::traits<A>::size; ++i) /* Visit each component needed by this calculation. */ \
            out[i] = glsl::NAME(x[i]); /* Store the computed component in the independent result. */ \
        return out; /* Return the completed result. */ \
    } /* Finish this generated overload. */
SHDR_DETAIL_FLOAT_UNARY(radians, x* T(0.01745329251994329576923690768489L)) // Convert degrees to radians for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(degrees, x* T(57.295779513082320876798154814105L)) // Convert radians to degrees for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(sin, std::sin(x)) // Compute the sine for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(cos, std::cos(x)) // Compute the cosine for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(tan, std::tan(x)) // Compute the tangent for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(asin, std::asin(x)) // Compute the inverse sine for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(acos, std::acos(x)) // Compute the inverse cosine for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(atan, std::atan(x)) // Compute the inverse tangent for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(sinh, std::sinh(x)) // Compute the hyperbolic sine for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(cosh, std::cosh(x)) // Compute the hyperbolic cosine for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(tanh, std::tanh(x)) // Compute the hyperbolic tangent for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(asinh, std::asinh(x)) // Compute the inverse hyperbolic sine for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(acosh, std::acosh(x)) // Compute the inverse hyperbolic cosine for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(atanh, std::atanh(x)) // Compute the inverse hyperbolic tangent for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(exp, std::exp(x)) // Raise e to the input power for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(log, std::log(x)) // Compute the natural logarithm for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(exp2, std::exp2(x)) // Raise two to the input power for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(log2, std::log2(x)) // Compute the base-two logarithm for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(sqrt, std::sqrt(x)) // Compute the square root for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(inversesqrt, T(1) / std::sqrt(x)) // Compute the reciprocal square root for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(floor, std::floor(x)) // Round toward negative infinity for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(trunc, std::trunc(x)) // Round toward zero for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(round, std::round(x)) // Round to the nearest integer for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(ceil, std::ceil(x)) // Round toward positive infinity for scalars and vector components.
SHDR_DETAIL_FLOAT_UNARY(fract, x - std::floor(x)) // Keep the fraction above the floor for scalars and vector components.
#undef SHDR_DETAIL_FLOAT_UNARY // Remove this local code generator macro.
template <class T> typename std::enable_if<std::is_arithmetic<T>::value, T>::type abs(T x) { // Accept the matching input types and dimensions.
    return x < T(0) ? -x : x; // Return a nonnegative magnitude.
} // Finish the abs operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
abs(const A& a) { // Return a nonnegative magnitude.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::abs(a[i]); // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the abs operation.
template <class T> typename std::enable_if<std::is_arithmetic<T>::value, T>::type sign(T x) { // Accept the matching input types and dimensions.
    return x > T(0) ? T(1) : (x < T(0) ? T(-1) : T(0)); // Return minus one, zero or one from the input sign.
} // Finish the sign operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
sign(const A& a) { // Return minus one, zero or one from the input sign.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::sign(a[i]); // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the sign operation.
/** @brief Round halfway cases to the nearest even integer, independent of the
 * host's floating-point rounding mode. Nonfinite inputs are preserved.
 */
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, T>::type roundEven(T x) { // Enable this overload only for compatible input types.
    if (!std::isfinite(x)) // Keep nonfinite inputs unchanged.
        return x; // Round halfway values to the nearest even integer.
    T lo = std::floor(x), fraction = x - lo; // Separate the lower integer from its fractional remainder.
    if (fraction < T(0.5)) // Choose the lower integer below the halfway point.
        return lo; // Round halfway values to the nearest even integer.
    if (fraction > T(0.5)) // Choose the upper integer above the halfway point.
        return lo + T(1); // Round halfway values to the nearest even integer.
    return std::fmod(lo, T(2)) == T(0) ? lo : lo + T(1); // Round halfway values to the nearest even integer.
} // Finish the roundEven operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
roundEven(const A& a) { // Round halfway values to the nearest even integer.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::roundEven(a[i]); // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the roundEven operation.
#define SHDR_DETAIL_MATH_BINARY(NAME, ENABLE, EXPR) /* Generate the matching scalar and vector overloads. */ \
    template <class T> typename std::enable_if<ENABLE, T>::type NAME(T x, T y) { /* Accept the matching input types and dimensions. */ \
        return static_cast<T>(EXPR); /* Round halfway values to the nearest even integer. */ \
    } /* Finish this generated overload. */ \
    template <class A, class B> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<detail::binary<A, B>::valid, /* Enable this overload only for compatible input types. */ \
                            typename detail::binary<A, B>::result>::type /* Preserve the selected component type and result shape. */ \
    NAME(const A& a, const B& b) { /* Start the matching math overload. */ \
        typedef typename detail::binary<A, B>::scalar T; /* Keep the chosen component type for the calculation. */ \
        typename detail::binary<A, B>::result out; /* Allocate an independent result with the correct shape. */ \
        for (std::size_t i = 0; i < detail::binary<A, B>::size; ++i) { /* Visit each component needed by this calculation. */ \
            out[i] = glsl::NAME(static_cast<T>(detail::component(a, i)), /* Store the computed component in the independent result. */ \
                                static_cast<T>(detail::component(b, i))); /* Convert the second operand component to the result type. */ \
        } /* Finish this component or matrix pass. */ \
        return out; /* Return the completed result. */ \
    } /* Finish this generated overload. */
SHDR_DETAIL_MATH_BINARY(atan, std::is_floating_point<T>::value, std::atan2(x, y)) // Compute the inverse tangent for scalars and vector components.
SHDR_DETAIL_MATH_BINARY(pow, std::is_floating_point<T>::value, std::pow(x, y)) // Raise each base to its exponent.
SHDR_DETAIL_MATH_BINARY(mod, std::is_floating_point<T>::value, x - y * std::floor(x / y)) // Use the GLSL floor-based remainder.
SHDR_DETAIL_MATH_BINARY(min, std::is_arithmetic<T>::value, y < x ? y : x) // Choose the smaller value.
SHDR_DETAIL_MATH_BINARY(max, std::is_arithmetic<T>::value, x < y ? y : x) // Choose the larger value.
SHDR_DETAIL_MATH_BINARY(step, std::is_floating_point<T>::value, y < x ? T(0) : T(1)) // Return zero below the edge and one otherwise.
#undef SHDR_DETAIL_MATH_BINARY // Remove this local code generator macro.
/** @brief Fused multiply-add.  final rounding may differ across CPU/GPU. */
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, T>::type fma(T a, T b, T c) { // Enable this overload only for compatible input types.
    return std::fma(a, b, c); // Multiply and add with one final rounding.
} // Finish the fma operation.
template <class A, class B, class C> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
fma(const A& a, const B& b, const C& c) { // Multiply and add with one final rounding.
    static_assert(detail::binary<A, B>::valid && detail::binary<A, C>::valid, // Check this type or dimension requirement at compile time.
                  "fma dimensions mismatch"); // Explain the rejected input: fma dimensions mismatch.
    typedef typename detail::traits<A>::scalar T; // Keep the chosen component type for the calculation.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::fma( // Store the computed component in the independent result.
            a[i], static_cast<T>(detail::component(b, i)), static_cast<T>(detail::component(c, i))); // Supply the matching three components to fused multiply-add.
    return out; // Return the completed result.
} // Finish the fma operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<T>::value, T>::type clamp(T x, T lower, T upper) { // Enable this overload only for compatible input types.
    return glsl::min(glsl::max(x, lower), upper); // Keep a value between its lower and upper limits.
} // Finish the clamp operation.
template <class A, class B, class C> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
clamp(const A& a, const B& lower, const C& upper) { // Keep a value between its lower and upper limits.
    static_assert(detail::binary<A, B>::valid && detail::binary<A, C>::valid, // Check this type or dimension requirement at compile time.
                  "clamp dimensions mismatch"); // Explain the rejected input: clamp dimensions mismatch.
    typedef typename detail::traits<A>::scalar T; // Keep the chosen component type for the calculation.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::clamp(a[i], // Store the computed component in the independent result.
                             static_cast<T>(detail::component(lower, i)), // Use the matching lower bound component.
                             static_cast<T>(detail::component(upper, i))); // Use the matching upper bound component.
    return out; // Return the completed result.
} // Finish the clamp operation.
template <class T, class U> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && // Enable this overload only for compatible input types.
                            !std::is_same<U, bool>::value, // Complete the type and dimension rule for this overload.
                        T>::type // Complete the type and dimension rule for this overload.
mix(T x, T y, U selector) { // Blend values or select one with a boolean.
    T a = static_cast<T>(selector); // Convert the interpolation amount to the component type.
    return x * (T(1) - a) + y * a; // Blend values or select one with a boolean.
} // Finish the mix operation.
template <class T, class B> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_arithmetic<T>::value && std::is_same<B, bool>::value, T>::type // Enable this overload only for compatible input types.
mix(T x, T y, B a) { // Blend values or select one with a boolean.
    return a ? y : x; // Blend values or select one with a boolean.
} // Finish the mix operation.
template <class A, class B, class C> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
mix(const A& a, const B& b, const C& selector) { // Blend values or select one with a boolean.
    static_assert(detail::binary<A, B>::valid && detail::binary<A, C>::valid, // Check this type or dimension requirement at compile time.
                  "mix dimensions mismatch"); // Explain the rejected input: mix dimensions mismatch.
    typedef typename detail::traits<A>::scalar T; // Keep the chosen component type for the calculation.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::mix( // Store the computed component in the independent result.
            a[i], static_cast<T>(detail::component(b, i)), detail::component(selector, i)); // Blend or select the matching input components.
    return out; // Return the completed result.
} // Finish the mix operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, T>::type // Enable this overload only for compatible input types.
smoothstep(T lower, T upper, T x) { // Interpolate smoothly between two edges.
    T t = glsl::clamp((x - lower) / (upper - lower), T(0), T(1)); // Map the input into a clamped zero-to-one interval.
    return t * t * (T(3) - T(2) * t); // Interpolate smoothly between two edges.
} // Finish the smoothstep operation.
template <class A, class B, class C> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<C>::vector, typename detail::result<C>::type>::type // Enable this overload only for compatible input types.
smoothstep(const A& lower, const B& upper, const C& x) { // Interpolate smoothly between two edges.
    static_assert(detail::binary<C, A>::valid && detail::binary<C, B>::valid, // Check this type or dimension requirement at compile time.
                  "smoothstep dimensions mismatch"); // Explain the rejected input: smoothstep dimensions mismatch.
    typedef typename detail::traits<C>::scalar T; // Keep the chosen component type for the calculation.
    typename detail::result<C>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<C>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::smoothstep(static_cast<T>(detail::component(lower, i)), // Store the computed component in the independent result.
                                  static_cast<T>(detail::component(upper, i)), // Use the matching upper interpolation edge.
                                  x[i]); // Interpolate this input component between its edges.
    return out; // Return the completed result.
} // Finish the smoothstep operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, bool>::type isnan(T x) { // Enable this overload only for compatible input types.
    return std::isnan(x); // Check whether the value is not a number.
} // Finish the isnan operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, bool>::type isinf(T x) { // Enable this overload only for compatible input types.
    return std::isinf(x); // Check whether the value is infinite.
} // Finish the isinf operation.
#define SHDR_DETAIL_CLASSIFICATION(NAME) /* Generate the matching scalar and vector overloads. */ \
    template <class A> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<detail::traits<A>::vector, vec<bool, detail::traits<A>::size>>::type /* Enable this overload only for compatible input types. */ \
    NAME(const A& a) { /* Start the matching math overload. */ \
        vec<bool, detail::traits<A>::size> out; /* Keep an independent value for the result or aliased inputs. */ \
        for (std::size_t i = 0; i < detail::traits<A>::size; ++i) /* Visit each component needed by this calculation. */ \
            out[i] = glsl::NAME(a[i]); /* Store the computed component in the independent result. */ \
        return out; /* Return the completed result. */ \
    } /* Finish this generated overload. */
SHDR_DETAIL_CLASSIFICATION(isnan) // Check whether the value is not a number.
SHDR_DETAIL_CLASSIFICATION(isinf) // Check whether the value is infinite.
#undef SHDR_DETAIL_CLASSIFICATION // Remove this local code generator macro.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, T>::type modf(T x, T& integer) { // Enable this overload only for compatible input types.
    return std::modf(x, &integer); // Separate fractional and integral components.
} // Finish the modf operation.
/** @brief Split a vector into signed fractional and integer parts.
 * The return value contains fractional components.  integer receives the
 * integral components. Both arguments may be swizzles, including selections
 * of the same vector, because the input is copied before output assignment.
 */
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector, // Enable this overload only for compatible input types.
                        typename detail::result<A>::type>::type // Preserve the selected component type and result shape.
modf(const A& x, B&& integer) { // Separate fractional and integral components.
    static_assert(detail::traits<A>::size == detail::traits<B>::size, "modf dimensions mismatch"); // Require matching component counts.
    static_assert( // Check this type or dimension requirement at compile time.
        std::is_same<typename detail::traits<A>::scalar, typename detail::traits<B>::scalar>::value, // Require the input and output component types to match.
        "modf component types must match"); // Explain the rejected input: modf component types must match.
    static_assert(detail::writable<B>::value, "modf output requires writable unique components"); // Prevent repeated destination components.
    typename detail::result<A>::type input(x), out, integral; // Preserve the selected component type and result shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::modf(input[i], integral[i]); // Store the computed component in the independent result.
    integer = integral; // Publish all integral components after reading the input.
    return out; // Return the completed result.
} // Finish the modf operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, T>::type frexp(T x, int& exponent) { // Enable this overload only for compatible input types.
    return std::frexp(x, &exponent); // Separate a significand and its binary exponent.
} // Finish the frexp operation.
/** @brief Return binary significands and write matching signed exponents.
 * Each nonzero component satisfies x = significand * 2^exponent. Output may be
 * an integer vector or a writable swizzle with matching component count.
 */
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector, // Enable this overload only for compatible input types.
                        typename detail::result<A>::type>::type // Preserve the selected component type and result shape.
frexp(const A& x, B&& exponent) { // Separate a significand and its binary exponent.
    static_assert(detail::traits<A>::size == detail::traits<B>::size, "frexp dimensions mismatch"); // Require matching component counts.
    static_assert(std::is_same<typename detail::traits<B>::scalar, std::int32_t>::value, // Check this type or dimension requirement at compile time.
                  "frexp requires signed integer exponent components"); // Explain the rejected input: frexp requires signed integer exponent components.
    static_assert(detail::writable<B>::value, "frexp output requires writable unique components"); // Prevent repeated destination components.
    typename detail::result<A>::type input(x), out; // Allocate an independent result with the correct shape.
    vec<std::int32_t, detail::traits<A>::size> powers; // Use the selected vector or matrix shape for this value.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) { // Visit each component needed by this calculation.
        int e = 0; // Prepare a signed exponent for this component.
        out[i] = std::frexp(input[i], &e); // Store the computed component in the independent result.
        powers[i] = e; // Store the binary exponent for this component.
    } // Finish this component or matrix pass.
    exponent = powers; // Publish all exponents after reading the input.
    return out; // Return the completed result.
} // Finish the frexp operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, T>::type ldexp(T x, int exponent) { // Enable this overload only for compatible input types.
    return std::ldexp(x, exponent); // Scale a significand by a power of two.
} // Finish the ldexp operation.
/** @brief Multiply each component by two raised to its signed integer exponent. */
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector, // Enable this overload only for compatible input types.
                        typename detail::result<A>::type>::type // Preserve the selected component type and result shape.
ldexp(const A& x, const B& exponent) { // Scale a significand by a power of two.
    static_assert(detail::traits<A>::size == detail::traits<B>::size, "ldexp dimensions mismatch"); // Require matching component counts.
    static_assert(std::is_same<typename detail::traits<B>::scalar, std::int32_t>::value, // Check this type or dimension requirement at compile time.
                  "ldexp requires signed integer exponent components"); // Explain the rejected input: ldexp requires signed integer exponent components.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = std::ldexp(x[i], exponent[i]); // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the ldexp operation.
/** @} */

/** @name Geometry
 * Geometry uses scalar dot products and GLSL's usual component formulas.
 * normalize requires nonzero length. Inputs to reflect, refract and faceforward
 * should obey GLSL's normalized-normal requirements.
 * @{ */
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<std::is_floating_point<T>::value, T>::type dot(T a, T b) { // Enable this overload only for compatible input types.
    return a * b; // Sum products of corresponding components.
} // Finish the dot operation.
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector, // Enable this overload only for compatible input types.
                        typename detail::traits<A>::scalar>::type // Preserve the selected component type and result shape.
dot(const A& a, const B& b) { // Sum products of corresponding components.
    static_assert(detail::traits<A>::size == detail::traits<B>::size, "dot dimensions mismatch"); // Require matching component counts.
    typedef typename detail::traits<A>::scalar T; // Keep the chosen component type for the calculation.
    T out(0); // Start the scalar accumulator at zero.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) { // Visit each component needed by this calculation.
        out += a[i] * b[i]; // Add this component product to the running total.
    } // Finish this component or matrix pass.
    return out; // Return the completed result.
} // Finish the dot operation.
template <class T> typename std::enable_if<std::is_floating_point<T>::value, T>::type length(T x) { // Accept the matching input types and dimensions.
    return glsl::abs(x); // Return the component count or geometric length.
} // Finish the length operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::traits<A>::scalar>::type // Enable this overload only for compatible input types.
length(const A& a) { // Return the component count or geometric length.
    return glsl::sqrt(glsl::dot(a, a)); // Return the component count or geometric length.
} // Finish the length operation.
template <class A, class B> auto distance(const A& a, const B& b) -> decltype(glsl::length(a - b)) { // Accept the matching input types and dimensions.
    return glsl::length(a - b); // Measure the length of the difference.
} // Finish the distance operation.
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector && // Enable this overload only for compatible input types.
                            detail::traits<A>::size == 3 && detail::traits<B>::size == 3, // Complete the type and dimension rule for this overload.
                        vec<typename detail::traits<A>::scalar, 3>>::type // Preserve the selected component type and result shape.
cross(const A& a, const B& b) { // Compute the three-dimensional cross product.
    typedef typename detail::traits<A>::scalar T; // Keep the chosen component type for the calculation.
    return vec<T, 3>( // Compute the three-dimensional cross product.
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]); // Use the three signed two-by-two minors for the cross product.
} // Finish the cross operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::valid, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
normalize(const A& a) { // Scale a nonzero value to unit length.
    return a / glsl::length(a); // Scale a nonzero value to unit length.
} // Finish the normalize operation.
template <class A, class B, class C> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::valid, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
faceforward(const A& normal, const B& incident, const C& reference) { // Choose the normal direction facing the incident vector.
    return glsl::dot(reference, incident) < 0 ? normal : -normal; // Choose the normal direction facing the incident vector.
} // Finish the faceforward operation.
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::valid, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
reflect(const A& incident, const B& normal) { // Reflect an incident vector around its normal.
    typedef typename detail::traits<A>::scalar T; // Keep the chosen component type for the calculation.
    return incident - T(2) * glsl::dot(normal, incident) * normal; // Reflect an incident vector around its normal.
} // Finish the reflect operation.
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::valid, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
refract(const A& incident, const B& normal, typename detail::traits<A>::scalar eta) { // Refract an incident vector using the index ratio.
    typedef typename detail::traits<A>::scalar T; // Keep the chosen component type for the calculation.
    T d = glsl::dot(normal, incident); // Measure the incident component along the normal.
    T k = T(1) - eta * eta * (T(1) - d * d); // Compute the refraction discriminant before taking its root.
    return k < T(0) ? typename detail::result<A>::type(T(0)) // Refract an incident vector using the index ratio.
                    : eta * incident - (eta * d + glsl::sqrt(k)) * normal; // Apply the transmitted-direction formula when refraction is possible.
} // Finish the refract operation.
/** @} */

/** @name Vector relational functions
 * operator== compares the whole vector and returns bool. These builtins return
 * component masks. logicalNot is the C++ spelling of GLSL's reserved `not`.
 * @{ */
#define SHDR_DETAIL_RELATIONAL(NAME, OP) /* Generate the matching scalar and vector overloads. */ \
    template <class A, class B> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector, /* Enable this overload only for compatible input types. */ \
                            vec<bool, detail::traits<A>::size>>::type /* Use the selected vector or matrix shape for this value. */ \
    NAME(const A& a, const B& b) { /* Start the matching math overload. */ \
        static_assert(detail::traits<A>::size == detail::traits<B>::size, /* Require matching component counts. */ \
                      "relational dimensions mismatch"); /* Explain the rejected input: relational dimensions mismatch. */ \
        vec<bool, detail::traits<A>::size> out; /* Keep an independent value for the result or aliased inputs. */ \
        for (std::size_t i = 0; i < detail::traits<A>::size; ++i) /* Visit each component needed by this calculation. */ \
            out[i] = a[i] OP b[i]; /* Store the computed component in the independent result. */ \
        return out; /* Return the completed result. */ \
    } /* Finish this generated overload. */
SHDR_DETAIL_RELATIONAL(lessThan, <) // Provide the requested component operation.
SHDR_DETAIL_RELATIONAL(lessThanEqual, <=) // Provide the requested component operation.
SHDR_DETAIL_RELATIONAL(greaterThan, >) // Provide the requested component operation.
SHDR_DETAIL_RELATIONAL(greaterThanEqual, >=) // Provide the requested component operation.
SHDR_DETAIL_RELATIONAL(equal, ==) // Provide the requested component operation.
SHDR_DETAIL_RELATIONAL(notEqual, !=) // Provide the requested component operation.
#undef SHDR_DETAIL_RELATIONAL // Remove this local code generator macro.
template <class A> typename std::enable_if<detail::traits<A>::vector, bool>::type any(const A& a) { // Accept the matching input types and dimensions.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        if (a[i]) // Handle the exceptional or boundary case before continuing.
            return true; // Report that every required component matched.
    return false; // Report that at least one component differs.
} // Finish the any operation.
template <class A> typename std::enable_if<detail::traits<A>::vector, bool>::type all(const A& a) { // Accept the matching input types and dimensions.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        if (!a[i]) // Handle the exceptional or boundary case before continuing.
            return false; // Report that at least one component differs.
    return true; // Report that every required component matched.
} // Finish the all operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, vec<bool, detail::traits<A>::size>>::type // Enable this overload only for compatible input types.
logicalNot(const A& a) { // Invert each boolean component.
    vec<bool, detail::traits<A>::size> out; // Keep an independent value for the result or aliased inputs.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = !a[i]; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the logicalNot operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, vec<bool, detail::traits<A>::size>>::type // Enable this overload only for compatible input types.
_not(const A& a) { // Start the matching math overload.
    return glsl::logicalNot(a); // Invert each boolean component.
} // Finish the _not operation.
/** @} */
/** @name 32-bit bit operations
 * Reinterpretation uses memcpy and requires IEEE binary32 floats and 32-bit
 * integers. Bitfield ranges must satisfy 0 <= offset,bits and offset+bits<=32.
 * The 64-bit host intermediates in extended multiplication are never exposed
 * to shared shaders. Bit searches and counts return signed integer results.
 * @{ */
namespace detail { // Keep these names inside the detail namespace.
template <class To, class From> To bit_copy(const From& value) { // Accept the matching input types and dimensions.
    static_assert(sizeof(To) == sizeof(From), "bit conversion requires equal sizes"); // Check this type or dimension requirement at compile time.
    To out; // Allocate a destination with the same bit width.
    std::memcpy(&out, &value, sizeof(out)); // Copy the bits with the alias-safe byte-copy operation.
    return out; // Return the completed result.
} // Finish the bit_copy operation.
inline std::uint32_t bit_mask(int bits) { // Start the matching math overload.
    return bits == 32 ? UINT32_MAX : (bits == 0 ? 0u : (std::uint32_t(1) << bits) - 1u); // Invert each boolean component.
} // Finish the bit_mask operation.
template <class T> // Accept the matching input types and dimensions.
struct integer32 : std::integral_constant<bool, // Describe this input or selection at compile time.
                                          std::is_integral<T>::value && // Require integer components for the bit operation.
                                              !std::is_same<T, bool>::value && sizeof(T) == 4> {}; // Complete the type and dimension rule for this overload.
} // namespace detail
inline std::uint32_t floatBitsToUint(float x) { // Copy float bits into an unsigned integer.
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, // Check this type or dimension requirement at compile time.
                  "IEEE binary32 float required"); // Explain the rejected input: IEEE binary32 float required.
    return detail::bit_copy<std::uint32_t>(x); // Copy float bits into an unsigned integer.
} // Finish the floatBitsToUint operation.
inline std::int32_t floatBitsToInt(float x) { // Copy float bits into a signed integer.
    return detail::bit_copy<std::int32_t>(x); // Copy float bits into a signed integer.
} // Finish the floatBitsToInt operation.
inline float uintBitsToFloat(std::uint32_t x) { // Copy unsigned integer bits into a float.
    return detail::bit_copy<float>(x); // Copy unsigned integer bits into a float.
} // Finish the uintBitsToFloat operation.
inline float intBitsToFloat(std::int32_t x) { // Copy signed integer bits into a float.
    return detail::bit_copy<float>(x); // Copy signed integer bits into a float.
} // Finish the intBitsToFloat operation.
#define SHDR_DETAIL_BIT_VECTOR(NAME, OUT_T) /* Generate the matching scalar and vector overloads. */ \
    template <class A> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<detail::traits<A>::vector, vec<OUT_T, detail::traits<A>::size>>::type /* Enable this overload only for compatible input types. */ \
    NAME(const A& a) { /* Start the matching math overload. */ \
        vec<OUT_T, detail::traits<A>::size> out; /* Keep an independent value for the result or aliased inputs. */ \
        for (std::size_t i = 0; i < detail::traits<A>::size; ++i) /* Visit each component needed by this calculation. */ \
            out[i] = glsl::NAME(a[i]); /* Store the computed component in the independent result. */ \
        return out; /* Return the completed result. */ \
    } /* Finish this generated overload. */
SHDR_DETAIL_BIT_VECTOR(floatBitsToUint, std::uint32_t) // Copy float bits into an unsigned integer.
SHDR_DETAIL_BIT_VECTOR(floatBitsToInt, std::int32_t) // Copy float bits into a signed integer.
SHDR_DETAIL_BIT_VECTOR(uintBitsToFloat, float) // Copy unsigned integer bits into a float.
SHDR_DETAIL_BIT_VECTOR(intBitsToFloat, float) // Copy signed integer bits into a float.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<detail::integer32<T>::value, std::int32_t>::type bitCount(T x) { // Enable this overload only for compatible input types.
    std::uint32_t v = static_cast<std::uint32_t>(x); // Hold the exact integer bits used by this operation.
    std::int32_t count = 0; // Hold the exact integer bits used by this operation.
    while (v) { // Continue until the remaining bits reach the required form.
        v &= v - 1u; // Remove the least significant set bit.
        ++count; // Count the set bit that was removed.
    } // Finish this component or matrix pass.
    return count; // Count the set bits in a 32-bit value.
} // Finish the bitCount operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<detail::integer32<T>::value, std::int32_t>::type findLSB(T x) { // Enable this overload only for compatible input types.
    std::uint32_t v = static_cast<std::uint32_t>(x); // Hold the exact integer bits used by this operation.
    if (!v) // Handle the exceptional or boundary case before continuing.
        return -1; // Find the first set bit from the least significant end.
    std::int32_t position = 0; // Hold the exact integer bits used by this operation.
    while (!(v & 1u)) { // Continue until the remaining bits reach the required form.
        v >>= 1; // Move the next bit into the least significant position.
        ++position; // Advance the bit position being tested.
    } // Finish this component or matrix pass.
    return position; // Find the first set bit from the least significant end.
} // Finish the findLSB operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<detail::integer32<T>::value, std::int32_t>::type findMSB(T x) { // Enable this overload only for compatible input types.
    std::uint32_t v = static_cast<std::uint32_t>(x); // Hold the exact integer bits used by this operation.
    if (std::is_signed<T>::value && x < T(0)) // Handle the exceptional or boundary case before continuing.
        v = ~v; // For negative values, search for the highest zero bit.
    std::int32_t position = -1; // Hold the exact integer bits used by this operation.
    while (v) { // Continue until the remaining bits reach the required form.
        v >>= 1; // Move the next bit into the least significant position.
        ++position; // Advance the bit position being tested.
    } // Finish this component or matrix pass.
    return position; // Find the highest bit that differs from the sign bit.
} // Finish the findMSB operation.
SHDR_DETAIL_BIT_VECTOR(bitCount, std::int32_t) // Count the set bits in a 32-bit value.
SHDR_DETAIL_BIT_VECTOR(findLSB, std::int32_t) // Find the first set bit from the least significant end.
SHDR_DETAIL_BIT_VECTOR(findMSB, std::int32_t) // Find the highest bit that differs from the sign bit.
#undef SHDR_DETAIL_BIT_VECTOR // Remove this local code generator macro.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<detail::integer32<T>::value, T>::type bitfieldReverse(T x) { // Enable this overload only for compatible input types.
    std::uint32_t v = static_cast<std::uint32_t>(x), out = 0; // Hold the exact integer bits used by this operation.
    for (int i = 0; i < 32; ++i) { // Visit all 32 bits of the value.
        out = (out << 1) | (v & 1u); // Append the current low bit in reversed order.
        v >>= 1; // Move the next bit into the least significant position.
    } // Finish this component or matrix pass.
    return detail::bit_copy<T>(out); // Reverse all 32 bits.
} // Finish the bitfieldReverse operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
bitfieldReverse(const A& a) { // Reverse all 32 bits.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::bitfieldReverse(a[i]); // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the bitfieldReverse operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<detail::integer32<T>::value, T>::type // Enable this overload only for compatible input types.
bitfieldExtract(T value, int offset, int bits) { // Extract the requested bit range.
    if (bits == 0 || offset < 0 || bits < 0 || offset > 32 - bits) // Handle an empty bit field separately.
        return T(0); // Extract the requested bit range.
    std::uint32_t mask = detail::bit_mask(bits), // Prepare the mask for the requested bit range.
                  out = (static_cast<std::uint32_t>(value) >> offset) & mask; // Shift the selected field down and discard other bits.
    if (std::is_signed<T>::value && bits < 32 && (out & (std::uint32_t(1) << (bits - 1)))) // Handle the exceptional or boundary case before continuing.
        out |= ~mask; // Add the encoded field to its packed bit position.
    return detail::bit_copy<T>(out); // Extract the requested bit range.
} // Finish the bitfieldExtract operation.
template <class A> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector, typename detail::result<A>::type>::type // Enable this overload only for compatible input types.
bitfieldExtract(const A& a, int offset, int bits) { // Extract the requested bit range.
    typename detail::result<A>::type out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::bitfieldExtract(a[i], offset, bits); // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the bitfieldExtract operation.
template <class T> // Accept the matching input types and dimensions.
typename std::enable_if<detail::integer32<T>::value, T>::type // Enable this overload only for compatible input types.
bitfieldInsert(T base, T insert, int offset, int bits) { // Replace the requested bit range.
    if (bits == 0 || offset < 0 || bits < 0 || offset > 32 - bits) // Handle an empty bit field separately.
        return base; // Replace the requested bit range.
    std::uint32_t mask = detail::bit_mask(bits) << offset; // Prepare the mask for the requested bit range.
    std::uint32_t out = (static_cast<std::uint32_t>(base) & ~mask) | // Prepare the mask for the requested bit range.
                        ((static_cast<std::uint32_t>(insert) << offset) & mask); // Keep only insertion bits that belong to the selected field.
    return detail::bit_copy<T>(out); // Replace the requested bit range.
} // Finish the bitfieldInsert operation.
template <class A, class B> // Accept the matching input types and dimensions.
typename std::enable_if<detail::binary<A, B>::valid, typename detail::binary<A, B>::result>::type // Enable this overload only for compatible input types.
bitfieldInsert(const A& base, const B& insert, int offset, int bits) { // Replace the requested bit range.
    typedef typename detail::binary<A, B>::scalar T; // Keep the chosen component type for the calculation.
    typename detail::binary<A, B>::result out; // Allocate an independent result with the correct shape.
    for (std::size_t i = 0; i < detail::binary<A, B>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::bitfieldInsert(static_cast<T>(detail::component(base, i)), // Store the computed component in the independent result.
                                      static_cast<T>(detail::component(insert, i)), // Use this component as the replacement bit pattern.
                                      offset, // Use the requested first bit position.
                                      bits); // Use the requested number of replacement bits.
    return out; // Return the completed result.
} // Finish the bitfieldInsert operation.
inline std::uint32_t uaddCarry(std::uint32_t x, std::uint32_t y, std::uint32_t& carry) { // Add unsigned values and report the carry.
    std::uint64_t sum = std::uint64_t(x) + y; // Use a wider integer to retain carry or product bits.
    carry = std::uint32_t(sum >> 32); // Publish the extra word after the input values have been copied.
    return std::uint32_t(sum); // Add unsigned values and report the carry.
} // Finish the uaddCarry operation.
inline std::uint32_t usubBorrow(std::uint32_t x, std::uint32_t y, std::uint32_t& borrow) { // Subtract unsigned values and report the borrow.
    borrow = x < y ? 1u : 0u; // Publish the extra word after the input values have been copied.
    return x - y; // Subtract unsigned values and report the borrow.
} // Finish the usubBorrow operation.
inline void umulExtended(std::uint32_t x, std::uint32_t y, std::uint32_t& msb, std::uint32_t& lsb) { // Split a full unsigned product into two words.
    std::uint64_t product = std::uint64_t(x) * y; // Use a wider integer to retain carry or product bits.
    msb = std::uint32_t(product >> 32); // Publish the extra word after the input values have been copied.
    lsb = std::uint32_t(product); // Publish the extra word after the input values have been copied.
} // Finish the umulExtended operation.
inline void imulExtended(std::int32_t x, std::int32_t y, std::int32_t& msb, std::int32_t& lsb) { // Split a full signed product into two words.
    std::uint64_t product = static_cast<std::uint64_t>(std::int64_t(x) * y); // Use a wider integer to retain carry or product bits.
    msb = detail::bit_copy<std::int32_t>(std::uint32_t(product >> 32)); // Publish the extra word after the input values have been copied.
    lsb = detail::bit_copy<std::int32_t>(std::uint32_t(product)); // Publish the extra word after the input values have been copied.
} // Finish the imulExtended operation.
/** @brief Add unsigned vectors modulo 2^32 and write their carry flags. */
template <class A, class B, class C> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector && // Enable this overload only for compatible input types.
                            detail::traits<C>::vector, // Complete the type and dimension rule for this overload.
                        vec<std::uint32_t, detail::traits<A>::size>>::type // Use the selected vector or matrix shape for this value.
uaddCarry(const A& x, const B& y, C&& carry) { // Add unsigned values and report the carry.
    static_assert(detail::traits<A>::size == detail::traits<B>::size && // Require matching component counts.
                      detail::traits<A>::size == detail::traits<C>::size, // Complete the type and dimension rule for this overload.
                  "uaddCarry dimensions mismatch"); // Explain the rejected input: uaddCarry dimensions mismatch.
    static_assert(std::is_same<typename detail::traits<A>::scalar, std::uint32_t>::value && // Check this type or dimension requirement at compile time.
                      std::is_same<typename detail::traits<B>::scalar, std::uint32_t>::value && // Require the second input to contain unsigned 32-bit components.
                      std::is_same<typename detail::traits<C>::scalar, std::uint32_t>::value, // Require an unsigned 32-bit carry or borrow output.
                  "uaddCarry requires unsigned integer components"); // Explain the rejected input: uaddCarry requires unsigned integer components.
    static_assert(detail::writable<C>::value, // Require distinct writable output components.
                  "uaddCarry output requires writable unique components"); // Explain the rejected input: uaddCarry output requires writable unique components.
    vec<std::uint32_t, detail::traits<A>::size> a(x), b(y), out, flags; // Keep an independent value for the result or aliased inputs.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::uaddCarry(a[i], b[i], flags[i]); // Store the computed component in the independent result.
    carry = flags; // Publish the extra word after the input values have been copied.
    return out; // Return the completed result.
} // Finish the uaddCarry operation.
/** @brief Subtract unsigned vectors modulo 2^32 and write their borrow flags. */
template <class A, class B, class C> // Accept the matching input types and dimensions.
typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector && // Enable this overload only for compatible input types.
                            detail::traits<C>::vector, // Complete the type and dimension rule for this overload.
                        vec<std::uint32_t, detail::traits<A>::size>>::type // Use the selected vector or matrix shape for this value.
usubBorrow(const A& x, const B& y, C&& borrow) { // Subtract unsigned values and report the borrow.
    static_assert(detail::traits<A>::size == detail::traits<B>::size && // Require matching component counts.
                      detail::traits<A>::size == detail::traits<C>::size, // Complete the type and dimension rule for this overload.
                  "usubBorrow dimensions mismatch"); // Explain the rejected input: usubBorrow dimensions mismatch.
    static_assert(std::is_same<typename detail::traits<A>::scalar, std::uint32_t>::value && // Check this type or dimension requirement at compile time.
                      std::is_same<typename detail::traits<B>::scalar, std::uint32_t>::value && // Require the second input to contain unsigned 32-bit components.
                      std::is_same<typename detail::traits<C>::scalar, std::uint32_t>::value, // Require an unsigned 32-bit carry or borrow output.
                  "usubBorrow requires unsigned integer components"); // Explain the rejected input: usubBorrow requires unsigned integer components.
    static_assert(detail::writable<C>::value, // Require distinct writable output components.
                  "usubBorrow output requires writable unique components"); // Explain the rejected input: usubBorrow output requires writable unique components.
    vec<std::uint32_t, detail::traits<A>::size> a(x), b(y), out, flags; // Keep an independent value for the result or aliased inputs.
    for (std::size_t i = 0; i < detail::traits<A>::size; ++i) // Visit each component needed by this calculation.
        out[i] = glsl::usubBorrow(a[i], b[i], flags[i]); // Store the computed component in the independent result.
    borrow = flags; // Publish the extra word after the input values have been copied.
    return out; // Return the completed result.
} // Finish the usubBorrow operation.
#define SHDR_DETAIL_EXTENDED(NAME, T) /* Generate the matching scalar and vector overloads. */ \
    template <class A, class B, class C, class D> /* Accept the matching input types and dimensions. */ \
    typename std::enable_if<detail::traits<A>::vector && detail::traits<B>::vector && /* Enable this overload only for compatible input types. */ \
                            detail::traits<C>::vector && detail::traits<D>::vector>::type /* Complete the type and dimension rule for this overload. */ \
    NAME(const A& x, const B& y, C&& msb, D&& lsb) { /* Start the matching math overload. */ \
        static_assert(detail::traits<A>::size == detail::traits<B>::size && /* Require matching component counts. */ \
                          detail::traits<A>::size == detail::traits<C>::size && /* Complete the type and dimension rule for this overload. */ \
                          detail::traits<A>::size == detail::traits<D>::size, /* Complete the type and dimension rule for this overload. */ \
                      "extended multiplication dimensions mismatch"); /* Explain the rejected input: extended multiplication dimensions mismatch. */ \
        static_assert(std::is_same<typename detail::traits<A>::scalar, T>::value && /* Check this type or dimension requirement at compile time. */ \
                          std::is_same<typename detail::traits<B>::scalar, T>::value && /* Require the second input to use the selected integer type. */ \
                          std::is_same<typename detail::traits<C>::scalar, T>::value && /* Require the high output word to use the selected integer type. */ \
                          std::is_same<typename detail::traits<D>::scalar, T>::value, /* Require the low output word to use the selected integer type. */ \
                      "extended multiplication component types must match"); /* Explain the rejected input: extended multiplication component types must match. */ \
        static_assert(detail::writable<C>::value && detail::writable<D>::value, /* Require distinct writable output components. */ \
                      "extended multiplication requires writable unique output components"); /* Explain the rejected input: extended multiplication requires writable unique output components. */ \
        vec<T, detail::traits<A>::size> a(x), b(y), high, low; /* Keep an independent value for the result or aliased inputs. */ \
        for (std::size_t i = 0; i < detail::traits<A>::size; ++i) { /* Visit each component needed by this calculation. */ \
            glsl::NAME(a[i], b[i], high[i], low[i]); /* Compute both words for this pair of copied components. */ \
        } /* Finish this component or matrix pass. */ \
        msb = high; /* Publish the extra word after the input values have been copied. */ \
        lsb = low; /* Publish the extra word after the input values have been copied. */ \
    } /* Finish this generated overload. */
SHDR_DETAIL_EXTENDED(umulExtended, std::uint32_t) // Split a full unsigned product into two words.
SHDR_DETAIL_EXTENDED(imulExtended, std::int32_t) // Split a full signed product into two words.
#undef SHDR_DETAIL_EXTENDED // Remove this local code generator macro.
/** @} */

/** @name Packing
 * Normalized packing follows the GLSL clamp-and-round formulas.  unpacking
 * places the least significant byte/halfword in x. Half-float encoding rounds
 * to nearest even and preserves infinities and a quiet-NaN representation.
 * @{ */
inline std::uint32_t packUnorm2x16(const vec2& v) { // Pack two normalized values into 16-bit fields.
    std::uint32_t x = std::uint32_t(std::round(glsl::clamp(v.x, 0.0f, 1.0f) * 65535.0f)); // Hold the exact integer bits used by this operation.
    std::uint32_t y = std::uint32_t(std::round(glsl::clamp(v.y, 0.0f, 1.0f) * 65535.0f)); // Hold the exact integer bits used by this operation.
    return x | (y << 16); // Pack two normalized values into 16-bit fields.
} // Finish the packUnorm2x16 operation.
inline vec2 unpackUnorm2x16(std::uint32_t p) { // Decode two unsigned normalized 16-bit fields.
    return vec2(float(p & 65535u), float(p >> 16)) / 65535.0f; // Decode two unsigned normalized 16-bit fields.
} // Finish the unpackUnorm2x16 operation.
inline std::uint32_t packSnorm2x16(const vec2& v) { // Pack two signed normalized values into 16-bit fields.
    std::int32_t x = std::int32_t(std::round(glsl::clamp(v.x, -1.0f, 1.0f) * 32767.0f)); // Hold the exact integer bits used by this operation.
    std::int32_t y = std::int32_t(std::round(glsl::clamp(v.y, -1.0f, 1.0f) * 32767.0f)); // Hold the exact integer bits used by this operation.
    return (std::uint32_t(x) & 65535u) | (std::uint32_t(y) << 16); // Pack two signed normalized values into 16-bit fields.
} // Finish the packSnorm2x16 operation.
inline vec2 unpackSnorm2x16(std::uint32_t p) { // Decode two signed normalized 16-bit fields.
    std::int32_t x = std::int32_t(p & 65535u), y = std::int32_t(p >> 16); // Hold the exact integer bits used by this operation.
    if (x & 32768) // Handle the exceptional or boundary case before continuing.
        x -= 65536; // Extend the stored signed field into a signed scalar.
    if (y & 32768) // Handle the exceptional or boundary case before continuing.
        y -= 65536; // Extend the stored signed field into a signed scalar.
    return glsl::clamp(vec2(x, y) / 32767.0f, -1.0f, 1.0f); // Decode two signed normalized 16-bit fields.
} // Finish the unpackSnorm2x16 operation.
inline std::uint32_t packUnorm4x8(const vec4& v) { // Pack four normalized values into bytes.
    std::uint32_t out = 0; // Hold the exact integer bits used by this operation.
    for (std::size_t i = 0; i < 4; ++i) // Visit each component needed by this calculation.
        out |= std::uint32_t(std::round(glsl::clamp(v[i], 0.0f, 1.0f) * 255.0f)) << (i * 8); // Add the encoded field to its packed bit position.
    return out; // Return the completed result.
} // Finish the packUnorm4x8 operation.
inline vec4 unpackUnorm4x8(std::uint32_t p) { // Decode four unsigned normalized bytes.
    vec4 out; // Allocate four independent unpacked components.
    for (std::size_t i = 0; i < 4; ++i) // Visit each component needed by this calculation.
        out[i] = float((p >> (i * 8)) & 255u) / 255.0f; // Store the computed component in the independent result.
    return out; // Return the completed result.
} // Finish the unpackUnorm4x8 operation.
inline std::uint32_t packSnorm4x8(const vec4& v) { // Pack four signed normalized values into bytes.
    std::uint32_t out = 0; // Hold the exact integer bits used by this operation.
    for (std::size_t i = 0; i < 4; ++i) // Visit each component needed by this calculation.
        out |= (std::uint32_t(std::int32_t(std::round(glsl::clamp(v[i], -1.0f, 1.0f) * 127.0f))) & // Add the encoded field to its packed bit position.
                255u) // Keep only the low byte of the signed normalized encoding.
               << (i * 8); // Move that byte into its component position.
    return out; // Return the completed result.
} // Finish the packSnorm4x8 operation.
inline vec4 unpackSnorm4x8(std::uint32_t p) { // Decode four signed normalized bytes.
    vec4 out; // Allocate four independent unpacked components.
    for (std::size_t i = 0; i < 4; ++i) { // Visit each component needed by this calculation.
        int v = int((p >> (i * 8)) & 255u); // Extract this byte into a signed-capable scalar.
        if (v & 128) // Handle the exceptional or boundary case before continuing.
            v -= 256; // Extend an eight-bit negative value to the scalar type.
        out[i] = glsl::max(float(v) / 127.0f, -1.0f); // Store the computed component in the independent result.
    } // Finish this component or matrix pass.
    return out; // Return the completed result.
} // Finish the unpackSnorm4x8 operation.
namespace detail { // Keep these names inside the detail namespace.
inline std::uint16_t float_to_half(float value) { // Encode an IEEE half with ties rounded to even.
    std::uint32_t bits = glsl::floatBitsToUint(value), sign = (bits >> 16) & 32768u; // Separate the sign and payload bits for the conversion.
    int exponent = int((bits >> 23) & 255u) - 127 + 15; // Translate the exponent bias to the target float format.
    std::uint32_t mantissa = bits & 8388607u; // Keep the floating-point significand bits for encoding.
    if (((bits >> 23) & 255u) == 255u) // Handle the exceptional or boundary case before continuing.
        return std::uint16_t(sign | 31744u | (mantissa ? 512u : 0u)); // Encode an IEEE half with ties rounded to even.
    if (exponent >= 31) // Select the matching half-float exponent range.
        return std::uint16_t(sign | 31744u); // Encode an IEEE half with ties rounded to even.
    if (exponent <= 0) { // Select the matching half-float exponent range.
        if (exponent < -10) // Select the matching half-float exponent range.
            return std::uint16_t(sign); // Encode an IEEE half with ties rounded to even.
        mantissa |= 8388608u; // Restore the implicit leading significand bit.
        unsigned shift = unsigned(14 - exponent); // Choose the right shift for a half-float subnormal.
        std::uint32_t truncated = mantissa >> shift, remainder = mantissa & ((1u << shift) - 1u), // Keep the floating-point significand bits for encoding.
                      halfway = 1u << (shift - 1); // Identify the exact halfway rounding bit.
        if (remainder > halfway || (remainder == halfway && (truncated & 1u))) // Round up above halfway or for an odd halfway result.
            ++truncated; // Round the retained encoding to the next representable value.
        return std::uint16_t(sign | truncated); // Encode an IEEE half with ties rounded to even.
    } // Finish handling this condition.
    std::uint32_t truncated = (std::uint32_t(exponent) << 10) | (mantissa >> 13), // Keep the floating-point significand bits for encoding.
                  remainder = mantissa & 8191u; // Retain the discarded low bits for the rounding decision.
    if (remainder > 4096u || (remainder == 4096u && (truncated & 1u))) // Round up above halfway or for an odd halfway result.
        ++truncated; // Round the retained encoding to the next representable value.
    return std::uint16_t(sign | truncated); // Encode an IEEE half with ties rounded to even.
} // Finish the float_to_half operation.
inline float half_to_float(std::uint16_t h) { // Expand an IEEE half into a float.
    std::uint32_t sign = (std::uint32_t(h) & 32768u) << 16, mantissa = h & 1023u; // Separate the sign and payload bits for the conversion.
    int exponent = (h >> 10) & 31; // Translate the exponent bias to the target float format.
    if (exponent == 0) { // Select the matching half-float exponent range.
        if (!mantissa) // Handle the exceptional or boundary case before continuing.
            return glsl::uintBitsToFloat(sign); // Expand an IEEE half into a float.
        exponent = -14; // Set the exponent for a subnormal source value.
        while (!(mantissa & 1024u)) { // Continue until the remaining bits reach the required form.
            mantissa <<= 1; // Move the first significand bit toward the normalized position.
            --exponent; // Adjust the exponent while normalizing the significand.
        } // Finish this component or matrix pass.
        mantissa &= 1023u; // Remove the leading bit before storing the fraction.
        return glsl::uintBitsToFloat(sign | (std::uint32_t(exponent + 127) << 23) | // Expand an IEEE half into a float.
                                     (mantissa << 13)); // Place the expanded significand in the float fraction field.
    } // Finish handling this condition.
    if (exponent == 31) // Select the matching half-float exponent range.
        return glsl::uintBitsToFloat(sign | 2139095040u | (mantissa << 13)); // Expand an IEEE half into a float.
    return glsl::uintBitsToFloat(sign | (std::uint32_t(exponent - 15 + 127) << 23) | // Expand an IEEE half into a float.
                                 (mantissa << 13)); // Place the expanded significand in the float fraction field.
} // Finish the half_to_float operation.
} // namespace detail
inline std::uint32_t packHalf2x16(const vec2& v) { // Pack two IEEE half encodings into one word.
    return std::uint32_t(detail::float_to_half(v.x)) | // Pack two IEEE half encodings into one word.
           (std::uint32_t(detail::float_to_half(v.y)) << 16); // Place the second half encoding in the upper word.
} // Finish the packHalf2x16 operation.
inline vec2 unpackHalf2x16(std::uint32_t v) { // Expand the two IEEE half encodings.
    return vec2(detail::half_to_float(std::uint16_t(v)), // Expand the two IEEE half encodings.
                detail::half_to_float(std::uint16_t(v >> 16))); // Expand the upper half encoding into the second component.
} // Finish the unpackHalf2x16 operation.
/** @} */
} // namespace glsl
} // namespace shdr
#endif // Finish the math layer include guard.

// ---- runtime.hpp ----
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

// ---- async.hpp ----
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

// ---- egl.hpp ----
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

// ---- macros.hpp ----
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

#endif // SHDR_SINGLE_HEADER_INCLUDED
