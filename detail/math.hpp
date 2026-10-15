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
