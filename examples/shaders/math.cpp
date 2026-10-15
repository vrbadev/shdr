// Native GLSL compute source. The build compiles a generated C++11 adapter and
// embeds this source in its registered math_kernel factory automatically.
#ifdef __cplusplus // Expose host declarations only when a C++ compiler reads this shader.
#include "shdr.h" // Include shdr.h declarations.
#endif // Finish this conditional compilation branch.

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; // Group 1 independent invocations along x.
layout(std430, binding = 0) writeonly buffer FloatResults { // Write floating-point reference results through buffer binding 0.
    float results[]; // Store floating-point reference results.
}; // Finish this declaration.
layout(std430, binding = 1) writeonly buffer BitResults { // Write integer and packed-bit reference results through buffer binding 1.
    uint bits[]; // Store integer and packed-bit reference results.
}; // Finish this declaration.
uniform vec2 controls; // Supply runtime values for the absolute-value and exact-bit tests.

// Native GLSL out parameters become C++ references in the generated adapter.
void split(in vec2 value, out vec2 integral, out vec2 fractional) { // Split a vector into signed integer and fractional parts through out parameters.
    fractional = modf(value, integral); // Use modf to fill the integer out parameter and return the fractional part.
} // End this block.

void main() { // Record deterministic GLSL builtin and numeric results for CPU and GPU comparison.
    vec3 c = cross(vec3(2.0, 3.0, 4.0), vec3(5.0, 6.0, 7.0)); // Form a cross product whose three components are checked independently.
    results[0] = c.x; // Record the x component of the three-dimensional cross product.
    results[1] = c.y; // Record the y component of the three-dimensional cross product.
    results[2] = c.z; // Record the z component of the three-dimensional cross product.
    results[3] = dot(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)); // Record the scalar dot product of two vectors.
    results[4] = length(vec2(3.0, 4.0)); // Record the Euclidean length of a three-four vector.
    vec2 normalized = normalize(vec2(3.0, 4.0)); // Normalize a vector with a known three-four-five length.
    results[5] = normalized.x; // Record the normalized x component.
    results[6] = normalized.y; // Record the normalized y component.
    results[7] = reflect(vec3(1.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0)).y; // Record the reflected vector's y component.
    results[8] = refract(vec2(0.0, -1.0), vec2(0.0, 1.0), 0.5).y; // Record the refracted vector's y component.
    results[9] = mod(-3.5, 2.0); // Record floor-based mod with a negative dividend.
    results[10] = mod(3.5, -2.0); // Record floor-based mod with a negative divisor.
    results[11] = fract(-1.25); // Record the positive fractional part of a negative input.
    vec4 rounded = roundEven(vec4(0.5, 1.5, 2.5, -1.5)); // Exercise halfway rounding on positive and negative values.
    results[12] = rounded.x; // Record ties-to-even rounding of positive one-half.
    results[13] = rounded.y; // Record ties-to-even rounding of positive one-and-a-half.
    results[14] = rounded.z; // Record ties-to-even rounding of positive two-and-a-half.
    results[15] = rounded.w; // Record ties-to-even rounding of negative one-and-a-half.

    mat2 a = mat2(1.0, 2.0, 3.0, 4.0); // Specify the two-by-two test matrix in GLSL column order.
    mat3 b = mat3(3.0, 1.0, 3.0, 5.0, 5.0, 9.0, 2.0, 8.0, 2.0); // Specify the three-by-three determinant test matrix in GLSL column order.
    results[16] = determinant(a); // Record the determinant of the two-by-two matrix.
    results[17] = determinant(b); // Record the determinant of the three-by-three matrix.
    vec2 mv = a * vec2(2.0, 3.0); // Multiply a matrix by a column vector.
    vec2 vm = vec2(2.0, 3.0) * a; // Multiply a row vector by the same matrix.
    results[18] = mv.x; // Record the x component of matrix times column vector.
    results[19] = mv.y; // Record the y component of matrix times column vector.
    results[20] = vm.x; // Record the x component of row vector times matrix.
    results[21] = vm.y; // Record the y component of row vector times matrix.
    mat2 product = a * a; // Check the row-by-column products of ordinary matrix multiplication.
    results[22] = product[0][0]; // Record the first diagonal matrix product element.
    results[23] = product[1][1]; // Record the second diagonal matrix product element.
    mat2 identity = a * inverse(a); // Multiply a nonsingular matrix by its inverse.
    results[24] = identity[0][0]; // Record the identity product's first diagonal element.
    results[25] = identity[1][0]; // Record the identity product's off-diagonal element.
    mat3x2 rectangular = mat3x2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0); // Use a three-column, two-row matrix to check non-square dimensions.
    mat2 rectangular_product = rectangular * mat2x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0); // Multiply compatible rectangular dimensions into a two-by-two result.
    results[26] = rectangular_product[1][1]; // Record a rectangular matrix product element.
    mat2x3 transposed = transpose(rectangular); // Swap the rectangular matrix's row and column dimensions.
    results[27] = transposed[0][1]; // Record a rectangular transpose element.
    mat2x3 outer = outerProduct(vec3(1.0, 2.0, 3.0), vec2(4.0, 5.0)); // Create a matrix from the outer product of two differently sized vectors.
    results[28] = outer[1][2]; // Record an outer-product element.
    mat2 components = matrixCompMult(a, a); // Multiply matching matrix components independently.
    results[29] = components[1][1]; // Record a componentwise matrix product element.

    vec2 swapped = vec2(1.0, 2.0); // Start with distinct components so aliasing swizzle errors are visible.
    swapped.xy = swapped.yx; // Read both source components before assigning their reversed order.
    results[30] = swapped.x; // Record the x component after an aliasing swizzle assignment.
    results[31] = swapped.y; // Record the y component after an aliasing swizzle assignment.
    swapped += swapped.yx; // Snapshot the reversed source before applying both swizzle writes.
    results[32] = swapped.x; // Record the x component after an aliasing swizzle addition.
    results[33] = swapped.y; // Record the y component after an aliasing swizzle addition.
    results[34] = mix(vec2(2.0), vec2(4.0), 0.25).x; // Record floating-point vector interpolation.
    vec2 selected = mix(vec2(2.0, 3.0), vec2(4.0, 5.0), bvec2(true, false)); // Choose components from different inputs using a boolean selector vector.
    results[35] = float(selected.x); // Record boolean selection of the first vector component.
    results[36] = float(selected.y); // Record boolean selection of the second vector component.
    results[37] = smoothstep(0.0, 1.0, 0.5); // Record the midpoint of the cubic smoothstep curve.
    results[38] = step(1.0, 0.5); // Record a step input below the edge.
    results[39] = step(1.0, 1.0); // Record a step input exactly on the edge.
    results[40] = degrees(radians(90.0)); // Record the angle after a radians-degrees round trip.
    results[41] = inversesqrt(16.0); // Record the reciprocal square root of sixteen.
    vec2 integral; // Receive the signed integer parts through the helper's out parameter.
    vec2 fractional; // Receive the signed fractional parts through the helper's out parameter.
    split(vec2(-1.25, 2.5), integral, fractional); // Exercise vector out-parameter translation with positive and negative inputs.
    results[42] = integral.x; // Record the signed integer part produced through an out parameter.
    results[43] = fractional.x; // Record the signed fractional part produced through an out parameter.
    ivec2 exponent; // Receive the base-two exponents through frexp's out parameter.
    vec2 mantissa = frexp(vec2(4.0, 0.125), exponent); // Separate each input into a signed mantissa and a base-two exponent.
    vec2 restored = ldexp(mantissa, exponent); // Recombine those mantissas and exponents with ldexp.
    results[44] = restored.x; // Record a value reconstructed from binary mantissa and exponent.
    results[45] = ldexp(0.5, 3); // Record scalar reconstruction from a mantissa and exponent.
    results[46] = unpackHalf2x16(packHalf2x16(vec2(1.0, -2.0))).y; // Record a packed half-float round trip.
    results[47] = unpackSnorm2x16(packSnorm2x16(vec2(-1.0, 1.0))).x; // Record a packed signed-normalized round trip.
    results[48] = abs(controls.y); // Record the absolute value of a runtime uniform component.
    results[49] = clamp(0.5, 0.0, 1.0); // Record scalar clamping inside its interval.
    results[50] = faceforward(vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(0.0, 1.0)).y; // Record the normal chosen by faceforward.
    bvec2 inverted = not(bvec2(false, true)); // Apply componentwise GLSL logical negation.
    results[51] = float(inverted.x); // Record the first component of a logical vector inversion.
    results[52] = mix(2.0, 4.0, 0.25); // Record scalar floating-point interpolation.
    results[53] = pow(3.0, 2.0); // Record a scalar power.

    bits[0] = floatBitsToUint(controls.x); // Record the exact IEEE bits of a runtime floating-point value.
    bits[1] = floatBitsToUint(-0.0); // Record the IEEE sign bit of negative zero.
    bits[2] = uint(bitCount(-1)); // Record the number of set bits in a signed negative integer.
    bits[3] = uint(findLSB(8u)); // Record the least-significant set-bit position.
    bits[4] = uint(findMSB(-2)); // Record the signed most-significant set-bit position.
    bits[5] = bitfieldReverse(1u); // Record a complete reversal of the low input bit.
    bits[6] = uint(bitfieldExtract(240, 4, 4)); // Record an extracted signed bit field.
    bits[7] = bitfieldInsert(0xff00u, 3u, 0, 4); // Record an inserted low unsigned bit field.
    uint carry; // Receive the unsigned-addition carry through an out parameter.
    uint sum = uaddCarry(0xffffffffu, 2u, carry); // Add across the uint limit to check the result and carry separately.
    bits[8] = sum; // Record the wrapped sum from unsigned addition.
    bits[9] = carry; // Record the carry from that unsigned addition.
    uint hi; // Receive the high half of the full unsigned product.
    uint lo; // Receive the low half of the full unsigned product.
    umulExtended(0xffffffffu, 0xffffffffu, hi, lo); // Multiply two uint values into an exact two-word unsigned product.
    bits[10] = hi; // Record the high word of the full unsigned product.
    bits[11] = lo; // Record the low word of the full unsigned product.
    bits[12] = packHalf2x16(vec2(1.0, -2.0)); // Record two half-float encodings packed into one uint.
    bits[13] = packUnorm2x16(vec2(0.0, 1.0)); // Record two unsigned normalized components packed into one uint.
    bits[14] = packUnorm4x8(vec4(1.0, 0.0, 1.0, 0.0)); // Record four unsigned normalized bytes packed into one uint.
    uint borrow; // Receive the unsigned-subtraction borrow through an out parameter.
    uint difference = usubBorrow(0u, 1u, borrow); // Subtract across zero to check the wrapped difference and borrow separately.
    bits[15] = difference; // Record the wrapped difference from unsigned subtraction.
    bits[16] = borrow; // Record the borrow from that unsigned subtraction.
} // End this block.
