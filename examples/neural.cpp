/** @file neural.cpp
 * @brief Train and evaluate a dense ANN with full CPU/GPU gradient parity checks.
 */
#include "neural_checks.hpp" // Include neural_checks.hpp declarations.
int main(int argc, char** argv) { // Run this example and validate its results.
    return shdr_examples::binary_example_main(argc, argv, false); // Return the result of this operation.
} // End this block.
