/** @file lenet_support.hpp
 * @brief A complete modern LeNet-5-style classifier with seven native stages.
 * This version uses full C3 connectivity, fixed average pooling, ReLU and a
 * learned linear ten-class head. These are explicit modern adaptations of the
 * historical network. Inputs are 28-by-28 grayscale values between zero and one.
 * Model preparation may use a separate framework, but deployed inference uses
 * only standard C++11 and the shdr single-file header.
 */
#ifndef SHDR_EXAMPLES_LENET_SUPPORT_HPP // Include this model helper only once.
#define SHDR_EXAMPLES_LENET_SUPPORT_HPP // Remember that this model helper is available.
#include "shdr.h" // Use portable resources and CPU or GPU dispatches.
#include <algorithm> // Copy loaded model values into retained resources.
#include <cmath> // Reject nonfinite loaded model values.
#include <vector> // Accept a complete flattened parameter record.
namespace shdr_examples { // Keep the classifier helper with the other reusable examples.
/** @brief A fully connected C3, ReLU and average-pooling LeNet-5-style network.
 * Layer shapes are 1x28x28 to 6x28x28 to 6x14x14 to 16x10x10 to 16x5x5,
 * followed by feature vectors of 120, 84 and 10 values. First convolution uses
 * two implicit zero pixels on each edge. The parameter order matches PyTorch
 * convolution [out,in,ky,kx] and linear [out,in] tensors with adjacent biases.
 */
class lenet_classifier { // Keep model resources and their seven linked inference stages together.
    static unsigned // Return a validated unsigned image count.
    checked_batch(unsigned count) { // Validate dimensions before allocating intermediate features.
        if (!count || count > 512u) // Reject empty batches and batches beyond this helper's dispatch limit.
            throw shdr::error( // Keep the largest stage within minimum GLES work group limits.
                "LeNet batch count must be between 1 and 512"); // Keep the largest stage within minimum GLES work group limits.
        return count; // Return the validated number of independent images.
    } // Finish batch validation.
    unsigned batch_; // Remember the number of images sharing this classifier instance.
    std::unique_ptr<shdr::kernel> conv1_, pool1_, conv2_, pool2_, dense1_, dense2_, // Own the seven compiled shader instances.
        dense3_; // Own the seven compiled shader instances.
    shdr::pipeline pipeline_; // Keep the ordered inference dispatches ready for repeated execution.
    static shdr::extent // Return the dispatch size for a complete feature layer.
    groups(unsigned values) { // Cover every output feature with 64-wide work groups.
        return shdr::extent( // Build a one-dimensional work group count.
            values / 64u + // Count all completely filled 64-invocation groups.
            (values % 64u ? 1u : 0u)); // Include a final partial group when necessary.
    } // Finish group-count calculation.
    void convolution( // Configure one learned convolution stage.
        shdr::kernel& shader, // Configure the selected native shader instance.
        const shdr::buffer<float>& source, // Read features produced by the preceding layer.
        const shdr::buffer<float>& output, // Write features consumed by the following layer.
        unsigned width, // Specify the width of each input feature plane.
        unsigned height, // Specify the height of each input feature plane.
        unsigned input_channels, // Specify how many input feature planes each image has.
        unsigned output_channels, // Specify how many learned feature planes to produce.
        unsigned padding, // Specify the number of zero pixels around the input.
        unsigned weight_offset, // Locate the first weight of this layer.
        unsigned bias_offset) { // Connect one convolution stage to its exact model geometry.
        shader.bind("input_values", source) // Connect the preceding features as this layer's input.
            .bind("output_values", output) // Connect storage for this layer's output features.
            .bind( // Attach the shared model parameter resource.
                "parameters", // Select the native port for learned weights and biases.
                parameters); // Share the preceding features, following features and model parameters.
        shader.set<unsigned>("batch_count", batch_) // Process the configured number of images.
            .set<unsigned>("input_width", width) // Pass the input feature-plane width.
            .set<unsigned>("input_height", height); // Set the input geometry for every image.
        shader.set<unsigned>("input_channels", input_channels) // Pass the number of source feature planes.
            .set<unsigned>("output_channels", output_channels); // Set the two feature-plane counts.
        shader.set<unsigned>("kernel_size", 5u) // Use five-by-five learned convolution kernels.
            .set<unsigned>( // Set the following unsigned geometry option.
                "padding", // Select the zero-padding geometry option.
                padding); // Use five-by-five learned kernels with the requested zero padding.
        shader.set<unsigned>("weight_offset", weight_offset) // Pass the first weight index for this layer.
            .set<unsigned>("bias_offset", // Pass the first bias index for this layer.
                           bias_offset); // Locate this layer inside the shared flat model.
    } // Finish convolution resource wiring.
    void pooling(shdr::kernel& shader, // Configure one fixed average-pooling stage.
                 const shdr::buffer<float>& source, // Read features produced by the preceding layer.
                 const shdr::buffer<float>& output, // Write features consumed by the following layer.
                 unsigned width, // Specify the width of each input feature plane.
                 unsigned channels) { // Connect one fixed two-by-two average-pooling stage.
        shader.bind("input_values", source) // Connect the preceding features as this layer's input.
            .bind("output_values", output); // Keep pooling input and output in distinct resources.
        shader.set<unsigned>("batch_count", batch_) // Process the configured number of images.
            .set<unsigned>("channels", channels); // Process every image and channel independently.
        shader.set<unsigned>("input_width", width) // Pass the square feature-plane width.
            .set<unsigned>("input_height", width); // Set the square input plane used by this model.
    } // Finish pooling resource wiring.
    void dense(shdr::kernel& shader, // Configure one learned fully connected stage.
               const shdr::buffer<float>& source, // Read features produced by the preceding layer.
               const shdr::buffer<float>& output, // Write features consumed by the following layer.
               unsigned input_features, // Specify the length of each input feature vector.
               unsigned output_features, // Specify the length of each output feature vector.
               unsigned weight_offset, // Locate the first weight of this layer.
               unsigned bias_offset, // Locate the first bias of this layer.
               bool relu) { // Connect a dense layer to its flattened feature and parameter ranges.
        shader.bind("input_values", source) // Connect the preceding features as this layer's input.
            .bind("output_values", output) // Connect storage for this layer's output features.
            .bind("parameters", // Share the flat record of learned weights and biases.
                  parameters); // Share the input vector, output vector and learned model values.
        shader.set<unsigned>("batch_count", batch_) // Process the configured number of images.
            .set<unsigned>("input_features", input_features) // Pass the length of each source feature vector.
            .set<unsigned>("output_features", // Pass the length of each result feature vector.
                           output_features); // Set each image's vector dimensions.
        shader.set<unsigned>("weight_offset", weight_offset) // Pass the first weight index for this layer.
            .set<unsigned>("bias_offset", // Pass the first bias index for this layer.
                           bias_offset); // Locate this layer's row-major weights and biases.
        shader.set<unsigned>( // Pass the activation choice to the native stage.
            "activation", // Select linear output or ReLU output.
            relu ? 1u : 0u); // Keep the final logits linear while hidden layers use ReLU.
    } // Finish dense resource wiring.
  public: // Expose model resources and the complete inference pipeline.
    enum { // Publish the fixed image and parameter dimensions.
        parameter_count = 61706, // Count all weights and biases in this architecture.
        input_features = 784, // Count the 28-by-28 grayscale values in one image.
        class_count = 10 // Count the ten possible digit classes.
    }; // Publish the exact model and image sizes.
    shdr::buffer<float> inputs; // Store normalized 28-by-28 grayscale images consecutively.
    shdr::buffer<float> parameters; // Store all learned weights and biases in the documented order.
    shdr::buffer<float> conv1; // Retain the 6-by-28-by-28 first convolution features per image.
    shdr::buffer<float> pool1; // Retain the 6-by-14-by-14 first pooled features per image.
    shdr::buffer<float> conv2; // Retain the 16-by-10-by-10 second convolution features per image.
    shdr::buffer<float> pool2; // Retain the 16-by-5-by-5 second pooled features per image.
    shdr::buffer<float> hidden1; // Retain the 120 learned dense features per image.
    shdr::buffer<float> hidden2; // Retain the 84 learned dense features per image.
    shdr::buffer<float> logits; // Retain ten linear class scores per image.
    /** @brief Allocate and connect a batch of complete seven-stage classifiers. */
    explicit lenet_classifier( // Create one reusable complete classifier.
        unsigned batch_count = 32) // Construct one reusable batch-sized model instance.
        : batch_(checked_batch( // Validate the batch before any feature allocation.
              batch_count)), // Check the batch before allocating model or feature storage.
          conv1_(shdr::make_kernel("lenet_conv_kernel")), // Create the first learned convolution instance.
          pool1_(shdr::make_kernel( // Create the first fixed average-pooling instance.
              "lenet_pool_kernel")), // Create the first convolution and pooling stages.
          conv2_(shdr::make_kernel("lenet_conv_kernel")), // Create the second learned convolution instance.
          pool2_(shdr::make_kernel( // Create the second fixed average-pooling instance.
              "lenet_pool_kernel")), // Create independent instances for the second feature block.
          dense1_(shdr::make_kernel("lenet_dense_kernel")), // Create the 120-feature dense layer instance.
          dense2_(shdr::make_kernel("lenet_dense_kernel")), // Create the 84-feature dense layer instance.
          dense3_(shdr::make_kernel("lenet_dense_kernel")), // Create each learned dense layer.
          inputs(batch_ * 784u), // Allocate all normalized input images.
          parameters(parameter_count), // Allocate every image value and learned parameter.
          conv1(batch_ * 4704u), pool1(batch_ * 1176u), conv2(batch_ * 1600u), // Allocate the first three spatial feature records.
          pool2(batch_ * 400u), // Allocate the NCHW spatial intermediates.
          hidden1(batch_ * 120u), hidden2(batch_ * 84u), // Allocate the two learned hidden feature records.
          logits(batch_ * 10u) { // Allocate the dense intermediates and class logits.
        convolution(*conv1_, // Connect the first learned convolution.
                    inputs, // Read the normalized grayscale input images.
                    conv1, // Write the first learned spatial features.
                    28u, // Use the 28-pixel input dimension.
                    28u, // Use the 28-pixel input dimension.
                    1u, // Read one grayscale source plane.
                    6u, // Use the six first-block feature planes.
                    2u, // Pad each edge with two implicit zero pixels.
                    0u, // Start the first convolution weight record.
                    150u); // Wire padded first convolution with its adjacent six biases.
        pooling( // Connect the first average-pooling reduction.
            *pool1_, conv1, pool1, 28u, 6u); // Reduce each first-layer plane from 28 to 14 pixels.
        convolution(*conv2_, // Connect the second learned convolution.
                    pool1, // Read the first pooled spatial features.
                    conv2, // Use the second learned spatial feature record.
                    14u, // Use the 14-pixel first pooled dimension.
                    14u, // Use the 14-pixel first pooled dimension.
                    6u, // Use the six first-block feature planes.
                    16u, // Use the sixteen second-block feature planes.
                    0u, // Apply the second convolution over valid five-by-five input windows.
                    156u, // Start the second convolution weight record.
                    2556u); // Wire full C3 convolution over all six source planes.
        pooling(*pool2_, // Connect the second average-pooling reduction.
                conv2, // Use the second learned spatial feature record.
                pool2, // Use the final pooled spatial feature record.
                10u, // Use the ten-value dimension required by this layer.
                16u); // Reduce each second-layer plane from 10 to five pixels.
        dense(*dense1_, // Connect the first fully connected projection.
              pool2, // Use the final pooled spatial feature record.
              hidden1, // Use the 120-feature hidden representation.
              400u, // Read all sixteen five-by-five pooled planes.
              120u, // Use the 120-feature first dense representation.
              2572u, // Start the 400-to-120 dense weight record.
              50572u, // Start the 120 first dense biases.
              true); // Apply the flattened full C5 projection and ReLU.
        dense(*dense2_, // Connect the second fully connected projection.
              hidden1, // Use the 120-feature hidden representation.
              hidden2, // Use the 84-feature hidden representation.
              120u, // Use the 120-feature first dense representation.
              84u, // Use the 84-feature second dense representation.
              50692u, // Start the 120-to-84 dense weight record.
              60772u, // Start the 84 second dense biases.
              true); // Apply the 84-feature learned projection and ReLU.
        dense(*dense3_, // Connect the final linear class projection.
              hidden2, // Use the 84-feature hidden representation.
              logits, // Write the ten linear digit scores.
              84u, // Use the 84-feature second dense representation.
              10u, // Use the ten-value dimension required by this layer.
              60856u, // Start the 84-to-10 classifier weight record.
              61696u, // Start the ten final classifier biases.
              false); // Produce ten unnormalized class logits.
        pipeline_ = shdr::dispatch(*conv1_, groups(batch_ * 4704u)) >> // Run the first convolution before its pooled output is consumed.
                    shdr::dispatch(*pool1_, // Dispatch the first feature-plane reduction.
                                   groups(batch_ * 1176u)); // Link the first spatial feature block.
        pipeline_ += shdr::dispatch( // Append the next dependent native layer.
            *conv2_, groups(batch_ * 1600u)); // Append the second learned convolution.
        pipeline_ += shdr::dispatch( // Append the next dependent native layer.
            *pool2_, groups(batch_ * 400u)); // Append the second average-pooling stage.
        pipeline_ += // Append the next learned projection to the inference order.
            shdr::dispatch(*dense1_, groups(batch_ * 120u)); // Append the 120-feature dense stage.
        pipeline_ += // Append the next learned projection to the inference order.
            shdr::dispatch(*dense2_, groups(batch_ * 84u)); // Append the 84-feature dense stage.
        pipeline_ += shdr::dispatch( // Append the next dependent native layer.
            *dense3_, groups(batch_ * 10u)); // Append the final ten-class linear head.
    } // Finish complete model wiring.
    /** @brief Return the number of independent images in this model instance. */
    unsigned batch_count() const { // Expose the configured number of independent images.
        return batch_; // Return the validated batch size.
    } // Return the validated batch size from host metadata.
    /** @brief Copy an entire finite model parameter record into shared storage. */
    void load_parameters( // Load a complete finite flattened model.
        const std::vector<float>& values) { // Accept only the documented complete model layout.
        if (values.size() != parameter_count) // Require exactly the weights and biases of this architecture.
            throw shdr::error( // Reject a truncated or mismatched trained model.
                "LeNet parameters require exactly 61706 values"); // Reject a truncated or mismatched trained model.
        for (float value : values) // Inspect every model value before copying the record.
            if (!std::isfinite(value)) // Reject infinities and NaNs in loaded parameters.
                throw shdr::error( // Prevent invalid loaded values from entering inference.
                    "LeNet parameters must be finite"); // Prevent invalid loaded values from entering inference.
        std::copy(values.begin(), // Copy the complete validated parameter range.
                  values.end(), // Stop copying at the end of the model record.
                  parameters.data()); // Make the loaded model authoritative on the host.
    } // Finish model parameter loading.
    /** @brief Return all seven native inference stages in their data-flow order. */
    const shdr::pipeline& inference_pipeline() const { // Expose all seven linked inference stages.
        return pipeline_; // Return the reusable native CPU or GPU computation.
    } // Expose the same complete computation to CPU and GPU execution.
}; // Finish the reusable classifier helper.
} // namespace shdr_examples
#endif // Finish the classifier helper include guard.
