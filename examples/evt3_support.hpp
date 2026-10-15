/** @file evt3_support.hpp
 * @brief Reusable host wiring for a compact, streaming EVT3 example.
 * Native shaders parse packed words, resolve block state, scan output lengths,
 * and emit into disjoint intervals. Host readback is required only by a consumer.
 */
#ifndef SHDR_EXAMPLES_EVT3_SUPPORT_HPP // Open this include guard or optional feature branch.
#define SHDR_EXAMPLES_EVT3_SUPPORT_HPP // Define SHDR_EXAMPLES_EVT3_SUPPORT_HPP for the declarations that follow.
#include "shdr.h" // Include shdr.h declarations.
#include <algorithm> // Include algorithm declarations.
#include <cstdint> // Include cstdint declarations.
#include <random> // Include random declarations.
#include <vector> // Include vector declarations.

namespace shdr_examples { // Group these helpers within their namespace.

/** @brief Stateful four-stage decoder. reset_pipeline() adds a GPU/CPU reset stage.
 * Each uvec4 event stores X, packed Y/polarity/camera, and low/high uint32 timestamp.
 * Triggers use channel ID as X and set bit31 of the second component, with edge in bit0.
 * Output ordering follows input ordering. Metadata extensions are skipped. Use
 * separate decoder objects for independently queued streams or concurrent workers.
 */
class evt3_decoder { // Declare evt3 decoder.
    unsigned capacity_, block_size_, count_, blocks_; // Declare capacity for the following calculation.
    std::unique_ptr<shdr::kernel> summary_, seed_, prefix_, emit_, reset_; // Declare summary for the following calculation.
    shdr::buffer<unsigned> packed_, sizes_; // Declare packed for the following calculation.
    shdr::buffer<shdr::glsl::uvec4> summaries_, seeds_; // Declare summaries for the following calculation.
    shdr::pipeline stream_, independent_; // Declare stream for the following calculation.
    static unsigned capacity(unsigned n) { // Implement capacity for the operations below.
        if (!n || n > 0xffffffffu / 12u) // Check this condition before entering the following branch.
            throw shdr::error( // Report this failed operation.
                "EVT3 word capacity must be positive and fit worst-case output offsets"); // Provide this text argument for the operation above.
        return n; // Return n.
    } // End this block.
    static unsigned block(unsigned n) { // Implement block for the operations below.
        if (!n || n > 4096u) // Check this condition before entering the following branch.
            throw shdr::error("EVT3 block size must be between 1 and 4096 words"); // Report EVT3 block size must be between 1 and 4096 words.
        return n; // Return n.
    } // End this block.

  public: // Set public access for the following class members.
    shdr::buffer<shdr::glsl::uvec4> events; // Declare events for the following calculation.
    shdr::buffer<unsigned> offsets; // Declare offsets for the following calculation.
    shdr::buffer<shdr::glsl::uvec4> state; // Declare state for the following calculation.

    /** @brief Allocate reusable storage, including worst-case twelve events per word. */
    explicit evt3_decoder(unsigned max_words, unsigned block_size = 128) // Set block size from this expression.
        : capacity_(capacity(max_words)), block_size_(block(block_size)), count_(0), blocks_(0), // Declare capacity with the parameters continued below.
          summary_(shdr::make_kernel("evt3_summary_kernel")), // Create the linked evt3 summary kernel instance.
          seed_(shdr::make_kernel("evt3_seed_kernel")), // Create the linked evt3 seed kernel instance.
          prefix_(shdr::make_kernel("rle_offsets_kernel")), // Create the linked rle offsets kernel instance.
          emit_(shdr::make_kernel("evt3_emit_kernel")), // Create the linked evt3 emit kernel instance.
          reset_(shdr::make_kernel("evt3_reset_kernel")), packed_((capacity_ + 1u) / 2u), // Create the linked evt3 reset kernel instance.
          sizes_((capacity_ + block_size_ - 1u) / block_size_), summaries_(sizes_.size() * 3u), // Apply sizes to these arguments.
          seeds_(sizes_.size() * 2u), events(capacity_ * 12u), offsets(sizes_.size() + 1u), // Apply seeds to these arguments.
          state(2) { // Apply state to these arguments.
        summary_->bind("packed_words", packed_) // Connect this shader port to its application resource.
            .bind("summaries", summaries_) // Connect this shader port to its application resource.
            .set("block_size", block_size_); // Apply set to these arguments.
        seed_->bind("summaries", summaries_) // Connect this shader port to its application resource.
            .bind("seeds", seeds_) // Connect this shader port to its application resource.
            .bind("sizes", sizes_) // Connect this shader port to its application resource.
            .bind("stream_state", state); // Connect this shader port to its application resource.
        prefix_->bind("sizes", sizes_).bind("offsets", offsets); // Connect this shader port to its application resource.
        emit_->bind("packed_words", packed_) // Connect this shader port to its application resource.
            .bind("seeds", seeds_) // Connect this shader port to its application resource.
            .bind("offsets", offsets) // Connect this shader port to its application resource.
            .bind("events", events) // Connect this shader port to its application resource.
            .set("block_size", block_size_); // Apply set to these arguments.
        reset_->bind("stream_state", state); // Connect this shader port to its application resource.
        load(std::vector<std::uint16_t>()); // Apply load to these arguments.
    } // End this block.
    /** @brief Copy words into packed storage and adjust dispatches while preserving decoder state. */
    void load(const std::vector<std::uint16_t>& words) { // Implement load for the operations below.
        if (words.size() > capacity_) // Check this condition before entering the following branch.
            throw shdr::error("EVT3 batch exceeds allocated capacity"); // Report EVT3 batch exceeds allocated capacity.
        count_ = static_cast<unsigned>(words.size()); // Set count from this expression.
        blocks_ = (count_ + block_size_ - 1u) / block_size_; // Set blocks from this expression.
        unsigned* data = packed_.data(); // Set data from this expression.
        for (unsigned i = 0; i < (count_ + 1u) / 2u; ++i) { // Iterate over i for this calculation.
            data[i] = words[2u * i]; // Set data[i] from this expression.
            if (2u * i + 1u < count_) // Check this condition before entering the following branch.
                data[i] |= unsigned(words[2u * i + 1u]) << 16u; // Update data[i] with this contribution.
        } // End this block.
        summary_->set("word_count", count_); // Apply set to these arguments.
        emit_->set("word_count", count_); // Apply set to these arguments.
        seed_->set("blocks", blocks_); // Apply set to these arguments.
        prefix_->set("blocks", blocks_); // Apply set to these arguments.
        const shdr::extent groups(std::max(1u, (blocks_ + 63u) / 64u)); // Construct groups with these initial values.
        stream_ = shdr::dispatch(*summary_, groups) >> shdr::dispatch(*seed_, shdr::extent(1)) >> // Add a shader dispatch with the required workgroup counts.
                  shdr::dispatch(*prefix_, shdr::extent(1)) >> shdr::dispatch(*emit_, groups); // Add a shader dispatch with the required workgroup counts.
        independent_ = shdr::dispatch(*reset_, shdr::extent(1)) >> // Add a shader dispatch with the required workgroup counts.
                       shdr::dispatch(*summary_, groups) >> // Add a shader dispatch with the required workgroup counts.
                       shdr::dispatch(*seed_, shdr::extent(1)) >> // Add a shader dispatch with the required workgroup counts.
                       shdr::dispatch(*prefix_, shdr::extent(1)) >> shdr::dispatch(*emit_, groups); // Add a shader dispatch with the required workgroup counts.
    } // End this block.
    /** @brief Execute this for consecutive batches from the same stream. */
    const shdr::pipeline& stream_pipeline() const { // Implement stream pipeline for the operations below.
        return stream_; // Return stream.
    } // End this block.
    /** @brief Execute this for independent batches, with deterministic initial state. */
    const shdr::pipeline& reset_pipeline() const { // Implement reset pipeline for the operations below.
        return independent_; // Return independent.
    } // End this block.
    /** @brief Return the encoded word count in the current batch. */
    unsigned word_count() const { // Implement word count for the operations below.
        return count_; // Return count.
    } // End this block.
    /** @brief Return the number of active block summaries and output intervals. */
    unsigned blocks() const { // Implement blocks for the operations below.
        return blocks_; // Return blocks.
    } // End this block.
    /** @brief Read compact output length, synchronizing this buffer after GPU work. */
    unsigned event_count() const { // Implement event count for the operations below.
        return offsets.data()[blocks_]; // Return offsets.data()[blocks ].
    } // End this block.
}; // Finish this declaration.

/** @brief Pack decoded CD events into X11/Y10/polarity1/time42 uint64 records.
 * This CPU consumer downloads completed GPU results and packs CD pixel events.
 * The lossless layout accepts X11, Y10 and time42 values, with range errors
 * reported as exceptions. The original uvec4 records support larger geometry.
 */
inline std::vector<std::uint64_t> // Declare std for the following calculation.
evt3_compact_cd_to64(const shdr::buffer<shdr::glsl::uvec4>& decoded, unsigned count) { // Apply evt3 compact cd to64 to these arguments.
    if (count > decoded.size()) // Check this condition before entering the following branch.
        throw shdr::error("EVT3 compact count exceeds decoded storage"); // Report EVT3 compact count exceeds decoded storage.
    const shdr::glsl::uvec4* events = decoded.data(); // Set events from this expression.
    std::vector<std::uint64_t> packed; // Declare packed for the following calculation.
    packed.reserve(count); // Apply reserve to these arguments.
    for (unsigned i = 0; i < count; ++i) { // Iterate over i for this calculation.
        const shdr::glsl::uvec4& event = events[i]; // Set event from this expression.
        if (event.y & 0x80000000u) // Check this condition before entering the following branch.
            continue; // Skip the remaining work for this loop iteration.
        const unsigned y = event.y & 2047u; // Set y from this expression.
        if (event.x >= 2048u || y >= 1024u) // Check this condition before entering the following branch.
            throw shdr::error("64-bit EVT3 CD packing requires X<2048 and Y<1024"); // Report 64-bit EVT3 CD packing requires X<2048 and Y<1024.
        if (event.w >= 1024u) // Check this condition before entering the following branch.
            throw shdr::error("64-bit EVT3 CD packing requires timestamps below 2^42 microseconds"); // Report 64-bit EVT3 CD packing requires timestamps below 2^42 microseconds.
        const std::uint64_t timestamp = (std::uint64_t(event.w) << 32u) | event.z; // Set timestamp from this expression.
        const unsigned polarity = (event.y >> 11u) & 1u; // Set polarity from this expression.
        packed.push_back(std::uint64_t(event.x) | (std::uint64_t(y) << 11u) | // Apply push back to these arguments.
                         (std::uint64_t(polarity) << 21u) | (timestamp << 22u)); // Apply uint64 t to these arguments.
    } // End this block.
    return packed; // Return packed.
} // End this block.

/** @brief Generate reproducible vector-heavy input, crossing blocks and timestamp wraps. */
inline std::vector<std::uint16_t> evt3_random_words(unsigned count, unsigned seed = 12345) { // Set seed from this expression.
    std::mt19937 random(seed); // Construct random with these initial values.
    std::vector<std::uint16_t> words; // Declare words for the following calculation.
    words.reserve(count); // Apply reserve to these arguments.
    unsigned high = 4093u; // Set high from this expression.
    for (unsigned row = 0; words.size() < count; ++row) { // Iterate over row for this calculation.
        const unsigned header[] = {0x8000u | (high & 4095u), // Declare header for the following calculation.
                                   0x6000u | (unsigned(random()) & 4095u), // Apply unsigned to these arguments.
                                   unsigned(random()) & 4095u, // Apply unsigned to these arguments.
                                   0x3000u | (unsigned(random()) % 1000u) | // Apply unsigned to these arguments.
                                       ((unsigned(random()) & 1u) << 11u)}; // Apply unsigned to these arguments.
        for (unsigned value : header) // Visit each value required by this loop.
            if (words.size() < count) // Check this condition before entering the following branch.
                words.push_back(static_cast<std::uint16_t>(value)); // Apply push back to these arguments.
        for (unsigned j = 0; j < 13 && words.size() < count; ++j) { // Iterate over j for this calculation.
            unsigned value; // Declare value for the following calculation.
            if (j == 4) // Check this condition before entering the following branch.
                value = 0xa000u | ((random() & 15u) << 8u) | (random() & 1u); // Set value from this expression.
            else if (j == 9) // Insert a scalar CD event between vector masks.
                value = 0x2000u | (random() & 4095u); // Set value from this expression.
            else // Alternate twelve-bit and eight-bit vector-mask words.
                value = (j & 1u ? 0x5000u : 0x4000u) | (random() & (j & 1u ? 255u : 4095u)); // Set value from this expression.
            words.push_back(static_cast<std::uint16_t>(value)); // Apply push back to these arguments.
        } // End this block.
        high = (high + 1u) & 4095u; // Set high from this expression.
    } // End this block.
    return words; // Return words.
} // End this block.
} // namespace shdr_examples
#endif // Finish this conditional compilation branch.
