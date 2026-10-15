// Estimate normal optical flow by fitting a local plane to event timestamps.
#ifdef __cplusplus // Include the runtime only when compiling the CPU adapter.
#include "shdr.h" // Share the portable resource and math declarations.
#endif // Finish the C++ declaration include guard.
layout(local_size_x = 64) in; // Assign one decoded event to each flow-estimation invocation.
layout(std430, binding = 0) readonly buffer Events { // Access decoded records in their original event order.
    uvec4 events[]; // Use decoded pixel or trigger records with paired timestamp words.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 1) readonly buffer Offsets { // Read each block's exclusive output interval and the final total.
    uint offsets[]; // Read exclusive event offsets and the final total event count.
}; // Finish this shader storage resource declaration.
layout(std430, binding = 2) writeonly buffer Flow { // Write one normal-flow result per decoded event.
    vec4 flow[]; // Store normal velocity in pixels per second, neighbor count and validity.
}; // Finish this shader storage resource declaration.
uniform uint blocks; // Receive the number of summarized input blocks.
uniform float radius; // Receive the spatial neighborhood radius in pixels.
uniform uint history; // Receive the maximum preceding records to inspect, capped at 128.
uniform uint time_window; // Receive the maximum neighbor age in microseconds.

void main() { // Fit a local timestamp plane and publish its observable normal velocity.
    uint index = gl_GlobalInvocationID.x; // Select the decoded event whose flow is being estimated.
    if (index >= offsets[blocks]) // Ignore invocations beyond the decoder's final compact event count.
        return; // Stop dispatch padding at the compact event-count boundary.
    uvec4 center = events[index]; // Load the current event's pixel metadata and complete timestamp.
    flow[index] = vec4(0.0); // Default to an invalid zero result until a usable plane is fitted.
    // Fit spatial neighborhoods around decoded CD pixel events.
    if ((center.y & 2147483648u) != 0u) // Recognize a trigger record before selecting a pixel neighborhood.
        return; // Retain the initialized zero flow for trigger records.
    float sx = 0.0, sy = 0.0, st = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0, sxt = 0.0, syt = 0.0; // Initialize first and second moments for a centered least-squares plane.
    uint neighbors = 0u, lookback = min(history, 128u); // Count accepted neighbors and bound the per-event search cost.
    ivec2 reference_offset = ivec2(0); // Remember the first nonzero integer pixel direction from the center.
    bool spatial_rank_two = false; // Require two independent directions before fitting a two-dimensional plane.
    for (uint step = 1u; step <= lookback && step <= index; ++step) { // Inspect only available preceding events within the bounded history.
        uvec4 previous = events[index - step]; // Load one candidate neighbor from the decoded stream.
        if ((previous.y & 2147483648u) != 0u || (previous.y & 6144u) != (center.y & 6144u)) // Reject triggers and events with a different polarity or camera.
            continue; // Exclude this candidate while preserving the accumulated accepted neighbors.
        // Subtract the timestamp pair exactly, including a low-word borrow.
        // Only differences fitting uint32 and the requested window become float.
        if (previous.w > center.w) // Reject timestamps newer than the center in their high word.
            continue; // Exclude this candidate while preserving the accumulated accepted neighbors.
        uint high = center.w - previous.w, borrow = center.z < previous.z ? 1u : 0u; // Subtract high words and determine the low-word borrow exactly.
        if (high < borrow) // Reject a future timestamp whose low-word borrow would underflow.
            continue; // Exclude this candidate while preserving the accumulated accepted neighbors.
        high -= borrow; // Apply the borrow to the remaining high-word difference.
        uint elapsed = center.z - previous.z; // Compute the low-word age with the defined unsigned wrapping subtraction.
        if (high != 0u || elapsed > time_window) // Reject ages beyond uint32 or the requested temporal neighborhood.
            continue; // Exclude this candidate while preserving the accumulated accepted neighbors.
        ivec2 offset = ivec2(int(previous.x) - int(center.x), int(previous.y & 2047u) - int(center.y & 2047u)); // Subtract the decoded eleven-bit pixel coordinates exactly as signed integers.
        float dx = float(offset.x), dy = float(offset.y); // Convert the small integer offsets for the floating-point moment calculation.
        if (dx * dx + dy * dy > radius * radius) // Reject candidates outside the circular pixel neighborhood.
            continue; // Exclude this candidate while preserving the accumulated accepted neighbors.
        if (!spatial_rank_two) { // Stop testing directions once the neighborhood is known to span a plane.
            if (reference_offset.x == 0 && reference_offset.y == 0) { // Find the first nonzero pixel direction from the center.
                reference_offset = offset; // Keep this direction for comparison with later accepted neighbors.
            } else if (offset.x * reference_offset.y != offset.y * reference_offset.x) { // Test independence with exact integer products bounded by eleven-bit coordinates.
                spatial_rank_two = true; // Record exact two-dimensional spatial rank from the integer cross product.
            } // Finish the exact spatial direction test.
        } // Skip later direction tests once two independent offsets are found.
        float dt = -float(elapsed); // Express an accepted earlier timestamp relative to the center in microseconds.
        sx += dx; // Accumulate the horizontal first moment.
        sy += dy; // Accumulate the vertical first moment.
        st += dt; // Accumulate the relative timestamp first moment.
        sxx += dx * dx; // Accumulate the horizontal squared moment.
        sxy += dx * dy; // Accumulate the horizontal and vertical cross moment.
        syy += dy * dy; // Accumulate the vertical squared moment.
        sxt += dx * dt; // Accumulate the horizontal and timestamp cross moment.
        syt += dy * dt; // Accumulate the vertical and timestamp cross moment.
        ++neighbors; // Count this accepted spatial and temporal neighbor.
    } // Finish this decoder or fitting branch.
    if (neighbors < 6u || !spatial_rank_two) // Require six neighbors and exact two-dimensional spatial rank.
        return; // Return the initialized zero result for insufficient or collinear neighbors.
    // Include the center sample and eliminate the intercept with count-scaled moments.
    // Subtract count-scaled integer moments before the final slope divisions.
    // The common sample-count factors cancel in the slope solve.
    float n = float(neighbors + 1u); // Include the center event at zero offset and zero relative time.
    float xx = n * sxx - sx * sx, xy = n * sxy - sx * sy, yy = n * syy - sy * sy; // Form spatial covariances with the common sample-count scale.
    float xt = n * sxt - sx * st, yt = n * syt - sy * st; // Form matching count-scaled space-time covariances.
    float determinant = xx * yy - xy * xy; // Measure invertibility with both covariance factors scaled by the sample count.
    if (determinant <= 0.0001 * n * n) // Preserve the original determinant threshold under the squared count scaling.
        return; // Return the initialized zero result for degenerate spatial covariance.
    float a = (xt * yy - yt * xy) / determinant, b = (yt * xx - xt * xy) / determinant; // Solve the horizontal and vertical timestamp slopes in microseconds per pixel.
    float gradient = a * a + b * b; // Compute the squared timestamp-gradient magnitude.
    if (gradient <= 0.000001) // Reject a plane with too little temporal slope to define finite velocity.
        return; // Return the initialized zero result for a timestamp slope below the velocity threshold.
    // Timestamps are microseconds, publish pixels/second, neighbor count, validity.
    flow[index] = vec4(vec2(a, b) * (1000000.0 / gradient), float(neighbors), 1.0); // Convert the observable normal velocity to pixels per second and mark this fit valid.
} // Finish this decoder or fitting branch.
