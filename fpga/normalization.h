// normalization.h — Fixed-Point Coordinate Normalization
//
// Converts raw event coordinates to the [0, 1) range used by the
// LocalGeometryEncoder. Uses LUT-based dividers for pxl_radius and
// t_radius scaling factors.
//
// Latency: 4 pipeline stages
// Resource: ~2K LUTs (LUT-based division)

#pragma once

#include <ap_fixed.h>
#include <hls_math.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
// These match FPGAParams in models/params.py
#define PXL_RADIUS 0.0225f   // Spatial scaling factor
#define T_RADIUS   0.01f     // Temporal scaling factor

// Fixed-point types
typedef ap_fixed<16,4>  norm_coord_t;   // Normalized coordinate [0, 8.0)
typedef ap_ufixed<16,4> raw_coord_t;    // Raw coordinate [0, 1.0)
typedef ap_fixed<24,8>  time_norm_t;    // Normalized time

// ---------------------------------------------------------------------------
// Coordinate normalization module
//
// Converts: raw_x → x / pxl_radius, raw_t → t / t_radius
//
// Pipeline stages:
//   1. Input register
//   2. Reciprocal lookup (LUT-based division)
//   3. Multiply
//   4. Output register
// ---------------------------------------------------------------------------
struct norm_input_t {
    raw_coord_t raw_x;
    raw_coord_t raw_y;
    ap_uint<32> timestamp;
    ap_uint<1>  polarity;
};

struct norm_output_t {
    norm_coord_t x;
    norm_coord_t y;
    time_norm_t  t;
    ap_uint<1>   polarity;
};

// Pre-computed reciprocal constants (synthesized as constants, not dividers)
// On FPGA, division by a constant becomes a multiply by reciprocal
static const norm_coord_t INV_PXL_RADIUS = norm_coord_t(1.0f / PXL_RADIUS);  // ≈ 44.44
static const time_norm_t  INV_T_RADIUS   = time_norm_t(1.0f / T_RADIUS);     // 100.0

void normalization(
    norm_input_t  event_in,
    norm_output_t& event_out
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE ap_none  port=event_in
    #pragma HLS INTERFACE ap_none  port=event_out
    #pragma HLS PIPELINE II=1

    // Stage 1: Scale normalized x, y coordinates
    event_out.x = norm_coord_t(event_in.raw_x) * INV_PXL_RADIUS;
    event_out.y = norm_coord_t(event_in.raw_y) * INV_PXL_RADIUS;

    // Stage 2: Normalize timestamp
    event_out.t = time_norm_t(event_in.timestamp) * INV_T_RADIUS;

    // Pass polarity through
    event_out.polarity = event_in.polarity;
}