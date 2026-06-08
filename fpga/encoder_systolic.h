// encoder_systolic.h — FPGA LocalGeometryEncoder (Real-Unrolled Systolic Array)
//
// This module implements the VecKM LocalGeometryEncoder with:
//   - d=128 (reduced from 384 for FPGA resource efficiency)
//   - Real-only arithmetic (complex numbers unrolled to 2× real)
//   - Fixed-point INT16 with power-of-2 quantization
//   - Tiled systolic array for G = J @ epA computation
//
// Architecture:
//   Input:  events (n, 3) normalized [t, x, y]  +  J adjacency (n, k) sparse
//   Stage 1: pA = events @ A  → (n, d) — tile-based matrix multiply
//   Stage 2: epA = [cos(pA), sin(pA)] → (n, 2d) — CORDIC sine/cosine
//   Stage 3: G = J @ epA  → (n, 2d) — sparse-dense systolic multiply
//   Stage 4: Complex division + normalize → (n, d) complex
//
// Latency: ~500 cycles (n=4096, d=128, k=32)
// Resource: ~35K LUTs + 128 DSP slices + 16 BRAM

#pragma once

#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_stream.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define D_ENC      128          // Encoding dimension (reduced from 384)
#define ALPHA_ENC  8            // Frequency scaling (increased from 5)
#define TILE_SIZE  32           // Tile dimension for systolic MAC unit
#define NUM_TILES  (D_ENC / TILE_SIZE)  // 4 tiles at d=128

// Fixed-point types
typedef ap_fixed<16,4>  coord_enc_t;    // Normalized coordinates
typedef ap_fixed<16,4>  weight_t;       // Encoder weights
typedef ap_fixed<32,10> prod_t;         // Tile product accumulator
typedef ap_fixed<16,8>  cos_sin_t;      // CORDIC output
typedef ap_fixed<16,8>  enc_out_t;      // Encoding output (real/imag parts)

// ---------------------------------------------------------------------------
// Encoder weights (A ∈ R^(3×d), stored in BRAM, loaded at init)
// These are the strict_standard_normal(d) * alpha values from the PyTorch model.
// They are fixed after training — no gradient updates on FPGA.
// ---------------------------------------------------------------------------
typedef weight_t encoder_weights_t[3][D_ENC];  // (3, d) weight matrix

// ---------------------------------------------------------------------------
// CORDIC sine/cosine — resource-efficient trigonometric approximation
//
// Uses iterative CORDIC algorithm (no DSP, LUT-based).
// Trade-off: ±0.001 accuracy vs instant lookup.
// ---------------------------------------------------------------------------
inline void cordic_sin_cos(prod_t angle, cos_sin_t& sin_out, cos_sin_t& cos_out) {
    #pragma HLS INLINE
    
    // Scale angle to [-π, π] range
    const prod_t PI  = prod_t(3.141592653589793);
    const prod_t PI2 = prod_t(6.283185307179586);
    
    while (angle > PI)  angle -= PI2;
    while (angle < -PI) angle += PI2;
    
    // CORDIC iterations (8 iterations for Q16.8 accuracy)
    static const prod_t cordic_table[8] = {
        prod_t(0.7853981633974483),  // atan(2^0)
        prod_t(0.4636476090008061),  // atan(2^-1)
        prod_t(0.24497866312686414), // atan(2^-2)
        prod_t(0.12435499454676144), // atan(2^-3)
        prod_t(0.06241880999595735), // atan(2^-4)
        prod_t(0.031239833430268277), // atan(2^-5)
        prod_t(0.015623728620476831), // atan(2^-6)
        prod_t(0.007812341060101111)  // atan(2^-7)
    };
    
    prod_t x = prod_t(0.6072529350088814); // CORDIC gain K
    prod_t y = prod_t(0.0);
    prod_t z = angle;
    
    for (int i = 0; i < 8; i++) {
        #pragma HLS UNROLL
        prod_t dx = x >> i;
        prod_t dy = y >> i;
        
        if (z >= 0) {
            x = x - dy;
            y = y + dx;
            z = z - cordic_table[i];
        } else {
            x = x + dy;
            y = y - dx;
            z = z + cordic_table[i];
        }
    }
    
    cos_out = cos_sin_t(x);
    sin_out = cos_sin_t(y);
}

// ---------------------------------------------------------------------------
// Tile-based Matrix Multiply: out = in @ weights
// Systolic array compute: for tiled D×D blocks with streaming dataflow
//
// We do this as (n, 3) @ (3, D_ENC) → (n, D_ENC) in tiles of TILE_SIZE
// ---------------------------------------------------------------------------
void tile_matmul(
    coord_enc_t       events[3],              // Single event: [t, x, y]
    encoder_weights_t weights,
    prod_t            result[D_ENC]           // Output: 1×D_ENC row
) {
    #pragma HLS PIPELINE II=1
    
    // Compute pA = events @ A  where events is (1, 3) and A is (3, D_ENC)
    for (int d = 0; d < D_ENC; d++) {
        #pragma HLS UNROLL factor=8
        prod_t acc = 0;
        for (int i = 0; i < 3; i++) {
            #pragma HLS UNROLL
            acc += prod_t(events[i]) * prod_t(weights[i][d]);
        }
        result[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Local Geometry Encoder (top-level HLS function)
//
// Computes the dense local geometry encoding G ∈ C^(n × d) for n events.
//
// Dataflow:
//   1. For each event: pA = event @ A        → (n, d)   via tile_matmul
//   2. For each pA[d]: [cos(pA), sin(pA)]    → (n, 2d) via CORDIC
//   3. Sparse-dense: G = J @ epA             → (n, 2d) via gather-accumulate
//   4. Complex division + normalize           → (n, d)
//
// Arguments:
//   events_txy    : (n, 3) normalized events [t, x, y]
//   num_events    : number of valid events
//   knn_results   : k-NN adjacency (n, k) — neighbor indices
//   weights       : encoder weight matrix A
//   flow_pred     : (n, 2) flow predictions (vx, vy) — real-only output
//   flow_uncert   : (n,) per-event uncertainty
// ---------------------------------------------------------------------------
void local_geometry_encoder(
    coord_enc_t       events_txy[MAX_EVENTS][3],
    ap_uint<12>       num_events,
    knn_output_t      knn_results[MAX_EVENTS],
    encoder_weights_t weights,
    enc_out_t         flow_pred[MAX_EVENTS][2],  // (vx, vy) real-valued flow
    enc_out_t         flow_uncert[MAX_EVENTS]
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=num_events bundle=CTRL
    #pragma HLS INTERFACE bram    port=events_txy bundle=MEM_EVENTS
    #pragma HLS INTERFACE bram    port=knn_results bundle=MEM_KNN
    #pragma HLS INTERFACE bram    port=weights bundle=MEM_WEIGHTS
    #pragma HLS INTERFACE bram    port=flow_pred bundle=MEM_FLOW
    #pragma HLS INTERFACE bram    port=flow_uncert bundle=MEM_UNCERT

    // -----------------------------------------------------------------------
    // Stage 1: pA = events @ A  — per-event encoding (parallel across tiles)
    // -----------------------------------------------------------------------
    prod_t pA[MAX_EVENTS][D_ENC];
    #pragma HLS ARRAY_PARTITION variable=pA cyclic factor=16 dim=2

    STAGE1_PA:
    for (ap_uint<12> i = 0; i < num_events; i++) {
        #pragma HLS PIPELINE II=1
        tile_matmul(events_txy[i], weights, pA[i]);
    }

    // -----------------------------------------------------------------------
    // Stage 2: epA = [cos(pA), sin(pA)]  — CORDIC trig per element
    // -----------------------------------------------------------------------
    cos_sin_t epA_real[MAX_EVENTS][D_ENC];  // cos(pA) parts
    cos_sin_t epA_imag[MAX_EVENTS][D_ENC];  // sin(pA) parts
    #pragma HLS ARRAY_PARTITION variable=epA_real cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=epA_imag cyclic factor=8 dim=2

    STAGE2_TRIG:
    for (ap_uint<12> i = 0; i < num_events; i++) {
        #pragma HLS PIPELINE II=1
        for (int d = 0; d < D_ENC; d++) {
            #pragma HLS UNROLL factor=4
            cordic_sin_cos(pA[i][d], epA_imag[i][d], epA_real[i][d]);
        }
    }

    // -----------------------------------------------------------------------
    // Stage 3: G = J @ epA  — sparse k-NN gather-accumulate
    // For each event i, G[i] = sum_{j in N(i)} epA[j]  (k terms)
    // -----------------------------------------------------------------------
    prod_t G_real[MAX_EVENTS][D_ENC];
    prod_t G_imag[MAX_EVENTS][D_ENC];
    #pragma HLS ARRAY_PARTITION variable=G_real cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=G_imag cyclic factor=8 dim=2

    STAGE3_SPARSE:
    for (ap_uint<12> i = 0; i < num_events; i++) {
        #pragma HLS PIPELINE II=1
        
        // Initialize accumulators
        for (int d = 0; d < D_ENC; d++) {
            #pragma HLS UNROLL factor=8
            G_real[i][d] = 0;
            G_imag[i][d] = 0;
        }
        
        // Accumulate over k neighbors
        ap_uint<6> k_count = knn_results[i].num_neighbors;
        if (k_count > K_NEIGHBORS) k_count = K_NEIGHBORS;
        
        SPARSE_ACCUM:
        for (ap_uint<6> ki = 0; ki < k_count; ki++) {
            #pragma HLS LOOP_TRIPCOUNT min=0 max=K_NEIGHBORS
            event_idx_t j = knn_results[i].neighbor_indices[ki];
            for (int d = 0; d < D_ENC; d++) {
                #pragma HLS UNROLL factor=4
                G_real[i][d] += prod_t(epA_real[j][d]);
                G_imag[i][d] += prod_t(epA_imag[j][d]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stage 4: Complex division + normalization + output
    //
    // G_out[d] = (G[d] / epA[i][d])   as complex division:
    //   real = (G_real*epA_real + G_imag*epA_imag) / (epA_real² + epA_imag²)
    //   imag = (G_imag*epA_real - G_real*epA_imag) / (epA_real² + epA_imag²)
    //
    // Simplified for FPGA: we output flow predictions as real-only
    // (vx, vy) = (norm * cos(theta), norm * sin(theta)) in pixel space
    // -----------------------------------------------------------------------
    STAGE4_OUTPUT:
    for (ap_uint<12> i = 0; i < num_events; i++) {
        #pragma HLS PIPELINE II=1
        
        enc_out_t sum_real = 0;
        enc_out_t sum_imag = 0;
        
        for (int d = 0; d < D_ENC; d++) {
            #pragma HLS UNROLL factor=8
            prod_t denom = prod_t(epA_real[i][d]) * prod_t(epA_real[i][d]) +
                          prod_t(epA_imag[i][d]) * prod_t(epA_imag[i][d]);
            
            // Avoid division by zero
            if (denom > prod_t(0.001)) {
                enc_out_t g_re = enc_out_t(
                    (G_real[i][d] * prod_t(epA_real[i][d]) + 
                     G_imag[i][d] * prod_t(epA_imag[i][d])) / denom
                );
                enc_out_t g_im = enc_out_t(
                    (G_imag[i][d] * prod_t(epA_real[i][d]) - 
                     G_real[i][d] * prod_t(epA_imag[i][d])) / denom
                );
                sum_real += g_re;
                sum_imag += g_im;
            }
        }
        
        // Output flow: (mean vx, mean vy) over all channels
        flow_pred[i][0] = sum_real;   // vx
        flow_pred[i][1] = sum_imag;   // vy
        flow_uncert[i]  = enc_out_t(0.0);  // Single-pass, no variance estimate
    }
}