// spatial_hash.h — FPGA k-NN Spatial Hash Implementation
//
// This module implements the O(n·k) k-nearest-neighbor graph construction
// using spatial grid hashing, replacing the O(n²) pairwise distance approach.
//
// Architecture:
//   Stage 1: Grid bucketing — each event is assigned to a spatial cell (BRAM hash table)
//   Stage 2: Per-event k-NN search — search 3×3 neighborhood of cells, top-k selection
//
// Latency: O(n·k) amortized, pipeline depth = 4 cycles per event
// Resource: ~12K LUTs + 48 BRAM blocks (4096 events, 32 neighbors)
//
// Fixed-point format: ap_fixed<16,4> for coordinates, normalized to [-8.0, 8.0)

#pragma once

#include "hls_compat.h"

// ---------------------------------------------------------------------------
// Configuration Constants (must match FPGAParams in models/params.py)
// ---------------------------------------------------------------------------
#define MAX_EVENTS      4096       // Ring buffer depth
#define K_NEIGHBORS     32         // k for k-NN graph
#define GRID_RADIUS     0.15f      // Spatial bin width (normalized coords)
#define MAX_CELL_EVENTS 128        // Maximum events per grid cell
#define MAX_CANDIDATES  (9 * MAX_CELL_EVENTS)  // 3×3 cells × max per cell

// Fixed-point types
typedef ap_fixed<16,4> coord_t;    // Coordinates: [-8.0, 8.0), Q4.12
typedef ap_fixed<32,8> dist_t;     // Squared distances: up to 256.0
typedef ap_uint<12> event_idx_t;   // Event index: 0..4095
typedef ap_uint<12> cell_idx_t;    // Cell index in hash table
typedef ap_int<16> cell_key_t;     // Signed cell grid key

// ---------------------------------------------------------------------------
// Event structure (same layout as ring_buffer.h)
// ---------------------------------------------------------------------------
struct event_packed_t {
    coord_t x;
    coord_t y;
    ap_uint<32> timestamp;
    ap_uint<1> polarity;
};

// ---------------------------------------------------------------------------
// k-NN adjacency output: for each event, list of k neighbor indices
// ---------------------------------------------------------------------------
struct knn_output_t {
    event_idx_t neighbor_indices[K_NEIGHBORS];
    union {
        ap_uint<6>  num_neighbors;           // Actual count (may be < K_NEIGHBORS)
        ap_uint<6>  count;                   // Alias for num_neighbors (testbench compat)
    };
};

// ---------------------------------------------------------------------------
// Grid Cell bucket (BRAM-stored linked list)
// ---------------------------------------------------------------------------
struct grid_cell_t {
    ap_uint<9> count;                    // Number of events in this cell (0..MAX_CELL_EVENTS)
    event_idx_t indices[MAX_CELL_EVENTS]; // Event indices stored in this cell
};

// ---------------------------------------------------------------------------
// Hash function: (cell_key_x, cell_key_y) → cell_idx_t
// Linear-probe hash with power-of-2 table size for BRAM efficiency
// ---------------------------------------------------------------------------
#define HASH_TABLE_SIZE 2048  // Must be power of 2, > #cells at 256×256 max

inline cell_idx_t cell_hash(cell_key_t cx, cell_key_t cy, ap_uint<12> probe) {
    #pragma HLS INLINE
    // Multiply-shift hash: distribute bits across table
    ap_uint<32> combined = (ap_uint<16>(cx) << 16) | ap_uint<16>(cy);
    combined = combined ^ (combined >> 11);
    combined = combined * 0x9E3779B9;  // Golden ratio
    combined = combined ^ (combined >> 16);
    return cell_idx_t((combined + probe) & (HASH_TABLE_SIZE - 1));
}

// ---------------------------------------------------------------------------
// Top-level k-NN adjacency computation
//
// Arguments:
//   events          : Input event array (read from ring buffer BRAM)
//   num_events      : Number of valid events (1..MAX_EVENTS)
//   knn_results     : Output array of k-NN neighbor indices
//
// Interface: AXI4-Stream for events input, AXI4-Lite for control, BRAM for output
// ---------------------------------------------------------------------------
void spatial_hash_knn(
    event_packed_t events[MAX_EVENTS],
    ap_uint<12>    num_events,
    knn_output_t   knn_results[MAX_EVENTS]
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=num_events bundle=CTRL
    #pragma HLS INTERFACE bram    port=events bundle=MEM_EVENTS
    #pragma HLS INTERFACE bram    port=knn_results bundle=MEM_KNN

    // -----------------------------------------------------------------------
    // Stage 1: Grid Bucketing — O(n) hash inserts
    // Pipeline: II=1, latency = num_events + HASH_TABLE_SIZE cycles
    // -----------------------------------------------------------------------
    grid_cell_t grid[HASH_TABLE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=grid cyclic factor=8 dim=1
    #pragma HLS BIND_STORAGE variable=grid type=RAM_T2P impl=BRAM

    // Initialize grid
    INIT_GRID:
    for (ap_uint<12> i = 0; i < HASH_TABLE_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        grid[i].count = 0;
    }

    // Bucket events into grid
    BUCKET_EVENTS:
    for (ap_uint<12> i = 0; i < num_events; i++) {
        #pragma HLS PIPELINE II=1
        cell_key_t cx = cell_key_t(events[i].x / coord_t(GRID_RADIUS));
        cell_key_t cy = cell_key_t(events[i].y / coord_t(GRID_RADIUS));

        // Linear-probe hash lookup
        ap_uint<12> probe = 0;
        cell_idx_t cell;
        HASH_LOOKUP:
        do {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            cell = cell_hash(cx, cy, probe);
            probe++;
        } while (grid[cell].count > 0 && probe < 16);

        // Store event index in cell bucket
        if (probe < 16 && grid[cell].count < MAX_CELL_EVENTS) {
            ap_uint<9> pos = grid[cell].count;
            grid[cell].indices[pos] = i;
            grid[cell].count++;
        }
    }

    // -----------------------------------------------------------------------
    // Stage 2: Per-event k-NN Search — O(n·k) amortized
    // Pipeline: II=1, latency = num_events × (9 × MAX_CELL_EVENTS + k²)
    // -----------------------------------------------------------------------
    KNN_LOOP:
    for (ap_uint<12> i = 0; i < num_events; i++) {
        #pragma HLS PIPELINE II=1
        coord_t ex = events[i].x;
        coord_t ey = events[i].y;
        cell_key_t cx = cell_key_t(ex / coord_t(GRID_RADIUS));
        cell_key_t cy = cell_key_t(ey / coord_t(GRID_RADIUS));

        // Collect candidates from 3×3 neighborhood
        event_idx_t candidate_indices[MAX_CANDIDATES];
        dist_t      candidate_dists[MAX_CANDIDATES];
        ap_uint<10> num_candidates = 0;

        // Search 9 neighboring cells
        SEARCH_CELLS:
        for (ap_int<2> dx = -1; dx <= 1; dx++) {
            for (ap_int<2> dy = -1; dy <= 1; dy++) {
                #pragma HLS LOOP_FLATTEN
                cell_key_t scx = cx + dx;
                cell_key_t scy = cy + dy;

                // Linear-probe lookup for search cell
                ap_uint<12> probe = 0;
                cell_idx_t cell;
                SEARCH_HASH:
                do {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    cell = cell_hash(scx, scy, probe);
                    probe++;
                } while (grid[cell].count > 0 && probe < 16);

                // Iterate events in this cell
                if (probe < 16) {
                    ap_uint<9> cell_count = grid[cell].count;
                    if (cell_count > MAX_CELL_EVENTS) cell_count = MAX_CELL_EVENTS;

                    SEARCH_CELL_EVENTS:
                    for (ap_uint<9> jj = 0; jj < cell_count; jj++) {
                        #pragma HLS LOOP_TRIPCOUNT min=0 max=MAX_CELL_EVENTS
                        event_idx_t j = grid[cell].indices[jj];
                        if (j == i) continue;  // Skip self

                        // Squared Euclidean distance (LUT-friendly, no sqrt)
                        dist_t dx_val = dist_t(ex - events[j].x);
                        dist_t dy_val = dist_t(ey - events[j].y);
                        dist_t dist_sq = dx_val * dx_val + dy_val * dy_val;

                        // Within radius check
                        dist_t r_sq = dist_t(coord_t(GRID_RADIUS * GRID_RADIUS));
                        if (dist_sq < r_sq && num_candidates < MAX_CANDIDATES) {
                            candidate_indices[num_candidates] = j;
                            candidate_dists[num_candidates] = dist_sq;
                            num_candidates++;
                        }
                    }
                }
            }
        }

        // Top-k selection via bubble-sort (k=32, small; maps to sorting network)
        // For larger k, replace with bitonic sorting network
        SELECT_TOP_K:
        for (ap_uint<6> ki = 0; ki < K_NEIGHBORS; ki++) {
            #pragma HLS LOOP_TRIPCOUNT min=32 max=32
            dist_t best_dist = dist_t(999999.0);
            event_idx_t best_idx = 0;
            ap_uint<10> best_pos = 0;

            SELECT_SCAN:
            for (ap_uint<10> ci = 0; ci < num_candidates; ci++) {
                #pragma HLS LOOP_TRIPCOUNT min=0 max=MAX_CANDIDATES
                #pragma HLS PIPELINE II=1
                if (candidate_dists[ci] < best_dist) {
                    best_dist = candidate_dists[ci];
                    best_idx = candidate_indices[ci];
                    best_pos = ci;
                }
            }

            if (num_candidates > 0) {
                knn_results[i].neighbor_indices[ki] = best_idx;
                candidate_dists[best_pos] = dist_t(999999.0);  // Mark as used
            } else {
                knn_results[i].neighbor_indices[ki] = i;  // Self-fallback
            }
        }
        knn_results[i].num_neighbors = (num_candidates > 0) ? 
            ap_uint<6>(hls::min(num_candidates, ap_uint<10>(K_NEIGHBORS))) : ap_uint<6>(0);
    }
}