// ring_buffer.h — FPGA Event Ring Buffer
//
// Continuous-write ring buffer for event camera AER events.
// Maps directly to dual-port BRAM on FPGA fabric.
//
// Interface:
//   Write: AER interface pushes events continuously (one per clock)
//   Read:  Burst-read all valid events for inference pipeline input
//
// Latency: 1 cycle write, 1 cycle per read burst
// Resource: 2 BRAM blocks (4096 × 32bit, dual-port)

#pragma once

#include "hls_compat.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define RING_BUFFER_SIZE 4096  // Power of 2 for efficient masking
#define RB_ADDR_MASK     (RING_BUFFER_SIZE - 1)

// ---------------------------------------------------------------------------
// Event type (packed: 48 bits total)
// Layout: [47:32] timestamp_hi | [31] timestamp_lo | [30:17] y (Q4.10) |
//         [16:3] x (Q4.10) | [2] polarity | [1:0] reserved
// ---------------------------------------------------------------------------
typedef ap_uint<48> event_raw_t;

// Unpacked event (for internal use)
struct event_unpacked_t {
    ap_ufixed<16,4> x;          // Normalized x coordinate [0, 1)
    ap_ufixed<16,4> y;          // Normalized y coordinate [0, 1)
    ap_uint<32>     timestamp;  // Microsecond timestamp
    ap_uint<1>      polarity;
};

// ---------------------------------------------------------------------------
// Ring Buffer Module
//
// Dual-port BRAM:
//   Port A: Write-only from AER interface (continuous stream)
//   Port B: Read-only for inference pipeline (burst)
//
// Arguments:
//   aer_valid       : AER data valid strobe (in)
//   aer_data        : Raw event data (in)
//   trigger_read    : Start burst read (in, pulse)
//   event_count     : Number of valid events in buffer (out)
//   read_data       : Burst-read event output (out, stream)
//   read_valid      : Read data valid (out, stream)
//   read_last       : Last event in burst (out, stream)
// ---------------------------------------------------------------------------
void ring_buffer(
    ap_uint<1>     aer_valid,
    event_raw_t    aer_data,
    ap_uint<1>     trigger_read,
    ap_uint<12>    event_count,
    hls::stream<event_unpacked_t>& read_data,
    hls::stream<ap_uint<1>>&      read_valid,
    hls::stream<ap_uint<1>>&      read_last
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=event_count bundle=CTRL
    #pragma HLS INTERFACE ap_none  port=aer_valid
    #pragma HLS INTERFACE ap_none  port=aer_data
    #pragma HLS INTERFACE ap_none  port=trigger_read
    #pragma HLS INTERFACE axis     port=read_data
    #pragma HLS INTERFACE axis     port=read_valid
    #pragma HLS INTERFACE axis     port=read_last

    // Dual-port BRAM storage
    static event_unpacked_t buffer[RING_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=buffer type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=4 dim=1

    static ap_uint<12> write_ptr = 0;
    #pragma HLS RESET variable=write_ptr

    // -----------------------------------------------------------------------
    // Write port (continuous AER event ingestion)
    // -----------------------------------------------------------------------
    if (aer_valid) {
        // Unpack raw event
        event_unpacked_t ev;
        ev.x        = ap_ufixed<16,4>(aer_data(13, 0))  / ap_ufixed<16,4>(16384.0);
        ev.y        = ap_ufixed<16,4>(aer_data(29, 16)) / ap_ufixed<16,4>(16384.0);
        ev.timestamp = ap_uint<32>(aer_data(47, 16));
        ev.polarity  = aer_data(2, 2);

        buffer[write_ptr & RB_ADDR_MASK] = ev;
        write_ptr++;
    }

    // -----------------------------------------------------------------------
    // Read port (burst-read for inference pipeline)
    // -----------------------------------------------------------------------
    if (trigger_read) {
        ap_uint<12> count = event_count;
        if (count > RING_BUFFER_SIZE) count = RING_BUFFER_SIZE;

        // Determine oldest event index (ring buffer semantics)
        ap_uint<12> read_ptr;
        if (write_ptr >= count) {
            read_ptr = write_ptr - count;
        } else {
            read_ptr = write_ptr + RING_BUFFER_SIZE - count;
        }

        BURST_READ:
        for (ap_uint<12> i = 0; i < count; i++) {
            #pragma HLS PIPELINE II=1
            ap_uint<12> addr = (read_ptr + i) & RB_ADDR_MASK;
            read_data.write(buffer[addr]);
            read_valid.write(1);
            read_last.write((i == count - 1) ? ap_uint<1>(1) : ap_uint<1>(0));
        }
    }
}