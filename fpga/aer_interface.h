// aer_interface.h — Address-Event Representation (AER) Camera Interface
//
// Receives event data from event-based vision sensors (Prophesee/Inivation)
// via parallel AER bus or SPI. Converts raw AER packets to normalized
// event structures for the inference pipeline.
//
// Supported cameras:
//   - Prophesee Gen3/Gen4 (parallel AER + sync)
//   - Inivation DAVIS/ATIS (parallel AER + IMU)
//   - CeleX-V (parallel, configurable resolution)
//
// Interface options:
//   1. Parallel AER:  address bus + request/acknowledge handshake
//   2. SPI slave:     4-wire SPI with event packet framing
//   3. AXI4-Stream:   direct FPGA fabric connection for chip-on-board
//
// Latency: 1 cycle per event (parallel), 4 cycles (SPI)
// Resource: ~1K LUTs (parallel), ~2K LUTs (SPI)

#pragma once

#include "hls_compat.h"

// ---------------------------------------------------------------------------
// Camera configuration
// ---------------------------------------------------------------------------
#define SENSOR_WIDTH 640   // Pixels
#define SENSOR_HEIGHT 480  // Pixels
#define SENSOR_MAX_ADDR (SENSOR_WIDTH * SENSOR_HEIGHT)

// AER bus interface signals (parallel mode)
struct aer_bus_t {
    ap_uint<19> address;  // Pixel address (0..640*480-1)
    ap_uint<10> x;        // Pixel x (0..639) — convenience alias
    ap_uint<9> y;         // Pixel y (0..479) — convenience alias
    ap_uint<1> pol;       // Polarity (1=ON, 0=OFF)
    ap_uint<1> polarity;  // ON (1) or OFF (0) event (alias for pol)
    ap_uint<1> req;       // Request strobe (camera → FPGA)
    ap_uint<1> ack;       // Acknowledge strobe (FPGA → camera)
};

// Normalized event output (same as ring_buffer.h event_unpacked_t)
struct aer_event_out_t {
    ap_ufixed<16, 4> x;     // Normalized x [0.0, 1.0)
    ap_ufixed<16, 4> y;     // Normalized y [0.0, 1.0)
    ap_uint<32> timestamp;  // Microsecond timestamp
    ap_uint<1> polarity;    // Event polarity
};

// ---------------------------------------------------------------------------
// AER Parallel Interface (4-phase handshake)
//
// Implements the standard AER handshake protocol:
//   1. Camera asserts REQ + sets ADDR + POL
//   2. FPGA captures data, asserts ACK
//   3. Camera deasserts REQ
//   4. FPGA deasserts ACK
//
// Events stream out on AXI4-Stream for the ring buffer.
// ---------------------------------------------------------------------------
void aer_parallel_interface(aer_bus_t& aer_bus,
                            ap_uint<64> aer_timestamp,  // System timestamp (cycles or μs)
                            hls::stream<aer_event_out_t>& event_stream) {
#pragma HLS INTERFACE s_axilite port = return bundle = CTRL
#pragma HLS INTERFACE s_axilite port = aer_timestamp bundle = CTRL
#pragma HLS INTERFACE ap_none port = aer_bus.address
#pragma HLS INTERFACE ap_none port = aer_bus.polarity
#pragma HLS INTERFACE ap_none port = aer_bus.req
#pragma HLS INTERFACE ap_none port = aer_bus.ack
#pragma HLS INTERFACE axis port = event_stream
#pragma HLS PIPELINE II = 1

    // 4-phase handshake state machine — static: the state register must
    // persist across calls (an automatic variable here is uninitialized
    // garbage every invocation)
    static enum { IDLE, CAPTURE, WAIT_REQ_LOW, WAIT_ACK_LOW } state = IDLE;
#pragma HLS RESET variable = state

    static aer_event_out_t captured_event;
#pragma HLS RESET variable = captured_event

    switch (state) {
        case IDLE:
            if (aer_bus.req == 1) {
                // Phase 1: Capture event data
                ap_uint<19> addr = aer_bus.address;
                captured_event.x = ap_ufixed<16, 4>(ap_ufixed<20, 10>(addr % SENSOR_WIDTH) /
                                                    ap_ufixed<20, 10>(SENSOR_WIDTH));
                captured_event.y = ap_ufixed<16, 4>(ap_ufixed<20, 10>(addr / SENSOR_WIDTH) /
                                                    ap_ufixed<20, 10>(SENSOR_HEIGHT));
                captured_event.polarity = aer_bus.polarity;
                captured_event.timestamp = ap_uint<32>(aer_timestamp & 0xFFFFFFFF);

                aer_bus.ack = 1;
                state = CAPTURE;
            } else {
                aer_bus.ack = 0;
            }
            break;

        case CAPTURE:
            // Phase 2: Output event on stream
            event_stream.write(captured_event);
            state = WAIT_REQ_LOW;
            break;

        case WAIT_REQ_LOW:
            // Phase 3: Wait for camera to deassert REQ
            if (aer_bus.req == 0) {
                aer_bus.ack = 0;
                state = WAIT_ACK_LOW;
            }
            break;

        case WAIT_ACK_LOW:
            // Phase 4: Complete, return to IDLE
            state = IDLE;
            break;
    }
}

// ---------------------------------------------------------------------------
// SPI Slave Event Interface
//
// Receives events over 4-wire SPI:
//   - SCLK: clock from camera (up to 50MHz)
//   - MOSI: event data from camera
//   - SS:   frame sync (event start/end)
//   - MISO: unused (or used for configuration readback)
//
// Event frame format (48 bits):
//   [47:32] timestamp[31:16]
//   [31]    timestamp[15]
//   [30:17] y[13:0]  — pixel y coordinate
//   [16:3]  x[13:0]  — pixel x coordinate
//   [2]     polarity  — ON/OFF
//   [1:0]   reserved
// ---------------------------------------------------------------------------
void aer_spi_interface(ap_uint<1> spi_sclk, ap_uint<1> spi_mosi, ap_uint<1> spi_ss,
                       ap_uint<64> aer_timestamp, hls::stream<aer_event_out_t>& event_stream) {
#pragma HLS INTERFACE s_axilite port = return bundle = CTRL
#pragma HLS INTERFACE s_axilite port = aer_timestamp bundle = CTRL
#pragma HLS INTERFACE ap_none port = spi_sclk
#pragma HLS INTERFACE ap_none port = spi_mosi
#pragma HLS INTERFACE ap_none port = spi_ss
#pragma HLS INTERFACE axis port = event_stream

    static ap_uint<48> shift_reg = 0;
    static ap_uint<6> bit_count = 0;
    static ap_int<1> sclk_prev = 0;
#pragma HLS RESET variable = shift_reg
#pragma HLS RESET variable = bit_count
#pragma HLS RESET variable = sclk_prev

    // Detect rising edge of SS (start of frame)
    if (spi_ss == 1) {
        bit_count = 0;
        shift_reg = 0;
    }

    // Detect rising edge of SCLK → sample MOSI
    if (spi_sclk == 1 && sclk_prev == 0 && spi_ss == 0) {
        // Shift in one bit (MSB first)
        shift_reg = (shift_reg << 1) | ap_uint<48>(spi_mosi);
        bit_count++;
    }
    sclk_prev = spi_sclk;

    // Complete event frame (48 bits received)
    if (bit_count >= 48) {
        // Unpack 48-bit frame
        ap_uint<14> x_raw = shift_reg(16, 3);   // bits [16:3]
        ap_uint<14> y_raw = shift_reg(30, 17);  // bits [30:17]
        ap_uint<1> pol = shift_reg(2, 2);       // bit  [2]

        aer_event_out_t ev;
        ev.x = ap_ufixed<16, 4>(ap_ufixed<20, 10>(x_raw) / ap_ufixed<20, 10>(SENSOR_WIDTH));
        ev.y = ap_ufixed<16, 4>(ap_ufixed<20, 10>(y_raw) / ap_ufixed<20, 10>(SENSOR_HEIGHT));
        ev.polarity = pol;
        ev.timestamp = ap_uint<32>(aer_timestamp & 0xFFFFFFFF);

        event_stream.write(ev);
        bit_count = 0;  // Reset for next event
    }
}