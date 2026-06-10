// top_level.cpp — FPGA Top-Level Collision Avoidance Pipeline
//
// Integrates all HLS modules into a single AXI-interfaced IP block:
//   1. AER camera interface → event stream
//   2. Ring buffer → burst-read window of N events
//   3. Normalization → fixed-point coordinate scaling
//   4. k-NN spatial hash → O(n·k) adjacency graph
//   5. LocalGeometryEncoder → per-event normal flow (vx, vy)
//   6. PWM output → 4 motor channels from velocity commands
//
// Target: Zynq UltraScale+ MPSoC (XCZU9EG) / Kria K26 SOM
// Clock: 100MHz PL fabric
// Interface: AXI4-Lite for ARM control, AXI4-Stream for data

#include "hls_compat.h"
#include "ring_buffer.h"
#include "normalization.h"
#include "spatial_hash.h"
#include "encoder_systolic.h"
#include "pwm_output.h"
#include "aer_interface.h"

// ---------------------------------------------------------------------------
// AXI4-Stream data type for event flow (synthesis only)
// ---------------------------------------------------------------------------
#ifndef __SIMULATION__
typedef ap_axiu<48, 0, 0, 0> axis_event_t;
#endif

// ---------------------------------------------------------------------------
// Top-Level Control Registers (AXI4-Lite addressable)
// ---------------------------------------------------------------------------
struct control_regs_t {
    ap_uint<1>   enable;           // Global enable (0=idle, 1=running)
    ap_uint<1>   enable_motors;    // Motor arming (0=disarmed, 1=armed)
    ap_uint<32>  inference_period; // Cycles between inference runs (e.g., 100000 for 1kHz)
    velocity_t   manual_vx;        // Manual override velocity (for testing)
    velocity_t   manual_vy;
    velocity_t   manual_vz;
    velocity_t   manual_yaw;
    ap_uint<1>   manual_mode;      // 1=manual control, 0=autonomous evasion
};

// ---------------------------------------------------------------------------
// Top-Level Function
//
// Arguments:
//   ctrl_regs       : AXI4-Lite control register bank
//   aer_bus         : AER parallel camera bus
//   aer_timestamp   : System timestamp for event tagging
//   motor_out       : 4-channel PWM output struct
//   debug_flow      : Debug output: first event's (vx, vy) for monitoring
// ---------------------------------------------------------------------------
void collision_avoidance_top(
    control_regs_t&    ctrl_regs,
    aer_bus_t&         aer_bus,
    ap_uint<64>        aer_timestamp,
    motor_outputs_t&   motor_out,
    enc_out_t          debug_flow[2]           // (vx, vy) of first event for debug
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=ctrl_regs bundle=CTRL
    #pragma HLS INTERFACE ap_none   port=aer_timestamp
    #pragma HLS INTERFACE ap_none   port=motor_out
    #pragma HLS INTERFACE s_axilite port=debug_flow bundle=DEBUG

    // -------------------------------------------------------------------
    // Internal state
    // -------------------------------------------------------------------
    static hls::stream<aer_event_out_t> event_fifo("event_fifo");
    #pragma HLS STREAM variable=event_fifo depth=16

    static event_unpacked_t ring_buffer_events[RING_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=ring_buffer_events type=RAM_T2P impl=BRAM
    static ap_uint<12> rb_write_ptr = 0;
    static ap_uint<12> rb_count = 0;
    #pragma HLS RESET variable=rb_write_ptr
    #pragma HLS RESET variable=rb_count

    static event_packed_t spatial_hash_events[MAX_EVENTS];
    #pragma HLS BIND_STORAGE variable=spatial_hash_events type=RAM_T2P impl=BRAM
    static knn_output_t knn_results[MAX_EVENTS];
    #pragma HLS BIND_STORAGE variable=knn_results type=RAM_T2P impl=BRAM

    static coord_enc_t enc_events[MAX_EVENTS][3];
    #pragma HLS BIND_STORAGE variable=enc_events type=RAM_T2P impl=BRAM
    static enc_out_t flow_pred[MAX_EVENTS][2];
    #pragma HLS BIND_STORAGE variable=flow_pred type=RAM_T2P impl=BRAM
    static enc_out_t flow_uncert[MAX_EVENTS];

    static encoder_weights_t enc_weights;
    #pragma HLS BIND_STORAGE variable=enc_weights type=ROM_T2P impl=BRAM
    static bool weights_init = false;
    if (!weights_init) {
        WEIGHTS_INIT:
        for (int i = 0; i < 3; i++) {
            for (int d = 0; d < D_ENC; d++) {
                #pragma HLS PIPELINE II=1
                enc_weights[i][d] = ENCODER_WEIGHTS[i][d];
            }
        }
        weights_init = true;
    }

    // -------------------------------------------------------------------
    // Pipeline state machine
    // -------------------------------------------------------------------
    enum pipeline_state_t {
        IDLE,
        COLLECT_EVENTS,
        NORMALIZE,
        KNN_GRAPH,
        ENCODE_FLOW,
        COMPUTE_EVASION,
        OUTPUT_MOTORS
    };
    static pipeline_state_t state = IDLE;
    static ap_uint<32> cycle_counter = 0;
    #pragma HLS RESET variable=state
    #pragma HLS RESET variable=cycle_counter

    // -------------------------------------------------------------------
    // Per-cycle pipeline execution
    // -------------------------------------------------------------------
    switch (state) {
        // ---------------------------------------------------------------
        case IDLE:
            if (ctrl_regs.enable) {
                state = COLLECT_EVENTS;
            }
            cycle_counter = 0;
            break;

        // ---------------------------------------------------------------
        case COLLECT_EVENTS:
            // Ingest events from AER camera into ring buffer
            // Done continuously; this state waits until enough events
            aer_parallel_interface(aer_bus, aer_timestamp, event_fifo);
            
            // Drain FIFO into ring buffer
            while (!event_fifo.empty() && rb_count < RING_BUFFER_SIZE) {
                #pragma HLS PIPELINE II=1
                aer_event_out_t aer_ev = event_fifo.read();
                event_unpacked_t rb_ev;
                rb_ev.x = aer_ev.x;
                rb_ev.y = aer_ev.y;
                rb_ev.timestamp = aer_ev.timestamp;
                rb_ev.polarity = aer_ev.polarity;
                ring_buffer_events[rb_write_ptr] = rb_ev;
                rb_write_ptr = (rb_write_ptr + 1) & RB_ADDR_MASK;
                rb_count++;
            }

            cycle_counter++;
            // Trigger inference when buffer has enough events or timeout
            if (rb_count >= RING_BUFFER_SIZE / 2 || 
                cycle_counter >= ctrl_regs.inference_period) {
                state = NORMALIZE;
            }
            break;

        // ---------------------------------------------------------------
        case NORMALIZE:
            // Convert raw events to normalized (t, x, y) format
            // and pack into spatial_hash_events array
            {
                ap_uint<12> count = rb_count;
                if (count > MAX_EVENTS) count = MAX_EVENTS;

                ap_uint<12> read_ptr;
                if (rb_write_ptr >= count) {
                    read_ptr = rb_write_ptr - count;
                } else {
                    read_ptr = rb_write_ptr + RING_BUFFER_SIZE - count;
                }

                NORM_LOOP:
                for (ap_uint<12> i = 0; i < count; i++) {
                    #pragma HLS PIPELINE II=1
                    ap_uint<12> addr = (read_ptr + i) & RB_ADDR_MASK;
                    event_unpacked_t ev = ring_buffer_events[addr];

                    // Normalize: x/pixel_radius, y/pixel_radius, t/time_radius
                    static const coord_enc_t INV_PXL = coord_enc_t(1.0f / PXL_RADIUS);
                    static const coord_enc_t INV_T   = coord_enc_t(1.0f / T_RADIUS);

                    spatial_hash_events[i].x = coord_enc_t(ev.x) * INV_PXL;
                    spatial_hash_events[i].y = coord_enc_t(ev.y) * INV_PXL;
                    spatial_hash_events[i].timestamp = ev.timestamp;

                    enc_events[i][0] = coord_enc_t(ev.timestamp) * INV_T;
                    enc_events[i][1] = spatial_hash_events[i].x;
                    enc_events[i][2] = spatial_hash_events[i].y;
                }
                rb_count = count; // Update count to actual processed events
            }
            state = KNN_GRAPH;
            break;

        // ---------------------------------------------------------------
        case KNN_GRAPH:
            // Compute k-NN adjacency using spatial hash
            spatial_hash_knn(spatial_hash_events, rb_count, knn_results);
            state = ENCODE_FLOW;
            break;

        // ---------------------------------------------------------------
        case ENCODE_FLOW:
            // Run LocalGeometryEncoder for per-event flow
            local_geometry_encoder(
                enc_events, rb_count, knn_results, 
                enc_weights, flow_pred, flow_uncert
            );
            state = COMPUTE_EVASION;
            break;

        // ---------------------------------------------------------------
        case COMPUTE_EVASION:
            // Aggregate flow vectors to compute evasion command
            // Simple strategy: average flow from all events → escape opposite
            // (Full collision predictor logic runs on ARM via AXI)
            {
                enc_out_t sum_vx = 0;
                enc_out_t sum_vy = 0;
                ap_uint<12> count = rb_count;
                if (count == 0) count = 1;

                FLOW_AGGREGATE:
                for (ap_uint<12> i = 0; i < count; i++) {
                    #pragma HLS PIPELINE II=1
                    sum_vx += flow_pred[i][0];
                    sum_vy += flow_pred[i][1];
                }

                // Mean flow direction → escape opposite direction
                velocity_t mean_vx = velocity_t(sum_vx / enc_out_t(count));
                velocity_t mean_vy = velocity_t(sum_vy / enc_out_t(count));

                // Debug output: first event's flow
                debug_flow[0] = flow_pred[0][0];
                debug_flow[1] = flow_pred[0][1];

                // Generate motor outputs
                // Evasion: move away from mean flow direction
                // Vertical: slight altitude increase when threat detected
                velocity_t escape_vx = -mean_vx * velocity_t(2.0);   // Opposite direction
                velocity_t escape_vy = -mean_vy * velocity_t(2.0);
                velocity_t escape_vz = velocity_t(0.3);              // Slight climb
                velocity_t escape_yaw = velocity_t(0.0);             // No yaw

                if (ctrl_regs.manual_mode) {
                    pwm_output(
                        ctrl_regs.manual_vx,
                        ctrl_regs.manual_vy,
                        ctrl_regs.manual_vz,
                        ctrl_regs.manual_yaw,
                        ctrl_regs.enable_motors,
                        motor_out
                    );
                } else {
                    pwm_output(
                        escape_vx, escape_vy, escape_vz, escape_yaw,
                        ctrl_regs.enable_motors,
                        motor_out
                    );
                }
            }
            state = OUTPUT_MOTORS;
            break;

        // ---------------------------------------------------------------
        case OUTPUT_MOTORS:
            // Motors updated via combinational logic in pwm_output
            // Loop back to collecting events
            cycle_counter = 0;
            state = COLLECT_EVENTS;
            break;
    }
}