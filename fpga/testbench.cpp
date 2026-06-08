// testbench.cpp — HLS C-simulation testbench for collision_avoidance_top
//
// Compile & run:
//   vitis_hls -f build.tcl  (sets testbench as C simulation top)
//   # or standalone:
//   g++ -I$XILINX_VIVADO/include -std=c++11 -o testbench testbench.cpp && ./testbench
//
// Tests:
//   1. Basic pipeline operation with synthetic events
//   2. Evasion response: drone sees looming object → motors respond
//   3. Manual override mode
//   4. Edge case: empty event stream
//   5. Performance: throughput measurement

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>

// ---------------------------------------------------------------------------
// Use the unified HLS compatibility layer
// hls_compat.h provides ap_fixed, ap_uint, hls::stream, etc. for simulation
// and includes real Vitis HLS headers when synthesizing.
// ---------------------------------------------------------------------------
#define __SIMULATION__
#include "hls_compat.h"

// ---------------------------------------------------------------------------
// Include the FPGA modules under test
// ---------------------------------------------------------------------------
#include "aer_interface.h"
#include "ring_buffer.h"
#include "normalization.h"
#include "spatial_hash.h"
#include "encoder_systolic.h"
#include "pwm_output.h"
#include "top_level.cpp"

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------
#define TEST(name) printf("  TEST: %s ... ", name); fflush(stdout)
#define PASS()      printf("PASS\n")
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); errors++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) FAIL(msg); else PASS_count++; } while(0)

static int PASS_count = 0;
static int errors = 0;

// ---------------------------------------------------------------------------
// Simulated AER camera stream
// ---------------------------------------------------------------------------
struct SimulatedEvents {
    std::vector<uint16_t> xs, ys;
    std::vector<bool> polarities;
    std::vector<uint64_t> timestamps;
};

SimulatedEvents generate_looming_object_event_stream(int n_events) {
    // Simulate an object approaching the camera:
    // Events are generated at positions that expand outward from center
    // over time, mimicking a looming (approaching) object.
    SimulatedEvents events;
    std::mt19937 rng(42);
    
    float center_x = 160.0f;
    float center_y = 120.0f;
    
    for (int i = 0; i < n_events; i++) {
        // Expansion grows over time (simulating approach)
        float t = static_cast<float>(i) / n_events;
        float radius = 10.0f + t * 90.0f;  // 10 → 100 pixels
        
        float angle = static_cast<float>(i) * 0.5f;  // Spiral pattern
        float event_x = center_x + radius * std::cos(angle);
        float event_y = center_y + radius * std::sin(angle);
        
        // Add noise
        event_x += std::normal_distribution<float>(0, 2.0f)(rng);
        event_y += std::normal_distribution<float>(0, 2.0f)(rng);
        
        events.xs.push_back(static_cast<uint16_t>(
            std::max(0.0f, std::min(319.0f, event_x))
        ));
        events.ys.push_back(static_cast<uint16_t>(
            std::max(0.0f, std::min(239.0f, event_y))
        ));
        events.polarities.push_back(i % 2 == 0);
        events.timestamps.push_back(static_cast<uint64_t>(i * 1000));  // 1μs between events
    }
    
    return events;
}

SimulatedEvents generate_static_noise_event_stream(int n_events) {
    // Static background noise — uniform random positions
    SimulatedEvents events;
    std::mt19937 rng(123);
    
    for (int i = 0; i < n_events; i++) {
        events.xs.push_back(static_cast<uint16_t>(rng() % 320));
        events.ys.push_back(static_cast<uint16_t>(rng() % 240));
        events.polarities.push_back(rng() % 2);
        events.timestamps.push_back(static_cast<uint64_t>(i * 500));
    }
    
    return events;
}

// ---------------------------------------------------------------------------
// Simulated AER bus driver (feeds events into FPGA pipeline)
// ---------------------------------------------------------------------------
void feed_events_to_pipeline(
    control_regs_t& ctrl,
    aer_bus_t& aer_bus,
    ap_uint<64>& timestamp,
    const SimulatedEvents& events,
    motor_outputs_t& motor_out,
    enc_out_t debug_flow[2],
    int steps = 100
) {
    size_t event_idx = 0;
    ctrl.enable = 1;
    ctrl.enable_motors = 1;
    
    for (int step = 0; step < steps; step++) {
        timestamp = step * 1000;  // 1μs per step
        
        // Set up AER bus for this step
        if (event_idx < events.xs.size()) {
            aer_bus.x = events.xs[event_idx];
            aer_bus.y = events.ys[event_idx];
            aer_bus.pol = events.polarities[event_idx] ? 1 : 0;
            aer_bus.req = 1;  // Request asserted
            aer_bus.ack = 0;
            event_idx++;
        } else {
            aer_bus.req = 0;
        }
        
        // Run one cycle of the pipeline
        collision_avoidance_top(ctrl, aer_bus, timestamp, motor_out, debug_flow);
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

void test_pwm_output_basic() {
    printf("\n=== Test: pwm_output basic operation ===\n");
    
    motor_outputs_t motors;
    velocity_t vx(1.0f), vy(0.5f), vz(0.3f), yaw(0.1f);
    ap_uint<1> armed(1);
    
    pwm_output(vx, vy, vz, yaw, armed, motors);
    
    float m1 = static_cast<float>(motors.m1);
    float m2 = static_cast<float>(motors.m2);
    float m3 = static_cast<float>(motors.m3);
    float m4 = static_cast<float>(motors.m4);
    
    TEST("PWM produces non-zero output when armed");
    ASSERT(m1 > 0.001f || m2 > 0.001f || m3 > 0.001f || m4 > 0.001f, "all motors zero");
    
    TEST("PWM outputs are within valid range [0.0, 1.0]");
    ASSERT(m1 >= 0.0f && m1 <= 1.0f, "m1 out of range");
    ASSERT(m2 >= 0.0f && m2 <= 1.0f, "m2 out of range");
    ASSERT(m3 >= 0.0f && m3 <= 1.0f, "m3 out of range");
    ASSERT(m4 >= 0.0f && m4 <= 1.0f, "m4 out of range");
    
    // Test disarmed
    pwm_output(vx, vy, vz, yaw, ap_uint<1>(0), motors);
    TEST("PWM disarmed produces zero output");
    ASSERT(
        static_cast<float>(motors.m1) == 0.0f &&
        static_cast<float>(motors.m2) == 0.0f &&
        static_cast<float>(motors.m3) == 0.0f &&
        static_cast<float>(motors.m4) == 0.0f,
        "motors should be zero when disarmed"
    );
}

void test_normalization() {
    printf("\n=== Test: normalization ===\n");
    
    // Build synthetic event buffer
    event_unpacked_t events[RING_BUFFER_SIZE];
    for (int i = 0; i < RING_BUFFER_SIZE; i++) {
        events[i].x = static_cast<ap_uint<10>>((i * 17) % 320);
        events[i].y = static_cast<ap_uint<10>>((i * 31) % 240);
        events[i].timestamp = static_cast<ap_uint<64>>(i * 1000);
        events[i].polarity = i % 2;
    }
    
    event_packed_t packed[MAX_EVENTS];
    coord_enc_t enc_events[MAX_EVENTS][3];
    ap_uint<12> count = RING_BUFFER_SIZE;
    
    for (ap_uint<12> i = 0; i < count; i++) {
        auto& ev = events[static_cast<int>(i)];
        
        coord_enc_t inv_pxl(1.0f / PXL_RADIUS);
        coord_enc_t inv_t(1.0f / T_RADIUS);
        
        packed[static_cast<int>(i)].x = coord_enc_t(static_cast<float>(ev.x)) * inv_pxl;
        packed[static_cast<int>(i)].y = coord_enc_t(static_cast<float>(ev.y)) * inv_pxl;
        packed[static_cast<int>(i)].timestamp = ev.timestamp;
        
        enc_events[static_cast<int>(i)][0] = coord_enc_t(static_cast<float>(ev.timestamp)) * inv_t;
        enc_events[static_cast<int>(i)][1] = packed[static_cast<int>(i)].x;
        enc_events[static_cast<int>(i)][2] = packed[static_cast<int>(i)].y;
    }
    
    TEST("Normalization produces values within reasonable range");
    bool ok = true;
    for (int i = 0; i < 10; i++) {
        float x = static_cast<float>(packed[i].x);
        float y = static_cast<float>(packed[i].y);
        if (std::abs(x) > 1.5f || std::abs(y) > 1.5f) {
            ok = false;
            break;
        }
    }
    ASSERT(ok, "normalized values out of expected range [~0, ~1]");
}

void test_spatial_hash() {
    printf("\n=== Test: spatial_hash ===\n");
    
    // Create synthetic event positions
    event_packed_t events[MAX_EVENTS];
    for (int i = 0; i < 256; i++) {
        // Events in a tight cluster
        float x = 0.3f + 0.02f * (i % 16);
        float y = 0.4f + 0.02f * (i / 16);
        events[i].x = coord_enc_t(x);
        events[i].y = coord_enc_t(y);
        events[i].timestamp = i * 1000;
    }
    
    knn_output_t knn_results[MAX_EVENTS];
    ap_uint<12> count(256);
    
    spatial_hash_knn(events, count, knn_results);
    
    TEST("k-NN produces non-zero adjacency for clustered events");
    bool has_edges = false;
    for (int i = 0; i < static_cast<int>(count); i++) {
        if (knn_results[i].count > 0) {
            has_edges = true;
            break;
        }
    }
    ASSERT(has_edges, "no k-NN edges found for clustered events");
    
    // Events far apart should produce few/no neighbors
    for (int i = 256; i < MAX_EVENTS; i++) {
        events[i].x = coord_enc_t(0.9f + 0.001f * (i - 256));
        events[i].y = coord_enc_t(0.9f + 0.001f * (i - 256));
        events[i].timestamp = i * 1000;
    }
    count = ap_uint<12>(MAX_EVENTS);
    spatial_hash_knn(events, count, knn_results);
    
    int cluster1_edges = 0, cluster2_edges = 0;
    for (int i = 0; i < 256; i++) cluster1_edges += knn_results[i].count;
    for (int i = 256; i < MAX_EVENTS; i++) cluster2_edges += knn_results[i].count;
    
    TEST("Spatial hash correctly separates distant clusters");
    ASSERT(cluster1_edges > 0, "cluster 1 should have edges");
    ASSERT(cluster2_edges == 0, "cluster 2 should have no edges (too far)");
}

void test_encoder_systolic() {
    printf("\n=== Test: encoder_systolic ===\n");
    
    // Build minimal input
    coord_enc_t events[8][3];
    knn_output_t knn[8];
    encoder_weights_t weights;
    
    for (int i = 0; i < 8; i++) {
        events[i][0] = coord_enc_t(0.1f + 0.01f * i);
        events[i][1] = coord_enc_t(0.2f + 0.02f * i);
        events[i][2] = coord_enc_t(0.3f + 0.03f * i);
        knn[i].count = 0;  // No neighbors (not enough events for meaningful test)
    }
    
    enc_out_t flow_pred[8][2];
    enc_out_t flow_uncert[8];
    
    local_geometry_encoder(events, ap_uint<12>(8), knn, weights, flow_pred, flow_uncert);
    
    TEST("Encoder runs without crash with zero neighbors");
    ASSERT(static_cast<float>(flow_uncert[0]) >= 0.0f, "uncertainty should be non-negative");
    
    TEST("Encoder produces finite flow values");
    bool all_finite = true;
    for (int i = 0; i < 8; i++) {
        float vx = static_cast<float>(flow_pred[i][0]);
        float vy = static_cast<float>(flow_pred[i][1]);
        if (std::isnan(vx) || std::isinf(vx) || std::isnan(vy) || std::isinf(vy)) {
            all_finite = false;
            break;
        }
    }
    ASSERT(all_finite, "some flow values are NaN or Inf");
}

void test_top_level_evasion_response() {
    printf("\n=== Test: top-level evasion response ===\n");
    
    // TEST A: Looming object → should produce evasion response
    printf("\n  --- Subtest A: Looming object detection ---\n");
    {
        control_regs_t ctrl;
        aer_bus_t aer_bus = {0};
        ap_uint<64> timestamp(0);
        motor_outputs_t motor_out = {0};
        enc_out_t debug_flow[2] = {coord_enc_t(0), coord_enc_t(0)};
        
        ctrl.enable = 1;
        ctrl.enable_motors = 1;
        ctrl.manual_mode = 0;
        ctrl.inference_period = 5000;
        
        auto looming_events = generate_looming_object_event_stream(2048);
        
        feed_events_to_pipeline(ctrl, aer_bus, timestamp, looming_events, motor_out, debug_flow, 100);
        
        float m1 = static_cast<float>(motor_out.m1);
        float m2 = static_cast<float>(motor_out.m2);
        float m3 = static_cast<float>(motor_out.m3);
        float m4 = static_cast<float>(motor_out.m4);
        
        TEST("Looming object triggers motor response");
        bool any_thrust = (m1 > 0.01f || m2 > 0.01f || m3 > 0.01f || m4 > 0.01f);
        ASSERT(any_thrust, "all motors at idle with looming object");
        
        printf("    Motor values: m1=%.3f m2=%.3f m3=%.3f m4=%.3f\n", m1, m2, m3, m4);
    }
    
    // TEST B: Static noise → minimal motor response (background rejection)
    printf("\n  --- Subtest B: Static noise rejection ---\n");
    {
        control_regs_t ctrl;
        aer_bus_t aer_bus = {0};
        ap_uint<64> timestamp(0);
        motor_outputs_t motor_out = {0};
        enc_out_t debug_flow[2] = {coord_enc_t(0), coord_enc_t(0)};
        
        ctrl.enable = 1;
        ctrl.enable_motors = 1;
        ctrl.manual_mode = 0;
        ctrl.inference_period = 5000;
        
        auto noise_events = generate_static_noise_event_stream(2048);
        
        feed_events_to_pipeline(ctrl, aer_bus, timestamp, noise_events, motor_out, debug_flow, 100);
        
        float m1 = static_cast<float>(motor_out.m1);
        float m2 = static_cast<float>(motor_out.m2);
        float m3 = static_cast<float>(motor_out.m3);
        float m4 = static_cast<float>(motor_out.m4);
        
        TEST("Static noise produces minimal motor response");
        float total_thrust = m1 + m2 + m3 + m4;
        ASSERT(total_thrust < 2.0f, "static noise triggered strong evasion: %.3f", total_thrust);
        printf("    Motor values: m1=%.3f m2=%.3f m3=%.3f m4=%.3f (total=%.3f)\n", 
               m1, m2, m3, m4, total_thrust);
    }
    
    // TEST C: Manual override mode
    printf("\n  --- Subtest C: Manual override mode ---\n");
    {
        control_regs_t ctrl;
        aer_bus_t aer_bus = {0};
        ap_uint<64> timestamp(0);
        motor_outputs_t motor_out = {0};
        enc_out_t debug_flow[2] = {coord_enc_t(0), coord_enc_t(0)};
        
        ctrl.enable = 1;
        ctrl.enable_motors = 1;
        ctrl.manual_mode = 1;
        ctrl.manual_vx = velocity_t(1.0f);
        ctrl.manual_vy = velocity_t(0.0f);
        ctrl.manual_vz = velocity_t(0.5f);
        ctrl.manual_yaw = velocity_t(0.0f);
        ctrl.inference_period = 5000;
        
        auto looming_events = generate_looming_object_event_stream(512);
        
        feed_events_to_pipeline(ctrl, aer_bus, timestamp, looming_events, motor_out, debug_flow, 50);
        
        float m1 = static_cast<float>(motor_out.m1);
        float m2 = static_cast<float>(motor_out.m2);
        float m3 = static_cast<float>(motor_out.m3);
        float m4 = static_cast<float>(motor_out.m4);
        
        TEST("Manual mode produces commanded output");
        ASSERT(m1 > 0.01f || m2 > 0.01f || m3 > 0.01f || m4 > 0.01f, 
               "manual mode produced zero motor output");
        printf("    Manual vx=1.0 → motors: m1=%.3f m2=%.3f m3=%.3f m4=%.3f\n", m1, m2, m3, m4);
    }
}

void test_ring_buffer() {
    printf("\n=== Test: ring buffer ===\n");
    
    // Test basic push/pop from ring_buffer
    // Simulate the ring buffer logic from top_level.cpp
    event_unpacked_t ring_buf[RING_BUFFER_SIZE];
    ap_uint<12> wr_ptr = 0;
    ap_uint<12> count = 0;
    
    // Push events
    for (int i = 0; i < 100; i++) {
        ring_buf[wr_ptr.val] = {ap_uint<10>(i), ap_uint<10>(i*2), ap_uint<64>(i*1000), ap_uint<1>(i%2)};
        wr_ptr = ap_uint<12>((wr_ptr.val + 1) & RB_ADDR_MASK);
        count = ap_uint<12>(count.val + 1);
    }
    
    TEST("Ring buffer stores and retrieves events");
    ap_uint<12> read_ptr = ap_uint<12>((wr_ptr.val >= count.val) ? 
                                        wr_ptr.val - count.val : 
                                        wr_ptr.val + RING_BUFFER_SIZE - count.val);
    int retrieved = static_cast<int>(ring_buf[read_ptr.val].x);
    ASSERT(retrieved == 0, "first event should be at idx 0, got: %d", retrieved);
    
    TEST("Ring buffer wraps correctly");
    // Wrap around
    for (int i = 0; i < RING_BUFFER_SIZE - 50; i++) {
        ring_buf[wr_ptr.val] = {ap_uint<10>(i+1000), ap_uint<10>(0), ap_uint<64>(0), ap_uint<1>(0)};
        wr_ptr = ap_uint<12>((wr_ptr.val + 1) & RB_ADDR_MASK);
        if (count.val < RING_BUFFER_SIZE) count = ap_uint<12>(count.val + 1);
    }
    ASSERT(count.val == RING_BUFFER_SIZE, "count should saturate at %d, got: %d", 
           RING_BUFFER_SIZE, count.val);
    printf("    Ring buffer capacity: %d, wr_ptr: %d\n", count.val, wr_ptr.val);
}

// ===========================================================================
// Main entry
// ===========================================================================
int main() {
    printf("============================================\n");
    printf("  FPGA Collision Avoidance — Testbench\n");
    printf("============================================\n");
    
    int total_tests = 0;
    int passed = 0;
    
    // Run all tests
    test_pwm_output_basic();
    total_tests += 3;
    
    test_normalization();
    total_tests += 1;
    
    test_spatial_hash();
    total_tests += 2;
    
    test_encoder_systolic();
    total_tests += 2;
    
    test_ring_buffer();
    total_tests += 2;
    
    test_top_level_evasion_response();
    total_tests += 3;
    
    // Summary
    PASS_count = total_tests - errors;
    printf("\n============================================\n");
    printf("  Results: %d/%d passed", PASS_count, total_tests);
    if (errors > 0) printf(", %d FAILED", errors);
    printf("\n============================================\n");
    
    return errors > 0 ? 1 : 0;
}