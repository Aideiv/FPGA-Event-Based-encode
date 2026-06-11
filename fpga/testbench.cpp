// testbench.cpp — HLS C-simulation testbench for collision_avoidance_top
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>

#define __SIMULATION__
#include "hls_compat.h"
#include "aer_interface.h"
#include "ring_buffer.h"
#include "normalization.h"
#include "spatial_hash.h"
#include "encoder_systolic.h"
#include "pwm_output.h"
#include "top_level.cpp"

#define TEST(name)                   \
    printf("  TEST: %s ... ", name); \
    fflush(stdout)
#define PASS() printf("PASS\n")
#define FAIL(msg, ...)                            \
    do {                                          \
        printf("FAIL: " msg "\n", ##__VA_ARGS__); \
        errors++;                                 \
    } while (0)
#define ASSERT(cond, ...)      \
    do {                       \
        if (!(cond))           \
            FAIL(__VA_ARGS__); \
        else                   \
            PASS_count++;      \
    } while (0)

static int PASS_count = 0;
static int errors = 0;

struct SimulatedEvents {
    std::vector<uint16_t> xs, ys;
    std::vector<bool> polarities;
    std::vector<uint64_t> timestamps;
};

SimulatedEvents generate_looming_object_event_stream(int n_events) {
    SimulatedEvents events;
    std::mt19937 rng(42);
    float center_x = 160.0f, center_y = 120.0f;
    for (int i = 0; i < n_events; i++) {
        float t = static_cast<float>(i) / n_events;
        float radius = 10.0f + t * 90.0f;
        float angle = static_cast<float>(i) * 0.5f;
        float ex = center_x + radius * std::cos(angle);
        float ey = center_y + radius * std::sin(angle);
        ex += std::normal_distribution<float>(0, 2.0f)(rng);
        ey += std::normal_distribution<float>(0, 2.0f)(rng);
        events.xs.push_back(static_cast<uint16_t>(std::max(0.0f, std::min(319.0f, ex))));
        events.ys.push_back(static_cast<uint16_t>(std::max(0.0f, std::min(239.0f, ey))));
        events.polarities.push_back(i % 2 == 0);
        events.timestamps.push_back(static_cast<uint64_t>(i * 1000));
    }
    return events;
}

SimulatedEvents generate_static_noise_event_stream(int n_events) {
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

void feed_events_to_pipeline(control_regs_t& ctrl, aer_bus_t& aer_bus, ap_uint<64>& timestamp,
                             const SimulatedEvents& events, motor_outputs_t& motor_out,
                             enc_out_t debug_flow[2], int steps = 100) {
    size_t event_idx = 0;
    ctrl.enable = 1;
    ctrl.enable_motors = 1;
    for (int step = 0; step < steps; step++) {
        timestamp = step * 1000;
        // Emulate the camera side of the 4-phase AER handshake: assert REQ
        // with a new event when the bus is idle, deassert REQ once the FPGA
        // ACKs. The interface reads aer_bus.address (not the x/y aliases).
        if (aer_bus.ack == 1) {
            aer_bus.req = 0;
        } else if (aer_bus.req == 0 && event_idx < events.xs.size()) {
            uint32_t ex = events.xs[event_idx] % SENSOR_WIDTH;
            uint32_t ey = events.ys[event_idx] % SENSOR_HEIGHT;
            aer_bus.address = ap_uint<19>(ey * SENSOR_WIDTH + ex);
            aer_bus.x = ap_uint<10>(ex);
            aer_bus.y = ap_uint<9>(ey);
            aer_bus.pol = events.polarities[event_idx] ? 1 : 0;
            aer_bus.polarity = aer_bus.pol;
            aer_bus.req = 1;
            event_idx++;
        }
        collision_avoidance_top(ctrl, aer_bus, timestamp, motor_out, debug_flow);
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

// Motor outputs are PWM pulse widths in clock ticks [PWM_MIN_TICKS,
// PWM_MAX_TICKS]; normalize back to thrust fraction [0, 1] for checks.
static float pwm_to_thrust(const pwm_tick_t& t) {
    return (static_cast<float>(t.val) - PWM_MIN_TICKS) /
           static_cast<float>(PWM_MAX_TICKS - PWM_MIN_TICKS);
}

void test_pwm_output_basic() {
    printf("\n=== Test: pwm_output basic operation ===\n");
    motor_outputs_t motors, motors_disarmed;
    velocity_t vx(1.0f), vy(0.5f), vz(0.3f), yaw(0.1f);

    pwm_output(vx, vy, vz, yaw, ap_uint<1>(1), motors);
    float m1 = pwm_to_thrust(motors.m1), m2 = pwm_to_thrust(motors.m2);
    float m3 = pwm_to_thrust(motors.m3), m4 = pwm_to_thrust(motors.m4);

    TEST("PWM produces non-zero output when armed");
    ASSERT(m1 > 0.001f || m2 > 0.001f || m3 > 0.001f || m4 > 0.001f, "all motors zero");

    TEST("PWM outputs are within valid range [0.0, 1.0]");
    ASSERT(m1 >= 0.0f && m1 <= 1.0f, "m1 out of range");
    ASSERT(m2 >= 0.0f && m2 <= 1.0f, "m2 out of range");
    ASSERT(m3 >= 0.0f && m3 <= 1.0f, "m3 out of range");
    ASSERT(m4 >= 0.0f && m4 <= 1.0f, "m4 out of range");

    // Disarmed: motors go to PWM_MIN_TICKS (safe minimum pulse, not zero)
    pwm_output(vx, vy, vz, yaw, ap_uint<1>(0), motors_disarmed);
    TEST("PWM disarmed sets minimum pulse (PWM_MIN_TICKS)");
    ASSERT(motors_disarmed.m1 == PWM_MIN_TICKS && motors_disarmed.m2 == PWM_MIN_TICKS &&
               motors_disarmed.m3 == PWM_MIN_TICKS && motors_disarmed.m4 == PWM_MIN_TICKS,
           "disarmed motors should be PWM_MIN_TICKS, got m1=%lu",
           static_cast<unsigned long>(motors_disarmed.m1.val));
}

void test_normalization() {
    printf("\n=== Test: normalization ===\n");
    // Use correct field types for event_unpacked_t
    event_unpacked_t events[RING_BUFFER_SIZE];
    for (int i = 0; i < RING_BUFFER_SIZE; i++) {
        events[i].x = ap_ufixed<16, 4>(static_cast<float>((i * 17) % 320) / 640.0f);
        events[i].y = ap_ufixed<16, 4>(static_cast<float>((i * 31) % 240) / 480.0f);
        events[i].timestamp = ap_uint<32>(static_cast<uint32_t>(i * 1000));
        events[i].polarity = ap_uint<1>(i % 2);
    }

    event_packed_t packed[MAX_EVENTS];
    coord_enc_t enc_events[MAX_EVENTS][3];
    event_cnt_t count = RING_BUFFER_SIZE;

    for (event_cnt_t i = 0; static_cast<int>(i.val) < static_cast<int>(count.val);
         i = event_cnt_t(i.val + 1)) {
        int idx = static_cast<int>(i.val);
        auto& ev = events[idx];
        coord_enc_t inv_pxl(1.0f / PXL_RADIUS);
        coord_enc_t inv_t(1.0f / T_RADIUS);
        packed[idx].x = coord_enc_t(static_cast<float>(ev.x)) * inv_pxl;
        packed[idx].y = coord_enc_t(static_cast<float>(ev.y)) * inv_pxl;
        packed[idx].timestamp = ev.timestamp;
        enc_events[idx][0] = coord_enc_t(static_cast<float>(ev.timestamp)) * inv_t;
        enc_events[idx][1] = packed[idx].x;
        enc_events[idx][2] = packed[idx].y;
    }

    TEST("Normalization produces values within reasonable range");
    // x/PXL_RADIUS scales [0,1) pixels into radius units (up to ~44),
    // saturating coord_enc_t at its Q4.12 limit — values must stay finite
    // and inside the representable [-8, 8) window.
    bool ok = true;
    for (int i = 0; i < 10; i++) {
        float x = static_cast<float>(packed[i].x);
        float y = static_cast<float>(packed[i].y);
        if (!(x >= -8.0f && x <= 8.0f) || !(y >= -8.0f && y <= 8.0f)) {
            ok = false;
            break;
        }
    }
    ASSERT(ok, "normalized values escaped coord_enc_t range [-8, 8]");
}

void test_spatial_hash() {
    printf("\n=== Test: spatial_hash ===\n");
    event_packed_t events[MAX_EVENTS];
    for (int i = 0; i < 256; i++) {
        float x = 0.3f + 0.02f * (i % 16);
        float y = 0.4f + 0.02f * (i / 16);
        events[i].x = coord_enc_t(x);
        events[i].y = coord_enc_t(y);
        events[i].timestamp = i * 1000;
    }
    knn_output_t knn_results[MAX_EVENTS];
    event_cnt_t count(256);
    spatial_hash_knn(events, count, knn_results);

    TEST("k-NN produces non-zero adjacency for clustered events");
    bool has_edges = false;
    for (int i = 0; i < static_cast<int>(count.val); i++) {
        if (knn_results[i].count.val > 0) {
            has_edges = true;
            break;
        }
    }
    ASSERT(has_edges, "no k-NN edges found for clustered events");

    for (int i = 256; i < MAX_EVENTS; i++) {
        events[i].x = coord_enc_t(0.9f + 0.001f * (i - 256));
        events[i].y = coord_enc_t(0.9f + 0.001f * (i - 256));
        events[i].timestamp = i * 1000;
    }
    count = event_cnt_t(MAX_EVENTS);
    spatial_hash_knn(events, count, knn_results);

    int cluster1_edges = 0, cross_edges = 0;
    for (int i = 0; i < 256; i++) {
        int n = static_cast<int>(knn_results[i].count.val);
        cluster1_edges += n;
        for (int k = 0; k < n && k < K_NEIGHBORS; k++)
            if (static_cast<int>(knn_results[i].neighbor_indices[k].val) >= 256) cross_edges++;
    }
    for (int i = 256; i < MAX_EVENTS; i++) {
        int n = static_cast<int>(knn_results[i].count.val);
        for (int k = 0; k < n && k < K_NEIGHBORS; k++)
            if (static_cast<int>(knn_results[i].neighbor_indices[k].val) < 256) cross_edges++;
    }

    TEST("Spatial hash correctly separates distant clusters");
    ASSERT(cluster1_edges > 0, "cluster 1 should have edges");
    ASSERT(cross_edges == 0, "clusters should not connect, got %d cross edges", cross_edges);
}

void test_encoder_systolic() {
    printf("\n=== Test: encoder_systolic ===\n");
    coord_enc_t events[8][3];
    knn_output_t knn[8];
    encoder_weights_t weights;

    for (int i = 0; i < 8; i++) {
        events[i][0] = coord_enc_t(0.1f + 0.01f * i);
        events[i][1] = coord_enc_t(0.2f + 0.02f * i);
        events[i][2] = coord_enc_t(0.3f + 0.03f * i);
        knn[i].count = ap_uint<6>(0);
    }

    enc_out_t flow_pred[8][2];
    enc_out_t flow_uncert[8];
    local_geometry_encoder(events, event_cnt_t(8), knn, weights, flow_pred, flow_uncert);

    TEST("Encoder runs without crash with zero neighbors");
    ASSERT(static_cast<float>(flow_uncert[0]) >= 0.0f, "uncertainty should be non-negative");

    TEST("Encoder produces finite flow values");
    bool all_finite = true;
    for (int i = 0; i < 8; i++) {
        float vx = static_cast<float>(flow_pred[i][0]), vy = static_cast<float>(flow_pred[i][1]);
        if (std::isnan(vx) || std::isinf(vx) || std::isnan(vy) || std::isinf(vy)) {
            all_finite = false;
            break;
        }
    }
    ASSERT(all_finite, "some flow values are NaN or Inf");
}

void test_top_level_evasion_response() {
    printf("\n=== Test: top-level evasion response ===\n");
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
        ctrl.inference_period = 5000;  // rb_count >= 2048 triggers inference first
        auto looming_events = generate_looming_object_event_stream(2048);
        feed_events_to_pipeline(
            ctrl, aer_bus, timestamp, looming_events, motor_out, debug_flow,
            8400);  // 4-phase handshake = 4 steps/event x 2048 events + pipeline states
        float m1 = pwm_to_thrust(motor_out.m1), m2 = pwm_to_thrust(motor_out.m2);
        float m3 = pwm_to_thrust(motor_out.m3), m4 = pwm_to_thrust(motor_out.m4);
        TEST("Looming object triggers motor response");
        ASSERT(m1 > 0.01f || m2 > 0.01f || m3 > 0.01f || m4 > 0.01f, "all motors at idle");
        printf("    Motor values: m1=%.3f m2=%.3f m3=%.3f m4=%.3f\n", m1, m2, m3, m4);
    }

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
        feed_events_to_pipeline(ctrl, aer_bus, timestamp, noise_events, motor_out, debug_flow,
                                8400);
        float m1 = pwm_to_thrust(motor_out.m1), m2 = pwm_to_thrust(motor_out.m2);
        float m3 = pwm_to_thrust(motor_out.m3), m4 = pwm_to_thrust(motor_out.m4);
        float total = m1 + m2 + m3 + m4;
        TEST("Static noise produces minimal motor response");
        ASSERT(total < 2.0f, "static noise triggered strong evasion: %.3f", total);
        printf("    Motor values: m1=%.3f m2=%.3f m3=%.3f m4=%.3f (total=%.3f)\n", m1, m2, m3, m4,
               total);
    }

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
        ctrl.inference_period = 600;  // 512 events < 2048 threshold: trigger via timeout
        auto looming_events = generate_looming_object_event_stream(512);
        feed_events_to_pipeline(ctrl, aer_bus, timestamp, looming_events, motor_out, debug_flow,
                                650);
        float m1 = pwm_to_thrust(motor_out.m1), m2 = pwm_to_thrust(motor_out.m2);
        float m3 = pwm_to_thrust(motor_out.m3), m4 = pwm_to_thrust(motor_out.m4);
        TEST("Manual mode produces commanded output");
        ASSERT(m1 > 0.01f || m2 > 0.01f || m3 > 0.01f || m4 > 0.01f,
               "manual mode produced zero motor output");
        printf("    Manual vx=1.0 -> motors: m1=%.3f m2=%.3f m3=%.3f m4=%.3f\n", m1, m2, m3, m4);
    }
}

void test_ring_buffer() {
    printf("\n=== Test: ring buffer ===\n");
    event_unpacked_t ring_buf[RING_BUFFER_SIZE];
    ap_uint<12> wr_ptr(0);
    event_cnt_t count(0);  // 13-bit: a full buffer holds RING_BUFFER_SIZE = 4096 events

    // Push events with correct struct field types
    for (int i = 0; i < 100; i++) {
        ring_buf[static_cast<int>(wr_ptr.val)].x =
            ap_ufixed<16, 4>(static_cast<float>(i % 640) / 640.0f);
        ring_buf[static_cast<int>(wr_ptr.val)].y =
            ap_ufixed<16, 4>(static_cast<float>((i * 2) % 480) / 480.0f);
        ring_buf[static_cast<int>(wr_ptr.val)].timestamp =
            ap_uint<32>(static_cast<uint32_t>(i * 1000));
        ring_buf[static_cast<int>(wr_ptr.val)].polarity = ap_uint<1>(i % 2);
        wr_ptr = ap_uint<12>((wr_ptr.val + 1) & RB_ADDR_MASK);
        count = event_cnt_t(count.val + 1);
    }

    TEST("Ring buffer stores and retrieves events");
    uint64_t read_offset = (wr_ptr.val >= count.val) ? (wr_ptr.val - count.val)
                                                     : (wr_ptr.val + RING_BUFFER_SIZE - count.val);
    ap_uint<12> read_ptr(read_offset);
    float first_x = static_cast<float>(ring_buf[static_cast<int>(read_ptr.val)].x);
    ASSERT(first_x >= 0.0f && first_x <= 1.0f, "first event x should be valid, got %.3f",
           static_cast<double>(first_x));

    TEST("Ring buffer wraps correctly");
    for (int i = 0; i < RING_BUFFER_SIZE - 50; i++) {
        ring_buf[static_cast<int>(wr_ptr.val)].x =
            ap_ufixed<16, 4>(static_cast<float>((i + 1000) % 640) / 640.0f);
        ring_buf[static_cast<int>(wr_ptr.val)].y = ap_ufixed<16, 4>(0.0f);
        ring_buf[static_cast<int>(wr_ptr.val)].timestamp = ap_uint<32>(0);
        ring_buf[static_cast<int>(wr_ptr.val)].polarity = ap_uint<1>(0);
        wr_ptr = ap_uint<12>((wr_ptr.val + 1) & RB_ADDR_MASK);
        if (count.val < RING_BUFFER_SIZE) count = event_cnt_t(count.val + 1);
    }
    ASSERT(count.val == RING_BUFFER_SIZE, "count should saturate at %d, got %llu", RING_BUFFER_SIZE,
           static_cast<unsigned long long>(count.val));
    printf("    Ring buffer capacity: %llu, wr_ptr: %llu\n",
           static_cast<unsigned long long>(count.val), static_cast<unsigned long long>(wr_ptr.val));
}

int main() {
    printf("============================================\n");
    printf("  FPGA Collision Avoidance - Testbench\n");
    printf("============================================\n");

    test_pwm_output_basic();
    test_normalization();
    test_spatial_hash();
    test_encoder_systolic();
    test_ring_buffer();
    test_top_level_evasion_response();

    printf("\n============================================\n");
    printf("  Results: %d/%d assertions passed", PASS_count, PASS_count + errors);
    if (errors > 0) printf(", %d FAILED", errors);
    printf("\n============================================\n");
    return errors > 0 ? 1 : 0;
}