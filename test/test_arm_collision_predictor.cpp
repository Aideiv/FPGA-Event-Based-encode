// test_arm_collision_predictor.cpp — Unit tests for ARM collision_predictor.h
//
// Tests for the ARM-side collision prediction logic WITHOUT requiring FPGA.
// Uses Google Test framework or standalone CTest.
//
// Build:
//   g++ -std=c++17 -Iarm -I. -o test_arm_cp test_arm_collision_predictor.cpp -lpthread
//   ./test_arm_cp
//
// Or with CMake/CTest:
//   add_test(test_arm_cp test_arm_collision_predictor.cpp)

#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

#include "arm/collision_predictor.h"
#include "arm/evasion_controller.h"
#include "arm/fpga_interface.h"

using namespace drone;

// ---------------------------------------------------------------------------
// Simple test framework (no Google Test dependency)
// ---------------------------------------------------------------------------
static int g_passed = 0;
static int g_failed = 0;

#define TEST(name)                  \
    printf("  TEST: %-50s ", name); \
    fflush(stdout)
#define CHECK(cond, ...)         \
    do {                         \
        if (!(cond)) {           \
            printf("FAIL: ");    \
            printf(__VA_ARGS__); \
            printf("\n");        \
            g_failed++;          \
        } else {                 \
            printf("PASS\n");    \
            g_passed++;          \
        }                        \
    } while (0)

// ---------------------------------------------------------------------------
// Helper: generate synthetic event flow vectors
// ---------------------------------------------------------------------------
std::vector<EventFlow> make_looming_flow_vectors(int n, float expansion_rate = 1.0f) {
    // Simulate events from a looming object:
    // Outward radial flow from center, magnitude increasing outward
    std::vector<EventFlow> events;
    events.reserve(n);

    for (int i = 0; i < n; i++) {
        float angle = (float)i * 0.1f;
        float radius = 0.05f + 0.1f * ((float)i / n);  // 0.05 → 0.15

        float x = 0.5f + radius * cosf(angle);
        float y = 0.5f + radius * sinf(angle);

        // Outward radial flow
        float vx = radius * cosf(angle) * expansion_rate;
        float vy = radius * sinf(angle) * expansion_rate;

        events.push_back({vx, vy, x, y, (uint32_t)i});
    }
    return events;
}

std::vector<EventFlow> make_tangential_flow_vectors(int n) {
    // Simulate lateral motion: flow is tangential, NOT radial
    // Object moving left-to-right at constant depth
    std::vector<EventFlow> events;
    events.reserve(n);

    float center_y = 0.5f;
    float speed = 0.5f;

    for (int i = 0; i < n; i++) {
        float x = 0.1f + 0.8f * ((float)i / n);
        float y = center_y + 0.02f * sinf((float)i * 0.3f);

        // Uniform rightward flow
        float vx = speed;
        float vy = 0.0f;

        events.push_back({vx, vy, x, y, (uint32_t)i});
    }
    return events;
}

std::vector<EventFlow> make_random_flow_vectors(int n) {
    // Random noise: should produce no organized clusters
    std::vector<EventFlow> events;
    events.reserve(n);

    srand(42);
    for (int i = 0; i < n; i++) {
        float x = (float)rand() / RAND_MAX;
        float y = (float)rand() / RAND_MAX;
        float vx = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        float vy = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

        events.push_back({vx, vy, x, y, (uint32_t)i});
    }
    return events;
}

std::vector<EventFlow> make_multi_object_flow(int n) {
    // Two looming objects: left and right of center
    std::vector<EventFlow> events;
    events.reserve(n);

    for (int i = 0; i < n / 2; i++) {
        float angle = (float)i * 0.15f;
        float radius = 0.03f + 0.08f * ((float)i / (n / 2));

        // Left object
        float x1 = 0.2f + radius * cosf(angle);
        float y1 = 0.5f + radius * sinf(angle);
        float vx1 = radius * cosf(angle) * 0.8f;
        float vy1 = radius * sinf(angle) * 0.8f;
        events.push_back({vx1, vy1, x1, y1, (uint32_t)i});
    }

    for (int i = 0; i < n / 2; i++) {
        float angle = (float)i * 0.15f;
        float radius = 0.03f + 0.08f * ((float)i / (n / 2));

        // Right object
        float x2 = 0.8f + radius * cosf(angle);
        float y2 = 0.5f + radius * sinf(angle);
        float vx2 = radius * cosf(angle) * 1.2f;
        float vy2 = radius * sinf(angle) * 1.2f;
        events.push_back({vx2, vy2, x2, y2, (uint32_t)(n / 2 + i)});
    }

    return events;
}

// ===========================================================================
// TESTS
// ===========================================================================

void test_empty_input() {
    printf("\n=== Empty input ===\n");

    CollisionPredictor predictor;
    ThreatAssessment result = predictor.assess({});

    TEST("Empty input produces no threat");
    CHECK(!result.threat_detected, "threat should be false for empty input");

    TEST("Empty input produces empty objects list");
    CHECK(result.objects.empty(), "objects should be empty");

    TEST("Empty input produces max float TTC");
    CHECK(result.time_to_first_collision == std::numeric_limits<float>::max(),
          "TTC should be max float");
}

void test_looming_object_detected() {
    printf("\n=== Looming object detection ===\n");

    CollisionPredictor::Config cfg;
    cfg.cluster_radius = 0.15f;
    cfg.flow_similarity_threshold = 0.5f;
    cfg.min_events_per_cluster = 10;
    cfg.safety_time_threshold = 2.0f;
    CollisionPredictor predictor(cfg);

    auto events = make_looming_flow_vectors(500, 1.5f);
    ThreatAssessment result = predictor.assess(events);

    TEST("Looming object produces at least one cluster");
    CHECK(result.objects.size() >= 1, "no clusters found for looming object");

    TEST("Looming object has finite TTC");
    if (!result.objects.empty()) {
        CHECK(result.objects[0].ttc < 100.0f, "ttc should be finite: %.2f", result.objects[0].ttc);
    }

    TEST("Looming object triggers threat detection");
    CHECK(result.threat_detected, "threat should be detected");

    TEST("Looming object has non-zero urgency");
    CHECK(result.max_urgency > 0.0f, "urgency should be >0 for looming threat");

    printf("    Clusters: %zu, max_urgency: %.3f, ttc: %.2fs\n", result.objects.size(),
           result.max_urgency, result.time_to_first_collision);
}

void test_lateral_motion_no_false_alarm() {
    printf("\n=== Lateral motion (no false alarm) ===\n");

    CollisionPredictor::Config cfg;
    cfg.cluster_radius = 0.15f;
    cfg.flow_similarity_threshold = 0.5f;
    cfg.min_events_per_cluster = 50;
    CollisionPredictor predictor(cfg);

    auto events = make_tangential_flow_vectors(500);
    ThreatAssessment result = predictor.assess(events);

    TEST("Tangential flow may produce clusters but lower urgency");
    printf("    Objects: %zu, max_urgency: %.3f\n", result.objects.size(), result.max_urgency);

    // Note: tangential motion through a narrow FOV can still cluster,
    // but the radial-to-tangential ratio should identify it as non-looming
    CHECK(result.max_urgency < 0.9f, "urgency should not be critical for lateral motion: %.3f",
          result.max_urgency);
}

void test_random_noise_low_urgency() {
    printf("\n=== Random noise rejection ===\n");

    CollisionPredictor predictor;

    auto events = make_random_flow_vectors(500);
    ThreatAssessment result = predictor.assess(events);

    TEST("Random noise produces zero or low urgency");
    CHECK(result.max_urgency < 0.5f, "random noise should have low urgency: %.3f",
          result.max_urgency);

    printf("    Threat: %s, urgency: %.3f\n", result.threat_detected ? "YES" : "NO",
           result.max_urgency);
}

void test_multi_object_clustering() {
    printf("\n=== Multiple object clustering ===\n");

    CollisionPredictor::Config cfg;
    cfg.cluster_radius = 0.1f;
    cfg.min_events_per_cluster = 20;
    cfg.safety_time_threshold = 2.0f;
    CollisionPredictor predictor(cfg);

    auto events = make_multi_object_flow(800);
    ThreatAssessment result = predictor.assess(events);

    TEST("Multiple objects produce multiple clusters");
    CHECK(result.objects.size() >= 2, "expected >= 2 clusters, got %zu", result.objects.size());

    TEST("Multiple objects detected as threat");
    CHECK(result.threat_detected, "should detect multiple threats");

    printf("    Clusters: %zu, max_urgency: %.3f\n", result.objects.size(), result.max_urgency);

    // Print each cluster
    for (size_t i = 0; i < result.objects.size(); i++) {
        printf("    Cluster %zu: center=(%.3f,%.3f) extent=%.3f ttc=%.1fs urgency=%.2f events=%u\n",
               i, result.objects[i].center_x, result.objects[i].center_y,
               result.objects[i].spatial_extent, result.objects[i].ttc,
               result.objects[i].collision_urgency, result.objects[i].event_count);
    }
}

void test_safe_bearing() {
    printf("\n=== Safe bearing computation ===\n");

    CollisionPredictor predictor;

    // Single looming object on the right side
    auto events = make_looming_flow_vectors(300, 1.0f);
    // Shift all events to right side
    for (auto& ev : events) {
        ev.x = 0.3f + ev.x * 0.4f + 0.3f;  // Center around 0.6
        ev.y = 0.5f;
    }

    ThreatAssessment result = predictor.assess(events);

    TEST("Safe bearing is finite");
    CHECK(!std::isnan(result.safe_bearing), "safe_bearing should not be NaN");

    TEST("Safe bearing confidence is in [0, 1]");
    CHECK(result.safe_bearing_confidence >= 0.0f && result.safe_bearing_confidence <= 1.0f,
          "confidence should be [0,1]: %.3f", result.safe_bearing_confidence);

    printf("    Safe bearing: %.1f° (confidence: %.2f)\n", result.safe_bearing * 180.0f / M_PI,
           result.safe_bearing_confidence);
}

// ===========================================================================
// FPGA INTERFACE TESTS — FLOW bundle register decode
// ===========================================================================

void test_fpga_flow_register_roundtrip() {
    printf("\n=== FPGA flow register round-trip ===\n");

    FpgaInterface fpga;
    auto truth = make_looming_flow_vectors(500, 1.5f);
    for (size_t i = 0; i < truth.size(); i++) {
        fpga.write_register(REG_FLOW_DATA + 8 * i,
                            FpgaInterface::pack_position(truth[i].x, truth[i].y));
        fpga.write_register(REG_FLOW_DATA + 8 * i + 4,
                            FpgaInterface::pack_flow(truth[i].vx, truth[i].vy));
    }
    fpga.write_register(REG_FLOW_COUNT, static_cast<uint32_t>(truth.size()));
    fpga.write_register(REG_FLOW_SEQ, 1);

    std::vector<EventFlow> events;
    int n = fpga.read_flow_vectors(events, 4096);

    TEST("Round-trip returns all packed entries");
    CHECK(n == static_cast<int>(truth.size()) && events.size() == truth.size(),
          "expected %zu, got %d", truth.size(), n);

    TEST("Decoded values match within quantization error");
    bool within_eps = true;
    const float pos_eps = 1.0f / 4096.0f + 1e-5f;  // UQ4.12 LSB
    const float flow_eps = 1.0f / 256.0f + 1e-5f;  // Q8.8 LSB
    for (int i = 0; i < n; i++) {
        if (std::fabs(events[i].x - truth[i].x) > pos_eps ||
            std::fabs(events[i].y - truth[i].y) > pos_eps ||
            std::fabs(events[i].vx - truth[i].vx) > flow_eps ||
            std::fabs(events[i].vy - truth[i].vy) > flow_eps) {
            within_eps = false;
            break;
        }
    }
    CHECK(within_eps, "quantization error exceeded UQ4.12/Q8.8 bounds");

    TEST("Decoded flow still drives collision prediction end-to-end");
    CollisionPredictor::Config cfg;
    cfg.cluster_radius = 0.15f;
    cfg.flow_similarity_threshold = 0.5f;
    cfg.min_events_per_cluster = 10;
    cfg.safety_time_threshold = 2.0f;
    CollisionPredictor predictor(cfg);
    ThreatAssessment result = predictor.assess(events);
    CHECK(result.threat_detected && !result.objects.empty() && result.objects[0].ttc < 100.0f,
          "register-decoded flow failed to produce a finite-TTC threat");

    TEST("Zero count returns no events");
    fpga.write_register(REG_FLOW_COUNT, 0);
    CHECK(fpga.read_flow_vectors(events, 4096) == 0, "expected 0 events for zero count");

    TEST("Oversized count clamps to FLOW_MAX_OUT");
    fpga.write_register(REG_FLOW_COUNT, 5000);
    CHECK(fpga.read_flow_vectors(events, 4096) == FLOW_MAX_OUT, "expected clamp to %d",
          FLOW_MAX_OUT);

    TEST("max_events caps the returned batch");
    fpga.write_register(REG_FLOW_COUNT, static_cast<uint32_t>(truth.size()));
    CHECK(fpga.read_flow_vectors(events, 50) == 50, "expected 50 events");
}

// ===========================================================================
// EVASION CONTROLLER TESTS
// ===========================================================================

void test_evasion_controller_levels() {
    printf("\n=== Evasion controller level mapping ===\n");

    EvasionController::Config cfg;
    EvasionController controller(cfg);

    // Test: No threat → NONE level
    {
        ThreatAssessment assessment;
        assessment.threat_detected = false;
        assessment.objects = {};
        assessment.max_urgency = 0.0f;
        assessment.safe_bearing = 0.0f;

        EvasionCommand cmd = controller.compute_command(assessment);

        TEST("No threat → NONE level");
        CHECK(cmd.level == EvasionLevel::NONE, "expected NONE, got %s",
              evasion_level_name(cmd.level));

        TEST("No threat → zero velocity");
        CHECK(cmd.velocity_x == 0.0f && cmd.velocity_y == 0.0f && cmd.velocity_z == 0.0f,
              "expected zero velocity for NONE");
    }

    // Test: High urgency → EMERGENCY level
    {
        ThreatAssessment assessment;

        ObjectCluster obj;
        obj.id = 0;
        obj.center_x = 0.3f;
        obj.center_y = 0.5f;
        obj.spatial_extent = 0.1f;
        obj.ttc = 0.3f;  // Very close!
        obj.collision_urgency = 0.85f;
        obj.event_count = 100;

        assessment.objects.push_back(obj);
        assessment.max_urgency = 0.85f;
        assessment.safe_bearing = 2.0f;  // ~115° away from object
        assessment.safe_bearing_confidence = 0.7f;
        assessment.threat_detected = true;
        assessment.time_to_first_collision = 0.3f;

        EvasionCommand cmd = controller.compute_command(assessment);

        TEST("High urgency → EMERGENCY level");
        CHECK(cmd.level == EvasionLevel::EMERGENCY, "expected EMERGENCY, got %s",
              evasion_level_name(cmd.level));

        TEST("Emergency produces non-zero evasion velocity");
        CHECK(std::abs(cmd.velocity_x) > 0.1f || std::abs(cmd.velocity_y) > 0.1f,
              "expected evasion velocity in emergency");

        printf("    Emergency cmd: vx=%.2f vy=%.2f vz=%.2f yaw=%.2f\n", cmd.velocity_x,
               cmd.velocity_y, cmd.velocity_z, cmd.yaw_rate);
    }
}

void test_evasion_hysteresis() {
    printf("\n=== Evasion controller hysteresis ===\n");

    EvasionController::Config cfg;
    cfg.critical_to_warning_cycles = 2;
    cfg.warning_to_caution_cycles = 3;
    EvasionController controller(cfg);

    ThreatAssessment high_threat;
    high_threat.threat_detected = true;
    high_threat.max_urgency = 0.8f;

    ObjectCluster obj;
    obj.center_x = 0.3f;
    obj.center_y = 0.5f;
    obj.spatial_extent = 0.1f;
    obj.ttc = 0.4f;
    obj.collision_urgency = 0.8f;
    obj.event_count = 100;
    high_threat.objects.push_back(obj);
    high_threat.safe_bearing = -1.0f;
    high_threat.time_to_first_collision = 0.4f;

    ThreatAssessment low_threat = high_threat;
    low_threat.max_urgency = 0.05f;
    low_threat.threat_detected = false;
    low_threat.objects.clear();
    low_threat.time_to_first_collision = 100.0f;

    // Step 1: High threat should give EMERGENCY
    auto cmd1 = controller.compute_command(high_threat);
    TEST("Step 1: High threat → EMERGENCY");
    CHECK(cmd1.level == EvasionLevel::EMERGENCY, "expected EMERGENCY");

    // Step 2: Immediate low threat downgrades at most ONE level per cycle
    // (documented contract: "EMERGENCY → CRITICAL, never jump to NONE")
    auto cmd2 = controller.compute_command(low_threat);
    TEST("Step 2: Hysteresis limits downgrade to one step");
    CHECK(cmd2.level == EvasionLevel::CRITICAL, "hysteresis should step down to CRITICAL: got %s",
          evasion_level_name(cmd2.level));

    // Step 3: Sustained low threat should downgrade step by step
    auto cmd3 = controller.compute_command(low_threat);

    // Keep feeding low threats until we stabilize
    EvasionLevel last_level = EvasionLevel::EMERGENCY;
    std::vector<EvasionLevel> levels;
    levels.push_back(last_level);

    for (int i = 0; i < 20; i++) {
        auto cmd = controller.compute_command(low_threat);
        if (cmd.level != last_level) {
            levels.push_back(cmd.level);
            last_level = cmd.level;
        }
    }

    TEST("Hysteresis eventually downgrades to NONE");
    CHECK(last_level == EvasionLevel::NONE, "should reach NONE after sustained safety, got %s",
          evasion_level_name(last_level));

    printf("    Level transitions: ");
    for (auto lvl : levels) {
        printf("%s → ", evasion_level_name(lvl));
    }
    printf("STABLE\n");
}

void test_evasion_output_range() {
    printf("\n=== Evasion output range limits ===\n");

    EvasionController::Config cfg;
    cfg.max_horizontal_velocity = 5.0f;
    cfg.max_vertical_velocity = 3.0f;
    cfg.max_yaw_rate = 3.0f;
    EvasionController controller(cfg);

    ThreatAssessment urgent;
    urgent.threat_detected = true;
    urgent.max_urgency = 0.9f;

    ObjectCluster obj;
    obj.center_x = 0.1f;  // Far left
    obj.center_y = 0.5f;
    obj.spatial_extent = 0.15f;
    obj.ttc = 0.2f;
    obj.collision_urgency = 0.9f;
    obj.event_count = 200;
    urgent.objects.push_back(obj);
    urgent.safe_bearing = 1.5f;
    urgent.time_to_first_collision = 0.2f;

    EvasionCommand cmd = controller.compute_command(urgent);

    TEST("Velocity within max horizontal limit");
    float horiz = std::sqrt(cmd.velocity_x * cmd.velocity_x + cmd.velocity_y * cmd.velocity_y);
    CHECK(horiz <= cfg.max_horizontal_velocity * 1.01f, "horizontal velocity %.2f exceeds max %.2f",
          horiz, cfg.max_horizontal_velocity);

    TEST("Vertical velocity within max limit");
    CHECK(cmd.velocity_z >= 0.0f && cmd.velocity_z <= cfg.max_vertical_velocity * 1.01f,
          "vertical velocity %.2f out of range [0, %.2f]", cmd.velocity_z,
          cfg.max_vertical_velocity);

    TEST("Yaw rate within max limit");
    CHECK(std::abs(cmd.yaw_rate) <= cfg.max_yaw_rate * 1.01f, "yaw rate %.2f exceeds max %.2f",
          cmd.yaw_rate, cfg.max_yaw_rate);

    printf("    cmd: vx=%.2f vy=%.2f vz=%.2f yaw=%.2f |horiz|=%.2f\n", cmd.velocity_x,
           cmd.velocity_y, cmd.velocity_z, cmd.yaw_rate, horiz);
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    printf("============================================\n");
    printf("  ARM Collision Avoidance — Unit Tests\n");
    printf("============================================\n\n");

    printf("--- Collision Predictor Tests ---\n");
    test_empty_input();
    test_looming_object_detected();
    test_lateral_motion_no_false_alarm();
    test_random_noise_low_urgency();
    test_multi_object_clustering();
    test_safe_bearing();

    printf("\n--- FPGA Interface Tests ---\n");
    test_fpga_flow_register_roundtrip();

    printf("\n--- Evasion Controller Tests ---\n");
    test_evasion_controller_levels();
    test_evasion_hysteresis();
    test_evasion_output_range();

    printf("\n============================================\n");
    printf("  Results: %d/%d passed", g_passed, g_passed + g_failed);
    if (g_failed > 0) printf(", %d FAILED", g_failed);
    printf("\n============================================\n");

    return g_failed > 0 ? 1 : 0;
}