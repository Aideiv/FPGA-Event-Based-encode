// arm/drone_main.cpp — ARM Processing System Main Control Loop
//
// Runs on Zynq PS (ARM Cortex-A53 or R5). Integrates the collision
// prediction and evasion control pipeline with the FPGA fabric via
// AXI4-Stream and AXI4-Lite interfaces.
//
// Architecture:
//   while(1):
//     1. Read flow vectors from FPGA encoder output (AXI4-Lite FLOW bundle)
//     2. Convert to EventFlow structs
//     3. Run CollisionPredictor::assess() → ThreatAssessment
//     4. Run EvasionController::compute_command() → EvasionCommand
//     5. Write velocity commands to FPGA PWM controller (AXI4-Lite)
//     6. Sleep for control period (100Hz default)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <atomic>
#include <vector>

#include "collision_predictor.h"
#include "evasion_controller.h"
#include "fpga_interface.h"
#include "kalman_tracker.h"

// FreeRTOS or bare-metal alternatives
#ifdef FREERTOS
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#define SLEEP_MS(ms) vTaskDelay(pdMS_TO_TICKS(ms))
#else
#include <chrono>
#include <thread>
#define SLEEP_MS(ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms))
#endif

using namespace drone;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
static std::atomic<bool> g_running{true};
static std::atomic<bool> g_motors_armed{false};

// ---------------------------------------------------------------------------
// Telemetry / logging
// ---------------------------------------------------------------------------
void log_telemetry(const ThreatAssessment& assess, const EvasionCommand& cmd, size_t n_tracks,
                   size_t n_confirmed) {
    static uint32_t frame = 0;
    frame++;

    printf("[%06u] ", frame);

    if (assess.threat_detected) {
        printf("THREAT ur=%.2f safe=%.2f° obj=%zu ttc=%.2fs | ", assess.max_urgency,
               assess.safe_bearing * 180.0f / M_PI, assess.objects.size(),
               assess.time_to_first_collision);
    } else {
        printf("CLEAR                               | ");
    }

    printf("%-9s vx=%+5.2f vy=%+5.2f vz=%+5.2f yaw=%+5.2f | trk=%zu conf=%zu\n",
           evasion_level_name(cmd.level), cmd.velocity_x, cmd.velocity_y, cmd.velocity_z,
           cmd.yaw_rate, n_tracks, n_confirmed);
}

// ---------------------------------------------------------------------------
// Signal handler for graceful shutdown
// ---------------------------------------------------------------------------
void signal_handler(int sig) {
    printf("\nShutting down... (signal %d)\n", sig);
    g_running = false;
}

// ---------------------------------------------------------------------------
// Main control loop
// ---------------------------------------------------------------------------
int main(int /*argc*/, char** /*argv*/) {
    printf("=== Event-Based Drone Collision Avoidance ===\n");
    printf("ARM Processing System — Zynq MPSoC\n\n");

    // Register signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize FPGA interface
    FpgaInterface fpga;

    // Initialize collision prediction pipeline
    CollisionPredictor::Config pred_cfg;
    pred_cfg.safety_time_threshold = 1.5f;
    pred_cfg.critical_time_threshold = 0.5f;
    pred_cfg.cluster_radius = 0.05f;
    pred_cfg.flow_similarity_threshold = 0.7f;
    pred_cfg.min_events_per_cluster = 30;
    CollisionPredictor predictor(pred_cfg);

    // Initialize evasion controller
    EvasionController::Config evade_cfg;
    evade_cfg.max_horizontal_velocity = 5.0f;
    evade_cfg.max_vertical_velocity = 2.0f;
    evade_cfg.repulsion_strength = 1.0f;
    EvasionController evader(evade_cfg);

    // Initialize Kalman tracker
    // Config defaults are tuned for 100Hz — no overrides needed:
    //   dt=0.01s, process_noise=0.005, meas_noise=0.05
    //   max_age=30 frames (0.3s), min_hits_to_confirm=3 frames (30ms)
    KalmanTracker tracker;

    // Persistent track list — lives across loop iterations, passed by ref into tracker.update()
    std::vector<TrackedObject> tracks;

    // Control loop timing
    const int CONTROL_FREQ_HZ = 100;  // 100Hz control loop
    const int CONTROL_PERIOD_MS = 1000 / CONTROL_FREQ_HZ;

    // Enable FPGA pipeline (motors initially disarmed)
    fpga.enable(false);
    g_motors_armed = false;

    printf("FPGA pipeline enabled. Motors DISARMED.\n");
    printf("Control loop: %d Hz (%d ms period)\n", CONTROL_FREQ_HZ, CONTROL_PERIOD_MS);
    printf("Press Ctrl+C to stop.\n\n");

    // Main control loop
    std::vector<EventFlow> events;
    events.reserve(4096);

    uint32_t arm_countdown = CONTROL_FREQ_HZ * 2;  // Arm after 2 seconds

    while (g_running) {
        // Step 1: Read flow vectors from FPGA encoder
        int n_events = fpga.read_flow_vectors(events, 4096);

        // Step 2: Run collision prediction
        ThreatAssessment assessment;
        if (n_events > 0) {
            assessment = predictor.assess(events);
        }

        // Step 2b: Kalman tracker — always called, even with zero detections.
        // Empty detections = no new tracks created, but existing tracks age toward pruning.
        // Skipping this when n_events==0 would make tracks immortal during quiet frames.
        tracker.update(assessment.objects, tracks);

        // Confirmed tracks have seen >= 3 consecutive hits (30ms at 100Hz).
        // Single-frame blips are suppressed — the evasion controller never sees them.
        auto confirmed = tracker.get_confirmed_tracks(tracks);

        // Replace raw detections with Kalman-filtered confirmed-track positions.
        // Each ObjectCluster retains its original TTC/urgency metadata (latest_cluster)
        // but center_x/y are overridden with the smoothed KF estimate.
        assessment.objects.clear();
        for (const auto& t : confirmed) {
            ObjectCluster oc = t.latest_cluster;  // carries TTC, flow, size metadata
            oc.center_x = t.kf_state.x;           // KF-filtered position
            oc.center_y = t.kf_state.y;
            assessment.objects.push_back(oc);
        }
        // threat_detected now reflects confirmed track presence, not raw noisy detections
        assessment.threat_detected = !confirmed.empty();

        // Step 3: Compute evasion command
        EvasionCommand cmd = evader.compute_command(assessment);

        // Step 4: Send velocity command to FPGA PWM module
        if (g_motors_armed) {
            fpga.write_velocity_command(cmd);
        }

        // Step 5: Telemetry
        log_telemetry(assessment, cmd, tracks.size(), confirmed.size());

        // Auto-arm after startup countdown
        if (!g_motors_armed && arm_countdown > 0) {
            arm_countdown--;
            if (arm_countdown == 0) {
                g_motors_armed = true;
                fpga.write_register(REG_ENABLE_MOTORS, 1);
                printf("*** MOTORS ARMED ***\n");
            }
        }

        // Step 6: Sleep for control period
        SLEEP_MS(CONTROL_PERIOD_MS);
    }

    // Graceful shutdown
    printf("\nDisabling motors and FPGA pipeline...\n");
    fpga.disable();
    printf("Shutdown complete.\n");

    return 0;
}