// arm/test_arm_cp.cpp — Kalman tracker integration tests
//
// Tests:
//   Lifecycle  : confirmation threshold, max_age pruning, empty-detections invariant
//   Accuracy   : two-object separation, Kalman position smoothing
//   Integration: drone_main.cpp Step 2b threat gate (exact replication)
//
// Build + run: make test
// Binary:      ./test_arm_cp   (exit 0 = all pass, 1 = any failure)

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>

// Include order matters: kalman_tracker.h uses ObjectCluster but does not
// include collision_predictor.h itself. Include predictor first.
#include "collision_predictor.h"
#include "kalman_tracker.h"

using namespace drone;

// ---------------------------------------------------------------------------
// Minimal test framework (no external dependencies)
// ---------------------------------------------------------------------------
static int g_total  = 0;
static int g_passed = 0;
static bool g_test_failed = false;
static const char* g_test_name = "";

#define RUN_TEST(name) \
    do { g_test_name = #name; g_test_failed = false; g_total++; \
         printf("  [ RUN  ] %s\n", g_test_name); } while(0)

#define END_TEST \
    do { if (!g_test_failed) { \
             printf("  [ PASS ] %s\n\n", g_test_name); g_passed++; \
         } else { \
             printf("  [ FAIL ] %s\n\n", g_test_name); } } while(0)

#define EXPECT_EQ(a, b) \
    do { if ((a) != (b)) { \
        printf("    EXPECT_EQ failed %s:%d\n      left:  %s = %d\n      right: %s = %d\n", \
               __FILE__, __LINE__, #a, (int)(a), #b, (int)(b)); \
        g_test_failed = true; } } while(0)

#define EXPECT_TRUE(cond) \
    do { if (!(cond)) { \
        printf("    EXPECT_TRUE failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_test_failed = true; } } while(0)

#define EXPECT_FALSE(cond) \
    do { if ((cond)) { \
        printf("    EXPECT_FALSE failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_test_failed = true; } } while(0)

#define EXPECT_NEAR(a, b, tol) \
    do { float _diff = std::abs((float)(a) - (float)(b)); \
         if (_diff > (float)(tol)) { \
        printf("    EXPECT_NEAR failed %s:%d\n      |%s - %s| = %.5f > %.5f\n", \
               __FILE__, __LINE__, #a, #b, _diff, (float)(tol)); \
        g_test_failed = true; } } while(0)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// Build a minimal ObjectCluster at (cx, cy) with realistic defaults
static ObjectCluster make_cluster(float cx, float cy,
                                  float urgency = 0.8f, float ttc = 0.5f) {
    ObjectCluster c{};
    c.id              = 0;
    c.center_x        = cx;
    c.center_y        = cy;
    c.spatial_extent  = 0.05f;
    c.mean_flow_mag   = 1.0f;
    c.mean_flow_dir   = 0.0f;
    c.ttc             = ttc;
    c.collision_urgency = urgency;
    c.event_count     = 50;
    return c;
}

// Replicate drone_main.cpp Step 2b exactly.
// Returns a ThreatAssessment post-tracker: objects replaced with KF-filtered
// confirmed tracks, threat_detected recomputed from confirmed track count.
static ThreatAssessment run_integration_step(
    KalmanTracker& tracker,
    std::vector<TrackedObject>& tracks,
    std::vector<ObjectCluster> raw_objects,
    bool raw_threat = true)
{
    ThreatAssessment assessment;
    assessment.objects            = raw_objects;
    assessment.threat_detected    = raw_threat;
    assessment.max_urgency        = raw_threat ? 0.8f : 0.0f;
    assessment.safe_bearing       = 0.0f;
    assessment.safe_bearing_confidence = 1.0f;
    assessment.time_to_first_collision =
        raw_threat ? 0.5f : std::numeric_limits<float>::max();

    // --- drone_main.cpp Step 2b (verbatim) ---
    tracker.update(assessment.objects, tracks);

    auto confirmed = tracker.get_confirmed_tracks(tracks);

    assessment.objects.clear();
    for (const auto& t : confirmed) {
        ObjectCluster oc = t.latest_cluster;
        oc.center_x = t.kf_state.x;
        oc.center_y = t.kf_state.y;
        assessment.objects.push_back(oc);
    }
    assessment.threat_detected = !confirmed.empty();
    // -----------------------------------------

    return assessment;
}

// ---------------------------------------------------------------------------
// Test 1: Confirmation requires exactly min_hits_to_confirm (default 3) hits.
//
// Verifies that a track is NOT visible to the evasion controller on frames
// 1 and 2, but IS visible on frame 3. Prevents single-frame noise from
// triggering evasive manoeuvres.
// ---------------------------------------------------------------------------
void test_confirm_requires_3_hits() {
    RUN_TEST(test_confirm_requires_3_hits);

    KalmanTracker tracker;
    std::vector<TrackedObject> tracks;
    std::vector<ObjectCluster> det = { make_cluster(0.5f, 0.5f) };

    // Frame 1 — age=1, below threshold (< 3)
    tracker.update(det, tracks);
    EXPECT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracker.get_confirmed_tracks(tracks).size(), 0u);

    // Frame 2 — age=2, still below threshold
    tracker.update(det, tracks);
    EXPECT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracker.get_confirmed_tracks(tracks).size(), 0u);

    // Frame 3 — age=3, now confirmed (age >= min_hits_to_confirm)
    tracker.update(det, tracks);
    EXPECT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracker.get_confirmed_tracks(tracks).size(), 1u);

    END_TEST;
}

// ---------------------------------------------------------------------------
// Test 2: Track is pruned after max_age (default 30) consecutive empty frames.
//
// Verifies the exact boundary:
//   - 30 empty frames: frames_since_update = 30, condition (> 30) is false → alive
//   - 31 empty frames: frames_since_update = 31, condition (> 30) is true  → pruned
//
// This ensures stale tracks from occluded objects don't persist indefinitely
// and never cause phantom evasion responses.
// ---------------------------------------------------------------------------
void test_track_pruned_after_max_age() {
    RUN_TEST(test_track_pruned_after_max_age);

    KalmanTracker tracker;
    std::vector<TrackedObject> tracks;
    std::vector<ObjectCluster> det   = { make_cluster(0.5f, 0.5f) };
    std::vector<ObjectCluster> empty = {};

    // Confirm the track (3 hits, frames_since_update reset to 0 on last match)
    for (int i = 0; i < 3; i++) tracker.update(det, tracks);
    EXPECT_EQ(tracks.size(), 1u);

    // 30 empty frames: track should still be alive at exactly the boundary
    for (int i = 0; i < 30; i++) tracker.update(empty, tracks);
    EXPECT_EQ(tracks.size(), 1u);

    // 31st empty frame: frames_since_update = 31 > 30 → pruned
    tracker.update(empty, tracks);
    EXPECT_EQ(tracks.size(), 0u);

    END_TEST;
}

// ---------------------------------------------------------------------------
// Test 3: Calling update() with empty detections never creates tracks.
//
// Verifies the always-call invariant: the main loop calls tracker.update()
// every tick (even when n_events == 0) so stale tracks age correctly.
// A side effect must never be track creation.
// ---------------------------------------------------------------------------
void test_no_detections_no_tracks() {
    RUN_TEST(test_no_detections_no_tracks);

    KalmanTracker tracker;
    std::vector<TrackedObject> tracks;
    std::vector<ObjectCluster> empty = {};

    for (int i = 0; i < 20; i++) tracker.update(empty, tracks);

    EXPECT_EQ(tracks.size(), 0u);
    EXPECT_EQ(tracker.get_confirmed_tracks(tracks).size(), 0u);

    END_TEST;
}

// ---------------------------------------------------------------------------
// Test 4: Two spatially separated detections produce two independent tracks.
//
// Verifies that Mahalanobis gating prevents cross-association when objects
// are 0.8 apart. Positions are checked after confirmation so we also verify
// that each track converged toward its own detection, not a blend.
// ---------------------------------------------------------------------------
void test_two_objects_two_tracks() {
    RUN_TEST(test_two_objects_two_tracks);

    KalmanTracker tracker;
    std::vector<TrackedObject> tracks;
    std::vector<ObjectCluster> dets = {
        make_cluster(0.1f, 0.1f, 0.9f),
        make_cluster(0.9f, 0.9f, 0.7f),
    };

    // 3 frames to confirm both tracks
    for (int i = 0; i < 3; i++) tracker.update(dets, tracks);

    EXPECT_EQ(tracks.size(), 2u);
    EXPECT_EQ(tracker.get_confirmed_tracks(tracks).size(), 2u);

    // Each confirmed track must be near its original detection, not the other one
    auto conf = tracker.get_confirmed_tracks(tracks);
    bool found_near_01 = false, found_near_09 = false;
    for (const auto& t : conf) {
        if (std::abs(t.kf_state.x - 0.1f) < 0.05f &&
            std::abs(t.kf_state.y - 0.1f) < 0.05f)
            found_near_01 = true;
        if (std::abs(t.kf_state.x - 0.9f) < 0.05f &&
            std::abs(t.kf_state.y - 0.9f) < 0.05f)
            found_near_09 = true;
    }
    EXPECT_TRUE(found_near_01);
    EXPECT_TRUE(found_near_09);

    END_TEST;
}

// ---------------------------------------------------------------------------
// Test 5: Kalman filter reduces position noise below raw measurement noise.
//
// Feeds alternating +/- 0.04 noise around a true position of (0.5, 0.5).
// After 30 frames the KF estimate should have smaller error than the raw
// noise amplitude — confirming the filter is actively smoothing and not
// just passing measurements through.
// ---------------------------------------------------------------------------
void test_kalman_smoothing() {
    RUN_TEST(test_kalman_smoothing);

    KalmanTracker tracker;
    std::vector<TrackedObject> tracks;
    const float true_pos = 0.5f;
    const float noise    = 0.04f;

    for (int i = 0; i < 30; i++) {
        float sign = (i % 2 == 0) ? 1.0f : -1.0f;
        std::vector<ObjectCluster> det = {
            make_cluster(true_pos + sign * noise, true_pos + sign * noise)
        };
        tracker.update(det, tracks);
    }

    auto conf = tracker.get_confirmed_tracks(tracks);
    EXPECT_EQ(conf.size(), 1u);

    if (!conf.empty()) {
        // Filtered error should be strictly less than the raw noise amplitude
        float err_x = std::abs(conf[0].kf_state.x - true_pos);
        float err_y = std::abs(conf[0].kf_state.y - true_pos);
        EXPECT_TRUE(err_x < noise);
        EXPECT_TRUE(err_y < noise);
    }

    END_TEST;
}

// ---------------------------------------------------------------------------
// Test 6: Integration — threat gate replicates drone_main.cpp Step 2b.
//
// This test runs the exact integration logic we added to drone_main.cpp:
//   - Frames 1-2: raw threat present, but no confirmed tracks yet
//                 → assessment.threat_detected must be false
//                 → assessment.objects must be empty (evader is not triggered)
//   - Frame 3:    track confirmed
//                 → assessment.threat_detected must be true
//                 → assessment.objects has 1 entry with KF-filtered position
//   - After max_age+1 empty frames: track pruned
//                 → assessment.threat_detected must be false again
// ---------------------------------------------------------------------------
void test_integration_threat_gate() {
    RUN_TEST(test_integration_threat_gate);

    KalmanTracker tracker;
    std::vector<TrackedObject> tracks;
    std::vector<ObjectCluster> det   = { make_cluster(0.5f, 0.5f, 0.8f) };
    std::vector<ObjectCluster> empty = {};

    // Frames 1 and 2: raw threat, but evader must not see it yet
    for (int frame = 1; frame <= 2; frame++) {
        auto result = run_integration_step(tracker, tracks, det, true);
        EXPECT_FALSE(result.threat_detected);
        EXPECT_EQ(result.objects.size(), 0u);
    }

    // Frame 3: confirmed — evader now sees the threat
    {
        auto result = run_integration_step(tracker, tracks, det, true);
        EXPECT_TRUE(result.threat_detected);
        EXPECT_EQ(result.objects.size(), 1u);
        if (!result.objects.empty()) {
            // Reported position should be close to the detection (KF hasn't drifted far)
            EXPECT_NEAR(result.objects[0].center_x, 0.5f, 0.02f);
            EXPECT_NEAR(result.objects[0].center_y, 0.5f, 0.02f);
        }
    }

    // max_age+1 empty frames: track must be pruned, threat cleared
    ThreatAssessment last;
    for (int i = 0; i <= 31; i++) {
        last = run_integration_step(tracker, tracks, empty, false);
    }
    EXPECT_FALSE(last.threat_detected);
    EXPECT_EQ(last.objects.size(), 0u);

    END_TEST;
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
int main() {
    printf("ARM Kalman Tracker — Integration Tests\n");
    printf("=======================================\n\n");

    test_confirm_requires_3_hits();
    test_track_pruned_after_max_age();
    test_no_detections_no_tracks();
    test_two_objects_two_tracks();
    test_kalman_smoothing();
    test_integration_threat_gate();

    printf("=======================================\n");
    printf("Results: %d/%d passed\n\n", g_passed, g_total);

    return (g_passed == g_total) ? 0 : 1;
}
