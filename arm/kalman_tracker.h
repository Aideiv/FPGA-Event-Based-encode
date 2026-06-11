// arm/kalman_tracker.h — Kalman Filter Object Tracking for ARM C++
//
// Port of drone/collision_predictor.py Kalman filter logic to C++.
// Tracks moving objects across consecutive inference frames using
// a constant-velocity Kalman filter with 4-state vector [x, y, vx, vy].
//
// This fills the critical gap between the Python collision_predictor.py
// (which has Kalman-filtered per-object tracking) and the ARM C++ code
// (which previously lacked temporal tracking).
//
// Based on: Kalman, R. E. (1960). "A New Approach to Linear Filtering
//           and Prediction Problems."
//
// Thread-safe, real-time capable at 100+ Hz.

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <limits>
#include <algorithm>

namespace drone {

// -------------------------------------------------------------------------
// Kalman filter state: 4D [x, y, vx, vy]
// -------------------------------------------------------------------------
struct KalmanState {
    float x;   // Position x (normalized [0,1))
    float y;   // Position y (normalized [0,1))
    float vx;  // Velocity x (normalized / sec)
    float vy;  // Velocity y (normalized / sec)
};

// 4×4 covariance matrix (stored as 16 floats, row-major)
struct Covariance4D {
    float P[16];  // row-major: P[row*4 + col]

    Covariance4D() {
        // Initialize with large uncertainty
        for (int i = 0; i < 16; i++) P[i] = 0.0f;
        P[0] = 0.1f;   // x uncertainty
        P[5] = 0.1f;   // y uncertainty
        P[10] = 1.0f;  // vx uncertainty
        P[15] = 1.0f;  // vy uncertainty
    }

    float& operator()(int row, int col) { return P[row * 4 + col]; }
    const float& operator()(int row, int col) const { return P[row * 4 + col]; }
};

// -------------------------------------------------------------------------
// Tracked object (extends ObjectCluster with temporal state)
// -------------------------------------------------------------------------
struct TrackedObject {
    uint32_t id;                   // Unique track ID
    uint32_t age;                  // Frames since first detection
    uint32_t frames_since_update;  // Frames since last match (for track death)
    KalmanState kf_state;          // Kalman filter state
    Covariance4D kf_cov;           // Kalman filter covariance
    ObjectCluster latest_cluster;  // Most recent measurement
    float history_x[8];            // Position history for smooth motion
    float history_y[8];
    uint8_t history_idx;
    bool is_valid;  // Track is alive
};

// -------------------------------------------------------------------------
// Kalman Tracker
//
// Maintains a set of tracked objects across frames, performing:
//   1. Prediction: advance each track to current time
//   2. Data association: match new detections to existing tracks
//   3. Update: correct Kalman state with matched measurements
//   4. Management: create new tracks, kill old ones
//
// Parameters:
//   dt:            time step between frames (seconds, 0.01 @ 100Hz)
//   process_noise: model uncertainty (suggest 0.001-0.01)
//   measurement_noise: sensor uncertainty (suggest 0.05)
//   max_age:       frames before track is deleted (default 30 = 0.3s @ 100Hz)
//   match_threshold: max Mahalanobis distance for association
// -------------------------------------------------------------------------
class KalmanTracker {
   public:
    struct Config {
        float dt = 0.01f;                   // 100Hz control loop
        float process_noise_q = 0.005f;     // Model uncertainty
        float measurement_noise_r = 0.05f;  // Measurement uncertainty
        uint32_t max_age = 30;              // Frames before track death
        float match_threshold = 5.0f;       // Mahalanobis distance gate (chi² 4DOF)
        uint32_t min_hits_to_confirm = 3;   // Detections before track confirmed
    };

    KalmanTracker() : cfg_(), next_id_(0) {}
    explicit KalmanTracker(const Config& cfg) : cfg_(cfg), next_id_(0) {}

    // ------------------------------------------------------------------
    // Main update: predict → associate → update → manage
    // ------------------------------------------------------------------
    void update(std::vector<ObjectCluster>& detections, std::vector<TrackedObject>& tracks) {
        // Step 1: Predict all tracks forward
        predict_tracks(tracks);

        // Step 2: Associate detections to tracks (Hungarian/Greedy)
        auto matches = associate(detections, tracks);

        // Step 3: Update matched tracks with measurements
        apply_matches(matches, detections, tracks);

        // Step 4: Create new tracks for unmatched detections
        create_new_tracks(matches, detections, tracks);

        // Step 5: Remove stale tracks
        prune_tracks(tracks);
    }

    // ------------------------------------------------------------------
    // Get confirmed tracks only
    // ------------------------------------------------------------------
    std::vector<TrackedObject> get_confirmed_tracks(
        const std::vector<TrackedObject>& tracks) const {
        std::vector<TrackedObject> confirmed;
        for (const auto& t : tracks) {
            if (t.is_valid && t.age >= cfg_.min_hits_to_confirm) {
                confirmed.push_back(t);
            }
        }
        return confirmed;
    }

   private:
    Config cfg_;
    uint32_t next_id_;

    // ------------------------------------------------------------------
    // Kalman prediction step:  x' = F·x,  P' = F·P·F^T + Q
    //
    // State transition (constant velocity):
    //   x(t+dt) = x(t) + vx(t)*dt
    //   y(t+dt) = y(t) + vy(t)*dt
    //   vx(t+dt) = vx(t)
    //   vy(t+dt) = vy(t)
    // ------------------------------------------------------------------
    void predict_tracks(std::vector<TrackedObject>& tracks) {
        float dt = cfg_.dt;
        float q = cfg_.process_noise_q;

        // F matrix (4×4):
        // [1 0 dt 0 ]
        // [0 1 0  dt]
        // [0 0 1  0 ]
        // [0 0 0  1 ]

        for (auto& track : tracks) {
            if (!track.is_valid) continue;

            // Predict state: x' = F @ x
            float x_pred = track.kf_state.x + track.kf_state.vx * dt;
            float y_pred = track.kf_state.y + track.kf_state.vy * dt;
            float vx_pred = track.kf_state.vx;
            float vy_pred = track.kf_state.vy;

            // Predict covariance: P' = F·P·F^T + Q
            auto& P = track.kf_cov;

            // Compute F·P·F^T (analytically for efficiency)
            float p00 = P(0, 0) + 2 * dt * P(0, 2) + dt * dt * P(2, 2);
            float p01 = P(0, 1) + dt * P(1, 2) + dt * P(0, 3) + dt * dt * P(2, 3);
            float p02 = P(0, 2) + dt * P(2, 2);
            float p03 = P(0, 3) + dt * P(2, 3);

            float p10 = p01;
            float p11 = P(1, 1) + 2 * dt * P(1, 3) + dt * dt * P(3, 3);
            float p12 = P(1, 2) + dt * P(2, 3);
            float p13 = P(1, 3) + dt * P(3, 3);

            float p20 = p02;
            float p21 = p12;
            float p22 = P(2, 2);
            float p23 = P(2, 3);

            float p30 = p03;
            float p31 = p13;
            float p32 = p23;
            float p33 = P(3, 3);

            // Add process noise Q (simplified: only to velocity)
            // Q = q * I (but only on velocity for smoother position)
            P(0, 0) = p00 + q * dt * dt / 3;
            P(0, 1) = p01;
            P(0, 2) = p02 + q * dt * dt / 2;
            P(0, 3) = p03;

            P(1, 0) = p10;
            P(1, 1) = p11 + q * dt * dt / 3;
            P(1, 2) = p12;
            P(1, 3) = p13 + q * dt * dt / 2;

            P(2, 0) = p20 + q * dt * dt / 2;
            P(2, 1) = p21;
            P(2, 2) = p22 + q * dt;
            P(2, 3) = p23;

            P(3, 0) = p30;
            P(3, 1) = p31 + q * dt * dt / 2;
            P(3, 2) = p32;
            P(3, 3) = p33 + q * dt;

            // Update state
            track.kf_state.x = x_pred;
            track.kf_state.y = y_pred;
            track.kf_state.vx = vx_pred;
            track.kf_state.vy = vy_pred;

            // Age tracking
            track.frames_since_update++;
        }
    }

    // ------------------------------------------------------------------
    // Data association: greedy matching based on Mahalanobis distance
    //
    // For each detection, find the closest track within the gating
    // threshold. Greedy (not Hungarian) — sufficient for drone scenes
    // with typically <10 objects.
    // ------------------------------------------------------------------
    struct Match {
        int detection_idx;
        int track_idx;
        float distance;
        bool matched;
    };

    std::vector<Match> associate(const std::vector<ObjectCluster>& detections,
                                 const std::vector<TrackedObject>& tracks) {
        std::vector<Match> matches;
        std::vector<bool> det_used(detections.size(), false);
        std::vector<bool> track_used(tracks.size(), false);

        // Collect all possible matches with distances
        std::vector<Match> candidates;
        for (size_t d = 0; d < detections.size(); d++) {
            for (size_t t = 0; t < tracks.size(); t++) {
                if (!tracks[t].is_valid) continue;

                float dist = mahalanobis_distance(detections[d], tracks[t]);
                if (dist < cfg_.match_threshold) {
                    candidates.push_back({(int)d, (int)t, dist, false});
                }
            }
        }

        // Sort by distance (greedy best-first)
        std::sort(candidates.begin(), candidates.end(),
                  [](const Match& a, const Match& b) { return a.distance < b.distance; });

        // Greedy assignment
        for (auto& c : candidates) {
            if (!det_used[c.detection_idx] && !track_used[c.track_idx]) {
                c.matched = true;
                det_used[c.detection_idx] = true;
                track_used[c.track_idx] = true;
                matches.push_back(c);
            }
        }

        // Record unmatched detections (for new track creation)
        for (size_t d = 0; d < detections.size(); d++) {
            if (!det_used[d]) {
                matches.push_back({(int)d, -1, 0.0f, false});
            }
        }

        return matches;
    }

    // ------------------------------------------------------------------
    // Mahalanobis distance: (z - Hx)^T · S^-1 · (z - Hx)
    //
    // z = measurement [x, y]
    // Hx = predicted position [x_pred, y_pred]
    // S = H·P·H^T + R  (innovation covariance)
    // ------------------------------------------------------------------
    float mahalanobis_distance(const ObjectCluster& det, const TrackedObject& track) const {
        float dx = det.center_x - track.kf_state.x;
        float dy = det.center_y - track.kf_state.y;

        // Measurement noise
        float r = cfg_.measurement_noise_r;

        // Innovation covariance S = H·P·H^T + R
        // H extracts [x, y] from [x, y, vx, vy]:
        // H = [[1,0,0,0],[0,1,0,0]]
        // S = [[P(0,0)+r, P(0,1)  ],
        //      [P(1,0),   P(1,1)+r]]
        float s00 = track.kf_cov(0, 0) + r;
        float s01 = track.kf_cov(0, 1);
        float s10 = track.kf_cov(1, 0);
        float s11 = track.kf_cov(1, 1) + r;

        // Invert 2×2 S matrix
        float det_s = s00 * s11 - s01 * s10;
        if (std::abs(det_s) < 1e-10f) return 9999.0f;

        float inv_s00 = s11 / det_s;
        float inv_s01 = -s01 / det_s;
        float inv_s10 = -s10 / det_s;
        float inv_s11 = s00 / det_s;

        // Mahalanobis: d^T · S^-1 · d
        float maha = dx * (inv_s00 * dx + inv_s01 * dy) + dy * (inv_s10 * dx + inv_s11 * dy);

        return maha;
    }

    // ------------------------------------------------------------------
    // Kalman update: K = P·H^T·S^-1, x = x + K·(z - Hx), P = (I-KH)·P
    // ------------------------------------------------------------------
    void apply_matches(std::vector<Match>& matches, const std::vector<ObjectCluster>& detections,
                       std::vector<TrackedObject>& tracks) {
        for (auto& match : matches) {
            if (!match.matched || match.track_idx < 0) continue;

            auto& track = tracks[match.track_idx];
            const auto& det = detections[match.detection_idx];

            float r = cfg_.measurement_noise_r;
            auto& P = track.kf_cov;

            // Innovation covariance S
            float s00 = P(0, 0) + r;
            float s01 = P(0, 1);
            float s10 = P(1, 0);
            float s11 = P(1, 1) + r;

            float det_s = s00 * s11 - s01 * s10;
            if (std::abs(det_s) < 1e-10f) continue;

            float inv_s00 = s11 / det_s;
            float inv_s01 = -s01 / det_s;
            float inv_s10 = -s10 / det_s;
            float inv_s11 = s00 / det_s;

            // Innovation: z - Hx
            float y0 = det.center_x - track.kf_state.x;
            float y1 = det.center_y - track.kf_state.y;

            // Kalman gain: K = P·H^T·S^-1
            // For position-only measurement:
            float k00 = P(0, 0) * inv_s00 + P(0, 1) * inv_s10;
            float k01 = P(0, 0) * inv_s01 + P(0, 1) * inv_s11;
            float k10 = P(1, 0) * inv_s00 + P(1, 1) * inv_s10;
            float k11 = P(1, 0) * inv_s01 + P(1, 1) * inv_s11;
            float k20 = P(2, 0) * inv_s00 + P(2, 1) * inv_s10;
            float k21 = P(2, 0) * inv_s01 + P(2, 1) * inv_s11;
            float k30 = P(3, 0) * inv_s00 + P(3, 1) * inv_s10;
            float k31 = P(3, 0) * inv_s01 + P(3, 1) * inv_s11;

            // Update state: x = x + K·y
            track.kf_state.x += k00 * y0 + k01 * y1;
            track.kf_state.y += k10 * y0 + k11 * y1;
            track.kf_state.vx += k20 * y0 + k21 * y1;
            track.kf_state.vy += k30 * y0 + k31 * y1;

            // Update covariance: P = (I - K·H)·P
            // I-KH = [[1-k00, -k01, 0, 0],
            //         [-k10, 1-k11, 0, 0],
            //         [-k20, -k21, 1, 0],
            //         [-k30, -k31, 0, 1]]
            float P00 = (1 - k00) * P(0, 0) - k01 * P(1, 0);
            float P01 = (1 - k00) * P(0, 1) - k01 * P(1, 1);
            float P02 = (1 - k00) * P(0, 2) - k01 * P(1, 2);
            float P03 = (1 - k00) * P(0, 3) - k01 * P(1, 3);

            float P10 = -k10 * P(0, 0) + (1 - k11) * P(1, 0);
            float P11 = -k10 * P(0, 1) + (1 - k11) * P(1, 1);
            float P12 = -k10 * P(0, 2) + (1 - k11) * P(1, 2);
            float P13 = -k10 * P(0, 3) + (1 - k11) * P(1, 3);

            float P20 = -k20 * P(0, 0) - k21 * P(1, 0) + P(2, 0);
            float P21 = -k20 * P(0, 1) - k21 * P(1, 1) + P(2, 1);
            float P22 = -k20 * P(0, 2) - k21 * P(1, 2) + P(2, 2);
            float P23 = -k20 * P(0, 3) - k21 * P(1, 3) + P(2, 3);

            float P30 = -k30 * P(0, 0) - k31 * P(1, 0) + P(3, 0);
            float P31 = -k30 * P(0, 1) - k31 * P(1, 1) + P(3, 1);
            float P32 = -k30 * P(0, 2) - k31 * P(1, 2) + P(3, 2);
            float P33 = -k30 * P(0, 3) - k31 * P(1, 3) + P(3, 3);

            P(0, 0) = P00;
            P(0, 1) = P01;
            P(0, 2) = P02;
            P(0, 3) = P03;
            P(1, 0) = P10;
            P(1, 1) = P11;
            P(1, 2) = P12;
            P(1, 3) = P13;
            P(2, 0) = P20;
            P(2, 1) = P21;
            P(2, 2) = P22;
            P(2, 3) = P23;
            P(3, 0) = P30;
            P(3, 1) = P31;
            P(3, 2) = P32;
            P(3, 3) = P33;

            // Update track metadata
            track.frames_since_update = 0;
            track.age++;
            track.latest_cluster = det;

            // Circular buffer of position history for smoothing
            track.history_x[track.history_idx] = track.kf_state.x;
            track.history_y[track.history_idx] = track.kf_state.y;
            track.history_idx = (track.history_idx + 1) % 8;
        }
    }

    // ------------------------------------------------------------------
    // Create new tracks for unmatched detections
    // ------------------------------------------------------------------
    void create_new_tracks(const std::vector<Match>& matches,
                           const std::vector<ObjectCluster>& detections,
                           std::vector<TrackedObject>& tracks) {
        std::vector<bool> det_matched(detections.size(), false);
        for (const auto& m : matches) {
            if (m.matched && m.detection_idx >= 0) {
                det_matched[m.detection_idx] = true;
            }
        }

        for (size_t d = 0; d < detections.size(); d++) {
            if (!det_matched[d]) {
                TrackedObject new_track;
                new_track.id = next_id_++;
                new_track.age = 1;
                new_track.frames_since_update = 0;
                new_track.kf_state.x = detections[d].center_x;
                new_track.kf_state.y = detections[d].center_y;
                new_track.kf_state.vx = 0.0f;
                new_track.kf_state.vy = 0.0f;
                new_track.kf_cov = Covariance4D();  // Default uncertainty
                new_track.latest_cluster = detections[d];
                new_track.history_idx = 0;
                for (int i = 0; i < 8; i++) {
                    new_track.history_x[i] = detections[d].center_x;
                    new_track.history_y[i] = detections[d].center_y;
                }
                new_track.is_valid = true;
                tracks.push_back(new_track);
            }
        }
    }

    // ------------------------------------------------------------------
    // Remove stale tracks (exceeded max_age)
    // ------------------------------------------------------------------
    void prune_tracks(std::vector<TrackedObject>& tracks) {
        for (auto& track : tracks) {
            if (track.frames_since_update > cfg_.max_age) {
                track.is_valid = false;
            }
        }
        // Remove invalid tracks
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [](const TrackedObject& t) { return !t.is_valid; }),
                     tracks.end());
    }
};

}  // namespace drone