// arm/collision_predictor.h — ARM C++ Collision Predictor
//
// Runs on Zynq Processing System (ARM Cortex-A53 or R5).
// Computes Time-to-Collision (TTC) and threat assessment from per-event 
// flow vectors received from the FPGA fabric via AXI-stream.
//
// Based on Lee's τ hypothesis:  τ = θ / θ̇  (angular size / expansion rate)
// In event camera terms: θ ∝ cluster spatial extent, θ̇ ∝ mean outward flow
//
// Thread-safe, real-time capable at 100+ Hz assessment rate.

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <array>

namespace drone {

// -------------------------------------------------------------------------
// Per-event flow vector (received from FPGA encoder)
// -------------------------------------------------------------------------
struct EventFlow {
    float vx;       // Flow x component (pixels/frame)
    float vy;       // Flow y component (pixels/frame)
    float x;        // Event x coordinate (normalized [0,1))
    float y;        // Event y coordinate (normalized [0,1))
    uint32_t t;     // Timestamp
};

// -------------------------------------------------------------------------
// Detected moving object cluster
// -------------------------------------------------------------------------
struct ObjectCluster {
    uint32_t id;
    float center_x;         // Cluster center x (image space)
    float center_y;         // Cluster center y
    float spatial_extent;   // Max distance from center (radians)
    float mean_flow_mag;    // Mean flow magnitude (expansion rate)
    float mean_flow_dir;    // Mean flow direction (radians, 0=right)
    float ttc;              // Time to collision (seconds, ∞ if not looming)
    float collision_urgency; // [0, 1] normalized threat urgency
    uint32_t event_count;   // Number of events in this cluster
};

// -------------------------------------------------------------------------
// Threat assessment result
// -------------------------------------------------------------------------
struct ThreatAssessment {
    std::vector<ObjectCluster> objects;
    float max_urgency;              // Highest collision_urgency among all objects
    float safe_bearing;             // Direction (radians) of clearest path
    float safe_bearing_confidence;  // How clear that path is [0, 1]
    bool threat_detected;
    float time_to_first_collision;  // TTC of closest object
};

// -------------------------------------------------------------------------
// Collision Predictor
//
// Algorithm:
//   1. Receive flow vectors from FPGA encoder ring buffer
//   2. Cluster events with coherent flow (spatial + directional similarity)
//   3. For each cluster: compute TTC via looming (τ = θ/θ̇)
//      - θ = cluster radius in angular coordinates
//      - θ̇ = mean outward radial flow at cluster boundary
//   4. Compute urgency:  urgency = clamp(1 - τ/τ_critical, 0, 1)
//      where τ_critical = 1.0 seconds (configurable)
//   5. Find safe bearings: gaps between threat cones with margin
//   6. Output assessment to evasion controller
// -------------------------------------------------------------------------
class CollisionPredictor {
public:
    struct Config {
        float safety_time_threshold = 1.0f;     // TTC below this → evade
        float critical_time_threshold = 0.5f;   // TTC below this → hard evade
        float cluster_radius = 0.05f;           // Spatial clustering radius (normalized)
        float flow_similarity_threshold = 0.7f; // Cosine similarity for flow grouping
        float min_events_per_cluster = 50;      // Filter tiny clusters
        float safe_bearing_margin = 0.1f;       // Angular margin around objects (radians)
        float max_flow_magnitude = 5.0f;        // Clamp outlier flows
        float camera_fov = 1.2f;                // Field of view (radians) ~68°
    };

    CollisionPredictor() : cfg_() {}
    explicit CollisionPredictor(const Config& cfg) : cfg_(cfg) {}

    // ------------------------------------------------------------------
    // Main assessment function
    //   events: per-event flow vectors from FPGA encoder
    //   returns: threat assessment with object clusters and safe bearings
    // ------------------------------------------------------------------
    ThreatAssessment assess(const std::vector<EventFlow>& events) {
        ThreatAssessment result;
        result.max_urgency = 0.0f;
        result.safe_bearing = 0.0f;
        result.safe_bearing_confidence = 1.0f;
        result.threat_detected = false;
        result.time_to_first_collision = std::numeric_limits<float>::max();

        if (events.empty()) {
            return result;
        }

        // Step 1: Cluster events by spatial proximity + flow coherence
        result.objects = cluster_events(events);

        // Step 2: Compute TTC and urgency per cluster
        for (auto& obj : result.objects) {
            compute_ttc(obj, events);
            compute_urgency(obj);

            if (obj.collision_urgency > result.max_urgency) {
                result.max_urgency = obj.collision_urgency;
            }
            if (obj.ttc < result.time_to_first_collision) {
                result.time_to_first_collision = obj.ttc;
            }
        }

        result.threat_detected = result.max_urgency > 0.1f;

        // Step 3: Find safe bearing
        compute_safe_bearing(result);

        return result;
    }

private:
    Config cfg_;

    // ------------------------------------------------------------------
    // Spatial + flow clustering
    // Simple connected-components with flow direction similarity
    // ------------------------------------------------------------------
    std::vector<ObjectCluster> cluster_events(const std::vector<EventFlow>& events) {
        std::vector<ObjectCluster> clusters;
        std::vector<bool> visited(events.size(), false);

        for (size_t i = 0; i < events.size(); ++i) {
            if (visited[i]) continue;

            // Ignore events with near-zero flow (static background)
            float mag = std::sqrt(events[i].vx * events[i].vx + 
                                 events[i].vy * events[i].vy);
            if (mag < 0.01f) continue;

            // Start new cluster
            ObjectCluster cluster;
            cluster.id = static_cast<uint32_t>(clusters.size());
            cluster.center_x = 0.0f;
            cluster.center_y = 0.0f;
            cluster.event_count = 0;
            cluster.mean_flow_mag = 0.0f;
            cluster.mean_flow_dir = 0.0f;
            cluster.spatial_extent = 0.0f;
            cluster.collision_urgency = 0.0f;
            cluster.ttc = std::numeric_limits<float>::max();

            // BFS to find connected events
            std::vector<size_t> queue;
            queue.push_back(i);
            visited[i] = true;

            // Reference flow direction for this cluster
            float ref_flow_dir = std::atan2(events[i].vy, events[i].vx);

            float sum_vx = 0.0f, sum_vy = 0.0f;
            float sum_x = 0.0f, sum_y = 0.0f;
            float sum_mag = 0.0f;
            float max_dist_from_center = 0.0f;

            std::vector<size_t> cluster_members;

            while (!queue.empty()) {
                size_t idx = queue.back();
                queue.pop_back();
                cluster_members.push_back(idx);

                const auto& ev = events[idx];
                float ev_mag = std::sqrt(ev.vx * ev.vx + ev.vy * ev.vy);

                sum_vx += ev.vx;
                sum_vy += ev.vy;
                sum_x += ev.x;
                sum_y += ev.y;
                sum_mag += ev_mag;
                cluster.event_count++;

                // Find neighbors
                for (size_t j = 0; j < events.size(); ++j) {
                    if (visited[j]) continue;

                    float dx = ev.x - events[j].x;
                    float dy = ev.y - events[j].y;
                    float dist = std::sqrt(dx * dx + dy * dy);

                    if (dist < cfg_.cluster_radius) {
                        // Check flow direction similarity
                        float dir_j = std::atan2(events[j].vy, events[j].vx);
                        float flow_cos = std::cos(dir_j - ref_flow_dir);

                        if (flow_cos > cfg_.flow_similarity_threshold) {
                            visited[j] = true;
                            queue.push_back(j);
                        }
                    }
                }
            }

            if (cluster.event_count >= cfg_.min_events_per_cluster) {
                cluster.center_x = sum_x / cluster.event_count;
                cluster.center_y = sum_y / cluster.event_count;
                cluster.mean_flow_mag = sum_mag / cluster.event_count;
                cluster.mean_flow_dir = std::atan2(sum_vy, sum_vx);

                // Compute spatial extent (max distance from center)
                for (size_t idx : cluster_members) {
                    float dx = events[idx].x - cluster.center_x;
                    float dy = events[idx].y - cluster.center_y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    if (dist > max_dist_from_center) {
                        max_dist_from_center = dist;
                    }
                }
                cluster.spatial_extent = max_dist_from_center;

                clusters.push_back(cluster);
            }
        }

        return clusters;
    }

    // ------------------------------------------------------------------
    // Compute TTC using Lee's τ hypothesis
    // τ = θ / θ̇   where θ = angular radius, θ̇ = expansion rate
    //
    // For event cameras:
    //   θ ∝ cluster.spatial_extent
    //   θ̇ ∝ mean radial flow at cluster boundary
    //
    // Flow that is purely outward (radial from cluster center) 
    // indicates looming (approaching) motion.
    // Flow that is lateral indicates tangential motion (passing by).
    // ------------------------------------------------------------------
    void compute_ttc(ObjectCluster& obj, const std::vector<EventFlow>& events) {
        // Compute radial flow component: outward direction from cluster center
        float radial_sum = 0.0f;
        float tangential_sum = 0.0f;
        int count = 0;

        for (const auto& ev : events) {
            float dx = ev.x - obj.center_x;
            float dy = ev.y - obj.center_y;
            float dist = std::sqrt(dx * dx + dy * dy + 1e-8f);

            if (dist < obj.spatial_extent * 2.0f) {
                // Direction from center to event (radial direction)
                float radial_dir_x = dx / dist;
                float radial_dir_y = dy / dist;

                // Project flow onto radial direction
                float radial_flow = ev.vx * radial_dir_x + ev.vy * radial_dir_y;

                // Tangential component (perpendicular to radial)
                float tang_flow = 
                    ev.vx * (-radial_dir_y) + ev.vy * radial_dir_x;

                radial_sum += std::abs(radial_flow);
                tangential_sum += std::abs(tang_flow);
                count++;
            }
        }

        if (count == 0 || radial_sum < 1e-6f) {
            obj.ttc = std::numeric_limits<float>::max();
            return;
        }

        float mean_radial = radial_sum / count;

        // θ̇ = expansion rate (radial flow magnitude)
        float expansion_rate = mean_radial;

        // θ = angular extent of the object
        float angular_size = obj.spatial_extent;

        // τ = θ / θ̇ — both θ and θ̇ are in the same normalized image
        // units (extent in [0,1), flow in units/sec), so they cancel and
        // τ comes out directly in seconds. No FOV scaling: a previous
        // fov/θ factor here cancelled θ entirely, making TTC independent
        // of object size and ~30x too large.
        obj.ttc = angular_size / (expansion_rate + 1e-6f);

        // Clamp to realistic range
        if (obj.ttc < 0.05f) obj.ttc = 0.05f;
        if (obj.ttc > 100.0f) obj.ttc = std::numeric_limits<float>::max();

        // Store radial/tangential ratio for quality assessment
        // High radial/tangential = more likely to collide
        float quality = radial_sum / (radial_sum + tangential_sum + 1e-6f);
        obj.mean_flow_dir = quality; // Reuse field for quality metric
    }

    // ------------------------------------------------------------------
    // Compute collision urgency from TTC
    //   urgency = clamp(1 - τ/τ_safety, 0, 1)
    // ------------------------------------------------------------------
    void compute_urgency(ObjectCluster& obj) {
        if (obj.ttc > cfg_.safety_time_threshold * 10.0f) {
            obj.collision_urgency = 0.0f;
            return;
        }

        // Piecewise urgency curve:
        //   τ > τ_safety            → 0.0 (no urgency)
        //   τ_safety > τ > τ_critical → 0.1-0.5 (caution/warning)
        //   τ_critical > τ           → 0.5-1.0 (critical/emergency)
        if (obj.ttc > cfg_.safety_time_threshold) {
            obj.collision_urgency = 0.0f;
        } else if (obj.ttc > cfg_.critical_time_threshold) {
            float t = (cfg_.safety_time_threshold - obj.ttc) / 
                     (cfg_.safety_time_threshold - cfg_.critical_time_threshold);
            obj.collision_urgency = 0.1f + 0.4f * t;  // [0.1, 0.5]
        } else {
            float t = (cfg_.critical_time_threshold - obj.ttc) / 
                     cfg_.critical_time_threshold;
            obj.collision_urgency = 0.5f + 0.5f * std::min(t, 1.0f);  // [0.5, 1.0]
        }
    }

    // ------------------------------------------------------------------
    // Find safe bearing: scan azimuth for gap with largest clearance
    // ------------------------------------------------------------------
    void compute_safe_bearing(ThreatAssessment& result) {
        const int NUM_DIRECTIONS = 72;  // 5° resolution
        const float TWO_PI = 2.0f * M_PI;

        float best_score = -1.0f;
        float best_bearing = 0.0f;

        for (int i = 0; i < NUM_DIRECTIONS; ++i) {
            float bearing = TWO_PI * i / NUM_DIRECTIONS - M_PI;  // [-π, π]

            // Compute threat at this bearing
            float threat = 0.0f;
            for (const auto& obj : result.objects) {
                float obj_bearing = std::atan2(
                    obj.center_y - 0.5f,  // Center of image offset
                    obj.center_x - 0.5f
                );
                float bearing_diff = bearing - obj_bearing;

                // Normalize to [-π, π]
                while (bearing_diff > M_PI) bearing_diff -= TWO_PI;
                while (bearing_diff < -M_PI) bearing_diff += TWO_PI;

                // Gaussian threat: highest at object center, decays with angle
                float angle_dist = std::abs(bearing_diff);
                float margin = cfg_.safe_bearing_margin + obj.spatial_extent;
                if (angle_dist < margin) {
                    float threat_factor = 1.0f - angle_dist / margin;
                    threat += obj.collision_urgency * threat_factor;
                }
            }

            // Safety = 1 - threat. Higher is better.
            float safety = 1.0f - std::min(threat, 1.0f);
            if (safety > best_score) {
                best_score = safety;
                best_bearing = bearing;
            }
        }

        result.safe_bearing = best_bearing;
        result.safe_bearing_confidence = best_score;
    }
};

} // namespace drone