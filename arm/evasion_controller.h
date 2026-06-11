// arm/evasion_controller.h — ARM C++ Evasion Controller
//
// Runs on Zynq PS. Receives ThreatAssessment from CollisionPredictor,
// computes evasion velocity vector with graded response levels.
//
// 5-level response with hysteresis to prevent oscillation:
//   NONE (0.0-0.1)     → No action
//   CAUTION (0.1-0.3)   → Slow turn toward safe bearing
//   WARNING (0.3-0.5)   → Moderate lateral evasive action
//   CRITICAL (0.5-0.7)  → Hard pull-away + altitude change
//   EMERGENCY (0.7-1.0) → Maximum effort evasion, ignore comfort
//
// Hysteresis: downgrading levels requires sustained safety confirmation.
//   EMERGENCY → CRITICAL (one step down per cycle, never jump to NONE)
//   CRITICAL  → WARNING  (requires 3 cycles at lower urgency)
//   WARNING   → CAUTION  (requires 5 cycles at lower urgency)
//   CAUTION   → NONE     (requires 10 cycles at lower urgency)

#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include "collision_predictor.h"

namespace drone {

// -------------------------------------------------------------------------
// Evasion level enumeration
// -------------------------------------------------------------------------
enum class EvasionLevel { NONE = 0, CAUTION = 1, WARNING = 2, CRITICAL = 3, EMERGENCY = 4 };

inline const char* evasion_level_name(EvasionLevel lvl) {
    switch (lvl) {
        case EvasionLevel::NONE:
            return "NONE";
        case EvasionLevel::CAUTION:
            return "CAUTION";
        case EvasionLevel::WARNING:
            return "WARNING";
        case EvasionLevel::CRITICAL:
            return "CRITICAL";
        case EvasionLevel::EMERGENCY:
            return "EMERGENCY";
        default:
            return "UNKNOWN";
    }
}

// -------------------------------------------------------------------------
// Evasion command output
// -------------------------------------------------------------------------
struct EvasionCommand {
    float velocity_x;  // Forward/backward velocity (m/s)
    float velocity_y;  // Left/right velocity (m/s)
    float velocity_z;  // Up/down velocity (m/s)
    float yaw_rate;    // Yaw angular rate (rad/s)
    EvasionLevel level;
    std::string description;
};

// -------------------------------------------------------------------------
// Evasion Controller
//
// Computes evasion commands from threat assessments using:
//   - Potential-field repulsion from threat objects
//   - Attraction toward safe bearing
//   - Graded response with hysteresis
// -------------------------------------------------------------------------
class EvasionController {
   public:
    struct Config {
        // Velocity limits (m/s or rad/s)
        float max_horizontal_velocity = 5.0f;
        float max_vertical_velocity = 3.0f;
        float max_yaw_rate = 3.0f;  // rad/s (~172°/s)

        // Threat repulsion parameters
        float repulsion_strength = 1.0f;  // Potential field strength
        float repulsion_decay = 2.0f;     // Distance decay exponent
        float safety_distance = 2.0f;     // Meters — max repulsion range

        // Safe bearing attraction
        float attraction_strength = 0.5f;  // How strongly to seek safe bearing
        float lateral_gain = 0.7f;         // Lateral vs longitudinal evasion ratio

        // Vertical evasion
        float vertical_evasion_speed = 1.5f;  // Climb rate during evasion (m/s)

        // Hysteresis counts
        uint32_t critical_to_warning_cycles = 3;
        uint32_t warning_to_caution_cycles = 5;
        uint32_t caution_to_none_cycles = 10;

        // Minimum velocity for output
        float deadband = 0.05f;  // Output velocity deadband
    };

    EvasionController() : cfg_(), current_level_(EvasionLevel::NONE), level_persistence_(0) {}
    explicit EvasionController(const Config& cfg)
        : cfg_(cfg), current_level_(EvasionLevel::NONE), level_persistence_(0) {}

    // ------------------------------------------------------------------
    // Compute evasion command from threat assessment
    // ------------------------------------------------------------------
    EvasionCommand compute_command(const ThreatAssessment& assessment) {
        EvasionCommand cmd;
        cmd.velocity_x = 0.0f;
        cmd.velocity_y = 0.0f;
        cmd.velocity_z = 0.0f;
        cmd.yaw_rate = 0.0f;
        cmd.description = "NONE: holding position";

        if (!assessment.threat_detected || assessment.objects.empty()) {
            // No threats — gradual return to neutral
            cmd.level = apply_hysteresis(EvasionLevel::NONE);
            return cmd;
        }

        // Step 1: Determine target evasion level from max urgency
        EvasionLevel target_level = determine_level(assessment.max_urgency);

        // Step 2: Apply hysteresis
        cmd.level = apply_hysteresis(target_level);

        // Step 3: Compute evasion velocity based on level
        switch (cmd.level) {
            case EvasionLevel::NONE:
                // No action needed
                cmd.description = "NONE: holding position";
                break;

            case EvasionLevel::CAUTION: {
                // Gentle turn toward safe bearing
                float bearing_diff = assessment.safe_bearing;
                // Normalize to [-π, π]
                if (bearing_diff > M_PI) bearing_diff -= 2.0f * M_PI;
                if (bearing_diff < -M_PI) bearing_diff += 2.0f * M_PI;

                cmd.yaw_rate =
                    std::copysign(std::min(std::abs(bearing_diff) * 0.5f, cfg_.max_yaw_rate * 0.3f),
                                  bearing_diff);
                cmd.velocity_x = cfg_.max_horizontal_velocity * 0.2f;  // Slow forward
                cmd.description = "CAUTION: orienting toward safe bearing";
                break;
            }

            case EvasionLevel::WARNING: {
                // Active lateral evasion from most urgent object
                const auto& worst = find_most_urgent(assessment.objects);

                // Repulsion vector away from object
                float obj_x = worst.center_x - 0.5f;  // Center-offset
                float obj_y = worst.center_y - 0.5f;
                float dist = std::sqrt(obj_x * obj_x + obj_y * obj_y + 1e-6f);

                // Direction AWAY from object
                float rep_x = -obj_x / dist;
                float rep_y = -obj_y / dist;

                // Repulsion magnitude inversely proportional to distance
                float rep_mag =
                    cfg_.repulsion_strength * std::pow(1.0f / (dist + 0.1f), cfg_.repulsion_decay);
                rep_mag = std::min(rep_mag, cfg_.max_horizontal_velocity * 0.5f);

                cmd.velocity_x = rep_x * rep_mag;
                cmd.velocity_y = rep_y * rep_mag;

                // Slide perpendicular to threat direction for lateral evasion
                float slide_x = -rep_y;  // Perpendicular right
                float slide_y = rep_x;
                cmd.velocity_x += slide_x * cfg_.lateral_gain * rep_mag * 0.5f;
                cmd.velocity_y += slide_y * cfg_.lateral_gain * rep_mag * 0.5f;

                // Slight climb
                cmd.velocity_z = cfg_.vertical_evasion_speed * 0.3f;

                cmd.description = "WARNING: active avoidance";
                break;
            }

            case EvasionLevel::CRITICAL: {
                // Strong evasion from all threats combined
                float sum_rep_x = 0.0f;
                float sum_rep_y = 0.0f;

                for (const auto& obj : assessment.objects) {
                    if (obj.collision_urgency < 0.2f) continue;

                    float obj_x = obj.center_x - 0.5f;
                    float obj_y = obj.center_y - 0.5f;
                    float dist = std::sqrt(obj_x * obj_x + obj_y * obj_y + 1e-6f);

                    float rep_x = -obj_x / dist;
                    float rep_y = -obj_y / dist;
                    float weight = obj.collision_urgency;

                    sum_rep_x += rep_x * weight;
                    sum_rep_y += rep_y * weight;
                }

                float total_mag = std::sqrt(sum_rep_x * sum_rep_x + sum_rep_y * sum_rep_y + 1e-6f);
                float rep_mag = cfg_.max_horizontal_velocity * 0.7f;

                cmd.velocity_x = (sum_rep_x / total_mag) * rep_mag;
                cmd.velocity_y = (sum_rep_y / total_mag) * rep_mag;
                cmd.velocity_z = cfg_.vertical_evasion_speed * 0.7f;

                // Rapid yaw toward safe bearing
                float bearing_diff = assessment.safe_bearing;
                if (bearing_diff > M_PI) bearing_diff -= 2.0f * M_PI;
                if (bearing_diff < -M_PI) bearing_diff += 2.0f * M_PI;
                cmd.yaw_rate = std::copysign(cfg_.max_yaw_rate * 0.7f, bearing_diff);

                cmd.description = "CRITICAL: hard evasion";
                break;
            }

            case EvasionLevel::EMERGENCY: {
                // Maximum effort: full-throttle climb + lateral escape
                const auto& worst = find_most_urgent(assessment.objects);

                float obj_x = worst.center_x - 0.5f;
                float obj_y = worst.center_y - 0.5f;
                float dist = std::sqrt(obj_x * obj_x + obj_y * obj_y + 1e-6f);

                // Full power away from closest threat
                cmd.velocity_x = (-obj_x / dist) * cfg_.max_horizontal_velocity;
                cmd.velocity_y = (-obj_y / dist) * cfg_.max_horizontal_velocity;
                cmd.velocity_z = cfg_.max_vertical_velocity;  // Full climb

                // Hard yaw away
                float bearing_diff = assessment.safe_bearing;
                if (bearing_diff > M_PI) bearing_diff -= 2.0f * M_PI;
                if (bearing_diff < -M_PI) bearing_diff += 2.0f * M_PI;
                cmd.yaw_rate = std::copysign(cfg_.max_yaw_rate, bearing_diff);

                cmd.description = "EMERGENCY: maximum evasion!";
                break;
            }
        }

        // Apply deadband
        if (std::abs(cmd.velocity_x) < cfg_.deadband) cmd.velocity_x = 0.0f;
        if (std::abs(cmd.velocity_y) < cfg_.deadband) cmd.velocity_y = 0.0f;
        if (std::abs(cmd.velocity_z) < cfg_.deadband) cmd.velocity_z = 0.0f;

        return cmd;
    }

   private:
    Config cfg_;
    EvasionLevel current_level_;
    uint32_t level_persistence_;

    // ------------------------------------------------------------------
    // Map urgency value [0, 1] to evasion level
    // ------------------------------------------------------------------
    static EvasionLevel determine_level(float urgency) {
        if (urgency < 0.1f) return EvasionLevel::NONE;
        if (urgency < 0.3f) return EvasionLevel::CAUTION;
        if (urgency < 0.5f) return EvasionLevel::WARNING;
        if (urgency < 0.7f) return EvasionLevel::CRITICAL;
        return EvasionLevel::EMERGENCY;
    }

    // ------------------------------------------------------------------
    // Hysteresis: prevent rapid oscillation between levels
    // Upgrading (more urgent) is instant. Downgrading requires sustained
    // non-urgent assessments.
    // ------------------------------------------------------------------
    EvasionLevel apply_hysteresis(EvasionLevel target) {
        // Upgrading: instant escalation
        if (static_cast<int>(target) > static_cast<int>(current_level_)) {
            level_persistence_ = 0;
            current_level_ = target;
            return current_level_;
        }

        // Same level: maintain, reset persistence counter
        if (target == current_level_) {
            level_persistence_ = 0;
            return current_level_;
        }

        // Downgrading: require sustained lower-urgency assessments
        level_persistence_++;

        // Only downgrade ONE level at a time
        int current_int = static_cast<int>(current_level_);
        int target_int = static_cast<int>(target);

        if (current_int - target_int > 1) {
            // Limit to one-step downgrade
            target_int = current_int - 1;
            target = static_cast<EvasionLevel>(target_int);
        }

        // Check if persistence threshold met for this downgrade
        uint32_t required_cycles = 0;
        if (target_int == 0)
            required_cycles = cfg_.caution_to_none_cycles;
        else if (target_int == 1)
            required_cycles = cfg_.warning_to_caution_cycles;
        else if (target_int == 2)
            required_cycles = cfg_.critical_to_warning_cycles;
        else if (target_int == 3)
            required_cycles = 1;  // EMERGENCY→CRITICAL: 1 cycle

        if (level_persistence_ >= required_cycles) {
            level_persistence_ = 0;
            current_level_ = target;
        }

        return current_level_;
    }

    // ------------------------------------------------------------------
    // Find object with highest collision urgency
    // ------------------------------------------------------------------
    static const ObjectCluster& find_most_urgent(const std::vector<ObjectCluster>& objects) {
        const ObjectCluster* worst = &objects[0];
        for (const auto& obj : objects) {
            if (obj.collision_urgency > worst->collision_urgency) {
                worst = &obj;
            }
        }
        return *worst;
    }
};

}  // namespace drone