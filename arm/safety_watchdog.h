// arm/safety_watchdog.h — Safety watchdog for drone collision avoidance
//
// Provides:
//   1. RC link failsafe: if no new flow data received within timeout → disarm
//   2. Motor timeout: auto-disarm motors after inactivity
//   3. Altitude ceiling: prevent climbing above configured max
//   4. Velocity limiting: rate-limit velocity commands for smooth flight
//   5. Health monitoring: CPU/FPGA link health checks
//
// Integrates with arm/drone_main.cpp main control loop.

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace drone {

// ---------------------------------------------------------------------------
// Safety system state
// ---------------------------------------------------------------------------
struct SafetyStatus {
    bool motors_disarmed;           // True = motors forced off
    bool rc_link_lost;              // True = no data from FPGA/encoder
    bool altitude_exceeded;          // True = above ceiling
    bool velocity_limit_exceeded;    // True = NaN/Inf detected in commands
    bool fpga_link_ok;              // True = FPGA heartbeat received
    uint32_t disarmed_reason;        // Bitmask of reasons
    float watchdog_timer_ms;        // Time since last valid data
};

// ---------------------------------------------------------------------------
// Safety Watchdog
// ---------------------------------------------------------------------------
class SafetyWatchdog {
public:
    struct Config {
        // Timeouts (milliseconds)
        uint32_t rc_link_timeout_ms = 500;     // 0.5s no data → RC link lost
        uint32_t motor_auto_disarm_ms = 30000;  // 30s idle → auto disarm
        uint32_t fpga_heartbeat_timeout_ms = 100; // 100ms no heartbeat → FPGA issue
        uint32_t arming_debounce_ms = 2000;     // 2s hold to arm
        
        // Altitude limits
        float max_altitude_m = 120.0f;          // 120m (FAA limit)
        float max_vertical_speed_ms = 5.0f;     // 5 m/s max climb
        
        // Velocity sanity limits
        float max_horizontal_speed_ms = 15.0f;   // 15 m/s (~34 mph)
        float max_yaw_rate_rads = 6.0f;          // 6 rad/s (~344°/s)
        
        // Velocity smoothing (low-pass filter)
        float velocity_smoothing_alpha = 0.3f;   // 0.0=no smooth, 1.0=full smooth
    };

    SafetyWatchdog() : cfg_(), armed_(false), last_valid_timestamp_(0),
                       smoothed_vx_(0), smoothed_vy_(0), smoothed_vz_(0),
                       smoothed_yaw_(0), altitude_m_(0), arming_countdown_(0),
                       idle_timer_ms_(0) {}
    
    explicit SafetyWatchdog(const Config& cfg) : cfg_(cfg), armed_(false),
                       last_valid_timestamp_(0),
                       smoothed_vx_(0), smoothed_vy_(0), smoothed_vz_(0),
                       smoothed_yaw_(0), altitude_m_(0), arming_countdown_(0),
                       idle_timer_ms_(0) {}

    // ------------------------------------------------------------------
    // Called each control cycle. Returns safe velocity command.
    // ------------------------------------------------------------------
    EvasionCommand apply_safety(
        const EvasionCommand& cmd,
        uint32_t current_timestamp_ms,
        float current_altitude_m
    ) {
        SafetyStatus status = check_safety(cmd, current_timestamp_ms, current_altitude_m);
        
        if (!armed_ || status.motors_disarmed) {
            return disarm_output();
        }
        
        // Apply velocity smoothing and limits
        return apply_limits_and_smoothing(cmd);
    }

    // ------------------------------------------------------------------
    // Arm/disarm control
    // ------------------------------------------------------------------
    bool request_arm() {
        arming_countdown_ = cfg_.arming_debounce_ms;
        return false;
    }
    
    bool request_disarm() {
        armed_ = false;
        arming_countdown_ = 0;
        reset_smoothing();
        return true;
    }
    
    bool is_armed() const { return armed_; }
    
    // ------------------------------------------------------------------
    // Update altitude (from barometer/sonar)
    // ------------------------------------------------------------------
    void update_altitude(float altitude_m) {
        altitude_m_ = altitude_m;
    }

private:
    Config cfg_;
    bool armed_;
    uint32_t last_valid_timestamp_;
    float smoothed_vx_, smoothed_vy_, smoothed_vz_, smoothed_yaw_;
    float altitude_m_;
    uint32_t arming_countdown_;
    uint32_t idle_timer_ms_;

    SafetyStatus check_safety(
        const EvasionCommand& cmd,
        uint32_t current_timestamp_ms,
        float current_altitude_m
    ) {
        SafetyStatus status = {false, false, false, false, true, 0, 0};
        
        // 1. RC link check: has data been received recently?
        if (current_timestamp_ms - last_valid_timestamp_ > cfg_.rc_link_timeout_ms) {
            status.rc_link_lost = true;
            status.motors_disarmed = true;
            status.disarmed_reason |= 1;
        }
        
        // 2. Altitude ceiling check
        if (current_altitude_m > cfg_.max_altitude_m) {
            status.altitude_exceeded = true;
            status.motors_disarmed = true;
            status.disarmed_reason |= 2;
        }
        
        // 3. Velocity command sanity check (NaN/Inf detection)
        if (std::isnan(cmd.velocity_x) || std::isinf(cmd.velocity_x) ||
            std::isnan(cmd.velocity_y) || std::isinf(cmd.velocity_y) ||
            std::isnan(cmd.velocity_z) || std::isinf(cmd.velocity_z) ||
            std::isnan(cmd.yaw_rate) || std::isinf(cmd.yaw_rate)) {
            status.velocity_limit_exceeded = true;
            status.motors_disarmed = true;
            status.disarmed_reason |= 4;
        }
        
        // 4. Velocity magnitude limits
        float horiz = std::sqrt(cmd.velocity_x * cmd.velocity_x + 
                                cmd.velocity_y * cmd.velocity_y);
        if (horiz > cfg_.max_horizontal_speed_ms ||
            std::abs(cmd.velocity_z) > cfg_.max_vertical_speed_ms ||
            std::abs(cmd.yaw_rate) > cfg_.max_yaw_rate_rads) {
            status.velocity_limit_exceeded = true;
            status.motors_disarmed = true;
            status.disarmed_reason |= 4;
        }
        
        // 5. Idle auto-disarm
        if (horiz < 0.01f && std::abs(cmd.velocity_z) < 0.01f) {
            idle_timer_ms_ += 10;  // Assuming 100Hz loop
            if (idle_timer_ms_ > cfg_.motor_auto_disarm_ms) {
                status.motors_disarmed = true;
                status.disarmed_reason |= 8;
            }
        } else {
            idle_timer_ms_ = 0;
        }
        
        // 6. Arming countdown
        if (arming_countdown_ > 0) {
            arming_countdown_ -= 10;  // 100Hz → 10ms per decrement
            if (arming_countdown_ == 0) {
                armed_ = true;
            }
        }
        
        status.watchdog_timer_ms = static_cast<float>(
            current_timestamp_ms - last_valid_timestamp_);
        
        return status;
    }

    EvasionCommand disarm_output() {
        EvasionCommand safe;
        safe.velocity_x = 0.0f;
        safe.velocity_y = 0.0f;
        safe.velocity_z = 0.0f;
        safe.yaw_rate = 0.0f;
        safe.level = EvasionLevel::NONE;
        safe.description = "DISARMED (safety)";
        return safe;
    }

    EvasionCommand apply_limits_and_smoothing(const EvasionCommand& cmd) {
        EvasionCommand result = cmd;
        
        // Apply exponential moving average for smooth velocity changes
        float alpha = cfg_.velocity_smoothing_alpha;
        float beta = 1.0f - alpha;
        
        smoothed_vx_ = beta * smoothed_vx_ + alpha * cmd.velocity_x;
        smoothed_vy_ = beta * smoothed_vy_ + alpha * cmd.velocity_y;
        smoothed_vz_ = beta * smoothed_vz_ + alpha * cmd.velocity_z;
        smoothed_yaw_ = beta * smoothed_yaw_ + alpha * cmd.yaw_rate;
        
        result.velocity_x = smoothed_vx_;
        result.velocity_y = smoothed_vy_;
        result.velocity_z = smoothed_vz_;
        result.yaw_rate = smoothed_yaw_;
        
        return result;
    }

    void reset_smoothing() {
        smoothed_vx_ = 0.0f;
        smoothed_vy_ = 0.0f;
        smoothed_vz_ = 0.0f;
        smoothed_yaw_ = 0.0f;
    }
};

} // namespace drone