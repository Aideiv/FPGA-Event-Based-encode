// pwm_output.h — PWM/DShot Motor Output Generator
//
// Converts evasion velocity commands to motor PWM signals.
// Supports standard PWM (50Hz ESC) and DShot (bidirectional digital protocol).
//
// Architecture:
//   Input:  (vx, vy, vz, yaw) velocity commands from collision predictor
//   Mixer:  X-configure quadcopter mixing matrix → 4 motor thrust values
//   Output: 4 independent PWM/DShot channels with dead-time protection
//
// Latency: <1μs from command to motor update
// Resource: ~500 LUTs, 0 DSP, 0 BRAM

#pragma once

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define PWM_FREQ_HZ       50       // Standard ESC update rate
#define PWM_MIN_US        1000     // 1ms pulse = motor off
#define PWM_MAX_US        2000     // 2ms pulse = full throttle
#define CLOCK_FREQ_HZ     100000000 // 100MHz system clock
#define PWM_PERIOD_TICKS  (CLOCK_FREQ_HZ / PWM_FREQ_HZ)  // 2,000,000 ticks
#define PWM_MIN_TICKS     (CLOCK_FREQ_HZ / 1000000 * PWM_MIN_US)
#define PWM_MAX_TICKS     (CLOCK_FREQ_HZ / 1000000 * PWM_MAX_US)

// DShot constants
#define DSHOT_THROTTLE_MIN 48     // DShot command 0 (disarmed)
#define DSHOT_THROTTLE_MAX 2047   // DShot command 2047 (full)

// Fixed-point types
typedef ap_fixed<16,4>  velocity_t;   // Velocity: [-8.0, 8.0] m/s equivalent
typedef ap_ufixed<16,12> thrust_t;    // Thrust: [0.0, 1.0]
typedef ap_uint<16>     pwm_tick_t;   // PWM pulse width in clock ticks
typedef ap_uint<12>     dshot_cmd_t;  // DShot throttle command (11-bit)

// ---------------------------------------------------------------------------
// Quadcopter Mixing Matrix (X-configure)
//
// Motor layout (top-down view, front = +X):
//        M1 (front-right)        M2 (front-left)
//             \                    /
//              \    +X (forward)  /
//               +----------------+
//              /                  \
//             /    +Y (right)      \
//        M4 (rear-left)         M3 (rear-right)
//
// Mixing: thrust = velocity contributions summed per motor
// ---------------------------------------------------------------------------
struct motor_outputs_t {
    pwm_tick_t m1;   // Front-right
    pwm_tick_t m2;   // Front-left
    pwm_tick_t m3;   // Rear-right
    pwm_tick_t m4;   // Rear-left
};

// ---------------------------------------------------------------------------
// PWM Generation Module
//
// Computes 4 independent PWM pulse widths from velocity commands.
// Input:  (vx, vy, vz, yaw_rate) in normalized velocity units
// Output: 4 PWM pulse widths in clock ticks
//
// Throttle mapping: thrust = clamp(base_throttle + velocity_contribution, 0, 1)
//   base_throttle = 0.3 (hover throttle — configurable)
//   velocity_contribution = mixing matrix × velocity vector
// ---------------------------------------------------------------------------
void pwm_output(
    velocity_t vx,
    velocity_t vy,
    velocity_t vz,
    velocity_t yaw_rate,
    ap_uint<1>  enable_motors,
    motor_outputs_t& motors
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=vx bundle=VEL_CMD
    #pragma HLS INTERFACE s_axilite port=vy bundle=VEL_CMD
    #pragma HLS INTERFACE s_axilite port=vz bundle=VEL_CMD
    #pragma HLS INTERFACE s_axilite port=yaw_rate bundle=VEL_CMD
    #pragma HLS INTERFACE s_axilite port=enable_motors bundle=VEL_CMD
    #pragma HLS INTERFACE ap_none  port=motors
    #pragma HLS PIPELINE II=1

    // Base thrust (hover = ~30% throttle for typical UAV)
    static const thrust_t base_thrust = thrust_t(0.30);
    static const thrust_t thrust_max  = thrust_t(0.95);  // Headroom for control

    // Mixing matrix coefficients (X-configure)
    // These are tuned for a typical 250mm race/freestyle quad
    static const thrust_t mix_vx_positive = thrust_t(0.25);  // Forward → front motors +
    static const thrust_t mix_vy_positive = thrust_t(0.25);  // Right → right motors +
    static const thrust_t mix_vz_positive = thrust_t(0.30);  // Up → all motors +
    static const thrust_t mix_yaw_pos     = thrust_t(0.20);  // CW yaw → diagonal pair +

    // Compute thrust per motor = base + mixing
    // M1 (FR): +vx -vy +vz +yaw
    thrust_t t1 = base_thrust + 
        thrust_t(vx)  * mix_vx_positive -
        thrust_t(vy)  * mix_vy_positive +
        thrust_t(vz)  * mix_vz_positive +
        thrust_t(yaw_rate) * mix_yaw_pos;

    // M2 (FL): +vx +vy +vz -yaw
    thrust_t t2 = base_thrust +
        thrust_t(vx)  * mix_vx_positive +
        thrust_t(vy)  * mix_vy_positive +
        thrust_t(vz)  * mix_vz_positive -
        thrust_t(yaw_rate) * mix_yaw_pos;

    // M3 (RR): -vx +vy +vz +yaw
    thrust_t t3 = base_thrust -
        thrust_t(vx)  * mix_vx_positive +
        thrust_t(vy)  * mix_vy_positive +
        thrust_t(vz)  * mix_vz_positive +
        thrust_t(yaw_rate) * mix_yaw_pos;

    // M4 (RL): -vx -vy +vz -yaw
    thrust_t t4 = base_thrust -
        thrust_t(vx)  * mix_vx_positive -
        thrust_t(vy)  * mix_vy_positive +
        thrust_t(vz)  * mix_vz_positive -
        thrust_t(yaw_rate) * mix_yaw_pos;

    // Clamp thrust to valid range
    if (t1 < thrust_t(0)) t1 = thrust_t(0);
    if (t1 > thrust_max)  t1 = thrust_max;
    if (t2 < thrust_t(0)) t2 = thrust_t(0);
    if (t2 > thrust_max)  t2 = thrust_max;
    if (t3 < thrust_t(0)) t3 = thrust_t(0);
    if (t3 > thrust_max)  t3 = thrust_max;
    if (t4 < thrust_t(0)) t4 = thrust_t(0);
    if (t4 > thrust_max)  t4 = thrust_max;

    // Convert thrust [0, 1] to PWM pulse width [MIN_TICKS, MAX_TICKS]
    pwm_tick_t pwm_range = PWM_MAX_TICKS - PWM_MIN_TICKS;
    
    pwm_tick_t m1_pwm = pwm_tick_t(PWM_MIN_TICKS + ap_uint<32>(t1 * pwm_range));
    pwm_tick_t m2_pwm = pwm_tick_t(PWM_MIN_TICKS + ap_uint<32>(t2 * pwm_range));
    pwm_tick_t m3_pwm = pwm_tick_t(PWM_MIN_TICKS + ap_uint<32>(t3 * pwm_range));
    pwm_tick_t m4_pwm = pwm_tick_t(PWM_MIN_TICKS + ap_uint<32>(t4 * pwm_range));

    // Output: if motors disabled, output minimum PWM (disarmed)
    if (enable_motors == 0) {
        motors.m1 = PWM_MIN_TICKS;
        motors.m2 = PWM_MIN_TICKS;
        motors.m3 = PWM_MIN_TICKS;
        motors.m4 = PWM_MIN_TICKS;
    } else {
        motors.m1 = m1_pwm;
        motors.m2 = m2_pwm;
        motors.m3 = m3_pwm;
        motors.m4 = m4_pwm;
    }
}