// arm/fpga_interface.h — ARM-side AXI4-Lite interface to the FPGA pipeline
//
// Register map mirrors fpga/top_level.cpp:
//   CTRL bundle  @ FPGA_BASE_ADDR : control_regs_t fields
//   FLOW bundle  @ FPGA_FLOW_BASE : per-event (pos, flow) export
//
// NOTE: base addresses and FLOW offsets are design-time placeholders. The
// authoritative offsets come from the Vitis-generated
// xcollision_avoidance_top_hw.h after synthesis — confirm before flight.
#pragma once

#include <cstdint>
#include <vector>

#include "collision_predictor.h"
#include "evasion_controller.h"

// ---------------------------------------------------------------------------
// FPGA AXI Register Map (AXI4-Lite base addresses)
// These map to the control_regs_t struct in fpga/top_level.cpp
// ---------------------------------------------------------------------------
#define FPGA_BASE_ADDR 0x43C00000  // Example AXI address
#define REG_ENABLE (FPGA_BASE_ADDR + 0x00)
#define REG_ENABLE_MOTORS (FPGA_BASE_ADDR + 0x04)
#define REG_INFERENCE_PERIOD (FPGA_BASE_ADDR + 0x08)
#define REG_MANUAL_VX (FPGA_BASE_ADDR + 0x0C)
#define REG_MANUAL_VY (FPGA_BASE_ADDR + 0x10)
#define REG_MANUAL_VZ (FPGA_BASE_ADDR + 0x14)
#define REG_MANUAL_YAW (FPGA_BASE_ADDR + 0x18)
#define REG_MANUAL_MODE (FPGA_BASE_ADDR + 0x1C)
#define REG_EVENT_COUNT (FPGA_BASE_ADDR + 0x20)

// FLOW bundle: per-event flow export (see fpga/top_level.cpp FLOW_EXPORT).
// 2 words per entry i:
//   REG_FLOW_DATA + 8*i     bits[15:0]=x (UQ4.12)  bits[31:16]=y (UQ4.12)
//   REG_FLOW_DATA + 8*i + 4 bits[15:0]=vx (Q8.8)   bits[31:16]=vy (Q8.8)
#define FPGA_FLOW_BASE 0x43C10000
#define REG_FLOW_COUNT (FPGA_FLOW_BASE + 0x10)
#define REG_FLOW_SEQ (FPGA_FLOW_BASE + 0x18)
#define REG_FLOW_DATA (FPGA_FLOW_BASE + 0x2000)
#define FLOW_MAX_OUT 1024  // mirrors fpga/top_level.cpp — keep in sync

// Simulated register file spans CTRL base through the end of the FLOW window
#define REG_SPACE_WORDS ((FPGA_FLOW_BASE + 0x2000 + 8 * FLOW_MAX_OUT - FPGA_BASE_ADDR) / 4)

// ---------------------------------------------------------------------------
// Simulated FPGA register access (replace with actual MMIO for hardware)
// On actual Zynq: mmap /dev/mem → volatile pointer to FPGA AXI region
// ---------------------------------------------------------------------------
class FpgaInterface {
   public:
    FpgaInterface() {
        // On real hardware: mmap FPGA AXI region
        // void* ptr = mmap(NULL, 0x10000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, FPGA_BASE_ADDR);
        // registers_ = reinterpret_cast<volatile uint32_t*>(ptr);

        // Simulation: allocate local memory for testing
        registers_ = new volatile uint32_t[REG_SPACE_WORDS]();
    }

    ~FpgaInterface() { delete[] registers_; }

    void write_register(uint32_t offset, uint32_t value) {
        registers_[(offset - FPGA_BASE_ADDR) / 4] = value;
    }

    uint32_t read_register(uint32_t offset) { return registers_[(offset - FPGA_BASE_ADDR) / 4]; }

    // -----------------------------------------------------------------
    // Fixed-point packing — single source of truth for the FLOW word
    // format, shared by decode below and the register round-trip tests.
    // -----------------------------------------------------------------
    static uint32_t pack_position(float x, float y) {
        return (static_cast<uint32_t>(to_uq4_12(y)) << 16) | to_uq4_12(x);
    }

    static uint32_t pack_flow(float vx, float vy) {
        return (static_cast<uint32_t>(to_q8_8(vy)) << 16) | to_q8_8(vx);
    }

    // -----------------------------------------------------------------
    // Read per-event flow vectors from the FPGA FLOW bundle (AXI4-Lite).
    // Seqlock protocol: snapshot flow_seq, read count + data, reread
    // flow_seq — a mismatch means the FPGA exported mid-read; retry.
    // -----------------------------------------------------------------
    int read_flow_vectors(std::vector<drone::EventFlow>& events, int max_events) {
        events.clear();

        for (int attempt = 0; attempt < 3; ++attempt) {
            uint32_t seq_before = read_register(REG_FLOW_SEQ);

            uint32_t count = read_register(REG_FLOW_COUNT);
            if (count == 0) return 0;
            if (count > FLOW_MAX_OUT) count = FLOW_MAX_OUT;
            if (count > static_cast<uint32_t>(max_events)) count = max_events;

            events.reserve(count);
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t pos = read_register(REG_FLOW_DATA + 8 * i);
                uint32_t flw = read_register(REG_FLOW_DATA + 8 * i + 4);

                drone::EventFlow ev;
                ev.x = static_cast<uint16_t>(pos & 0xFFFF) / 4096.0f;  // UQ4.12
                ev.y = static_cast<uint16_t>(pos >> 16) / 4096.0f;
                ev.vx = static_cast<int16_t>(flw & 0xFFFF) / 256.0f;  // Q8.8
                ev.vy = static_cast<int16_t>(flw >> 16) / 256.0f;
                ev.t = i;  // Sequential within this batch
                events.push_back(ev);
            }

            if (read_register(REG_FLOW_SEQ) == seq_before) {
                return static_cast<int>(count);
            }
            events.clear();  // torn read — retry
        }
        return 0;
    }

    // -----------------------------------------------------------------
    // Write velocity command to FPGA PWM module
    // -----------------------------------------------------------------
    void write_velocity_command(const drone::EvasionCommand& cmd) {
        // Convert float velocities to fixed-point for FPGA
        // ap_fixed<16,4>: [-8.0, 8.0) range, Q4.12
        int32_t vx_fp = static_cast<int32_t>(cmd.velocity_x * 4096.0f);  // 2^12
        int32_t vy_fp = static_cast<int32_t>(cmd.velocity_y * 4096.0f);
        int32_t vz_fp = static_cast<int32_t>(cmd.velocity_z * 4096.0f);
        int32_t yaw_fp = static_cast<int32_t>(cmd.yaw_rate * 4096.0f);

        // Clamp to INT16 range
        auto clamp_int16 = [](int32_t v) -> uint32_t {
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            return static_cast<uint32_t>(v & 0xFFFF);
        };

        write_register(REG_MANUAL_VX, clamp_int16(vx_fp));
        write_register(REG_MANUAL_VY, clamp_int16(vy_fp));
        write_register(REG_MANUAL_VZ, clamp_int16(vz_fp));
        write_register(REG_MANUAL_YAW, clamp_int16(yaw_fp));

        // Set manual mode to feed computed commands to PWM
        write_register(REG_MANUAL_MODE, 1);
    }

    void enable(bool motors) {
        write_register(REG_ENABLE, 1);
        write_register(REG_ENABLE_MOTORS, motors ? 1 : 0);
        write_register(REG_INFERENCE_PERIOD, 100000);  // 1kHz inference trigger
    }

    void disable() {
        write_register(REG_ENABLE, 0);
        write_register(REG_ENABLE_MOTORS, 0);
    }

   private:
    // UQ4.12 with round-to-nearest, saturating at the 16-bit rails
    static uint16_t to_uq4_12(float v) {
        float s = v * 4096.0f + 0.5f;
        if (s < 0.0f) s = 0.0f;
        if (s > 65535.0f) s = 65535.0f;
        return static_cast<uint16_t>(s);
    }

    // Q8.8 two's-complement with round-to-nearest, saturating
    static uint16_t to_q8_8(float v) {
        int32_t s = static_cast<int32_t>(v * 256.0f + (v >= 0 ? 0.5f : -0.5f));
        if (s > 32767) s = 32767;
        if (s < -32768) s = -32768;
        return static_cast<uint16_t>(s & 0xFFFF);
    }

    volatile uint32_t* registers_;
};
