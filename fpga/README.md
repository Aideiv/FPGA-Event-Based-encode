# FPGA Collision Avoidance Pipeline

## Architecture

```
Event Camera (AER)
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ aer_interface │ →  │ ring_buffer  │ →  │normalization │ →  │spatial_hash │ →  │   encoder_   │ →  │  pwm_output  │
│    .h         │    │    .h        │    │    .h        │    │    .h       │    │  systolic.h  │    │    .h        │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
 AER parallel/SPI    Dual-port BRAM      x/pxl_radius       k-NN O(n·k)        Encoder(d=128)      4× PWM channels
 4-phase handshake   4096 × 48bit       t/t_radius          grid hashing        CORDIC + systolic   X-configure mixer
```

## Files

| File | Function | FPGA Resources |
|------|----------|---------------|
| `aer_interface.h` | AER parallel/SPI camera receiver | ~1-2K LUTs |
| `ring_buffer.h` | Dual-port BRAM event buffer (4096 deep) | 2 BRAM |
| `normalization.h` | Fixed-point coordinate normalization | ~2K LUTs |
| `spatial_hash.h` | O(n·k) k-NN adjacency via spatial hashing | ~12K LUTs + 48 BRAM |
| `encoder_systolic.h` | LocalGeometryEncoder (d=128, real-only) | ~35K LUTs + 128 DSP |
| `pwm_output.h` | PWM velocity → 4 motor channels | ~500 LUTs |
| `top_level.cpp` | Full pipeline integration (AXI4-Lite ctrl) | Total: ~88K LUTs + 224 DSP + 80 BRAM |

## Synthesis

### Prerequisites
- Vitis HLS 2023.1+ or Vivado HLS 2020.2+
- Target: Zynq UltraScale+ MPSoC (XCZU9EG) or Kria K26 SOM

### HLS Synthesis Commands

```bash
# Synthesize top-level
vitis_hls -f build.tcl

# Or per-module:
vitis_hls -f synth_aer.tcl
vitis_hls -f synth_spatial_hash.tcl
vitis_hls -f synth_encoder.tcl
vitis_hls -f synth_top.tcl
```

### Build Script (build.tcl)
```tcl
open_project collision_avoidance
set_top collision_avoidance_top
add_files top_level.cpp
add_files -tb testbench.cpp
open_solution "solution1"
set_part {xczu9eg-ffvb1156-2-i}
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog
```

## Configuration Constants

All constants match `FPGAParams` in `models/params.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_EVENTS` | 4096 | Ring buffer capacity |
| `K_NEIGHBORS` | 32 | k-NN graph size |
| `GRID_RADIUS` | 0.15 | Spatial hash cell width |
| `D_ENC` | 128 | Encoder dimension |
| `ALPHA_ENC` | 8 | Frequency encoding scale |
| `PXL_RADIUS` | 0.0225 | Coordinate spatial scale |
| `T_RADIUS` | 0.01 | Coordinate temporal scale |
| `PWM_FREQ_HZ` | 50 | ESC update rate |
| `CLOCK_FREQ_HZ` | 100000000 | System clock (100MHz) |

## AXI Register Map

Base address: `0x43C00000` (AXI4-Lite slave)

| Offset | Register | Access | Description |
|--------|----------|--------|-------------|
| 0x00 | `enable` | R/W | Global enable (0=idle, 1=run) |
| 0x04 | `enable_motors` | R/W | Motor arming |
| 0x08 | `inference_period` | R/W | Cycles between inference |
| 0x0C | `manual_vx` | R/W | Manual velocity X |
| 0x10 | `manual_vy` | R/W | Manual velocity Y |
| 0x14 | `manual_vz` | R/W | Manual velocity Z |
| 0x18 | `manual_yaw` | R/W | Manual yaw rate |
| 0x1C | `manual_mode` | R/W | 1=manual, 0=auto |
| 0x20 | `event_count` | RO | Events in ring buffer |

### FLOW bundle — per-event flow export

Base address: `0x43C10000` (second AXI4-Lite slave, bundle `FLOW`). Offsets
below follow standard Vitis s_axilite allocation but are **design-time
placeholders** — the generated `xcollision_avoidance_top_hw.h` is
authoritative after synthesis.

| Offset | Register | Access | Description |
|--------|----------|--------|-------------|
| 0x10 | `flow_count` | RO | Valid entries (0..1024) |
| 0x18 | `flow_seq` | RO | Export generation counter (seqlock) |
| 0x2000 + 8i | `flow_out[2i]` | RO | bits[15:0]=x (UQ4.12), bits[31:16]=y (UQ4.12) |
| 0x2004 + 8i | `flow_out[2i+1]` | RO | bits[15:0]=vx (Q8.8), bits[31:16]=vy (Q8.8) |

Read protocol (torn-read safe): read `flow_seq`, then `flow_count` and the
data words, then `flow_seq` again — if it changed, the FPGA exported a new
batch mid-read; retry. Export is bounded to the first 1024 events of a batch
(8KB window); the ARM clusterer needs only 30–50 events per object. See
`arm/fpga_interface.h` for the matching decode.

## Hardware Integration

### Camera Connection
- **Prophesee Gen4**: Parallel AER bus → `aer_parallel_interface`
- **Inivation DVXplorer**: SPI slave → `aer_spi_interface`
- Pin assignments: see constraints file

### Motor Output
- 4 independent PWM channels (50Hz ESC standard)
- X-configure mixing matrix built into `pwm_output.h`
- Motor arming via AXI register `enable_motors`

### Power
- FPGA PL: 1.0V core, 1.8V I/O
- Event camera: 5V via USB or dedicated regulator
- ESCs: Battery voltage (3S-6S LiPo) through PDB

## Benchmarks

| Metric | Value |
|--------|-------|
| Event ingestion rate | 10M events/sec (parallel AER) |
| Encoder latency | ~500 cycles (5μs @ 100MHz) |
| Full pipeline latency | ~600 cycles (6μs) |
| Control loop rate | 1kHz (configurable) |
| Power consumption | ~3.5W (PL only, estimated) |
| Resource utilization | ~32% of XCZU9EG |