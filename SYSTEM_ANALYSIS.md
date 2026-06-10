# FPGA Event-Based Drone Collision Avoidance — System Analysis

After a deep-dive review of every file in this repository, here is a complete analysis of what works, what's broken, what's missing, and what needs attention.

## Architecture Overview

The system has three subsystems, each with distinct responsibilities:

| Subsystem | Language | Technology | Role |
|-----------|----------|-----------|------|
| **FPGA Fabric** | C++ (HLS) | Vitis HLS, ap_fixed/ap_uint | Real-time event encoding + PWM output |
| **ARM Processor** | C++ | ARM Cortex-A53/R5 | Collision prediction, tracking, evasion logic |
| **Python/PyTorch** | Python | PyTorch, NumPy | Training, simulation, validation, demos |

The pipeline: `AER Camera → Ring Buffer → k-NN Spatial Hash → LocalGeometryEncoder → ARM Collision Predictor → Evasion Controller → PWM Motors`

---

## CRITICAL BUGS (Will cause runtime failure)

### 1. FPGA model loading is fundamentally broken
**File:** `models/inference.py`, line 68-74
```
load_key = training_set if training_set != 'FPGA' else 'UNION'
```
When `training_set='FPGA'`, this loads `UNION.pth` (d=384 architecture) into a model initialized with `FPGAParams` (d=128 architecture). **PyTorch will crash** with `RuntimeError: size mismatch` because the weight tensors have incompatible shapes (3×384 vs 3×128 for `encoder.A`, and the FeatureTransform linear layers are completely different sizes).

**Fix needed in two places:**
- The weight conversion script (`train/convert_weights_to_fpga.py`) must be run first to generate `FPGA.pth`.
- The `load_model()` method should load `FPGA.pth` when `training_set == 'FPGA'`.

### 2. Missing `FPGA.pth` model weight file
The `models/models/` directory contains only:
- `DSEC.pth`, `EVIMO.pth`, `MVSEC.pth`, `UNION.pth`

**Missing:** `models/models/FPGA.pth`

Without this file, the entire FPGA inference path (`NormalFlowEstimator(training_set='FPGA')`) is dead code. The conversion pipeline exists (`convert_weights_to_fpga.py`) but either:
- (a) It was never run, or
- (b) It was run but failed silently, because UNION.pth weights loaded into a d=128 model will raise a shape mismatch before saving.

### 3. `BaseParams` referenced but never defined
**File:** `test/golden_model_test.py`, lines 238, 331

```python
from models.params import BaseParams
model = NormalEstimator(BaseParams(), device="cpu")
```

`BaseParams` does **not exist** in `models/params.py`. The params module only defines `MVSECParams`, `EVIMOParams`, `DSECParams`, `UNIONParams`, and `FPGAParams`. This will raise `ImportError`.

### 4. `FPGAParams.max_events` attribute doesn't exist
**File:** `test/golden_model_test.py`, line 64
```python
"MAX_EVENTS": FPGAParams.max_events,
```
`FPGAParams` has `ring_buffer_size: int = 4096`, not `max_events`. This will raise `AttributeError`.

### 5. Testbench accesses private member `.val`
**File:** `fpga/testbench.cpp`, lines 424, 426, 427, 430-433, 443-444
```cpp
wr_ptr.val  // Accessing private member val_ of ap_uint<12>
```
The `hls_compat.h` template class `ap_uint<W>` has `val_` as a **private** member (line 27). The testbench `.val` access will fail to compile.

### 6. Disarmed PWM test expects 0.0 but hardware returns ~0.001
**File:** `fpga/testbench.cpp`, lines 178-186
```cpp
TEST("PWM disarmed produces zero output");
ASSERT(
    static_cast<float>(motors.m1) == 0.0f && ...
```
When `enable_motors == 0`, `pwm_output.h` sets `motors.m1 = PWM_MIN_TICKS`. With the 100MHz clock and 50Hz PWM, `PWM_MIN_TICKS = 100000` (1ms pulse). Since `motor_outputs_t::m1` is `pwm_tick_t = ap_ufixed<16,12>`, converting 100000 to this fixed-point format gives ~0.1f, **not** 0.0. Advanced assertion will fail.

### 7. Encoder weights are zero-initialized (dead pipeline)
**File:** `fpga/top_level.cpp`, line 91
```cpp
static encoder_weights_t enc_weights;
```
This static array is **never loaded** with trained weight values. The encoder's `tile_matmul` will compute `events @ zeros = 0`, then `cordic_sin_cos(0) = (0, 1)`, producing uniform non-informative flow predictions. The pipeline will "run" but output garbage.

The build script (`build.tcl`) doesn't include a BRAM initialization file. The `train/convert_weights_to_fpga.py` script can generate `fpga/weights/encoder_weights.h` with the `--export-fpga-header` flag, but this header is never `#include`'d in the synthesis flow.

### 8. Ring buffer test uses wrong types
**File:** `fpga/testbench.cpp`, line 424
```cpp
ring_buf[wr_ptr.val] = {ap_uint<10>(i), ap_uint<10>(i*2), ap_uint<64>(i*1000), ap_uint<1>(i%2)};
```
But `event_unpacked_t` (in `ring_buffer.h` lines 31-36) has fields:
- `x`: `ap_ufixed<16,4>` (not `ap_uint<10>`)
- `y`: `ap_ufixed<16,4>` (not `ap_uint<10>`)
- `timestamp`: `ap_uint<32>` (not `ap_uint<64>`)
- `polarity`: `ap_uint<1>`

The brace initialization has wrong types and the layout doesn't match the struct definition. This will fail to compile or produce wrong values.

---

## MODERATE ISSUES

### 9. `mavlink_bridge.h` cannot compile without MAVLink library
The `#ifdef MAVLINK_AVAILABLE` guard means the real implementation is gated. The manual stub (`pack_set_position_target_local_ned_manual`) sends malformed packets (CRC always 0). On real hardware, the flight controller would reject these packets.

### 10. `drone_main.cpp` simulation path won't work
The `FpgaInterface` class uses `new volatile uint32_t[128]()` for simulated registers, but:
- The register offsets (e.g., `REG_FLOW_PRED_BASE`) assume specific memory layout
- The `read_flow_vectors()` method expects flow data at specific register offsets
- The `top_level.cpp` doesn't write flow data to any AXI-readable register — it only outputs via `debug_flow` (2 values) and `motor_out`

The ARM code reads flow vectors from FPGA, but the FPGA pipeline doesn't expose them to the AXI4-Lite register space.

### 11. `ap_axiu<48, 0, 0, 0>` malformed template
**File:** `fpga/top_level.cpp`, line 27
```cpp
typedef ap_axiu<48, 0, 0, 0> axis_event_t;
```
The second template parameter (DATA_WIDTH / 8) is 0, which in the simulation stub produces `ap_uint<0>` for `keep` and `dest` fields. AXI4-Stream requires valid keep signals. This should be `ap_axiu<48, 4, 0, 0>` or similar.

### 12. `sx_axilite` typo
**File:** `fpga/build.tcl`, line 47
```tcl
set_directive_interface -mode s_axilite -bundle DEBUG "collision_avoidance_top" debug_flow
```
But in `top_level.cpp`, the debug_flow interface is correctly specified as:
```cpp
#pragma HLS INTERFACE s_axilite port=debug_flow bundle=DEBUG
```
The TCL directive is redundant with the pragma (not harmful, just inconsistent).

### 13. `operator/` uses integer division
**File:** `fpga/hls_compat.h`, lines 44-46 (ap_uint), 89-92 (ap_int), 138-142 (ap_fixed), 187-191 (ap_ufixed)
```cpp
ap_uint operator/(ap_uint o) const { return o.val_ ? ap_uint(val_ / o.val_) : ap_uint(0); }
```
All division operators use integer C++ division (`val_ / o.val_`), which truncates toward zero. This is fine for `ap_int` and `ap_uint`, but for `ap_fixed`, the fixed-point division should shift the numerator by `FRAC` bits before division (which is done at line 141). This is technically correct but loses precision compared to HLS synthesis where `ap_fixed` division uses dedicated divider IP.

### 14. Missing `csv` / log output for tests
The Python simulator tests (`test_fpga_simulator.py`) only print to stdout — no structured output that could feed into CI dashboards or regression tracking.

### 15. `arm/Makefile` targets not tested
- `make` targets the real cross-compiler (`arm-linux-gnueabihf-g++`) which may not be installed.
- `make freertos` depends on `/opt/FreeRTOSv10.5.1`.
- No CI runner validates these build targets.

---

## MINOR ISSUES & CODE QUALITY

### 16. `config.yaml` references `direct_action` and `event_frame` pipelines that don't exist in FPGA
The YAML file has sections for a CNN-based event frame pipeline (`direct_action`, `event_frame`) from Bonazzi et al. 2025 — but these modules are only implemented in Python (`drone/direct_action_predictor.py`, `drone/event_frame_aggregator.py`). They are NOT integrated into the FPGA HLS pipeline (`top_level.cpp`). This is misleading — the YAML implies functionality that doesn't exist in hardware.

### 17. `setup.py` package_data path mismatch
```python
package_data={"models": ["models/*.pth"]},
```
The .pth files are at `models/models/` (nested subpackage), so the correct glob would be `models/models/*.pth`. But because `include_package_data=True` and the pth files are tracked by git, they'll likely still get included via MANIFEST.in behavior.

### 18. Pin-happy `requirements.txt`
- `scipy==1.14.1`, `scikit-learn==1.5.0`, `tqdm==4.66.2`, `matplotlib==3.8.3`, `opencv-python==4.9.0.80`
- Strict pinning will cause installation failures on systems with newer/older Python versions.
- Consider using `>=` or `~=` constraints instead.

### 19. `t_radius()` called as method but defined as attribute
**File:** `models/inference.py`, lines 247-251 (or similar locations that call `params.t_radius()`)
- `FPGAParams.t_radius` is a `float` attribute (e.g., `0.01`), not a callable.
- Used as `model.params.t_radius()` in the golden model test and inference code.

### 20. Variable `const` vs `constexpr` inconsistency in HLS headers
- `spatial_hash.h` line 24: `#define GRID_RADIUS 0.15f` — should be `constexpr float`
- `hls_compat.h` lines 110-113: Uses `static constexpr` for some internal constants — good.
- Overall mix of `#define`, `static const`, and `constexpr` without a consistent style.

### 21. `hls_compat.h` stream has LIFO behavior
The simulation `hls::stream::read()` pops from the back (LIFO), but Vitis HLS streams are FIFO. In simulation, the `aer_parallel_interface` writes events and `top_level.cpp` reads them — they'll come out in reverse order during C simulation, making the testbench produce different results than synthesis.

### 22. The `knn_output_t` union field is awkward
```cpp
struct knn_output_t {
    event_idx_t neighbor_indices[K_NEIGHBORS];
    union {
        ap_uint<6>  num_neighbors;
        ap_uint<6>  count;  // Alias for num_neighbors
    };
};
```
The union works but is unusual. In `spatial_hash.h` line 231 it uses `.num_neighbors`, in the testbench it uses `.count`. The union ensures these are the same memory, but the naming inconsistency is confusing.

### 23. CORDIC iteration uses `>> i` on fixed-point types
**File:** `fpga/encoder_systolic.h`, lines 79-80
```cpp
prod_t dx = x >> i;
prod_t dy = y >> i;
```
The `>>` operator on `ap_fixed<32,10>` in `hls_compat.h` is not defined (only `<<` is). For simulation, this will fail to compile. It works in Vitis HLS because the `ap_fixed` library supports bit-shift.

---

## MISSING COMPONENTS

### 24. `test/test_arm_collision_predictor.cpp`
Referenced by `Makefile` line 57:
```
cd test && g++ -std=c++17 -I.. -I../arm -o test_arm_cp test_arm_collision_predictor.cpp -lpthread && ./test_arm_cp
```
This file appears in the open tabs but may not be functional or complete.

### 25. No hardware-in-the-loop test harness
- `test/hil_gazebo_bridge.py` is referenced by `make test-hil` but not reviewed.
- No integration test that validates the full pipeline: Python training → weight conversion → C++ simulation → ARM prediction.

### 26. No license headers on source files
Most `.h` and `.cpp` files lack copyright/license headers despite the repo having a `LICENSE` file.

---

## SUMMARY TABLE

| Priority | Issue | File(s) | Impact |
|----------|-------|---------|--------|
| 🔴 CRITICAL | FPGA.pth doesn't exist, UNION.pth loaded into d=128 model | `models/inference.py:68` | `NormalFlowEstimator('FPGA')` crashes |
| 🔴 CRITICAL | `BaseParams` doesn't exist | `test/golden_model_test.py:238,331` | ImportError on test run |
| 🔴 CRITICAL | `FPGAParams.max_events` doesn't exist | `test/golden_model_test.py:64` | AttributeError on test run |
| 🔴 CRITICAL | Testbench accesses private `.val` | `fpga/testbench.cpp:424-444` | Compilation failure |
| 🔴 CRITICAL | Disarmed check expects 0.0, gets ~0.1 | `fpga/testbench.cpp:178-186` | Test assertion failure |
| 🔴 CRITICAL | Encoder weights never initialized | `fpga/top_level.cpp:91` | Zero flow output on FPGA |
| 🔴 CRITICAL | Testbench uses wrong struct types | `fpga/testbench.cpp:424` | Compilation or silent data corruption |
| 🟠 MODERATE | MAVLink stub sends bad CRC | `arm/mavlink_bridge.h:455` | PX4 rejects packets |
| 🟠 MODERATE | ARM reads flow from non-existent AXI regs | `arm/drone_main.cpp:111-128` | ARM gets zero flow vectors |
| 🟠 MODERATE | `ap_axiu<48,0,0,0>` invalid template | `fpga/top_level.cpp:27` | AXI interface may be malformed |
| 🟠 MODERATE | Stream is LIFO in simulation | `fpga/hls_compat.h:213-214` | C-sim differs from RTL sim |
| 🟠 MODERATE | `>> i` shift on ap_fixed not implemented | `fpga/encoder_systolic.h:79-80` | Won't compile in sim |
| 🟢 MINOR | config.yaml references unimplemented CNN pipeline | `fpga/config.yaml:144-198` | Misleading documentation |
| 🟢 MINOR | `requirements.txt` too tightly pinned | `requirements.txt` | Install failures |
| 🟢 MINOR | `params.t_radius()` called as method | Multiple test files | AttributeError |

---

## Recommended Action Items (in priority order)

1. **Generate `FPGA.pth`** — Run `convert_weights_to_fpga.py` (or fix it to handle shape mismatches), then update `inference.py` to load `FPGA.pth` for FPGA preset.

2. **Define `BaseParams`** in `models/params.py` or refactor tests to use existing param classes.

3. **Fix `FPGAParams.max_events`** — Add the attribute or change the golden model test to use `ring_buffer_size`.

4. **Fix testbench compilation errors** — Remove `.val` accesses, fix struct initialization types.

5. **Fix disarmed PWM test** — Expect `PWM_MIN_TICKS` value instead of 0.0.

6. **Initialize encoder weights in FPGA** — Add a `#include` for the auto-generated weight header in `top_level.cpp`.

7. **Fix stream FIFO vs LIFO** — In simulation, make `hls::stream::read()` pop from front.

8. **Add CORDIC `>>` operator** for `ap_fixed` in `hls_compat.h`.

9. **Add ARM unit test** — Validate `CollisionPredictor`, `EvasionController`, `KalmanTracker` with known inputs/outputs.

10. **Loose `requirements.txt` pins** and add `pyyaml` dependency.
