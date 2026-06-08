# Security Analysis — FPGA Event-Based Drone Collision Avoidance

## Scope

This document covers the security considerations for deploying the FPGA-based event camera collision avoidance system on autonomous drones. It addresses hardware, firmware, communication, and adversarial resilience.

---

## Threat Model

### Attack Surfaces

| Surface | Vector | Impact | Mitigation |
|---------|--------|--------|------------|
| **Event camera spoofing** | Adversary projects false events (laser/laser-pointer pulses) into sensor | False collision detection → evasive maneuver into obstacle or forced landing | Temporal coherence check (real events have microsecond structure); polarity ratio check (real scenes have ~50/50 ON/OFF); contrast maximization rejects incoherent event bursts (§6.1) |
| **GPS spoofing** | RF injection of false GPS signals | Position drift, geofence violation | IMU dead-reckoning fallback; optical flow odometry from VecKM (no GPS needed for collision avoidance); MAVLink EKF fusion rejects inconsistent GPS updates |
| **IMU spoofing** | Acoustic resonance at MEMS gyro frequency | Attitude estimate corruption | Redundant IMU cross-check; flow-IMU consistency check (predicted flow from egomotion vs measured flow) |
| **RC link injection** | RF command injection on control frequency | Hijack drone control | Safety watchdog auto-disarms after 500ms of no valid data (§6.2); frequency-hopping (ELRS/Crossfire) at protocol level |
| **FPGA bitstream tampering** | Malicious bitstream loaded via JTAG/configuration port | Arbitrary motor control | Authenticated boot (Zynq secure boot with RSA-4096); bitstream encryption (AES-256-GCM); eFUSE key storage (§6.3) |
| **ARM firmware tampering** | Compromised ARM binary via OTA or physical access | Full system compromise | Secure boot chain: BootROM → FSBL → U-Boot → Linux kernel → application; signed firmware updates with hardware root of trust |
| **AXI bus snooping** | Physical probe on AXI interconnect | Data exfiltration (flow vectors, position) | AXI memory protection units (XMPU) restrict access per master; confidential data encrypted at rest in PL BRAM |
| **Supply chain** | Compromised dependency (PyTorch, OpenCV, etc.) | Backdoor in training or inference pipeline | Pinned dependency versions with SHA-256 hashes; `pip-audit` in CI; SBOM generated via `cyclonedx-python` |
| **Side-channel** | Power analysis of FPGA during encryption operations | Key extraction | Constant-time implementations; masking of critical operations; power analysis only feasible with physical access (drone in flight → not applicable) |
| **Denial of service** | High event-rate flood (laser projector, LED array) | Saturation of ring buffer → loss of real events | Ring buffer rate limiter; per-cell event count cap (MAX_CELL_EVENTS=128); priority queue for high-magnitude flow events |

---

## Secure Boot Chain

### Zynq UltraScale+ Secure Boot Flow

```
BootROM (immutable, mask ROM)
  │
  ▼
FSBL (First Stage Bootloader) — RSA-4096 signature verified
  │  └─ eFUSE: public key hash, secure boot enable bit
  ▼
ARM Trusted Firmware (ATF) + U-Boot — RSA-4096 signed
  │  └─ PL bitstream loaded with AES-256-GCM decryption
  ▼
Linux Kernel + Device Tree — signed FIT image
  │
  ▼
Application (drone_control) — signed binary
  │  └─ Runtime: OP-TEE trusted execution environment for key storage
```

### Configuration
```bash
# Enable secure boot in Vivado/Vitis
# 1. Generate RSA-4096 key pair
openssl genrsa -out secure_boot_private.pem 4096
openssl rsa -pubout -in secure_boot_private.pem -out secure_boot_public.pem

# 2. Program eFUSE with SHA-384 hash of public key
program_flash -fuse PPK0_HASH -hash $(sha384sum secure_boot_public.pem | cut -d' ' -f1)

# 3. Sign boot images
bootgen -image boot.bif -encrypt aes -arch zynqmp -w -o BOOT.BIN
```

---

## FPGA-Level Security

### Bitstream Encryption
The FPGA bitstream is encrypted with AES-256-GCM using the Zynq's built-in Device Key (stored in eFUSE, never accessible to software):

```tcl
# Vivado constraints for encrypted bitstream
set_property BITSTREAM.ENCRYPTION.ENCRYPT YES [current_design]
set_property BITSTREAM.ENCRYPTION.ENCRYPTKEYSELECT BBRAM [current_design]
set_property BITSTREAM.ENCRYPTION.KEY0 "AES256" [current_design]
```

### AXI Access Control
Memory protection units (XMPU) isolate the FPGA inference pipeline from the ARM application processor:

```c
// ARM side: configure AXI protection
// Inference pipeline (PL master) can ONLY write to flow prediction BRAM
// Cannot access DDR, peripherals, or motor control registers
XMPU_Config_Region(
    XMPU_BASEADDR,
    REGION_0,              // Region ID
    FLOW_PRED_BRAM_BASE,   // Start address
    0x10000,               // Size (64KB)
    XMPU_AP_RW,            // Read+Write access
    XMPU_TRUSTZONE_SECURE  // Secure world only
);
```

### Runtime Integrity
- **Heartbeat watchdog:** ARM core monitors a 100ms heartbeat from PL. No heartbeat → auto-disarm.
- **Flow sanity check:** NaN/Inf guard in `safety_watchdog.h` catches corrupted outputs.
- **Checksum verification:** PL flow prediction BRAM has CRC32 checksum read by ARM before acting on outputs.

---

## Communication Security

### MAVLink v2 Signing
All MAVLink communication uses signing with SHA-256:

```c
// Enable MAVLink2 signing with 32-byte shared secret
// Secret provisioned at manufacturing, unique per drone
#define MAVLINK_SIGNING_SECRET_KEY {0x...}  // 32 bytes, hardware-unique

// Each MAVLink packet includes:
//   - 6-byte signature (SHA-256 truncated)
//   - 2-byte timestamp (replay protection)
//   - Link ID (distinguish GCS from companion computer)
```

### Telemetry
- Telemetry stream is periodic (10Hz default) and contains NO raw event data (only state: evasion level, velocity, TTC)
- Full event log stored on-device for post-flight analysis; encrypted at rest
- Wireless telemetry uses AES-CCM at the radio link layer (SiK/ELRS)

---

## Supply Chain Security

### Dependency Verification

```bash
# CI/CD: verify all Python dependencies
pip-audit --requirement requirements.txt

# Generate Software Bill of Materials (SBOM)
cyclonedx-py --format json --output sbom.json requirements.txt

# Verify pretrained model hashes
sha256sum models/models/*.pth > models/models/SHA256SUMS
```

### Model Integrity
Pretrained weights (`models/models/*.pth`) are verified via SHA-256 hash before loading:

```python
import hashlib

def verify_model(path: str, expected_hash: str) -> bool:
    with open(path, 'rb') as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    return actual == expected_hash

# Hash embedded in firmware
WEIGHT_HASHES = {
    "FPGA.pth": "a1b2c3d4...",  # SHA-256
}
```

---

## Adversarial Robustness

### Robustness to Physical Attacks on Event Camera

1. **Laser dazzle / saturation:** Event cameras saturate locally (per-pixel) not globally (unlike frame cameras). Multiple ON events in a single pixel indicate saturation → flagged as anomalous.

2. **Projected patterns:** A coherent projected pattern (checkerboard, gratings) produces flow that is globally uniform. Real looming objects produce flow that diverges from a center. The `radial_to_tangential_ratio` metric in `collision_predictor.h` rejects globally-uniform flow.

3. **Temporal incoherence:** Real scenes produce events with Poisson-like temporal statistics (microsecond structure). A projected attack typically produces events at a fixed clock rate → detectable via inter-event interval histogram analysis.

### Robustness to Training-Time Attacks

- **Data poisoning:** Training data for VecKM is from established public datasets (MVSEC, DSEC, EVIMO) with known provenance. Custom training data undergoes distribution-shift detection before inclusion.
- **Backdoor weights:** Pretrained weights are cross-validated against holdout scenes before deployment.

---

## Privacy

### Data Minimization
- Event camera data is processed entirely on-device. No raw events are transmitted off-drone during flight.
- Post-flight log contains only: flight state (position, velocity, evasion level), diagnostic counters (event rate, pipeline latency), and no raw sensor data unless explicitly enabled for debugging.

### Data at Rest
- Post-flight logs encrypted with AES-256-GCM using device-unique key
- Logs automatically purged after 30 flights (configurable)

---

## Incident Response

### Emergency Disarm Triggers
The `safety_watchdog.h` provides multiple hardware-failsafe disarm paths:

| Trigger | Mechanism | Latency |
|---------|-----------|---------|
| RC link lost | 500ms timeout → disarm | 500ms |
| FPGA heartbeat lost | 100ms timeout → disarm | 100ms |
| Altitude ceiling exceeded | 120m limit → disarm | 10ms |
| NaN/Inf velocity | Range check → disarm | 10ms |
| Motor idle timeout | 30s idle → disarm | 30s |

### Manual Override
The `manual_mode` register in the FPGA control interface allows a human pilot to bypass the autonomous pipeline at any time:

```c
// ARM: set manual mode to override autonomous evasion
write_register(REG_MANUAL_MODE, 1);
write_register(REG_MANUAL_VX, pilot_vx);
write_register(REG_MANUAL_VY, pilot_vy);
// ... pilot has full authority over motors
```

---

## Security Roadmap

| Phase | Item | Timeline |
|-------|------|----------|
| **Phase 1** | SBOM generation, dependency pinning, pip-audit in CI | Week 1-2 |
| **Phase 2** | Zynq secure boot (RSA-4096) + bitstream encryption | Week 3-6 |
| **Phase 3** | MAVLink signing + AXI memory protection | Week 7-8 |
| **Phase 4** | Adversarial robustness evaluation (red-team testing) | Week 9-16 |
| **Phase 5** | Formal verification of safety-critical state machines | Week 17-20 |

---

## Contact

For security issues, do NOT file public issues. Contact: `security@example.com`

PGP key: `0x...` (available on keys.openpgp.org)