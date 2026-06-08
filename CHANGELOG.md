# Changelog

All notable changes to the FPGA Event-Based Drone Collision Avoidance project.

## [1.0.0] — 2026-06-08

### Added
- CI/CD pipeline (`.github/workflows/ci.yml`) with Python, ARM C++, and FPGA test workflows
- Root `Makefile` with `make test`, `make lint`, `make build`, `make ci` targets
- `SECURITY.md` covering threat model, secure boot, bitstream encryption, MAVLink signing, supply chain, adversarial robustness
- `CHANGELOG.md`, `CONTRIBUTING.md`, `CODEOWNERS`
- Updated `setup.py` with correct project name, author (Enotrium), and description
- Updated `LICENSE` copyright holder to Enotrium

### Changed
- Project renamed from `event-flow` → `fpga-event-drone` in setup.py

## [0.1.0] — 2024-12-01

### Added
- Initial release based on VecKM normal flow estimator (Yuan et al., ICCV 2025)
- Pretrained models for UNION, MVSEC, DSEC, EVIMO datasets
- Python training pipeline (`train/`)
- Python drone controller package with 12 modules (`drone/`)
- Dual-pipeline architecture: VecKM flow-based + Bonazzi CNN direct action
- Looming-based TTC prediction with Lee's τ hypothesis
- 5-level graded evasion with hysteresis
- FPGA HLS architecture proposal (7 modules: AER interface, ring buffer, normalization, spatial hash k-NN, encoder systolic array, PWM output, top-level pipeline)
- ARM C++ collision predictor and evasion controller
- Safety watchdog (RC failsafe, altitude ceiling, NaN guard, motor timeout)
- Python FPGA pipeline simulator with 5 test scenarios
- ARM C++ unit tests for collision prediction
- FPGA C++ testbench
- Egomotion estimation module
- Demo data and visualization