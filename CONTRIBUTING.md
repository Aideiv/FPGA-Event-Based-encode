# Contributing

## Development Setup

```bash
git clone https://github.com/Enotrium/FPGA-Event-Based-encode
cd FPGA-Event-Based-encode
pip install -e .
make install
```

## Running Tests

```bash
make ci          # Full CI: lint + test + build
make test        # All tests
make test-py     # Python pipeline tests only
make test-arm    # ARM C++ tests only
make test-fpga   # FPGA testbench only
make lint        # Lint check
```

## Project Structure

| Layer | Language | Directory | Purpose |
|-------|----------|-----------|---------|
| FPGA fabric | HLS C++ | `fpga/` | Real-time inference pipeline (Vitis HLS) |
| ARM controller | C++17 | `arm/` | Collision prediction, evasion, watchdog |
| Python training | Python 3.10+ | `train/` | VecKM model training and weight conversion |
| Python runtime | Python 3.10+ | `drone/` | Full-featured drone controller (sim/dev) |
| Python core | Python 3.10+ | `models/` | VecKM estimator and inference API |
| Tests | Python / C++ | `test/` | Unit and integration tests |

## Code Style

### Python
- Follow PEP 8
- Run `make lint` before committing
- Use type hints for public APIs

### C++ (ARM / FPGA HLS)
- C++17 for ARM, C++11 for HLS
- Header-only where possible (templates in `.h`)
- All HLS code must include appropriate `#pragma HLS` directives
- Use `namespace drone` for ARM code

### Commits
- Follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Reference issues in commit messages

## Pull Requests
1. Run `make ci` locally and ensure all tests pass
2. Include test coverage for new code
3. Update `CHANGELOG.md` under `[Unreleased]`
4. If modifying FPGA modules, update resource budget estimates
5. If modifying config constants, update all mirrored locations (`fpga/*.h`, `models/params.py`, `arm/*.h`, `fpga/config.yaml`)

## Mirroring Constants
Configuration constants in `fpga/config.yaml` MUST be synced across all three mirrors:

```yaml
# fpga/config.yaml
pipeline:
  k_neighbors: 32
  encoder_dim: 128
```
```python
# models/params.py
class FPGAParams:
    k_neighbors = 32
    encoder_dim = 128
```
```c
// fpga/spatial_hash.h
#define K_NEIGHBORS 32
#define D_ENC 128
```

A CI check verifies this on every push (see `test/golden_model_test.py`).

## Security
- Do **not** commit API keys, passwords, or private keys
- Report security issues privately (see `SECURITY.md`)
- All FPGA bitstreams must be encrypted in production