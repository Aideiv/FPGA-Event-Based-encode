#! /usr/bin/env python3
"""fpga_training_config.py — Training configuration for FPGA-optimized model (d=128).
#
# The original model (UNION) uses d=384, which produces excellent optical flow
# but is too large and slow for FPGA deployment. This config:
#   - Trains with d=128 for smaller encoder matrix (3×128 vs 3×384)
#   - Uses higher alpha=8 to compensate for reduced dimension
#   - Uses k-NN adjacency (k=32) to match FPGA spatial hash
#   - Uses same camera params (640×480, DSEC camera matrix)
#
# USAGE:
#   python train/s1_train.py --config train/fpga_training_config.py
#
# Or integrate with existing training pipeline:
#   python train/s1_train.py --params FPGAParams
#
# Then convert and export:
#   python train/convert_weights_to_fpga.py --input checkpoint.pth --output models/models/FPGA.pth --export-fpga-header
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# FPGA-Optimized Training Parameters
# ---------------------------------------------------------------------------
# These are designed to replace s0_params.py params for training with
# d=128 (FPGA-compatible) instead of d=384 (original).
#
# Key differences from original (UNION/DSEC):
#   1. d=128 instead of d=384 (3x smaller encoder matrix)
#   2. alpha=8 instead of alpha=5 (higher frequency encoding)
#   3. Uses k-NN adjacency (k=32) instead of radius-based
#   4. Reduced temporal radius (100μs window for fast events)
#   5. Smaller batch size (FPGA-limited BRAM capacity)
# ---------------------------------------------------------------------------

@dataclass
class FPGATrainingParams:
    # Model Architecture
    d: int = 128                    # Encoder dimension (FPGA: D_ENC=128)
    alpha: int = 8                  # Frequency encoding scale (FPGA: ALPHA=8)
    use_knn: bool = True            # Use k-NN graph (matches FPGA spatial_hash)
    k_neighbors: int = 32           # k-NN neighbors (FPGA: K_NEIGHBORS=32)
    
    # Normalization (must match fpga/normalization.h)
    pxl_radius: float = 0.0225     # Spatial normalization (FPGA: PXL_RADIUS)
    t_radius: float = 0.01         # Temporal normalization (FPGA: T_RADIUS)
    
    # Training ranges — filter training samples
    train_norm_range: tuple = (0.03, 3)  # Filter flow magnitude 0.03–3 px/ms
    val_norm_range: tuple = (0.03, 3)
    
    # Loss function
    loss_fn: str = 'mfl'           # Motion field loss (works best for flow)
    mfl_loss_lambda: float = 0.1   # Loss balance parameter
    
    # Camera geometry (for motion field computation)
    width: int = 640               # Image width (pixels)
    height: int = 480              # Image height (pixels)
    # Camera intrinsics — DSEC dataset (640×480, 4.86μm pixel)
    K: tuple = (569.7632987676102, 569.7632987676102, 335.0999870300293, 221.23667526245117)
    
    # Dataset paths
    dataset_path: str = './data'   # DSEC/MVSEC/EVIMO dataset root
    output_dir: str = './train/checkpoints/fpga'
    
    # Training hyperparameters
    batch_size: int = 4            # FPGA BRAM-limited; use small batch
    learning_rate: float = 1e-4    # Conservative LR for d=128 retrain
    weight_decay: float = 1e-5
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # If true, initialize A matrix from pre-trained d=384 weights
    # (truncated to first 128 columns, rescaled for alpha=8)
    init_from_pretrained: bool = True
    pretrained_path: str = 'models/models/UNION.pth'


# ---------------------------------------------------------------------------
# Usage instructions
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    params = FPGATrainingParams()
    
    print("FPGA-Trained Model Configuration")
    print("=" * 50)
    for key, val in params.__dict__.items():
        if not key.startswith('_'):
            print(f"  {key:20s} = {val}")
    
    print()
    print("To train:")
    print("  python train/s1_train.py --params FPGAParams")
    print()
    print("To convert for FPGA:")
    print("  python train/convert_weights_to_fpga.py \\")
    print("      --input train/checkpoints/fpga/best.pth \\")
    print("      --output models/models/FPGA.pth \\")
    print("      --export-fpga-header")
    print()
    print("To test simulation:")
    print("  python test/test_fpga_simulator.py --visualize")