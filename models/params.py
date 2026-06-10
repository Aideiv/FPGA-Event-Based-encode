from dataclasses import dataclass

@dataclass
class BaseParams:
    """Base parameter set for unit tests (uses large d=384 preset)."""
    d: int = 384
    alpha: float = 5
    pxl_radius: float = 0.0225
    t_radius: float = 0.01

@dataclass
class MVSECParams:
    # Model parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.025
   
@dataclass
class EVIMOParams:
    # Model parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01
    
@dataclass
class DSECParams:
    # Model parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01

@dataclass
class UNIONParams:
    # Model parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01

@dataclass
class FPGAParams:
    """FPGA-optimized parameter preset.
    
    Reduced d from 384→128 for 9× fewer FeatureTransform parameters (32K vs 294K).
    Increased alpha from 5→8 for comparable frequency encoding capacity.
    Fixed buffer size for real-time ring-buffer operation.
    """
    d: int = 128
    alpha: float = 8
    pxl_radius: float = 0.0225
    t_radius: float = 0.01
    ring_buffer_size: int = 4096
    max_events: int = 4096          # Alias for ring_buffer_size (test compat)
    k_neighbors: int = 32
    grid_radius: float = 0.15
    fixed_point_bits: int = 16
    fixed_point_int_bits: int = 4
    encoder_dim: int = 128          # Alias for d (test compat)
    alpha_enc: float = 8.0          # Alias for alpha (test compat)
