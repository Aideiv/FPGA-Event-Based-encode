from dataclasses import dataclass

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