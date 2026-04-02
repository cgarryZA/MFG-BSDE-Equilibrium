# config.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class EqnConfig:
    eqn_name: str
    total_time: float
    dim: int
    num_time_interval: int
    # SineBM-specific
    type: int = 3
    couple_coeff: float = 1.0
    num_hiddens: Optional[List[int]] = None
    N_simu: int = 500
    N_learn: int = 500
    drift_approx: str = "nn"
    # Flocking-specific
    beta: float = 0.0
    simul_seed: int = 123
    # ContXiong LOB-specific
    sigma_s: float = 0.3
    lambda_a: float = 1.0
    lambda_b: float = 1.0
    alpha: float = 1.5
    phi: float = 0.01
    discount_rate: float = 0.1
    q_max: float = 10.0
    s_init: float = 100.0
    # Jump model specific
    order_size: float = 1.0
    h_max: int = 10
    # Penalty type: "quadratic", "cubic", "exponential"
    penalty_type: str = "quadratic"
    gamma: float = 1.0  # for exponential penalty
    # McKean-Vlasov law encoder
    law_encoder_type: str = "moments"
    law_embed_dim: int = 16
    n_bins: int = 20
    # Adverse selection
    eta: float = 0.0  # adverse selection strength (0 = none)
    signal_decay: float = 0.9  # EMA decay for price signal


@dataclass
class OptConfig:
    lr_values: List[float]
    lr_boundaries: List[int]
    num_iterations: int = 0
    freq_resample: int = 20
    freq_update_drift: int = 50
    # DBDP-iterative specific
    num_sweep: int = 20
    num_iterations_perstep: int = 1000


@dataclass
class NetConfig:
    loss_type: str
    y_init_range: List[float]
    num_hiddens: List[int]
    batch_size: int
    valid_size: int
    logging_frequency: int
    dtype: str = "float64"
    verbose: bool = True
    # Flocking-specific
    simul_size: int = 1000
    # Per-solver optimizer configs (populated from JSON)
    opt_config1: Optional[OptConfig] = None
    opt_config2: Optional[OptConfig] = None
    opt_config3: Optional[OptConfig] = None
    # Flocking uses flat lr fields
    lr_values: Optional[List[float]] = None
    lr_boundaries: Optional[List[int]] = None
    num_iterations: int = 0


DELTA_CLIP = 50.0


@dataclass
class Config:
    eqn: EqnConfig
    net: NetConfig

    @classmethod
    def from_json(cls, json_path: Path) -> 'Config':
        with open(json_path) as f:
            data = json.load(f)

        eqn_data = {k: v for k, v in data['eqn_config'].items() if not k.startswith('_')}
        net_raw = {k: v for k, v in data['net_config'].items() if not k.startswith('_')}

        # Parse nested opt_config objects
        opt_configs = {}
        for key in ['opt_config1', 'opt_config2', 'opt_config3']:
            if key in net_raw:
                opt_raw = {k: v for k, v in net_raw.pop(key).items() if not k.startswith('_')}
                opt_configs[key] = OptConfig(**opt_raw)

        net_config = NetConfig(**net_raw, **opt_configs)

        return cls(
            eqn=EqnConfig(**eqn_data),
            net=net_config,
        )
