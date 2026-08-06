"""Training configuration for MEIP.

The defaults reproduce the 327K-parameter model reported in the paper:
N_int = 3 interaction blocks, pyramidal bond irreps 32x0e+16x1o+8x2e,
multi-task loss weights (E, F, S, M) = (1.9, 0.6, 0.1, 0.25), Huber delta = 0.1,
Adam at an initial learning rate of 7e-3 with cosine annealing, 30 epochs.

``SCALE_FACTOR`` scales the dynamic batch size (and the learning rate by
sqrt(SCALE_FACTOR)) to adapt to GPUs with different memory sizes.
"""

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

SCALE_FACTOR = float(os.environ.get("SCALE_FACTOR", "1"))


@dataclass
class TrainConfig:
    # Dataset
    dataset_path: str = "datasets/mptrj_tm_subset"
    subset_ratio: float = 1.0

    # Model architecture (paper defaults)
    dim_everywhere: int = 32
    n_conv: int = 3
    atom_fea_irreps_override: Optional[str] = None  # set in __post_init__
    bond_fea_irreps_override: Optional[str] = None  # set in __post_init__
    radial_MLP_dims: List[int] = field(default_factory=lambda: [32, 32, 32])
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [32, 32, 32])
    update_bond: bool = True
    enable_atom_nonlinearity: bool = False
    enable_bond_nonlinearity: bool = False
    enable_composition_model: bool = True
    enable_composition_model_train: bool = False

    # Graph construction
    algorithm: str = "legacy"  # "legacy" or "fast"
    enable_dihedral: bool = False
    enable_directed_bond_graph: bool = True
    atom_graph_cutoff: float = 6
    bond_graph_cutoff: float = 3
    cutoff_coeff: float = 8
    learnable_rbf: bool = True
    num_radial: int = 31
    num_angular: int = 31

    # Readout
    conv_dropout: float = 0
    read_out: str = "ave"
    gMLP_norm: str = "layer"
    readout_norm: str = "layer"
    mlp_first: bool = True
    is_intensive: bool = True
    non_linearity: str = "silu"

    # Batching (dynamic batch sampler, DDP-aware)
    use_ddp_dynamic_batch: bool = True
    dynamic_total_size: int = int(80000 * SCALE_FACTOR)
    batch_size: int = int(100000 * SCALE_FACTOR)
    dynamic_atom_weight: float = 1.0
    dynamic_bond_weight: float = 2.0
    dynamic_utilization_threshold: float = 0.8
    dynamic_max_search_attempts: int = 50
    find_unused_parameters: bool = True

    # Optimization
    lr_scale: float = 7 * math.sqrt(SCALE_FACTOR)  # learning_rate = 1e-3 * lr_scale
    max_epochs: int = 30
    optimizer: str = "Adam"  # Adam, AdamW, SGD
    scheduler: str = "CosLR"  # CosLR, BypassLR
    weight_decay: float = 0
    scheduler_params: Optional[dict] = None

    # Multi-task loss weights
    loss_type: str = "static"
    dwa_window_size: int = 3
    energy_loss_ratio: float = 1.9
    force_loss_ratio: float = 0.6
    stress_loss_ratio: float = 0.1
    mag_loss_ratio: float = 0.25

    # DataLoader
    num_workers: int = 2
    prefetch_factor: int = 1

    # Data augmentation (disabled for the paper runs)
    enable_augmentation: bool = False
    augmentation_prob: float = 0.5
    use_o3: bool = True
    augmentation_seed: Optional[int] = None

    # Reproducibility
    torch_seed: int = 42
    data_seed: int = 42

    def __post_init__(self):
        dim = self.dim_everywhere
        if self.atom_fea_irreps_override is None:
            self.atom_fea_irreps_override = f"{dim}x0e"
        if self.bond_fea_irreps_override is None:
            # Pyramidal 2:1:0.5 channel allocation over l = 0, 1, 2
            self.bond_fea_irreps_override = f"{dim}x0e+{dim // 2}x1o+{dim // 4}x2e"
        self.mlp_hidden_dims = [dim, dim, dim]
        self.radial_MLP_dims = [dim, dim, dim]

    def model_kwargs(self) -> dict:
        """Keyword arguments used to construct the MEIP model."""
        return {
            "composition_model": "MPtrj" if self.enable_composition_model else None,
            "n_conv": self.n_conv,
            "update_bond": self.update_bond,
            "enable_atom_nonlinearity": self.enable_atom_nonlinearity,
            "enable_bond_nonlinearity": self.enable_bond_nonlinearity,
            "radial_MLP_dims": self.radial_MLP_dims,
            "atom_fea_irreps_override": self.atom_fea_irreps_override,
            "bond_fea_irreps_override": self.bond_fea_irreps_override,
            "atom_fea_dim": self.dim_everywhere,
            "bond_fea_dim": self.dim_everywhere,
            "dihedral_fea_dim": 0,
            "num_radial": self.num_radial,
            "num_angular": self.num_angular,
            "conv_dropout": self.conv_dropout,
            "read_out": self.read_out,
            "gMLP_norm": self.gMLP_norm,
            "readout_norm": self.readout_norm,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "mlp_first": self.mlp_first,
            "is_intensive": self.is_intensive,
            "non_linearity": self.non_linearity,
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "bond_graph_cutoff": self.bond_graph_cutoff,
            "graph_converter_algorithm": self.algorithm,
            "cutoff_coeff": self.cutoff_coeff,
            "learnable_rbf": self.learnable_rbf,
        }
