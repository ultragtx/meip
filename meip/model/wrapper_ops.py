"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
from typing import List, Optional

import torch
from e3nn import o3

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUET_AVAILABLE = True
except ImportError:
    # Fall back to pure e3nn ops when cuequivariance is not installed
    CUET_AVAILABLE = False

@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = True
    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    layout_str: str = "mul_ir"
    group: str = "O3"
    optimize_all: bool = True  # Set to True to enable all optimizations
    optimize_linear: bool = True
    optimize_channelwise: bool = True
    optimize_symmetric: bool = True
    optimize_fctp: bool = True

    def __post_init__(self):
        if self.enabled and CUET_AVAILABLE:
            self.layout_str = self.layout
            self.layout = getattr(cue, self.layout)
            self.group = getattr(cue, self.group)
        if not CUET_AVAILABLE:
            self.enabled = False


cueq_config = CuEquivarianceConfig()

class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_linear)
        ):
            return cuet.Linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                use_fallback=True,
            )

        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class TensorProduct:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)
        ):
            return cuet.ChannelWiseTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )

        return o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class FullyConnectedTensorProduct:
    """Wrapper around o3.FullyConnectedTensorProduct/cuet.FullyConnectedTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)
        ):
            return cuet.FullyConnectedTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                use_fallback=True,
            )

        return o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
