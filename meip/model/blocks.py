"""Equivariant convolution blocks used by the MEIP model.

The atom convolution updates atom features and intermediate bond features
from atom--bond messages, and the bond convolution propagates information
across neighboring bonds (three-body correlations) on the bond graph.
"""

from __future__ import annotations

from typing import List

import torch
from e3nn import nn, o3
from torch import Tensor

from meip.model.basis import RadialBessel
from meip.model.wrapper_ops import FullyConnectedTensorProduct, Linear


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum-aggregate rows of ``src`` into ``dim_size`` bins given by ``index``."""
    out = src.new_zeros(dim_size, src.size(1))
    out.index_add_(0, index, src)
    return out


def apply_radial_weights(features: Tensor, weights: Tensor, irreps_out: o3.Irreps) -> Tensor:
    """Apply per-channel radial weights to features of arbitrary irreps.

    Scalar (l=0) channels are weighted element-wise; higher-order channels are
    weighted per irrep so that equivariance is preserved.

    Args:
        features: [batch, irreps_out.dim]
        weights: [batch, weight_dim] where weight_dim equals the total
            multiplicity of ``irreps_out``
        irreps_out: irreps layout of ``features``
    """
    outputs = []
    weight_idx = 0

    for (mul, ir), slice_obj in zip(irreps_out, irreps_out.slices()):
        irrep_features = features[:, slice_obj]  # [batch, mul * ir.dim]
        irrep_weights = weights[:, weight_idx : weight_idx + mul]  # [batch, mul]
        weight_idx += mul

        if ir.l == 0:
            # Scalars: element-wise weighting
            weighted = irrep_features * irrep_weights
        else:
            # Vectors/tensors: weight each irrep as a whole
            irrep_features = irrep_features.reshape(-1, mul, ir.dim)
            weighted = irrep_features * irrep_weights.unsqueeze(-1)
            weighted = weighted.reshape(-1, mul * ir.dim)

        outputs.append(weighted)

    return torch.cat(outputs, dim=1)


class AtomConvEq(torch.nn.Module):
    """Equivariant atom convolution.

    Combines the features of the two atoms forming each directed bond with the
    equivariant bond features through a Clebsch--Gordan tensor product,
    modulates the result with radial weights, and sum-aggregates the messages
    onto the center atoms.
    """

    def __init__(
        self,
        *,
        atom_fea_irreps: o3.Irreps,
        bond_vec_fea_irreps: o3.Irreps,
        bond_fea_irreps: o3.Irreps,
        bond_weight_irreps: o3.Irreps,
        radial_MLP_dims: List[int] = [64, 64, 64],
        is_first_layer: bool = False,
        is_last_layer: bool = False,
        enable_mlp_after_concat: bool = False,
    ):
        super().__init__()

        self.atom_fea_irreps = atom_fea_irreps
        self.bond_vec_fea_irreps = bond_vec_fea_irreps
        self.bond_fea_irreps = bond_fea_irreps
        self.bond_weight_irreps = bond_weight_irreps
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer
        self.enable_mlp_after_concat = enable_mlp_after_concat

        if enable_mlp_after_concat:
            self.atom_mlp = nn.FullyConnectedNet(
                [self.atom_fea_irreps.dim * 2] + [self.atom_fea_irreps.dim] + [self.atom_fea_irreps.dim],
                act=torch.nn.functional.silu,
            )
        else:
            self.atom_mlp = None

        self.conv_tp = FullyConnectedTensorProduct(
            irreps_in1=self.atom_fea_irreps + self.atom_fea_irreps
            if not self.enable_mlp_after_concat
            else self.atom_fea_irreps,
            irreps_in2=self.bond_vec_fea_irreps if self.is_first_layer else self.bond_fea_irreps,
            irreps_out=self.atom_fea_irreps if self.is_last_layer else self.bond_fea_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        self.radial_MLP_dims = radial_MLP_dims
        # The last layer outputs atom_fea_irreps and therefore needs a matching
        # number of radial weights; earlier layers output bond_fea_irreps.
        if self.is_last_layer:
            self.weight_numel = sum(mul for mul, _ in self.atom_fea_irreps)
        else:
            self.weight_numel = sum(mul for mul, _ in self.bond_fea_irreps)

        self.radial_mlp = nn.FullyConnectedNet(
            [self.bond_weight_irreps.dim] + self.radial_MLP_dims + [self.weight_numel],
            act=torch.nn.functional.silu,
        )

        if not self.is_last_layer:
            self.atom_fea_eqv_to_inv = Linear(
                irreps_in=self.bond_fea_irreps,
                irreps_out=self.atom_fea_irreps,
                internal_weights=True,
                shared_weights=True,
            )

    def forward(
        self,
        atom_feas: torch.Tensor,
        directed_bond_vec_feas: torch.Tensor,
        directed_bond_weights_ag: torch.Tensor,
        directed_bond_feas: torch.Tensor | None,
        atom_graph: torch.Tensor,
    ):
        center_atom_feas = torch.index_select(atom_feas, 0, atom_graph[:, 0])
        nbr_atom_feas = torch.index_select(atom_feas, 0, atom_graph[:, 1])

        concated_atom_feas = torch.cat([center_atom_feas, nbr_atom_feas], dim=1)
        if self.enable_mlp_after_concat:
            concated_atom_feas = self.atom_mlp(concated_atom_feas)

        if directed_bond_feas is not None:
            updated_directed_bond_feas = self.conv_tp(concated_atom_feas, directed_bond_feas)
        else:
            updated_directed_bond_feas = self.conv_tp(concated_atom_feas, directed_bond_vec_feas)

        radial_weights = self.radial_mlp(directed_bond_weights_ag)
        updated_directed_bond_feas = apply_radial_weights(
            updated_directed_bond_feas,
            radial_weights,
            self.atom_fea_irreps if self.is_last_layer else self.bond_fea_irreps,
        )

        updated_atom_feas_eqv = scatter_sum(
            updated_directed_bond_feas, index=atom_graph[:, 0], dim_size=atom_feas.size(0)
        )

        if not self.is_last_layer:
            updated_atom_feas_inv = self.atom_fea_eqv_to_inv(updated_atom_feas_eqv)
        else:
            updated_atom_feas_inv = updated_atom_feas_eqv

        # Residual connection for atom features
        updated_atom_feas_inv = updated_atom_feas_inv + atom_feas

        # Residual connection for bond features
        if directed_bond_feas is not None and not self.is_last_layer:
            updated_directed_bond_feas = updated_directed_bond_feas + directed_bond_feas

        return updated_atom_feas_inv, updated_directed_bond_feas


class BondConvEq(torch.nn.Module):
    """Equivariant bond convolution over the bond graph.

    Refines the equivariant features of each directed bond using tensor
    products with the features of adjacent bonds sharing the same center atom,
    which encodes three-body angular correlations.
    """

    def __init__(
        self,
        bond_fea_irreps: o3.Irreps,
        bond_weight_irreps: o3.Irreps,
        radial_MLP_dims: List[int] = [64, 64, 64],
        enable_nonlinearity: bool = False,
    ):
        super().__init__()

        self.bond_fea_irreps = bond_fea_irreps
        self.bond_weight_irreps = bond_weight_irreps

        self.self_tp = FullyConnectedTensorProduct(
            irreps_in1=self.bond_fea_irreps,
            irreps_in2=self.bond_fea_irreps,
            irreps_out=self.bond_fea_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        self.radial_MLP_dims = radial_MLP_dims
        self.weight_numel = sum(mul for mul, ir in self.bond_fea_irreps)
        self.radial_mlp = nn.FullyConnectedNet(
            [self.bond_weight_irreps.dim] + self.radial_MLP_dims + [self.weight_numel],
            act=torch.nn.functional.silu,
        )

        if enable_nonlinearity:
            # Apply the activation to scalar channels only, which keeps the
            # operation equivariant and numerically stable.
            self.norm_activation = ScalarOnlyActivation(
                irreps_in=self.bond_fea_irreps, activation=torch.nn.functional.silu
            )

    def forward(
        self,
        directed_bond_feas: torch.Tensor,
        directed_bond_weights_bg: torch.Tensor,
        bond_graph: torch.Tensor,
    ):
        bond_feas_i = torch.index_select(directed_bond_feas, 0, bond_graph[:, 0])
        bond_feas_j = torch.index_select(directed_bond_feas, 0, bond_graph[:, 1])

        # Pairwise interaction between adjacent bonds
        interaction = self.self_tp(bond_feas_i, bond_feas_j)

        bond_weights_bg_i = torch.index_select(directed_bond_weights_bg, 0, bond_graph[:, 0])
        bond_weights_bg_j = torch.index_select(directed_bond_weights_bg, 0, bond_graph[:, 1])
        weight = bond_weights_bg_i * bond_weights_bg_j
        radial_weights = self.radial_mlp(weight)
        interaction = apply_radial_weights(interaction, radial_weights, self.bond_fea_irreps)

        if hasattr(self, "norm_activation"):
            interaction = self.norm_activation(interaction)

        # Aggregate messages back onto the bonds
        updated_bond_feas = scatter_sum(
            interaction, index=bond_graph[:, 0], dim_size=directed_bond_feas.size(0)
        )

        # Residual connection
        updated_bond_feas = updated_bond_feas + directed_bond_feas

        return updated_bond_feas


class ScalarOnlyActivation(torch.nn.Module):
    """Apply an activation to scalar (l=0) channels and pass higher-order
    channels through unchanged.

    This preserves equivariance while avoiding the numerical issues of
    norm-based activations on vector channels.
    """

    def __init__(self, irreps_in: o3.Irreps, activation=torch.nn.functional.silu):
        super().__init__()
        self.irreps_in = irreps_in
        self.activation = activation

    def forward(self, x):
        outputs = []
        for (mul, ir), slice_obj in zip(self.irreps_in, self.irreps_in.slices()):
            irrep_features = x[:, slice_obj]

            if ir.l == 0:
                outputs.append(self.activation(irrep_features))
            else:
                outputs.append(irrep_features)

        return torch.cat(outputs, dim=1)


class BondEncoder(torch.nn.Module):
    """Encode a chemical bond given the positions of the two atoms forming it."""

    def __init__(
        self,
        *,
        atom_graph_cutoff: float = 5,
        bond_graph_cutoff: float = 3,
        num_radial: int = 9,
        cutoff_coeff: int = 5,
        learnable: bool = False,
    ) -> None:
        """Initialize the bond encoder.

        Args:
            atom_graph_cutoff (float): The cutoff for constructing the atom graph.
                Default = 5
            bond_graph_cutoff (float): The cutoff for constructing the bond graph.
                Default = 3
            num_radial (int): The number of radial components. Default = 9
            cutoff_coeff (int): Strength of the graph cutoff smoothness. Default = 5
            learnable (bool): Whether the frequencies of the radial basis expansion
                are learnable. Default = False
        """
        super().__init__()
        self.rbf_expansion_ag = RadialBessel(
            num_radial=num_radial,
            cutoff=atom_graph_cutoff,
            smooth_cutoff=cutoff_coeff,
            learnable=learnable,
        )
        self.rbf_expansion_bg = RadialBessel(
            num_radial=num_radial,
            cutoff=bond_graph_cutoff,
            smooth_cutoff=cutoff_coeff,
            learnable=learnable,
        )

    def forward(
        self,
        center: Tensor,
        neighbor: Tensor,
        undirected2directed: Tensor,
        image: Tensor,
        lattice: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute the radial bases and unit direction vector of each directed bond.

        Args:
            center (Tensor): Cartesian coordinates of center atoms
                [n_directed_bond, 3]
            neighbor (Tensor): Cartesian coordinates of neighbor atoms
                [n_directed_bond, 3]
            undirected2directed (Tensor): mapping from an undirected bond to one of
                its directed bonds [n_undirected_bond]
            image (Tensor): periodic image specifying the location of each
                neighboring atom [n_directed_bond, 3]
            lattice (Tensor): lattice of the structure [3, 3]

        Returns:
            bond_basis_ag (Tensor): radial basis in the atom graph
                [n_directed_bond, num_radial]
            bond_basis_bg (Tensor): radial basis in the bond graph
                [n_directed_bond, num_radial]
            bond_vectors (Tensor): normalized bond direction vectors
                [n_directed_bond, 3]
            smooth_factor_ag (Tensor): smooth cutoff factors in the atom graph
                [n_directed_bond]
            smooth_factor_bg (Tensor): smooth cutoff factors in the bond graph
                [n_directed_bond]
        """
        # Shift neighbors into the periodic image closest to the center atom
        neighbor = neighbor + image @ lattice
        bond_vectors = center - neighbor
        bond_lengths = torch.norm(bond_vectors, dim=1)
        assert (bond_lengths == 0).sum() == 0
        bond_vectors = bond_vectors / bond_lengths[:, None]

        bond_basis_ag, smooth_factor_ag = self.rbf_expansion_ag(bond_lengths, return_smooth_factor=True)
        bond_basis_bg, smooth_factor_bg = self.rbf_expansion_bg(bond_lengths, return_smooth_factor=True)
        return (
            bond_basis_ag,
            bond_basis_bg,
            bond_vectors,
            smooth_factor_ag.squeeze(-1),
            smooth_factor_bg.squeeze(-1),
        )
