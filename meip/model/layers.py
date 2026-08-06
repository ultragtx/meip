"""Readout layers that map batched atom features to per-crystal features."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from meip.model.functions import MLP, aggregate


class GraphPooling(nn.Module):
    """Pooling the sub-graphs in the batched graph."""

    def __init__(self, *, average: bool = False) -> None:
        """Args:
        average (bool): whether to average the features.
        """
        super().__init__()
        self.average = average

    def forward(self, atom_feas: Tensor, atom_owner: Tensor) -> Tensor:
        """Merge the atom features that belong to same graph in a batched graph.

        Args:
            atom_feas (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, atom_fea_dim or 1]
            atom_owner (Tensor): graph indices for each atom.
                [num_batch_atoms]

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, atom_fea_dim or 1]
        """
        return aggregate(atom_feas, atom_owner, average=self.average)


class GraphAttentionReadOut(nn.Module):
    """Multi Head Attention Read Out Layer
    merge the information from atom_feas to crystal_fea.
    """

    def __init__(
        self,
        atom_fea_dim: int,
        num_head: int = 3,
        hidden_dim: int = 32,
        *,
        average=False,
    ) -> None:
        """Initialize the layer.

        Args:
            atom_fea_dim (int): atom feature dimension
            num_head (int): number of attention heads used
            hidden_dim (int): dimension of hidden layer
            average (bool): whether to average the features
        """
        super().__init__()
        self.key = MLP(
            input_dim=atom_fea_dim, output_dim=num_head, hidden_dim=hidden_dim
        )
        self.softmax = nn.Softmax(dim=0)
        self.average = average

    def forward(self, atom_feas: Tensor, atom_owner: Tensor) -> Tensor:
        """Merge the atom features that belong to same graph in a batched graph.

        Args:
            atom_feas (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, atom_fea_dim]
            atom_owner (Tensor): graph indices for each atom.
                [num_batch_atoms]

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, atom_fea_dim]
        """
        crystal_feas = []
        weights = self.key(atom_feas)  # [n_batch_atom, n_heads]
        bin_count = torch.bincount(atom_owner)
        start_index = 0
        for n_atom in bin_count:
            # find atoms belong to this crystal. shape = [n_atom, atom_fea_dim]
            atom_fea = atom_feas[start_index : start_index + n_atom, :]

            # find weight belong to these atoms. shape = [n_atom, n_heads]
            weight = self.softmax(weights[start_index : start_index + n_atom, :])

            # Weighted summation from multiple attention heads
            crystal_fea = (atom_fea.T @ weight).view(-1)  # [n_heads * atom_fea_dim]

            # Normalize the crystal feature if the model output is intensive
            if self.average:
                crystal_fea /= n_atom

            crystal_feas.append(crystal_fea)
            start_index += n_atom
        return torch.stack(crystal_feas, dim=0)
