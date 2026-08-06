from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def aggregate(data: Tensor, owners: Tensor, *, average=True, num_owner=None) -> Tensor:
    """Aggregate rows in data by specifying the owners.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        owners (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """ # bond--: data: [560, 64], owners: [560], num_owner: 636,
    bin_count = torch.bincount(owners) # [628]
    bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1)) # [628]

    # If there exist 5 owners, but only the first 4 appear in the owners tensor,
    # we would like the output to have shape [5, fea_dim] with the last row = 0
    # So, we need to optionally add the fifth owner as a row with zero in the output
    if (num_owner is not None) and (bin_count.shape[0] != num_owner):
        difference = num_owner - bin_count.shape[0]
        bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])

    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bin_count.shape[0], data.shape[1]]) # [636, 64]
    output = output.index_add_(0, owners, data) # [636, 64]
    if average:
        output = (output.T / bin_count).T
    return output


class MLP(nn.Module):
    """Multi-Layer Perceptron used for non-linear regression."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int = 1,
        hidden_dim: int | Sequence[int] | None = (64, 64),
        dropout: float = 0,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (list[int] | int]): a list of integers or a single integer
                representing the number of hidden units in each layer of the MLP.
                Default = [64, 64]
            dropout (float): the dropout rate before each linear layer. Default: 0
            activation (str, optional): The name of the activation function to use
                in the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            bias (bool): whether to use bias in each Linear layers.
                Default = True
        """
        super().__init__()
        if hidden_dim is None or hidden_dim == 0:
            layers = [nn.Dropout(dropout), nn.Linear(input_dim, output_dim, bias=bias)]
        elif isinstance(hidden_dim, int):
            layers = [
                nn.Linear(input_dim, hidden_dim, bias=bias),
                find_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim, bias=bias),
            ]
        elif isinstance(hidden_dim, Sequence):
            layers = [
                nn.Linear(input_dim, hidden_dim[0], bias=bias),
                find_activation(activation),
            ]
            if len(hidden_dim) != 1:
                for h_in, h_out in itertools.pairwise(hidden_dim):
                    layers.append(nn.Linear(h_in, h_out, bias=bias))
                    layers.append(find_activation(activation))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim[-1], output_dim, bias=bias))
        else:
            raise TypeError(
                f"{hidden_dim=} must be an integer, a list of integers, or None."
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        return self.layers(x)


class GatedMLP_Orig(nn.Module):
    """Gated MLP
    similar model structure is used in CGCNN and M3GNet.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | list[int] | None = None,
        dropout: float = 0,
        activation: str = "silu",
        norm: str = "batch",
        bias: bool = True,
    ) -> None:
        """Initialize a gated MLP.

        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (list[int] | int]): a list of integers or a single integer
                representing the number of hidden units in each layer of the MLP.
                Default = None
            dropout (float): the dropout rate before each linear layer.
                Default: 0
            activation (str, optional): The name of the activation function to use in
                the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            norm (str, optional): The name of the normalization layer to use on the
                updated atom features. Must be one of "batch", "layer", or None.
                Default = "batch"
            bias (bool): whether to use bias in each Linear layers.
                Default = True
        """
        super().__init__()
        self.mlp_core = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.activation = find_activation(activation)
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.bn1 = find_normalization(name=norm, dim=output_dim)
        self.bn2 = find_normalization(name=norm, dim=output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        if self.norm is None:
            core = self.activation(self.mlp_core(x))
            gate = self.sigmoid(self.mlp_gate(x))
        else:
            core = self.activation(self.bn1(self.mlp_core(x)))
            gate = self.sigmoid(self.bn2(self.mlp_gate(x)))
        return core * gate

class GatedMLP_v2(nn.Module):
    """Gated MLP
    similar model structure is used in CGCNN and M3GNet.
    v2: input is a list of tensors
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int,
        *,
        hidden_dim: int | list[int] | None = None,
        dropout: float = 0,
        activation: str = "silu",
        norm: str = "batch",
        bias: bool = True,
    ) -> None:
        super().__init__()

        input_dim = sum(input_dims)

        self.mlp_core = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.activation = find_activation(activation)
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.bn1 = find_normalization(name=norm, dim=output_dim)
        self.bn2 = find_normalization(name=norm, dim=output_dim)

    def forward(self, x: list[Tensor]) -> Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        x = torch.cat(x, dim=1)
        if self.norm is None:
            core = self.activation(self.mlp_core(x))
            gate = self.sigmoid(self.mlp_gate(x))
        else:
            core = self.activation(self.bn1(self.mlp_core(x)))
            gate = self.sigmoid(self.bn2(self.mlp_gate(x)))
        return core * gate


class AttentionMLP_v1(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        output_dim: int,
        *,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "silu",
        norm: str = "batch",
        bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim else output_dim

        self.input_dims = input_dims
        self.total_input_dim = sum(input_dims)
        self.hidden_dim = hidden_dim
        # self.n_fields = len(input_dims)

        self.proj_layers = nn.ModuleList([
            nn.Identity() if d == hidden_dim else nn.Linear(d, hidden_dim, bias=bias)
            for d in input_dims
        ])

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

        self.norm = find_normalization(name=norm, dim=hidden_dim) if norm else None

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            find_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        # x_list: list of Tensors with shape [batch, dim_i]
        # assert len(x_list) == self.n_fields, f"Expected {self.n_fields} inputs, got {len(x_list)}"

        projected = [
            proj(xi).unsqueeze(1) for proj, xi in zip(self.proj_layers, x_list)
        ]  # [batch, 1, hidden_dim] * n

        tokens = torch.cat(projected, dim=1)  # [batch, n_fields, hidden_dim]

        # Self-attention over fields
        attn_output, _ = self.attn(tokens, tokens, tokens)

        # Pooling: mean over fields
        pooled = attn_output.mean(dim=1)  # [batch, hidden_dim]

        if self.norm:
            pooled = self.norm(pooled)

        output = self.mlp_out(pooled)
        return output


GatedMLP = GatedMLP_v2

class ScaledSiLU(torch.nn.Module):
    """Scaled Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a scaled SiLU."""
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self._activation(x) * self.scale_factor


def find_activation(name: str) -> nn.Module:
    """Return an activation function using name."""
    try:
        return {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "scaledsilu": ScaledSiLU,
            "gelu": nn.GELU,
            "softplus": nn.Softplus,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
        }[name.lower()]()
    except KeyError as exc:
        raise NotImplementedError from exc


def find_normalization(name: str, dim: int | None = None) -> nn.Module | None:
    """Return an normalization function using name."""
    if name is None:
        return None
    return {
        "batch": nn.BatchNorm1d(dim),
        "layer": nn.LayerNorm(dim),
    }.get(name.lower(), None)
