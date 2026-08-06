from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import torch
from pymatgen.core import Structure
from torch import Tensor, nn

from meip import PredTask
from meip.graph import CrystalGraph, CrystalGraphConverter
from meip.graph.crystalgraph import TORCH_DTYPE
from meip.model.composition_model import AtomRef
from meip.model.encoders import (
    AtomEmbedding,
)
from meip.model.functions import MLP, GatedMLP, find_normalization
from meip.model.blocks import (
    AtomConvEq,
    BondConvEq,
    BondEncoder,
)
from meip.model.layers import (
    GraphAttentionReadOut,
    GraphPooling,
)
from meip.utils import determine_device
from meip.model.basis import CutoffPolynomial
from e3nn import o3

if TYPE_CHECKING:
    from typing_extensions import Self


module_dir = os.path.dirname(os.path.abspath(__file__))


class MEIP(nn.Module):
    """Magnetic-moment-supervised E(3)-Equivariant Interatomic Potential.

    An E(3)-equivariant dual-graph message-passing network that takes a crystal
    graph as input and predicts energy, forces, stress, and absolute on-site
    magnetic moments.
    """

    def __init__(
        self,
        *,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        atom_fea_irreps_override: str | None = None,  # Override atom irreps
        bond_fea_irreps_override: str | None = None,  # Override bond irreps
        radial_MLP_dims: Sequence[int],
        angle_fea_dim: int = 64,
        dihedral_fea_dim: int = 64,
        composition_model: str | nn.Module = "MPtrj",
        num_radial: int = 12,
        num_angular: int = 12,
        n_conv: int = 4,
        atom_conv_hidden_dim: Sequence[int] | int = 64,
        update_bond: bool = True,
        bond_conv_hidden_dim: Sequence[int] | int = 64,
        enable_vector_message: bool = False,
        enable_local_attention: bool = False,
        conv_dropout: float = 0,
        read_out: str = "ave",
        mlp_hidden_dims: Sequence[int] | int = (64, 64, 64),
        mlp_dropout: float = 0,
        mlp_first: bool = True,
        is_intensive: bool = True,
        non_linearity: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        atom_graph_cutoff: float = 6,
        bond_graph_cutoff: float = 3,
        graph_converter_algorithm: Literal["legacy", "fast"] = "fast",
        cutoff_coeff: int = 8,
        learnable_rbf: bool = True,
        gMLP_norm: str | None = "layer",  # noqa: N803
        readout_norm: str | None = "layer",
        version: str | None = None,
        enable_atom_nonlinearity: bool = False,
        enable_bond_nonlinearity: bool = False,
        magmom_readout: Literal["legacy", "normed"] = "legacy",
        **kwargs,
    ) -> None:
        """Initialize MEIP.

        Args:
            atom_fea_dim (int): dimension of the scalar atom features.
            bond_fea_dim (int): base multiplicity used to build the default bond
                feature irreps.
            atom_fea_irreps_override (str | None): explicit e3nn irreps string for
                the atom features, e.g. ``"32x0e"``. Overrides ``atom_fea_dim``.
            bond_fea_irreps_override (str | None): explicit e3nn irreps string for
                the equivariant bond features, e.g. ``"32x0e+16x1o+8x2e"``.
            radial_MLP_dims (Sequence[int]): hidden dims of the radial weight MLPs.
            composition_model (str | nn.Module): elemental reference energies used
                to normalize the target energies ("MPtrj" for the pre-fitted
                weights, or a custom AtomRef module).
            num_radial (int): number of radial Bessel basis functions.
            n_conv (int): number of interaction blocks.
            update_bond (bool): whether to refine bond features with bond
                convolutions on the bond graph.
            mlp_hidden_dims (Sequence[int] | int): hidden dims of the energy
                readout MLP.
            is_intensive (bool): whether the energy target is per-atom (intensive).
            atom_graph_cutoff (float): cutoff radius (A) of the atom graph.
            bond_graph_cutoff (float): cutoff radius (A) of the bond graph.
            graph_converter_algorithm ("legacy" | "fast"): graph converter backend.
            cutoff_coeff (int): strength of the smooth graph cutoff.
            learnable_rbf (bool): whether radial basis frequencies are learnable.
            magmom_readout ("legacy" | "normed"): magnetic-moment head variant.
        """
        # Store model args for reconstruction
        self.model_args = {
            key: val
            for key, val in locals().items()
            if key not in {"self", "__class__", "kwargs"}
        }
        self.model_args.update(kwargs)
        if version:
            self.model_args["version"] = version

        super().__init__()
        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_dim = bond_fea_dim
        self.is_intensive = is_intensive
        self.n_conv = n_conv
        self.cutoff_coeff = cutoff_coeff
        self.enable_vector_message = enable_vector_message
        self.enable_local_attention = enable_local_attention
        self.update_bond = update_bond
        self.magmom_readout = magmom_readout

        # Optionally, define composition model
        if isinstance(composition_model, nn.Module):
            self.composition_model = composition_model
        elif isinstance(composition_model, str):
            self.composition_model = AtomRef(is_intensive=is_intensive)
            self.composition_model.initialize_from(composition_model)
        else:
            self.composition_model = None

        if self.composition_model is not None:
            # fixed composition_model weights
            for param in self.composition_model.parameters():
                param.requires_grad = False

        # Define Crystal Graph Converter
        self.graph_converter = CrystalGraphConverter(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            algorithm=graph_converter_algorithm,
            verbose=kwargs.pop("converter_verbose", False),
        )

        # Define embedding layers
        self.atom_embedding = AtomEmbedding(atom_feature_dim=atom_fea_dim)
        self.bond_basis_expansion = BondEncoder(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            num_radial=num_radial,
            cutoff_coeff=cutoff_coeff,
            learnable=learnable_rbf,
        )

        sh_irreps = o3.Irreps.spherical_harmonics(3)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=False, normalization="component"
        )  # no re-normalization: bond vectors are already unit-length

        # Atom feature irreps - configurable for parameter tuning
        if atom_fea_irreps_override is not None:
            atom_fea_irreps = o3.Irreps(atom_fea_irreps_override)
        else:
            atom_fea_irreps = o3.Irreps(f"{atom_fea_dim}x0e")  # Default: scalar only

        # Bond feature irreps - configurable for parameter tuning
        if bond_fea_irreps_override is not None:
            bond_fea_irreps = o3.Irreps(bond_fea_irreps_override)
        else:
            # Default: scalars + vectors + rank-2 tensors
            bond_fea_irreps = o3.Irreps(f"{bond_fea_dim}x0e+{bond_fea_dim//2}x1o+{bond_fea_dim//4}x2e")
            # Other options (commented for reference):
            # bond_fea_irreps = o3.Irreps(f"{bond_fea_dim}x0e+{bond_fea_dim//2}x1o+{bond_fea_dim//4}x2e+{bond_fea_dim//6}x3o")
            # bond_fea_irreps = o3.Irreps(f"{bond_fea_dim}x0e+{bond_fea_dim//2}x1o")
            # bond_fea_irreps = o3.Irreps(f"{bond_fea_dim}x0e")
            # bond_fea_irreps = o3.Irreps(f"{bond_fea_dim}x0e+{bond_fea_dim}x1o+{bond_fea_dim}x2e")

        print(">>> Using atom_fea_irreps:", atom_fea_irreps)
        print(">>> Using bond_fea_irreps:", bond_fea_irreps)

        bond_weight_irreps = o3.Irreps(f"{num_radial}x0e")

        atom_graph_layers = [
            AtomConvEq(
                atom_fea_irreps=atom_fea_irreps,
                bond_vec_fea_irreps=sh_irreps,
                bond_fea_irreps=bond_fea_irreps,
                bond_weight_irreps=bond_weight_irreps,
                radial_MLP_dims=radial_MLP_dims,
                is_first_layer=layer_idx == 0,
                is_last_layer=layer_idx == n_conv - 1,
            )
            for layer_idx in range(n_conv)
        ]
        self.atom_conv_layers = nn.ModuleList(atom_graph_layers)

        if update_bond:
            bond_graph_layers = [
                BondConvEq(
                    bond_fea_irreps=bond_fea_irreps,
                    bond_weight_irreps=bond_weight_irreps,
                    radial_MLP_dims=radial_MLP_dims,
                    enable_nonlinearity=enable_bond_nonlinearity,
                )
                for _ in range(n_conv - 1)
            ]
            self.bond_conv_layers = nn.ModuleList(bond_graph_layers)
        else:
            self.bond_conv_layers = [None for _ in range(n_conv - 1)]


        # Define readout layer
        self.readout_norm = find_normalization(readout_norm, dim=atom_fea_dim)
        # Dedicated magnetic moment head (normalized scalar features + non-negative activation)
        if self.magmom_readout == "normed":
            self.magmom_norm = nn.LayerNorm(atom_fea_dim)
        self.magmom_head = nn.Linear(atom_fea_dim, 1)
        self.mlp_first = mlp_first
        if mlp_first:
            self.read_out_type = "sum"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=False)
        elif read_out in {"attn", "weighted"}:
            self.read_out_type = "attn"
            num_heads = kwargs.pop("num_heads", 3)
            self.pooling = GraphAttentionReadOut(
                atom_fea_dim, num_head=num_heads, average=True
            )
            input_dim = atom_fea_dim * num_heads
        else:
            self.read_out_type = "ave"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=True)
        if kwargs.pop("final_mlp", "MLP") in {"normal", "MLP"}: # <--- True
            self.mlp = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dims,
                output_dim=1,
                dropout=mlp_dropout,
                activation=non_linearity,
            )
        else:
            self.mlp = nn.Sequential(
                GatedMLP(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dims,
                    output_dim=mlp_hidden_dims[-1],
                    dropout=mlp_dropout,
                    norm=gMLP_norm,
                    activation=non_linearity,
                ),
                nn.Linear(in_features=mlp_hidden_dims[-1], out_features=1),
            )

        version_str = f" v{version}" if version else ""
        print(f"MEIP{version_str} initialized with {self.n_params:,} parameters")

    @property
    def version(self) -> str | None:
        """Return the version of the loaded checkpoint."""
        return self.model_args.get("version")

    @property
    def n_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        graphs: Sequence[CrystalGraph],
        *,
        task: PredTask = "e",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor]:
        """Get prediction associated with input graphs
        Args:
            graphs (List): a list of CrystalGraphs
            task (str): the prediction task. One of 'e', 'em', 'ef', 'efs', 'efsm'.
                Default = 'e'
            return_site_energies (bool): whether to return per-site energies,
                only available if self.mlp_first == True
                Default = False
            return_atom_feas (bool): whether to return the atom features before last
                conv layer.
                Default = False
            return_crystal_feas (bool): whether to return crystal feature.
                Default = False
        Returns:
            model output (dict).
        """
        # Optionally, make composition model prediction
        comp_energy = (
            0 if self.composition_model is None else self.composition_model(graphs)
        ) # size: [batch_size, 1]

        # Make batched graph
        batched_graph = BatchedGraph.from_graphs(
            graphs,
            bond_basis_expansion=self.bond_basis_expansion,
            angle_basis_expansion=None,
            dihedral_basis_expansion=None,
            compute_stress="s" in task,
            cutoff_coeff=self.cutoff_coeff,
        )

        # Pass to model
        prediction = self._compute(
            batched_graph,
            compute_force="f" in task,
            compute_stress="s" in task,
            compute_magmom="m" in task,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )
        prediction["e"] += comp_energy
        if return_site_energies and self.composition_model is not None:
            site_energy_shifts = self.composition_model.get_site_energies(graphs)
            prediction["site_energies"] = [
                i + j
                for i, j in zip(
                    prediction["site_energies"], site_energy_shifts, strict=True
                )
            ]
        return prediction

    def _compute(
        self,
        g: BatchedGraph,
        *,
        compute_force: bool = False,
        compute_stress: bool = False,
        compute_magmom: bool = False,
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Get Energy, Force, Stress, Magmom associated with input graphs
        force = - d(Energy)/d(atom_positions)
        stress = 1/V * d(Energy)/d(strain).

        Args:
            g (BatchedGraph): batched graph
            compute_force (bool): whether to compute force.
                Default = False
            compute_stress (bool): whether to compute stress.
                Default = False
            compute_magmom (bool): whether to compute magmom.
                Default = False
            return_site_energies (bool): whether to return per-site energies,
                only available if self.mlp_first == True
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False

        Returns:
            prediction (dict): containing the fields:
                e (Tensor) : energy of structures [batch_size, 1]
                f (Tensor) : force on atoms [num_batch_atoms, 3]
                s (Tensor) : stress of structure [3 * batch_size, 3]
                m (Tensor) : magnetic moments of sites [num_batch_atoms, 3]
        """
        prediction = {}
        atoms_per_graph = torch.bincount(g.atom_owners)
        prediction["atoms_per_graph"] = atoms_per_graph

        # Embed atoms; atomic numbers are shifted to 0-based indices
        atom_feas = self.atom_embedding(g.atomic_numbers - 1)  # [n_atoms, atom_fea_dim]

        directed_bond_feas = None
        # BondEncoder already computed radial bases per directed bond
        directed_bond_weights_ag = g.bond_bases_ag  # [num_directed, num_radial]
        directed_bond_weights_bg = g.bond_bases_bg
        directed_bond_vecs = g.bond_vectors  # [num_directed, 3]
        directed_bond_vec_feas = self.spherical_harmonics(directed_bond_vecs)  # [num_directed, sh_dim]

        # Message Passing
        for idx, (
            atom_layer,
            bond_layer,
        ) in enumerate(
            zip(
                self.atom_conv_layers[:-1],
                self.bond_conv_layers,
                strict=False,
            )
        ):
            # Atom Conv
            atom_feas, directed_bond_feas = atom_layer(
                atom_feas,
                directed_bond_vec_feas,
                directed_bond_weights_ag,
                directed_bond_feas,
                g.batched_atom_graph,
            )

            if bond_layer and len(g.batched_directed_bond_graph) != 0:
                directed_bond_feas = bond_layer(
                    directed_bond_feas,
                    directed_bond_weights_bg,
                    g.batched_directed_bond_graph,
                )

            if idx == self.n_conv - 2:
                if return_atom_feas:
                    prediction["atom_fea"] = torch.split(
                        atom_feas, atoms_per_graph.tolist()
                    )

        # Last conv layer
        atom_feas, _ = self.atom_conv_layers[-1](
            atom_feas,
            directed_bond_vec_feas,
            directed_bond_weights_ag,
            directed_bond_feas,
            g.batched_atom_graph,
        )

        if compute_magmom:
            prediction["m"] = self._compute_magmom(atom_feas, atoms_per_graph)

        if self.readout_norm is not None:
            atom_feas = self.readout_norm(atom_feas)

        # Aggregate nodes and ReadOut
        if self.mlp_first:
            energies = self.mlp(atom_feas)
            energy = self.pooling(energies, g.atom_owners).view(-1)
            if return_site_energies:
                prediction["site_energies"] = torch.split(
                    energies.squeeze(1), atoms_per_graph.tolist()
                )
            if return_crystal_feas:
                prediction["crystal_fea"] = self.pooling(atom_feas, g.atom_owners)
        else:  # ave or attn to create crystal_fea first
            crystal_feas = self.pooling(atom_feas, g.atom_owners)
            energy = self.mlp(crystal_feas).view(-1) * atoms_per_graph
            if return_crystal_feas:
                prediction["crystal_fea"] = crystal_feas

        # Compute force using vector features
        if compute_force:
            forces = torch.autograd.grad(
                energy.sum(), g.atom_positions, create_graph=True, retain_graph=True
            )
            prediction["f"] = [-1 * force_dim for force_dim in forces]

        # Compute stress
        if compute_stress:
            stress = torch.autograd.grad(
                energy.sum(), g.strains, create_graph=True, retain_graph=True
            )
            # Convert Stress unit from eV/A^3 to GPa
            scale = 1 / g.volumes * 160.21766208
            stress = [i * j for i, j in zip(stress, scale, strict=False)]
            prediction["s"] = stress

        # Normalize energy if model is intensive
        if self.is_intensive:
            energy /= atoms_per_graph
        prediction["e"] = energy

        return prediction

    def _compute_magmom(self, atom_feas: Tensor, atoms_per_graph: Tensor) -> list[Tensor]:
        """Compute per-site magnetic moments from atom features.
        Two modes:
        - "normed": LayerNorm + Linear + Softplus (default)
        - "legacy": abs(Linear)
        Returns a list of tensors per graph.
        """
        if self.magmom_readout == "legacy":
            magmom = torch.abs(self.magmom_head(atom_feas))
        else:  # "normed"
            magmom_feat = self.magmom_norm(atom_feas)
            magmom = torch.nn.functional.softplus(self.magmom_head(magmom_feat))
        return list(torch.split(magmom.view(-1), atoms_per_graph.tolist()))

    def predict_structure(
        self,
        structure: Structure | Sequence[Structure],
        *,
        task: PredTask = "efsm",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 16,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Predict from pymatgen.core.Structure.

        Args:
            structure (Structure | Sequence[Structure]): structure or a list of
                structures to predict.
            task (str): can be 'e' 'ef', 'em', 'efs', 'efsm'
                Default = "efsm"
            return_site_energies (bool): whether to return per-site energies.
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False
            batch_size (int): batch_size for predict structures.
                Default = 16

        Returns:
            prediction (dict): dict or list of dict containing the fields:
                e (Tensor) : energy of structures float in eV/atom
                f (Tensor) : force on atoms [num_atoms, 3] in eV/A
                s (Tensor) : stress of structure [3, 3] in GPa
                m (Tensor) : magnetic moments of sites [num_atoms, 3] in Bohr
                    magneton mu_B
        """
        if self.graph_converter is None:
            raise ValueError("graph_converter cannot be None!")

        structures = [structure] if isinstance(structure, Structure) else structure

        graphs = [self.graph_converter(struct) for struct in structures]
        return self.predict_graph(
            graphs,
            task=task,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
            batch_size=batch_size,
        )

    def predict_graph(
        self,
        graph: CrystalGraph | Sequence[CrystalGraph],
        *,
        task: PredTask = "efsm",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 16,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Predict from CrustalGraph.

        Args:
            graph (CrystalGraph | Sequence[CrystalGraph]): CrystalGraph(s) to predict.
            task (PredTask): one of 'e', 'ef', 'em', 'efs', 'efsm'
                Default = "efsm"
            return_site_energies (bool): whether to return per-site energies.
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False
            batch_size (int): batch_size for predict structures.
                Default = 16

        Returns:
            prediction (dict): dict or list of dict containing the fields:
                e (Tensor) : energy of structures float in eV/atom
                f (Tensor) : force on atoms [num_atoms, 3] in eV/A
                s (Tensor) : stress of structure [3, 3] in GPa
                m (Tensor) : magnetic moments of sites [num_atoms, 3] in Bohr
                    magneton mu_B
        """
        if not isinstance(graph, CrystalGraph | Sequence):
            raise TypeError(
                f"{type(graph)=} must be CrystalGraph or list of CrystalGraphs"
            )
        valid_tasks = get_args(PredTask)
        if task not in valid_tasks:
            raise ValueError(f"Invalid {task=}. Must be one of {valid_tasks}.")

        model_device = next(self.parameters()).device

        graphs = [graph] if isinstance(graph, CrystalGraph) else graph
        self.eval()
        predictions: list[dict[str, Tensor]] = [{} for _ in range(len(graphs))]
        n_steps = math.ceil(len(graphs) / batch_size)
        for step in range(n_steps):
            prediction = self.forward(
                [
                    g.to(model_device)
                    for g in graphs[batch_size * step : batch_size * (step + 1)]
                ],
                task=task,
                return_site_energies=return_site_energies,
                return_atom_feas=return_atom_feas,
                return_crystal_feas=return_crystal_feas,
            )
            for key in {
                "e",
                "f",
                "s",
                "m",
                "site_energies",
                "atom_fea",
                "crystal_fea",
            } & {*prediction}:
                for idx, tensor in enumerate(prediction[key]):
                    predictions[step * batch_size + idx][key] = (
                        tensor.cpu().detach().numpy()
                    )

        return predictions[0] if len(graphs) == 1 else predictions

    def as_dict(self) -> dict:
        """Return the MEIP weights and args in a dictionary."""
        return {"state_dict": self.state_dict(), "model_args": self.model_args}

    def todict(self) -> dict:
        """Needed for ASE JSON serialization when saving MEIP potential to
        trajectory files.
        """
        return {"model_name": type(self).__name__, "model_args": self.model_args}

    @classmethod
    def from_dict(cls, dct: dict, **kwargs) -> Self:
        """Build a MEIP from a saved dictionary."""
        model = cls(**dct["model_args"], **kwargs)
        model.load_state_dict(dct["state_dict"])
        return model

    @classmethod
    def from_file(cls, path: str, **kwargs) -> Self:
        """Build a MEIP from a saved file."""
        state = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        return cls.from_dict(state["model"], **kwargs)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        use_device: str | None = None,
        check_cuda_mem: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> Self:
        """Load a trained MEIP model from a checkpoint file and move it to the
        requested device.

        Args:
            path (str): path to a checkpoint saved by the trainer or
                ``MEIP.as_dict``.
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device
                is automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use the cuda device with the most
                available memory. Default = False
            verbose (bool): whether to print model device information.
                Default = True
        """
        model = cls.from_file(path, **kwargs)

        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)
        model = model.to(device)
        if verbose:
            print(f"MEIP will run on {device}")
        return model


@dataclass
class BatchedGraph:
    """Batched crystal graph for parallel computing.

    Attributes:
        atomic_numbers (Tensor): atomic numbers vector
            [num_batch_atoms]
        bond_bases_ag (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_ag, num_radial]
        bond_bases_bg (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_bg, num_radial]
        angle_bases (Tensor): angle bases vector
            [num_batch_angles, num_angular]
        dihedral_bases (Tensor): dihedral bases vector
            [num_batch_dihedrals, num_angular]
        smooth_factors_ag (Tensor): smooth factors vector for atom_graph
            [num_batch_bonds_ag]
        smooth_factors_bg (Tensor): smooth factors vector for atom_graph
            [num_batch_bonds_bg]
        smooth_factors_atom_graph (Tensor): smooth factors vector for atom_graph
            [num_batch_atoms]
        batched_atom_graph (Tensor) : batched atom graph adjacency list
            [num_batch_bonds, 2]
        batched_bond_graph (Tensor) : bond graph adjacency list
            [num_batch_angles, 3]
        batched_dihedral_graph (Tensor) : dihedral graph adjacency list
            [num_batch_dihedrals, 7]
        atom_owners (Tensor): graph indices for each atom, used aggregate batched
            graph back to single graph
            [num_batch_atoms]
        directed2undirected (Tensor): the utility tensor used to quickly
            map directed edges to undirected edges in graph
            [num_directed]
        atom_positions (list[Tensor]): cartesian coordinates of the atoms
            from structures
            [[num_atoms_1, 3], [num_atoms_2, 3], ...]
        strains (list[Tensor]): a list of strains that's initialized to be zeros
            [[3, 3], [3, 3], ...]
        volumes (Tensor): the volume of each structure in the batch
            [batch_size]
        bond_vectors (Tensor): normalized bond vectors
            [num_batch_bonds, 3]

        # Precomputed data for local attention
        batch_size (int): number of graphs in the batch
        max_atoms (int): maximum number of atoms in any graph
        global_to_local (Tensor): mapping from global atom index to local index within its graph
            [num_batch_atoms]
        max_neighbors (int): maximum number of neighbors for any atom
        la_neighbor_indices (Tensor): precomputed neighbor indices for each atom
            [batch_size, max_atoms, max_neighbors]
        la_neighbor_distances (Tensor): precomputed normalized distances (r/cutoff) for each neighbor
            [batch_size, max_atoms, max_neighbors]
        la_atom_indices (list): precomputed atom indices for each graph
            List of length batch_size, each element is a tensor of atom indices
    """
    atom_graph_cutoff: float
    bond_graph_cutoff: float

    atomic_numbers: Tensor
    bond_bases_ag: Tensor
    bond_bases_bg: Tensor
    angle_bases: Tensor
    dihedral_bases: Tensor
    smooth_factors_ag: Tensor
    smooth_factors_bg: Tensor
    smooth_factors_atom_graph: Tensor
    batched_atom_graph: Tensor
    batched_bond_graph: Tensor
    batched_directed_bond_graph: Tensor
    batched_dihedral_graph: Tensor
    atom_owners: Tensor
    directed2undirected: Tensor
    atom_positions: Sequence[Tensor]
    strains: Sequence[Tensor]
    volumes: Sequence[Tensor] | Tensor
    bond_vectors: Tensor

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[CrystalGraph],
        bond_basis_expansion: nn.Module,
        angle_basis_expansion: nn.Module,
        dihedral_basis_expansion: nn.Module,
        *,
        compute_stress: bool = False,
        cutoff_coeff: float = 8,
    ) -> Self:
        """Featurize and assemble a list of graphs.

        Args:
            graphs (list[Tensor]): a list of CrystalGraphs
            bond_basis_expansion (nn.Module): bond basis expansion layer in MEIP
            angle_basis_expansion (nn.Module): angle basis expansion layer in MEIP
            compute_stress (bool): whether to compute stress. Default = False

        Returns:
            BatchedGraph: assembled graphs ready for batched MEIP forward pass
        """
        atom_graph_cutoff = graphs[0].atom_graph_cutoff
        bond_graph_cutoff = graphs[0].bond_graph_cutoff

        atomic_numbers, atom_positions = [], []
        strains, volumes = [], []
        bond_bases_ag, bond_bases_bg, angle_bases, dihedral_bases = [], [], [], []
        smooth_factors_ag, smooth_factors_bg = [], []
        smooth_factors_atom_graph = []
        batched_atom_graph, batched_bond_graph, batched_dihedral_graph = [], [], []
        batched_directed_bond_graph = []
        directed2undirected = []
        atom_owners = []
        bond_vectors = []
        atom_offset_idx = n_undirected = 0
        n_directed = 0

        for graph_idx, graph in enumerate(graphs):
            # Atoms
            n_atom = graph.atomic_number.shape[0]
            atomic_numbers.append(graph.atomic_number)

            # Lattice: attach a differentiable strain tensor when stress is needed
            if compute_stress:
                strain = graph.lattice.new_zeros([3, 3], requires_grad=True)
                lattice = graph.lattice @ (
                    torch.eye(3, dtype=TORCH_DTYPE).to(strain.device) + strain
                )
            else:
                strain = None
                lattice = graph.lattice
            # abs() guards against negative volumes under improper O(3) operations
            volumes.append(torch.linalg.det(lattice).abs())
            strains.append(strain)

            # Bonds: convert fractional to Cartesian coordinates
            atom_cart_coords = graph.atom_frac_coord @ lattice
            if graph.atom_graph.dim() == 1:
                # This is to avoid structure with all atoms isolated
                graph.atom_graph = graph.atom_graph.reshape(0, 2)

            # Radial bases and unit vectors are computed per directed bond
            bond_basis_ag, bond_basis_bg, bond_vectors_graph, smooth_factor_ag, smooth_factor_bg = bond_basis_expansion(
                center=atom_cart_coords[graph.atom_graph[:, 0]],
                neighbor=atom_cart_coords[graph.atom_graph[:, 1]],
                undirected2directed=graph.undirected2directed,
                image=graph.neighbor_image,
                lattice=lattice,
            )
            atom_positions.append(atom_cart_coords)
            bond_bases_ag.append(bond_basis_ag)
            bond_bases_bg.append(bond_basis_bg)
            smooth_factors_ag.append(smooth_factor_ag)
            smooth_factors_bg.append(smooth_factor_bg)

            # Look up smooth factors per atom-graph adjacency entry
            smooth_factor_atom_graph = smooth_factor_ag[graph.directed2undirected]
            smooth_factors_atom_graph.append(smooth_factor_atom_graph)

            # BondEncoder returns center - neighbor; flip to the
            # neighbor - center (receiver - sender) convention
            bond_vectors.append(-bond_vectors_graph)

            # Offset indices so that graphs concatenate into one disjoint batch graph
            batched_atom_graph.append(graph.atom_graph + atom_offset_idx)
            directed2undirected.append(graph.directed2undirected + n_undirected)

            # Angles
            # Here we use directed edges to calculate angles, and
            # keep only the undirected graph index in the bond_graph,
            # So the number of columns in bond_graph reduce from 5 to 3
            if len(graph.bond_graph) != 0:
                bond_graph = graph.bond_graph.new_zeros([graph.bond_graph.shape[0], 3])
                bond_graph[:, 0] = graph.bond_graph[:, 0] + atom_offset_idx
                bond_graph[:, 1] = graph.bond_graph[:, 1] + n_undirected
                bond_graph[:, 2] = graph.bond_graph[:, 3] + n_undirected
                batched_bond_graph.append(bond_graph)

            if len(graph.directed_bond_graph) != 0:
                # graph.directed_bond_graph: [num_pairs, 3] => [atom_center, left_de, right_de]
                # Only the directed-edge pair [left_de, right_de] is needed;
                # offset the directed-edge indices by n_directed for batching
                dbg = graph.directed_bond_graph.new_zeros([graph.directed_bond_graph.shape[0], 2])
                dbg[:, 0] = graph.directed_bond_graph[:, 1] + n_directed
                dbg[:, 1] = graph.directed_bond_graph[:, 2] + n_directed
                batched_directed_bond_graph.append(dbg)  # [num_pairs, 2]

            # Graph index of each atom, used to aggregate the batch back per crystal
            atom_owners.append(torch.ones(n_atom, requires_grad=False) * graph_idx)
            atom_offset_idx += n_atom
            n_undirected += len(bond_basis_ag)
            n_directed += len(graph.directed2undirected)

        # Make Torch Tensors
        atomic_numbers = torch.cat(atomic_numbers, dim=0)
        bond_bases_ag = torch.cat(bond_bases_ag, dim=0)
        bond_bases_bg = torch.cat(bond_bases_bg, dim=0)
        smooth_factors_ag = torch.cat(smooth_factors_ag, dim=0)
        smooth_factors_bg = torch.cat(smooth_factors_bg, dim=0)
        smooth_factors_atom_graph = torch.cat(smooth_factors_atom_graph, dim=0)

        angle_bases = (
            torch.cat(angle_bases, dim=0) if len(angle_bases) != 0 else torch.tensor([])
        )
        dihedral_bases = (
            torch.cat(dihedral_bases, dim=0) if len(dihedral_bases) != 0 else torch.tensor([])
        )
        batched_atom_graph = torch.cat(batched_atom_graph, dim=0)
        if batched_bond_graph:
            batched_bond_graph = torch.cat(batched_bond_graph, dim=0)
        else:  # when bond graph is empty or disabled
            batched_bond_graph = torch.tensor([])

        if batched_directed_bond_graph:
            batched_directed_bond_graph = torch.cat(batched_directed_bond_graph, dim=0)
        else: # when directed bond graph is empty or disabled
            batched_directed_bond_graph = torch.tensor([])

        if batched_dihedral_graph:
            batched_dihedral_graph = torch.cat(batched_dihedral_graph, dim=0)
        else: # when dihedral graph is empty or disabled
            batched_dihedral_graph = torch.tensor([])

        atom_owners = (
            torch.cat(atom_owners, dim=0).type(torch.int32).to(atomic_numbers.device)
        )
        directed2undirected = torch.cat(directed2undirected, dim=0)
        volumes = torch.tensor(volumes, dtype=TORCH_DTYPE, device=atomic_numbers.device)
        bond_vectors = torch.cat(bond_vectors, dim=0)

        batched_graph = cls(
            atomic_numbers=atomic_numbers, # [28]
            bond_bases_ag=bond_bases_ag, # [1272, 31]
            bond_bases_bg=bond_bases_bg, # [1272, 31]
            angle_bases=angle_bases, # [560, 31]
            dihedral_bases=dihedral_bases, # [976, 31]
            smooth_factors_ag=smooth_factors_ag, # [1272]
            smooth_factors_bg=smooth_factors_bg, # [1272]
            smooth_factors_atom_graph=smooth_factors_atom_graph, # [1272]
            batched_atom_graph=batched_atom_graph, # [1272, 2]
            batched_bond_graph=batched_bond_graph, # [560, 3]
            batched_directed_bond_graph=batched_directed_bond_graph, # [num_pairs, 2]
            batched_dihedral_graph=batched_dihedral_graph, # [0]
            atom_owners=atom_owners, # [28]
            directed2undirected=directed2undirected, # [1272]
            atom_positions=atom_positions, # list([28, 3])
            strains=strains, # list([3, 3])
            volumes=volumes, # list(tensor1)
            bond_vectors=bond_vectors, # [1272, 3]
            atom_graph_cutoff=atom_graph_cutoff, # 6
            bond_graph_cutoff=bond_graph_cutoff, # 3
        )

        return batched_graph
