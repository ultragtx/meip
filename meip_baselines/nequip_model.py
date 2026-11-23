"""NequIP baseline wrapped for the MEIP training and inference pipeline.

Wraps ``nequip.model.NequIPGNNModel`` so that it consumes MEIP crystal graphs and
predicts energy, forces, stress, and (through an added linear head) absolute
on-site magnetic moments.
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, get_args

import torch
import torch.nn as nn
from e3nn import o3
from pymatgen.core import Structure

from meip import PredTask
from meip.graph import CrystalGraph, CrystalGraphConverter
from meip.model.composition_model import AtomRef
from meip.utils import determine_device

# NequIP imports
from nequip.model.nequip_models import NequIPGNNModel
from nequip.data import AtomicDataDict
from nequip.data._keys import (
    POSITIONS_KEY,
    EDGE_INDEX_KEY,
    EDGE_CELL_SHIFT_KEY,
    CELL_KEY,
    PBC_KEY,
    ATOMIC_NUMBERS_KEY,
    ATOM_TYPE_KEY,
    BATCH_KEY,
    NUM_NODES_KEY,
    TOTAL_ENERGY_KEY,
    FORCE_KEY,
    STRESS_KEY,
    PER_ATOM_ENERGY_KEY,
    NODE_FEATURES_KEY,
)
from nequip.utils.global_state import set_global_state, global_state_initialized


if TYPE_CHECKING:
    from typing_extensions import Self


class NequIP(nn.Module):
    """CHGNet-compatible wrapper for NequIP model.
    
    This wrapper adapts NequIP's E(3)-equivariant graph neural network
    to work with CHGNet's data format and training infrastructure.
    """
    
    def __init__(
        self,
        *,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        radial_MLP_dims: Sequence[int] = (64, 64, 64),
        angle_fea_dim: int = 64,
        dihedral_fea_dim: int = 64,
        composition_model: str | torch.nn.Module | None = "MPtrj",
        num_radial: int = 8,
        num_angular: int = 8,
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
        graph_converter_algorithm: Literal["legacy", "fast", "original"] = "original",
        cutoff_coeff: int = 8,
        learnable_rbf: bool = True,
        gMLP_norm: str | None = "layer",
        readout_norm: str | None = "layer",
        version: str | None = None,
        enable_atom_nonlinearity: bool = False,
        enable_bond_nonlinearity: bool = False,
        # NequIP-specific parameters
        seed: int = 42,
        l_max: int = 1,
        parity: bool = True,
        radial_mlp_depth: int = 2,
        radial_mlp_width: int = 64,
        polynomial_cutoff_p: int = 6,
        avg_num_neighbors: float | None = 65.0,
        qm9_energy_shifts: dict[int, float] | None = None,  # QM9 atomic energy shifts
        per_type_energy_scales: dict[int, float] | None = None,  # Per-type force RMS for scaling
        num_features_per_l: list[int] | None = None,  # Per-l multiplicities (e.g., [64, 32, 16] for l=0,1,2)
        **kwargs,
    ):
        """Initialize NequIP model.
        
        Args:
            atom_fea_dim: Atom feature dimension (maps to num_features in NequIP)
            bond_fea_dim: Bond feature dimension (not directly used in NequIP)
            radial_MLP_dims: Radial MLP dimensions
            composition_model: Composition model for energy shifts ("MPtrj" or None)
            num_radial: Number of Bessel basis functions
            n_conv: Number of interaction layers
            atom_graph_cutoff: Cutoff radius for neighbor finding
            is_intensive: Whether to predict intensive (per-atom) energy
            seed: Random seed for reproducibility (NequIP requirement)
            l_max: Maximum rotation order for features (1 or 2)
            parity: Whether to include features with odd parity
            radial_mlp_depth: Depth of radial MLP
            radial_mlp_width: Width of radial MLP
            polynomial_cutoff_p: Polynomial cutoff exponent
            avg_num_neighbors: Average number of neighbors for normalization
            num_features_per_l: List of feature multiplicities for each l order.
                For l_max=2 with parity=True, provide [n0, n1, n2] which creates:
                n0x0e + n0x0o + n1x1e + n1x1o + n2x2e + n2x2o
                Example: [64, 32, 16] creates "64x0e+64x0o+32x1e+32x1o+16x2e+16x2o"
                If None, uses uniform atom_fea_dim for all l.
        """
        super().__init__()
        
        # Store model args early, before creating any local variables
        # Only store the actual constructor parameters
        import types
        self.model_args = {}
        for key, val in locals().items():
            if key not in {"self", "__class__", "kwargs"} and not isinstance(val, (types.ModuleType, type)):
                self.model_args[key] = val
        self.model_args.update(kwargs)
        if version:
            self.model_args["version"] = version
        
        # Initialize NequIP global state if not already initialized
        # IMPORTANT: NequIP by default uses float64, but CHGNet uses float32
        # We need to hack the global dtype before initializing
        if not global_state_initialized():
            # Temporarily override NequIP's _GLOBAL_DTYPE to float32
            import nequip.utils.global_dtype as global_dtype_module
            original_dtype = global_dtype_module._GLOBAL_DTYPE
            global_dtype_module._GLOBAL_DTYPE = torch.float32
            
            set_global_state(allow_tf32=False, warn_on_override=False)
            
            # Restore original (though we want it to stay float32)
            # global_dtype_module._GLOBAL_DTYPE = original_dtype
            
        # Ensure global default is float32
        torch.set_default_dtype(torch.float32)
            
        self.is_intensive = is_intensive
        self.atom_graph_cutoff = atom_graph_cutoff
        
        # Build type names for all elements (1-94)
        # NequIP uses type names like "type0", "type1", etc.
        num_elements = 94
        type_names = [f"type{i}" for i in range(num_elements)]
        
        # Get atomic energy shifts
        # Priority: qm9_energy_shifts > composition_model > None
        # NequIP 0.16.1+ requires dict format with ALL type_names as keys
        if qm9_energy_shifts is not None:
            # Use provided QM9 energy shifts
            # IMPORTANT: NequIP requires ALL type_names in the dict
            # CRITICAL: atom_types are 0-indexed (Z-1), so type0=H, type5=C, etc.
            per_type_energy_shifts = {f"type{i}": 0.0 for i in range(num_elements)}
            for z, energy in qm9_energy_shifts.items():
                if 1 <= z <= num_elements:  # z is atomic number (1-indexed)
                    type_idx = z - 1  # Convert to 0-indexed type
                    per_type_energy_shifts[f"type{type_idx}"] = float(energy)
            print(f"Using QM9 energy shifts for {len(qm9_energy_shifts)} atom types")
        elif isinstance(composition_model, str) and composition_model == "MPtrj":
            atom_ref = AtomRef(is_intensive=is_intensive)
            atom_ref.initialize_from_MPtrj_subset()
            energy_list = atom_ref.fc.weight.data.view(-1).detach().clone().tolist()
            # Convert to dict format - MUST include ALL type_names
            per_type_energy_shifts = {f"type{i}": 0.0 for i in range(num_elements)}
            for z, energy in enumerate(energy_list):
                if z < num_elements:
                    per_type_energy_shifts[f"type{z}"] = float(energy)
            print(f"Using MPtrj energy shifts")
        else:
            per_type_energy_shifts = None
            print("WARNING: No energy shifts provided - training may be difficult!")
        
        # Process per_type_energy_scales (for optimal NequIP training)
        # This is CRITICAL for achieving NequIP paper-level performance!
        # Scales should be set to per-type force RMS from training data
        nequip_per_type_scales = None
        if per_type_energy_scales is not None:
            # Convert from atomic number keyed dict to NequIP format
            nequip_per_type_scales = {f"type{i}": 1.0 for i in range(num_elements)}
            for z, scale in per_type_energy_scales.items():
                if 1 <= z <= num_elements:
                    type_idx = z - 1  # Convert to 0-indexed
                    nequip_per_type_scales[f"type{type_idx}"] = float(scale)
            print(f"Using per_type_energy_scales for {len(per_type_energy_scales)} atom types")
            for z, scale in sorted(per_type_energy_scales.items()):
                print(f"  Z={z}: scale={scale:.6f} eV/Å")
        
        # Determine num_features for NequIP
        # NequIP natively supports per-l multiplicities via num_features parameter
        if num_features_per_l is not None:
            # Use custom per-l multiplicities
            # Example: [64, 32, 16] for l_max=2 means 64 scalars, 32 vectors, 16 rank-2 tensors
            assert len(num_features_per_l) == l_max + 1, (
                f"num_features_per_l must have length l_max+1 ({l_max+1}), "
                f"but got {len(num_features_per_l)}: {num_features_per_l}"
            )
            nequip_num_features = num_features_per_l
            print(f"Using custom num_features: {num_features_per_l}")
            print(f"  This creates irreps with multiplicities: ", end="")
            for l, n in enumerate(num_features_per_l):
                print(f"{n}x(l={l})", end=" ")
            print()
        else:
            # Use uniform multiplicity (default NequIP behavior)
            nequip_num_features = atom_fea_dim
        
        # Prepare model kwargs
        model_kwargs = {
            # Mandatory NequIP arguments
            'seed': seed,
            'model_dtype': "float32",
            'type_names': type_names,
            # Model architecture
            'r_max': atom_graph_cutoff,
            'num_layers': n_conv,
            'l_max': l_max,
            'parity': parity,
            'num_features': nequip_num_features,  # Can be int or List[int]
            'radial_mlp_depth': radial_mlp_depth,
            'radial_mlp_width': radial_mlp_width,
            'num_bessels': num_radial,
            'bessel_trainable': learnable_rbf,
            'polynomial_cutoff_p': polynomial_cutoff_p,
            'avg_num_neighbors': avg_num_neighbors,
            'per_type_energy_shifts': per_type_energy_shifts,
            'per_type_energy_scales': nequip_per_type_scales,
            'do_derivatives': True,  # Enable force/stress computation
        }
        
        # Create NequIP model
        # NequIP requires seed, model_dtype, and type_names as mandatory arguments
        self.model = NequIPGNNModel(**model_kwargs)
        
        # Define Crystal Graph Converter
        self.graph_converter = CrystalGraphConverter(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            algorithm=graph_converter_algorithm,
            verbose=kwargs.pop("converter_verbose", False),
        )
        
        # Composition model (set to None, as NequIP handles it internally via per_type_energy_shifts)
        self.composition_model = None
        
        # Magnetic moment prediction layer
        # NequIP doesn't natively support magmom, so we add a simple readout
        self.magmom_norm = nn.LayerNorm(atom_fea_dim)
        self.magmom_readout = nn.Linear(atom_fea_dim, 1)
        
        # CRITICAL: Convert all NequIP model parameters to float32
        # NequIP uses float64 by default, but CHGNet uses float32
        self.model = self.model.to(torch.float32)
        self.magmom_norm = self.magmom_norm.to(torch.float32)
        self.magmom_readout = self.magmom_readout.to(torch.float32)
        
        version_str = f" v{version}" if version else ""
        print(f"NequIP{version_str} initialized with {self.n_params:,} parameters")

    
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
    ) -> dict[str, torch.Tensor]:
        """Get prediction associated with input graphs.
        
        Args:
            graphs: A list of CrystalGraphs
            task: The prediction task. One of 'e', 'em', 'ef', 'efs', 'efsm'.
            return_site_energies: Whether to return per-site energies.
            return_atom_feas: Whether to return atom features.
            return_crystal_feas: Whether to return crystal features.
            
        Returns:
            Model output dictionary.
        """
        # Convert CrystalGraph to NequIP data format
        nequip_data = self._create_nequip_data(graphs)
        
        # Pass to model
        prediction = self._compute(
            nequip_data,
            compute_force="f" in task,
            compute_stress="s" in task,
            compute_magmom="m" in task,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )
        
        return prediction
    
    def _create_nequip_data(self, graphs: Sequence[CrystalGraph]) -> dict[str, torch.Tensor]:
        """Convert CrystalGraphs to NequIP AtomicDataDict format."""
        device = next(self.parameters()).device
        
        positions_list = []
        atomic_numbers_list = []
        atom_types_list = []
        edge_index_list = []
        edge_cell_shift_list = []
        cell_list = []
        batch_list = []
        num_nodes_list = []
        
        atom_offset = 0
        for graph_idx, graph in enumerate(graphs):
            n_atoms = graph.atomic_number.shape[0]
            
            # Convert fractional to cartesian coordinates
            # Detach to create a clean tensor without computation history
            # Ensure float32 dtype for NequIP compatibility
            cart_coords = (graph.atom_frac_coord @ graph.lattice).detach().to(torch.float32)
            positions_list.append(cart_coords.to(device))
            
            # Atomic numbers (NequIP expects 0-indexed atom types)
            atomic_numbers = graph.atomic_number.to(device)
            atomic_numbers_list.append(atomic_numbers)
            # atom_types is 0-indexed (atomic_number - 1)
            atom_types_list.append(atomic_numbers - 1)
            
            # Edge index [2, num_edges] - NequIP format
            if graph.atom_graph.dim() == 1:
                graph.atom_graph = graph.atom_graph.reshape(0, 2)
            edge_index = graph.atom_graph.to(device).t().contiguous()  # [2, num_edges]
            edge_index_list.append(edge_index + atom_offset)
            
            # Edge cell shifts (periodic boundary conditions)
            # Ensure float32 dtype
            if len(graph.atom_graph) > 0:
                edge_cell_shift = graph.neighbor_image.to(torch.float32).to(device)  # [num_edges, 3]
            else:
                edge_cell_shift = torch.empty(0, 3, dtype=torch.float32, device=device)
            edge_cell_shift_list.append(edge_cell_shift)
            
            # Cell (lattice vectors) - detach to avoid gradient issues
            # Ensure float32 dtype
            cell_list.append(graph.lattice.detach().to(torch.float32).to(device).unsqueeze(0))  # [1, 3, 3]
            
            # Batch information
            batch_list.append(torch.full((n_atoms,), graph_idx, dtype=torch.long, device=device))
            num_nodes_list.append(n_atoms)
            
            atom_offset += n_atoms
        
        # Concatenate all data
        # Ensure all tensors are float32 to match model dtype
        nequip_data = {
            POSITIONS_KEY: torch.cat(positions_list, dim=0).to(torch.float32),
            ATOMIC_NUMBERS_KEY: torch.cat(atomic_numbers_list, dim=0),
            ATOM_TYPE_KEY: torch.cat(atom_types_list, dim=0),
            EDGE_INDEX_KEY: torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty(2, 0, dtype=torch.long, device=device),
            EDGE_CELL_SHIFT_KEY: torch.cat(edge_cell_shift_list, dim=0).to(torch.float32) if edge_cell_shift_list else torch.empty(0, 3, dtype=torch.float32, device=device),
            CELL_KEY: torch.cat(cell_list, dim=0).to(torch.float32),  # [batch_size, 3, 3]
            PBC_KEY: torch.tensor([[True, True, True]] * len(graphs), device=device),
            BATCH_KEY: torch.cat(batch_list, dim=0),
            NUM_NODES_KEY: torch.tensor(num_nodes_list, dtype=torch.long, device=device),
        }
        
        return nequip_data
    
    def _compute(
        self,
        data: dict[str, torch.Tensor],
        *,
        compute_force: bool = False,
        compute_stress: bool = False,
        compute_magmom: bool = False,
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Compute energy, forces, stress, and magnetic moments.
        
        Args:
            data: NequIP-format data dictionary
            compute_force: Whether to compute forces
            compute_stress: Whether to compute stress
            compute_magmom: Whether to compute magnetic moments
            return_site_energies: Whether to return per-site energies
            return_atom_feas: Whether to return atom features
            return_crystal_feas: Whether to return crystal features
            
        Returns:
            Prediction dictionary
        """
        prediction = {}
        
        # Get batch information
        batch = data[BATCH_KEY]
        atoms_per_graph = torch.bincount(batch)
        prediction["atoms_per_graph"] = atoms_per_graph
        
        # Note: NequIP's ForceStressOutput will handle requires_grad for positions
        # We need to temporarily disable do_derivatives if not computing force/stress
        original_do_derivatives = None
        if not compute_force and not compute_stress:
            # Find the ForceStressOutput module and temporarily disable derivatives
            for module in self.model.modules():
                if hasattr(module, 'do_derivatives'):
                    original_do_derivatives = module.do_derivatives
                    module.do_derivatives = False
                    break
        
        # Run NequIP model
        output = self.model(data)
        
        # Restore original do_derivatives setting
        if original_do_derivatives is not None:
            for module in self.model.modules():
                if hasattr(module, 'do_derivatives'):
                    module.do_derivatives = original_do_derivatives
                    break
        
        # Extract total energy
        if TOTAL_ENERGY_KEY in output:
            energy = output[TOTAL_ENERGY_KEY]  # Total energy per structure
            
            # NequIP returns total energy with shape [batch_size, 1] or [batch_size]
            # Ensure it's [batch_size]
            energy = energy.view(-1)  # Flatten to [batch_size]
            
            # NOTE: When is_intensive=True, we want per-atom energy
            # However, dataset now provides per-atom targets, so model should also output per-atom
            # Convert NequIP's total energy to per-atom energy
            if self.is_intensive:
                # Convert to per-atom energy
                atoms_per_graph_f = atoms_per_graph.to(energy.dtype).view(-1)
                energy = energy / torch.clamp(atoms_per_graph_f, min=1)
            
            # Convert to float32 for CHGNet compatibility
            prediction["e"] = energy.to(torch.float32)
        
        # Extract forces
        if compute_force and FORCE_KEY in output:
            forces = output[FORCE_KEY].to(torch.float32)  # [total_atoms, 3]
            prediction["f"] = list(torch.split(forces, atoms_per_graph.tolist()))
        
        # Extract stress
        if compute_stress and STRESS_KEY in output:
            stress = output[STRESS_KEY].to(torch.float32)  # [batch_size, 3, 3]
            # Convert eV/Å³ to GPa
            stress_gpa = stress * 160.21766208
            prediction["s"] = list(torch.unbind(stress_gpa, dim=0))
        
        # Compute magnetic moments from node features
        if compute_magmom and NODE_FEATURES_KEY in output:
            node_feats = output[NODE_FEATURES_KEY].to(torch.float32)
            node_feats_normed = self.magmom_norm(node_feats)
            magmom = torch.abs(self.magmom_readout(node_feats_normed))
            prediction["m"] = list(torch.split(magmom.view(-1), atoms_per_graph.tolist()))
        
        # Return site energies
        if return_site_energies and PER_ATOM_ENERGY_KEY in output:
            per_atom_energy = output[PER_ATOM_ENERGY_KEY].to(torch.float32)
            prediction["site_energies"] = list(torch.split(per_atom_energy, atoms_per_graph.tolist()))
        
        # Return atom features
        if return_atom_feas and NODE_FEATURES_KEY in output:
            node_feats = output[NODE_FEATURES_KEY].to(torch.float32)
            prediction["atom_fea"] = list(torch.split(node_feats, atoms_per_graph.tolist()))
        
        # Return crystal features (averaged node features)
        if return_crystal_feas and NODE_FEATURES_KEY in output:
            node_feats = output[NODE_FEATURES_KEY].to(torch.float32)
            crystal_feas = []
            start_idx = 0
            for n_atoms in atoms_per_graph:
                end_idx = start_idx + n_atoms
                crystal_feas.append(node_feats[start_idx:end_idx].mean(dim=0))
                start_idx = end_idx
            prediction["crystal_fea"] = torch.stack(crystal_feas)
        
        return prediction
    
    def predict_structure(
        self,
        structure: Structure | Sequence[Structure],
        *,
        task: PredTask = "efsm",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 16,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Predict from pymatgen.core.Structure.
        
        Args:
            structure: Structure or list of structures to predict.
            task: Prediction task, one of 'e', 'ef', 'em', 'efs', 'efsm'
            return_site_energies: Whether to return per-site energies.
            return_atom_feas: Whether to return atom features.
            return_crystal_feas: Whether to return crystal features.
            batch_size: Batch size for prediction.
            
        Returns:
            Prediction dictionary or list of dictionaries.
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
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Predict from CrystalGraph.
        
        Args:
            graph: CrystalGraph(s) to predict.
            task: Prediction task, one of 'e', 'ef', 'em', 'efs', 'efsm'
            return_site_energies: Whether to return per-site energies.
            return_atom_feas: Whether to return atom features.
            return_crystal_feas: Whether to return crystal features.
            batch_size: Batch size for prediction.
            
        Returns:
            Prediction dictionary or list of dictionaries.
        """
        if not isinstance(graph, (CrystalGraph, Sequence)):
            raise TypeError(
                f"{type(graph)=} must be CrystalGraph or list of CrystalGraphs"
            )
        valid_tasks = get_args(PredTask)
        if task not in valid_tasks:
            raise ValueError(f"Invalid {task=}. Must be one of {valid_tasks}.")
            
        model_device = next(self.parameters()).device
        
        graphs = [graph] if isinstance(graph, CrystalGraph) else graph
        self.eval()
        predictions: list[dict[str, torch.Tensor]] = [{} for _ in range(len(graphs))]
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
                "e", "f", "s", "m",
                "site_energies", "atom_fea", "crystal_fea",
            } & set(prediction.keys()):
                for idx, tensor in enumerate(prediction[key]):
                    predictions[step * batch_size + idx][key] = (
                        tensor.cpu().detach().numpy()
                    )
                    
        return predictions[0] if len(graphs) == 1 else predictions
    
    def as_dict(self) -> dict:
        """Return the NequIP weights and args in a dictionary."""
        return {"state_dict": self.state_dict(), "model_args": self.model_args}
    
    def todict(self) -> dict:
        """Needed for ASE JSON serialization."""
        return {"model_name": type(self).__name__, "model_args": self.model_args}
    
    @classmethod
    def from_dict(cls, dct: dict, **kwargs) -> "NequIP":
        """Build a NequIP from a saved dictionary."""
        chgnet = cls(**dct["model_args"], **kwargs)
        chgnet.load_state_dict(dct["state_dict"])
        return chgnet
    
    @classmethod
    def from_file(cls, path: str, **kwargs) -> "NequIP":
        """Build a NequIP from a saved file."""
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
    ):
        """Load a trained model from a checkpoint file and move it to the requested
        device ("cpu", "cuda", or "mps"; auto-selected when None).
        """
        model = cls.from_file(path, **kwargs)
        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)
        model = model.to(device)
        if verbose:
            print(f"{cls.__name__} will run on {device}")
        return model
