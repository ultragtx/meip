"""MACE baseline wrapped for the MEIP training and inference pipeline.

Subclasses ``mace.modules.models.ScaleShiftMACE`` so that it consumes MEIP crystal
graphs and predicts energy, forces, stress, and (through an added linear head)
absolute on-site magnetic moments.
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, get_args
if TYPE_CHECKING:
    from typing_extensions import Self

import torch
from e3nn import o3
from pymatgen.core import Structure

from meip import PredTask
from meip.graph import CrystalGraph, CrystalGraphConverter
from meip.model.composition_model import AtomRef
from meip.utils import determine_device

from mace.modules.models import ScaleShiftMACE
from mace.modules.blocks import RealAgnosticResidualInteractionBlock, LinearReadoutBlock
from mace.modules.wrapper_ops import CuEquivarianceConfig


class MACE(ScaleShiftMACE):
    def __init__(
        self, 
        *,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        radial_MLP_dims: Sequence[int],
        angle_fea_dim: int = 64,
        dihedral_fea_dim: int = 64,
        composition_model: str | torch.nn.Module = "MPtrj",
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
        qm9_energy_shifts: dict[int, float] | None = None,  # QM9 atomic energy shifts
        avg_num_neighbors: float | None = None,  # Dataset-specific average neighbors
        atomic_inter_scale: float | None = None,  # Atomic interaction scale
        atomic_inter_shift: float | None = None,  # Atomic interaction shift
        hidden_irreps_override: str | None = None,  # Override hidden irreps for parameter tuning
        **kwargs,
    ):
        mace_config = {}
        mace_config['r_max'] = atom_graph_cutoff
        mace_config['num_bessel'] = num_radial
        mace_config['num_polynomial_cutoff'] = 6
        mace_config['max_ell'] = 3
        mace_config['interaction_cls'] = RealAgnosticResidualInteractionBlock
        mace_config['interaction_cls_first'] = RealAgnosticResidualInteractionBlock
        mace_config['num_interactions'] = n_conv
        mace_config['num_elements'] = 94 # same as default max_num_elements of AtomRef
        
        # Hidden irreps - configurable for parameter tuning
        if hidden_irreps_override is not None:
            mace_config['hidden_irreps'] = o3.Irreps(hidden_irreps_override)
            print(f"Using custom hidden_irreps: {hidden_irreps_override}")
        else:
            # Default: scalars + vectors + rank-2 tensors
            mace_config['hidden_irreps'] = o3.Irreps(f"{atom_fea_dim}x0e+{atom_fea_dim}x1o+{atom_fea_dim}x2e")
        
        mace_config['MLP_irreps'] = o3.Irreps(f"16x0e") # Both mp-small and mp-large use this value
        
        # Get atomic energy shifts
        # Priority: qm9_energy_shifts > composition_model > None
        if qm9_energy_shifts is not None:
            # Use provided QM9 energy shifts (format: {Z: energy})
            # Convert to tensor of length 94 (for all elements)
            atomic_energies = torch.zeros(mace_config['num_elements'])
            for z, energy in qm9_energy_shifts.items():
                if 0 < z <= mace_config['num_elements']:
                    atomic_energies[z - 1] = float(energy)  # Z=1 -> index 0
            mace_config['atomic_energies'] = atomic_energies
            print(f"Using QM9 energy shifts for {len(qm9_energy_shifts)} atom types")
        elif isinstance(composition_model, str):
            # Get atomic energies from AtomRef composition model
            atom_ref = AtomRef(is_intensive=is_intensive)
            atom_ref.initialize_from_MPtrj_subset()
            mace_config['atomic_energies'] = atom_ref.fc.weight.data.view(-1).detach().clone()
            print(f"Using MPtrj energy shifts")
        else:
            mace_config['atomic_energies'] = torch.zeros(mace_config['num_elements'])
            print("WARNING: No energy shifts provided - training may be difficult!")
        
        # Set average number of neighbors (dataset-specific)
        if avg_num_neighbors is not None:
            mace_config['avg_num_neighbors'] = avg_num_neighbors
            print(f"Using provided avg_num_neighbors: {avg_num_neighbors:.2f}")
        else:
            # Default: Materials Project crystal dataset value
            mace_config['avg_num_neighbors'] = 65.00457000732422
            print(f"Using default avg_num_neighbors: {mace_config['avg_num_neighbors']:.2f} (MPtrj crystals)")
        
        # Set atomic interaction scale and shift
        # For pre-trained models, these are fitted values (from MPtrj-fitted model)
        # For training from scratch, use identity (1.0 and 0.0) - let model learn these
        if atomic_inter_scale is not None:
            mace_config['atomic_inter_scale'] = atomic_inter_scale
        else:
            # Default to identity for training from scratch
            # Use 0.8041538754478097 only for fine-tuning from pre-trained MPtrj model
            mace_config['atomic_inter_scale'] = 1.0  # Identity (training from scratch)
        
        if atomic_inter_shift is not None:
            mace_config['atomic_inter_shift'] = atomic_inter_shift
        else:
            # Default to identity for training from scratch
            # Use 0.16409696359187365 only for fine-tuning from pre-trained MPtrj model
            mace_config['atomic_inter_shift'] = 0.0  # Identity (training from scratch)
        
        print(f"Using atomic_inter_scale={mace_config['atomic_inter_scale']:.4f}, "
              f"atomic_inter_shift={mace_config['atomic_inter_shift']:.4f}")
            
        mace_config['atomic_numbers'] = torch.arange(1, mace_config['num_elements']+1)
        mace_config['correlation'] = 3
        mace_config['gate'] = torch.nn.SiLU()
        mace_config['pair_repulsion'] = False
        mace_config['distance_transform'] = "Agnesi"
        mace_config['radial_MLP'] = [atom_fea_dim, atom_fea_dim, atom_fea_dim]
        mace_config['radial_type'] = "bessel"
        mace_config['heads'] = ['Default']
        mace_config['cueq_config'] = CuEquivarianceConfig(
            enabled=False,
            optimize_all=False,
        )
        mace_config['lammps_mliap'] = None
        
        # Store model args for reconstruction
        self.model_args = {
            key: val
            for key, val in locals().items()
            # mace_config/atom_ref are derived locals, rebuilt from the args above
            if key not in {"self", "__class__", "kwargs", "mace_config", "atom_ref"}
        }
        self.model_args.update(kwargs)
        if version:
            self.model_args["version"] = version

        super().__init__(**mace_config)
        
        # Define Crystal Graph Converter
        self.graph_converter = CrystalGraphConverter(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            algorithm=graph_converter_algorithm,
            verbose=kwargs.pop("converter_verbose", False),
        )
        
        # composition model should always be None for MACE, because it has extra logic in train
        self.composition_model = None
        # Whether the target energy is intensive (per-atom). Align with other models.
        self.is_intensive = is_intensive

        
        # Add magnetic moment prediction layer (similar to CHGNet's site_wise layer)
        # Since we're working with the final concatenated node_feats, use a simple linear layer
        # but ensure it follows MACE's pattern for consistency
        self.hidden_irreps = mace_config['hidden_irreps']
        # Last layer in MACE is scalar-only; record its scalar dim precisely
        try:
            self.last_layer_scalar_dim = o3.Irreps(str(self.hidden_irreps[0])).dim
        except Exception:
            self.last_layer_scalar_dim = self.hidden_irreps.count(o3.Irrep(0, 1))
        # A light normalization before magmom readout to stabilize scale across models
        self.magmom_norm = torch.nn.LayerNorm(self.last_layer_scalar_dim)
        self.magmom_readout = LinearReadoutBlock(
            irreps_in=o3.Irreps(f"{self.last_layer_scalar_dim}x0e"),
            irrep_out=o3.Irreps(f"1x0e"),
            cueq_config=mace_config['cueq_config'],
        )
        
        version_str = f" v{version}" if version else ""
        print(f"MACE{version_str} initialized with {self.n_params:,} parameters")

    def forward(
        self,
        graphs: Sequence[CrystalGraph],
        *,
        task: PredTask = "e",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict[str, torch.Tensor]:
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

        # Make simplified batched graph
        batched_graph = self._create_batched_graph(graphs)

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

        return prediction

    def _create_batched_graph(self, graphs: Sequence[CrystalGraph]):
        """Create a simplified batched graph for MACE."""
        # Get device from model parameters
        device = next(self.parameters()).device
        
        atomic_numbers = []
        atom_positions = []
        atom_owners = []
        batched_atom_graph = []
        lattices = []
        shifts_list = []
        unit_shifts_list = []
        
        atom_offset = 0
        for graph_idx, graph in enumerate(graphs):
            n_atoms = graph.atomic_number.shape[0]
            atomic_numbers.append(graph.atomic_number.to(device))
            
            # Convert fractional coordinates to cartesian coordinates
            cart_coords = graph.atom_frac_coord @ graph.lattice
            atom_positions.append(cart_coords.to(device))
            atom_owners.extend([graph_idx] * n_atoms)
            lattices.append(graph.lattice.to(device))
            
            # Handle edge case where atom_graph is 1D
            if graph.atom_graph.dim() == 1:
                graph.atom_graph = graph.atom_graph.reshape(0, 2)
            
            batched_atom_graph.append(graph.atom_graph.to(device) + atom_offset)
            
            # Compute shifts for this graph
            if len(graph.atom_graph) > 0:
                # atom_graph is actually directed edges, and neighbor_image corresponds to each directed edge
                unit_shifts = graph.neighbor_image.to(device)  # [num_directed_edges, 3]
                
                # Convert unit shifts to cartesian shifts using this graph's lattice
                lattice = graph.lattice.to(device)  # [3, 3]
                # Convert fractional unit shifts to cartesian using the same convention as coords: cart = frac @ lattice
                shifts = unit_shifts @ lattice  # [num_directed_edges, 3]
                
                shifts_list.append(shifts)
                unit_shifts_list.append(unit_shifts)
            else:
                # Empty graph case
                shifts_list.append(torch.empty(0, 3, device=device))
                unit_shifts_list.append(torch.empty(0, 3, device=device))
            
            atom_offset += n_atoms
        
        # Create a simple object to hold the batched graph data (detached from gradients)
        return type('SimpleBatchedGraph', (), {
            'atomic_numbers': torch.cat(atomic_numbers, dim=0).detach(),
            'atom_positions': [pos.detach() for pos in atom_positions],
            'atom_owners': torch.tensor(atom_owners, dtype=torch.int32, device=device),
            'batched_atom_graph': torch.cat(batched_atom_graph, dim=0).detach() if batched_atom_graph else torch.empty(0, 2, dtype=torch.long, device=device),
            'lattices': [lat.detach() for lat in lattices],
            'shifts': torch.cat(shifts_list, dim=0).detach() if shifts_list else torch.empty(0, 3, device=device),
            'unit_shifts': torch.cat(unit_shifts_list, dim=0).detach() if unit_shifts_list else torch.empty(0, 3, device=device),
        })()

    def _compute(
        self,
        g,
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
            g: simplified batched graph
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

        # Convert simplified batched graph to MACE data format
        mace_data = self._convert_simple_graph_to_mace_data(g)
        
        # Call MACE forward method
        mace_output = super().forward(
            mace_data,
            compute_force=compute_force,
            compute_stress=compute_stress,
            # Enable symmetric displacement construction only when needed by stress/virials
            compute_displacement=bool(compute_stress),
            compute_virials=False,
            training=self.training,
        )
        
        # Convert MACE output back to CHGNet format
        energy = mace_output["energy"]  # total energy per structure
        # Convert to per-atom energy to match dataset target (energy_per_atom)
        if isinstance(energy, torch.Tensor):
            energy = energy.view(-1)
            if self.is_intensive:
                atoms_per_graph_f = atoms_per_graph.to(energy.dtype)
                energy = energy / torch.clamp(atoms_per_graph_f, min=1)
        prediction["e"] = energy

        # Handle forces
        if compute_force and mace_output["forces"] is not None:
            prediction["f"] = list(torch.split(mace_output["forces"], atoms_per_graph.tolist()))

        # Handle stress 
        if compute_stress and mace_output["stress"] is not None:
            # MACE stress already applies 1/V, and is in eV/Å^3; convert to GPa only.
            stress = mace_output["stress"]  # [batch_size, 3, 3], eV/Å^3
            stress_gpa = stress * 160.21766208
            prediction["s"] = list(torch.unbind(stress_gpa, dim=0))

        # Handle magnetic moments using node features
        if compute_magmom:
            if mace_output.get("node_feats") is not None:
                node_feats_concat = mace_output["node_feats"]
                
                # Extract only the last layer scalar features from concatenated node_feats
                node_feats_last = node_feats_concat[:, -self.last_layer_scalar_dim:]
                node_feats_last = self.magmom_norm(node_feats_last)
                
                # Create dummy node_heads for the readout (all zeros for single head)
                num_atoms = node_feats_last.shape[0]
                node_heads = torch.zeros(num_atoms, dtype=torch.long, device=node_feats_last.device)
                
                # Predict magnetic moments using LinearReadoutBlock
                # Apply absolute value to ensure non-negative magnetic moments
                magmom = torch.abs(self.magmom_readout(node_feats_last, node_heads))
                prediction["m"] = list(torch.split(magmom.view(-1), atoms_per_graph.tolist()))
            else:
                assert False
                # Fallback to zeros if node features are not available
                magmom = torch.zeros_like(g.atomic_numbers, dtype=torch.float32)
                prediction["m"] = list(torch.split(magmom.view(-1), atoms_per_graph.tolist()))

        # Handle site energies
        if return_site_energies and mace_output.get("node_energy") is not None:
            node_energy = mace_output["node_energy"]
            prediction["site_energies"] = list(torch.split(node_energy, atoms_per_graph.tolist()))

        # Handle atom features
        if return_atom_feas and mace_output.get("node_feats") is not None:
            node_feats = mace_output["node_feats"]
            prediction["atom_fea"] = list(torch.split(node_feats, atoms_per_graph.tolist()))

        # Handle crystal features  
        if return_crystal_feas:
            # Aggregate node features to get crystal features
            if mace_output.get("node_feats") is not None:
                node_feats = mace_output["node_feats"]
                # Simple averaging per graph
                crystal_feas = []
                start_idx = 0
                for n_atoms in atoms_per_graph:
                    end_idx = start_idx + n_atoms
                    crystal_feas.append(node_feats[start_idx:end_idx].mean(dim=0))
                    start_idx = end_idx
                prediction["crystal_fea"] = torch.stack(crystal_feas)

        return prediction
        
    def _convert_simple_graph_to_mace_data(self, g) -> dict[str, torch.Tensor]:
        """Convert simplified batched graph to MACE data format."""
        device = g.atomic_numbers.device
        
        # Create one-hot encoding for atomic numbers
        max_atomic_number = 94  # Same as in MACE config
        node_attrs = torch.zeros(
            (g.atomic_numbers.shape[0], max_atomic_number), 
            device=device
        )
        node_attrs.scatter_(1, g.atomic_numbers.unsqueeze(1) - 1, 1)
        
        # Create batch indices
        batch = g.atom_owners.to(torch.int64).to(device)
        
        # Create edge index from batched_atom_graph  
        edge_index = g.batched_atom_graph.to(device).t().contiguous()
        
        # Create positions from atom_positions (data is already detached from SimpleBatchedGraph)
        positions = torch.cat([pos.to(device) for pos in g.atom_positions], dim=0)
        
        # Stack lattices for all graphs (data is already detached from SimpleBatchedGraph)
        cell = torch.stack([lat.to(device) for lat in g.lattices])  # [num_graphs, 3, 3]
        
        # Use precomputed shifts and unit_shifts
        shifts = g.shifts.to(device)
        unit_shifts = g.unit_shifts.to(device)
        
        # Create ptr for batch information
        atoms_per_graph = torch.bincount(batch)
        ptr = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),
            torch.cumsum(atoms_per_graph, dim=0)
        ])
        
        # Create head information (for multi-head models, simplified to single head)
        head = torch.zeros_like(batch)
        
        return {
            "node_attrs": node_attrs,
            "batch": batch,
            "edge_index": edge_index,
            "positions": positions,
            "cell": cell,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "ptr": ptr,
            "head": head,
        }
        
    @property
    def version(self) -> str | None:
        """Return the version of the loaded checkpoint."""
        return self.model_args.get("version")

    @property
    def n_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    


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
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
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
        """Return the CHGNet weights and args in a dictionary."""
        return {"state_dict": self.state_dict(), "model_args": self.model_args}

    def todict(self) -> dict:
        """Needed for ASE JSON serialization when saving CHGNet potential to
        trajectory file (https://github.com/CederGroupHub/chgnet/issues/48).
        """
        return {"model_name": type(self).__name__, "model_args": self.model_args}
    
    @classmethod
    def from_dict(cls, dct: dict, **kwargs):
        """Build a CHGNet from a saved dictionary."""
        chgnet = cls(**dct["model_args"], **kwargs)
        chgnet.load_state_dict(dct["state_dict"])
        return chgnet

    @classmethod
    def from_file(cls, path: str, **kwargs):
        """Build a CHGNet from a saved file."""
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
