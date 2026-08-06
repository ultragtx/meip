from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self

TORCH_DTYPE = torch.float32


class CrystalGraph:
    """A data class for crystal graph."""

    def __init__(
        self,
        atomic_number: Tensor,
        atom_frac_coord: Tensor,
        atom_graph: Tensor,
        atom_graph_cutoff: float,
        neighbor_image: Tensor,
        directed2undirected: Tensor,
        undirected2directed: Tensor,
        bond_graph: Tensor,
        bond_graph_cutoff: float,
        directed_bond_graph: Tensor,
        dihedral_graph: Tensor,
        dihedral_graph_cutoff: float,
        lattice: Tensor,
        graph_id: str | None = None,
        mp_id: str | None = None,
        composition: str | None = None,
    ) -> None:
        """Initialize the crystal graph.

        Attention! This data class is not intended to be created manually. CrystalGraph
        should be returned by a CrystalGraphConverter

        Args:
            atomic_number (Tensor): the atomic numbers of atoms in the structure
                [n_atom]
            atom_frac_coord (Tensor): the fractional coordinates of the atoms
                [n_atom, 3]
            atom_graph (Tensor): a directed graph adjacency list,
                (center atom indices, neighbor atom indices, undirected bond index)
                for bonds in bond_fea
                [num_directed_bonds, 2]
            atom_graph_cutoff (float): the cutoff radius to draw edges in atom_graph
            neighbor_image (Tensor): the periodic image specifying the location of
                neighboring atom
                see: https://github.com/materialsproject/pymatgen/blob/ca2175c762e37ea7
                c9f3950ef249bc540e683da1/pymatgen/core/structure.py#L1485-L1541
                [num_directed_bonds, 3]
            directed2undirected (Tensor): the mapping from directed edge index to
                undirected edge index for the atom graph
                [num_directed_bonds]
            undirected2directed (Tensor): the mapping from undirected edge index to
                one of its directed edge index, this is essentially the inverse
                mapping of the directed2undirected this tensor is needed for
                computation efficiency.
                Note that num_directed_bonds = 2 * num_undirected_bonds
                [num_undirected_bonds]
            bond_graph (Tensor): a directed graph adjacency list,
                (atom indices, 1st undirected bond idx, 1st directed bond idx,
                2nd undirected bond idx, 2nd directed bond idx) for angles in angle_fea
                [n_angle, 5]
            bond_graph_cutoff (float): the cutoff bond length to include bond
                as nodes in bond_graph
            lattice (Tensor): lattices of the input structure
                [3, 3]
            graph_id (str | None): an id to keep track of this crystal graph
                Default = None
            mp_id (str | None): Materials Project id of this structure
                Default = None
            composition: Chemical composition of the compound, used just for
                better tracking of the graph
                Default = None.

        Raises:
            ValueError: if len(directed2undirected) != 2 * len(undirected2directed)
        """
        super().__init__()
        self.atomic_number = atomic_number
        self.atom_frac_coord = atom_frac_coord
        self.atom_graph = atom_graph
        self.atom_graph_cutoff = atom_graph_cutoff
        self.neighbor_image = neighbor_image
        self.directed2undirected = directed2undirected
        self.undirected2directed = undirected2directed
        self.bond_graph = bond_graph
        self.bond_graph_cutoff = bond_graph_cutoff
        self.directed_bond_graph = directed_bond_graph
        self.dihedral_graph = dihedral_graph
        self.dihedral_graph_cutoff = dihedral_graph_cutoff
        self.lattice = lattice
        self.graph_id = graph_id
        self.mp_id = mp_id
        self.composition = composition


    def to(self, device: str = "cpu") -> CrystalGraph:
        """Move the graph to a device. Default = 'cpu'."""
        return CrystalGraph(
            atomic_number=self.atomic_number.to(device),
            atom_frac_coord=self.atom_frac_coord.to(device),
            atom_graph=self.atom_graph.to(device),
            atom_graph_cutoff=self.atom_graph_cutoff,
            neighbor_image=self.neighbor_image.to(device),
            directed2undirected=self.directed2undirected.to(device),
            undirected2directed=self.undirected2directed.to(device),
            bond_graph=self.bond_graph.to(device),
            bond_graph_cutoff=self.bond_graph_cutoff,
            directed_bond_graph=self.directed_bond_graph.to(device),
            dihedral_graph=self.dihedral_graph.to(device),
            dihedral_graph_cutoff=self.dihedral_graph_cutoff,
            lattice=self.lattice.to(device),
            graph_id=self.graph_id,
            mp_id=self.mp_id,
            composition=self.composition,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph to a dictionary."""
        return {
            "atomic_number": self.atomic_number,
            "atom_frac_coord": self.atom_frac_coord,
            "atom_graph": self.atom_graph,
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "neighbor_image": self.neighbor_image,
            "directed2undirected": self.directed2undirected,
            "undirected2directed": self.undirected2directed,
            "bond_graph": self.bond_graph,
            "bond_graph_cutoff": self.bond_graph_cutoff,
            "directed_bond_graph": self.directed_bond_graph,
            "dihedral_graph": self.dihedral_graph,
            "dihedral_graph_cutoff": self.dihedral_graph_cutoff,
            "lattice": self.lattice,
            "graph_id": self.graph_id,
            "mp_id": self.mp_id,
            "composition": self.composition,
        }

    def save(self, fname: str | None = None, save_dir: str = ".") -> str:
        """Save the graph to a file.

        Args:
            fname (str, optional): File name. Defaults to None.
            save_dir (str, optional): Directory to save the file. Defaults to ".".

        Returns:
            str: The path to the saved file.
        """
        if fname is not None:
            save_name = os.path.join(save_dir, fname)
        elif self.graph_id is not None:
            save_name = os.path.join(save_dir, f"{self.graph_id}.pt")
        else:
            save_name = os.path.join(save_dir, f"{self.composition}.pt")
        torch.save(self.to_dict(), f=save_name)
        return save_name

    @classmethod
    def from_file(cls, file_name: str) -> Self:
        """Load a crystal graph from a file.

        Args:
            file_name (str): The path to the file.

        Returns:
            CrystalGraph: The loaded graph.
        """
        return torch.load(file_name)

    @classmethod
    def from_dict(cls, dic: dict[str, Any]) -> Self:
        """Load a CrystalGraph from a dictionary."""
        return cls(**dic)

    def __repr__(self) -> str:
        """String representation of the graph."""
        composition = self.composition
        atom_graph_cutoff = self.atom_graph_cutoff
        bond_graph_cutoff = self.bond_graph_cutoff
        dihedral_graph_cutoff = self.dihedral_graph_cutoff
        atom_graph_len = self.atom_graph
        n_atoms = len(self.atomic_number)
        atom_graph_len = len(self.atom_graph)
        bond_graph_len = len(self.bond_graph)
        dihedral_graph_len = len(self.dihedral_graph)
        return (
            f"CrystalGraph({composition=}, {atom_graph_cutoff=}, {bond_graph_cutoff=}, "
            f"{dihedral_graph_cutoff=}, {n_atoms=}, {atom_graph_len=}, {bond_graph_len=}, {dihedral_graph_len=})"
        )

    @property
    def num_isolated_atoms(self) -> int:
        """Number of isolated atoms given the atom graph cutoff
        Isolated atoms are disconnected nodes in the atom graph
        that will not get updated in MEIP.
        These atoms will always have calculated force equal to zero.

        With the default MEIP atom graph cutoff radius, only ~ 0.1% of MPtrj dataset
        structures has isolated atoms.
        """
        return len(self.atomic_number) - torch.unique(self.atom_graph[:, 0]).numel()

    def to_dict_w_requires_grad(self) -> dict[str, Any]:
        """Convert to a dict, recording requires_grad info to support serialization."""
        requires_grad_flags = {}
        dtype_flags = {}
        data = {}

        for key in self.__dict__:
            if isinstance(getattr(self, key), torch.Tensor):
                requires_grad_flags[key] = getattr(self, key).requires_grad
                dtype_flags[key] = str(getattr(self, key).dtype)
                # Convert the tensor to a numpy array and ensure primitive types
                data[key] = getattr(self, key).detach().cpu().numpy().tolist()
            else:
                data[key] = getattr(self, key)

        data["requires_grad_flags"] = requires_grad_flags
        data["dtype_flags"] = dtype_flags
        return data

    @classmethod
    def from_dict_w_requires_grad(cls, data: dict[str, Any]) -> Self:
        """Restore a CrystalGraph from a dict, including requires_grad info."""
        # Remove the requires_grad_flags and dtype_flags keys first, since they are not class attributes
        requires_grad_flags = data.pop("requires_grad_flags", {})
        dtype_flags = data.pop("dtype_flags", {})

        # Create the object
        crystal_graph = cls.from_dict(data)

        # Restore requires_grad settings
        for key, requires_grad in requires_grad_flags.items():
            # Get the data type
            dtype = eval(dtype_flags[key])  # Convert the string to a torch.dtype
            # Convert the list back to a numpy array, then to a tensor, preserving the original dtype
            tensor_data = data[key]

            # Handle the numpy array case
            if isinstance(tensor_data, np.ndarray):
                # Choose the numpy dtype matching the torch dtype
                if dtype == torch.float32:
                    np_dtype = np.float32
                elif dtype == torch.float64:
                    np_dtype = np.float64
                elif dtype == torch.int32:
                    np_dtype = np.int32
                elif dtype == torch.int64:
                    np_dtype = np.int64
                elif dtype == torch.bool:
                    np_dtype = np.bool_
                else:
                    raise ValueError(f"Unsupported dtype {dtype}")

                # Ensure the numpy array has the correct dtype
                if tensor_data.dtype != np_dtype:
                    tensor_data = tensor_data.astype(np_dtype)

            # Convert to a tensor
            tensor_data = torch.tensor(tensor_data, requires_grad=requires_grad, dtype=dtype)
            setattr(crystal_graph, key, tensor_data)

        return crystal_graph


    def is_equal(self, other: Self) -> bool:
        """Compare if two CrystalGraph instances are equal.

        This method compares all attributes of two CrystalGraph instances, including:
        - Basic attributes (composition, cutoffs, IDs)
        - Tensor shapes
        - Tensor values
        - Tensor requires_grad properties

        Args:
            other (Self): Another CrystalGraph instance to compare with

        Returns:
            bool: True if all attributes are equal, False otherwise
        """
        # Compare basic attributes
        if self.composition != other.composition:
            return False
        if self.atom_graph_cutoff != other.atom_graph_cutoff:
            return False
        if self.bond_graph_cutoff != other.bond_graph_cutoff:
            return False
        if self.dihedral_graph_cutoff != other.dihedral_graph_cutoff:
            return False
        if self.graph_id != other.graph_id:
            return False
        if self.mp_id != other.mp_id:
            return False

        # Compare tensor shapes
        if self.atomic_number.shape != other.atomic_number.shape:
            return False
        if self.atom_frac_coord.shape != other.atom_frac_coord.shape:
            return False
        if self.atom_graph.shape != other.atom_graph.shape:
            return False
        if self.neighbor_image.shape != other.neighbor_image.shape:
            return False
        if self.directed2undirected.shape != other.directed2undirected.shape:
            return False
        if self.undirected2directed.shape != other.undirected2directed.shape:
            return False
        if self.bond_graph.shape != other.bond_graph.shape:
            return False
        if self.dihedral_graph.shape != other.dihedral_graph.shape:
            return False
        if self.lattice.shape != other.lattice.shape:
            return False

        # Compare tensor values
        if not torch.allclose(self.atomic_number, other.atomic_number):
            return False
        if not torch.allclose(self.atom_frac_coord, other.atom_frac_coord):
            return False
        if not torch.allclose(self.atom_graph, other.atom_graph):
            return False
        if not torch.allclose(self.neighbor_image, other.neighbor_image):
            return False
        if not torch.allclose(self.directed2undirected, other.directed2undirected):
            return False
        if not torch.allclose(self.undirected2directed, other.undirected2directed):
            return False
        if not torch.allclose(self.bond_graph, other.bond_graph):
            return False
        if not torch.allclose(self.dihedral_graph, other.dihedral_graph):
            return False
        if not torch.allclose(self.lattice, other.lattice):
            return False

        # Compare requires_grad properties
        if self.atomic_number.requires_grad != other.atomic_number.requires_grad:
            return False
        if self.atom_frac_coord.requires_grad != other.atom_frac_coord.requires_grad:
            return False
        if self.atom_graph.requires_grad != other.atom_graph.requires_grad:
            return False
        if self.neighbor_image.requires_grad != other.neighbor_image.requires_grad:
            return False
        if self.directed2undirected.requires_grad != other.directed2undirected.requires_grad:
            return False
        if self.undirected2directed.requires_grad != other.undirected2directed.requires_grad:
            return False
        if self.bond_graph.requires_grad != other.bond_graph.requires_grad:
            return False
        if self.dihedral_graph.requires_grad != other.dihedral_graph.requires_grad:
            return False
        if self.lattice.requires_grad != other.lattice.requires_grad:
            return False

        return True
