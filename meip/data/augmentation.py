"""
Data augmentation utilities for O(3) and SO(3) transformations.
Implements random rotations and reflections for crystal structures and their properties.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from pymatgen.core.structure import Structure
from pymatgen.core.operations import SymmOp
from scipy.spatial.transform import Rotation


class O3Augmentation:
    """
    O(3) augmentation for crystal structures and their properties.

    O(3) includes both rotations and reflections.
    SO(3) includes only rotations (no reflections).

    For equivariant models, the following transformations are applied:
    - Structure coordinates: R @ coords
    - Forces: R @ forces
    - Stress: R @ stress @ R.T
    - Magmom (if scalar): unchanged
    - Energy: unchanged
    """

    def __init__(
        self,
        enable_augmentation: bool = True,
        augmentation_prob: float = 0.5,
        use_o3: bool = True,  # True for O(3), False for SO(3)
        random_seed: Optional[int] = None
    ):
        """
        Initialize O3 augmentation.

        Args:
            enable_augmentation: Whether to enable augmentation
            augmentation_prob: Probability of applying augmentation
            use_o3: True for O(3) (rotations + reflections), False for SO(3) (rotations only)
            random_seed: Random seed for reproducibility
        """
        self.enable_augmentation = enable_augmentation
        self.augmentation_prob = augmentation_prob
        self.use_o3 = use_o3

        if random_seed is not None:
            np.random.seed(random_seed)

    def should_augment(self) -> bool:
        """Determine if augmentation should be applied based on probability."""
        if not self.enable_augmentation:
            return False
        return np.random.random() < self.augmentation_prob

    def generate_random_rotation_matrix(self) -> np.ndarray:
        """Generate a random rotation matrix."""
        # Generate random rotation using scipy
        rotation = Rotation.random()
        rotation_matrix = rotation.as_matrix()

        if self.use_o3:
            # For O(3), randomly add reflection (determinant = -1)
            if np.random.random() < 0.5:
                # Add reflection by negating one axis (inversion through a plane)
                axis = np.random.randint(3)  # Choose which axis to reflect
                reflection = np.eye(3)
                reflection[axis, axis] = -1  # Reflect through plane perpendicular to this axis
                rotation_matrix = reflection @ rotation_matrix

        return rotation_matrix.astype(np.float32)

    def transform_structure(self, structure: Structure, rotation_matrix: np.ndarray) -> Structure:
        """
        Apply rotation/reflection to complete crystal structure.

        For complete crystal rotation, the lattice and atomic coordinates are rotated
        consistently such that fractional coordinates remain unchanged. This is the
        correct behavior for O(3) equivariant models.

        Args:
            structure: Original pymatgen Structure
            rotation_matrix: 3x3 rotation/reflection matrix

        Returns:
            Transformed Structure with rotated lattice, preserving fractional coordinates
        """
        # For complete crystal rotation in pymatgen:
        # - Fractional coordinates remain unchanged
        # - New lattice matrix = old_lattice @ R.T
        # - New cartesian coordinates = R @ old_cartesian (automatically satisfied)

        # Transform lattice matrix using the correct pymatgen formula
        # In pymatgen: cart = lattice.matrix.T @ frac.T
        # For rotation: new_cart = R @ cart = R @ (lattice.matrix.T @ frac.T)
        # We want: new_cart = new_lattice.matrix.T @ frac.T
        # So: R @ lattice.matrix.T = new_lattice.matrix.T
        # Therefore: new_lattice.matrix = (R @ lattice.matrix.T).T = lattice.matrix @ R.T
        new_lattice_matrix = structure.lattice.matrix @ rotation_matrix.T

        # Create new lattice
        from pymatgen.core.lattice import Lattice
        new_lattice = Lattice(new_lattice_matrix)

        # Fractional coordinates remain unchanged for complete crystal rotation
        new_frac_coords = structure.frac_coords.copy()

        # Create new structure with same fractional coordinates but rotated lattice
        new_structure = Structure(
            lattice=new_lattice,
            species=[site.specie for site in structure.sites],
            coords=new_frac_coords,
            coords_are_cartesian=False
        )

        # Verify the transformation is correct (for debugging)
        expected_cart_coords = (rotation_matrix @ structure.cart_coords.T).T
        actual_cart_coords = new_structure.cart_coords
        cart_diff = np.max(np.abs(actual_cart_coords - expected_cart_coords))

        if cart_diff > 1e-10:
            print(f"Warning: Cartesian coordinate transformation error: {cart_diff:.2e}")

        return new_structure

    def transform_structure_with_symmop(self, structure: Structure, rotation_matrix: np.ndarray) -> Structure:
        """
        Apply rotation/reflection to complete crystal structure using pymatgen SymmOp.

        This method uses pymatgen's SymmOp for structure transformation, which provides
        a more standardized approach compared to manual matrix operations.

        Args:
            structure: Original pymatgen Structure
            rotation_matrix: 3x3 rotation/reflection matrix

        Returns:
            Transformed Structure using SymmOp
        """
        # Create SymmOp from rotation matrix (no translation)
        rotation_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=rotation_matrix,
            translation_vec=[0, 0, 0]
        )

        # Apply the operation to the structure
        transformed_structure = structure.copy()
        transformed_structure.apply_operation(rotation_op)

        return transformed_structure

    def transform_forces(self, forces: torch.Tensor, rotation_matrix: np.ndarray) -> torch.Tensor:
        """
        Transform forces under rotation/reflection.
        Forces transform as: f_new = R @ f_old

        Args:
            forces: Original forces tensor (N_atoms, 3)
            rotation_matrix: 3x3 rotation/reflection matrix

        Returns:
            Transformed forces tensor
        """
        rotation_tensor = torch.tensor(rotation_matrix, dtype=forces.dtype, device=forces.device)

        # Forces transform as vectors: f_new = R @ f_old
        transformed_forces = (rotation_tensor @ forces.T).T

        return transformed_forces

    def transform_stress(self, stress: torch.Tensor, rotation_matrix: np.ndarray) -> torch.Tensor:
        """
        Transform stress tensor under rotation/reflection.
        Stress transforms as: σ_new = R @ σ_old @ R.T

        Args:
            stress: Original stress tensor (3, 3) or (6,) in Voigt notation
            rotation_matrix: 3x3 rotation/reflection matrix

        Returns:
            Transformed stress tensor in same format as input
        """
        rotation_tensor = torch.tensor(rotation_matrix, dtype=stress.dtype, device=stress.device)

        if stress.shape == (6,):
            # Convert from Voigt notation to 3x3 matrix
            stress_matrix = self.voigt_to_matrix(stress)
        elif stress.shape == (3, 3):
            stress_matrix = stress
        else:
            raise ValueError(f"Unsupported stress tensor shape: {stress.shape}")

        # Transform stress: σ_new = R @ σ_old @ R.T
        transformed_stress_matrix = rotation_tensor @ stress_matrix @ rotation_tensor.T

        if stress.shape == (6,):
            # Convert back to Voigt notation
            return self.matrix_to_voigt(transformed_stress_matrix)
        else:
            return transformed_stress_matrix

    def transform_magmom(self, magmom: torch.Tensor, rotation_matrix: np.ndarray) -> torch.Tensor:
        """
        Transform magnetic moments.
        For scalar magnetic moments (magnitude only), no transformation needed.
        For vector magnetic moments, transform as vectors.

        Args:
            magmom: Magnetic moment tensor
            rotation_matrix: 3x3 rotation/reflection matrix

        Returns:
            Transformed magnetic moment tensor
        """
        if magmom.dim() == 1:
            # Scalar magnetic moments (magnitude only) - no transformation
            return magmom
        elif magmom.dim() == 2 and magmom.shape[1] == 3:
            # Vector magnetic moments - transform as vectors
            rotation_tensor = torch.tensor(rotation_matrix, dtype=magmom.dtype, device=magmom.device)
            return (rotation_tensor @ magmom.T).T
        else:
            # Unknown format, return unchanged
            return magmom

    @staticmethod
    def voigt_to_matrix(voigt_stress: torch.Tensor) -> torch.Tensor:
        """Convert stress from Voigt notation (6,) to matrix (3,3)."""
        matrix = torch.zeros(3, 3, dtype=voigt_stress.dtype, device=voigt_stress.device)
        matrix[0, 0] = voigt_stress[0]  # σ_xx
        matrix[1, 1] = voigt_stress[1]  # σ_yy
        matrix[2, 2] = voigt_stress[2]  # σ_zz
        matrix[1, 2] = matrix[2, 1] = voigt_stress[3]  # σ_yz
        matrix[0, 2] = matrix[2, 0] = voigt_stress[4]  # σ_xz
        matrix[0, 1] = matrix[1, 0] = voigt_stress[5]  # σ_xy
        return matrix

    @staticmethod
    def matrix_to_voigt(stress_matrix: torch.Tensor) -> torch.Tensor:
        """Convert stress from matrix (3,3) to Voigt notation (6,)."""
        voigt = torch.zeros(6, dtype=stress_matrix.dtype, device=stress_matrix.device)
        voigt[0] = stress_matrix[0, 0]  # σ_xx
        voigt[1] = stress_matrix[1, 1]  # σ_yy
        voigt[2] = stress_matrix[2, 2]  # σ_zz
        voigt[3] = stress_matrix[1, 2]  # σ_yz
        voigt[4] = stress_matrix[0, 2]  # σ_xz
        voigt[5] = stress_matrix[0, 1]  # σ_xy
        return voigt

    def augment_data(
        self,
        structure: Structure,
        targets: dict,
        use_symmop: bool = False,
    ) -> Tuple[Structure, dict]:
        """
        Apply augmentation to structure and targets.

        Args:
            structure: Original pymatgen Structure
            targets: Dictionary containing targets (e, f, s, m)
            use_symmop: If True, use SymmOp-based structure transformation.
                       If False, use the original manual matrix method.

        Returns:
            Tuple of (augmented_structure, augmented_targets)
        """
        if not self.should_augment():
            return structure, targets

        # Generate random rotation/reflection matrix
        rotation_matrix = self.generate_random_rotation_matrix()

        # Transform structure using selected method
        if use_symmop:
            augmented_structure = self.transform_structure_with_symmop(structure, rotation_matrix)
        else:
            augmented_structure = self.transform_structure(structure, rotation_matrix)

        # Transform targets
        augmented_targets = {}

        for key, value in targets.items():
            if value is None:
                augmented_targets[key] = None
                continue

            if key == "e":
                # Energy is invariant under rotations/reflections
                augmented_targets[key] = value
            elif key == "f":
                # Transform forces
                augmented_targets[key] = self.transform_forces(value, rotation_matrix)
            elif key == "s":
                # Transform stress
                augmented_targets[key] = self.transform_stress(value, rotation_matrix)
                # Disable stress augmentation
                # augmented_targets[key] = value
            elif key == "m":
                # Transform magnetic moments
                augmented_targets[key] = self.transform_magmom(value, rotation_matrix)
            else:
                # Unknown target, keep unchanged
                augmented_targets[key] = value

        return augmented_structure, augmented_targets
