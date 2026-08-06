from __future__ import annotations

import torch
from torch import Tensor, nn
from e3nn import o3
from meip.model.basis import Fourier, RadialBessel


class AtomEmbedding(nn.Module):
    """Encode an atom by its atomic number using a learnable embedding layer."""

    def __init__(self, atom_feature_dim: int, max_num_elements: int = 94) -> None:
        """Initialize the Atom featurizer.

        Args:
            atom_feature_dim (int): dimension of atomic embedding.
            max_num_elements (int): maximum number of elements in the dataset.
                Default = 94
        """
        super().__init__()
        self.embedding = nn.Embedding(max_num_elements, atom_feature_dim)

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        """Convert the structure to a atom embedding tensor.

        Args:
            atomic_numbers (Tensor): [n_atom, 1].

        Returns:
            atom_fea (Tensor): atom embeddings [n_atom, atom_feature_dim].
        """
        return self.embedding(atomic_numbers)


class BondEncoder(nn.Module):
    """Encode a chemical bond given the positions of two atoms using Gaussian
    distance.
    """

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
            atom_graph_cutoff (float): The cutoff for constructing AtomGraph default = 5
            bond_graph_cutoff (float): The cutoff for constructing BondGraph default = 3
            num_radial (int): The number of radial component. Default = 9
            cutoff_coeff (int): Strength for graph cutoff smoothness. Default = 5
            learnable(bool): Whether the frequency in rbf expansion is learnable.
                Default = False
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
        center: Tensor, # [1272, 3]
        neighbor: Tensor, # [1272, 3]
        undirected2directed: Tensor, # [1264]
        image: Tensor, # [1272, 3]
        lattice: Tensor, # [3, 3]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute the pairwise distance between 2 3d coordinates.

        Args:
            center (Tensor): 3d cartesian coordinates of center atoms [n_directed_bond, 3]
            neighbor (Tensor): 3d cartesian coordinates of neighbor atoms [n_directed_bond, 3]
            undirected2directed (Tensor): mapping from undirected bond to one of its
                directed bond [n_undirected_bond]
            image (Tensor): the periodic image specifying the location of neighboring
                atom [n_directed_bond, 3]
            lattice (Tensor): the lattice of this structure [3, 3]

        Returns:
            bond_basis_ag (Tensor): the bond basis in AtomGraph [n_undirected_bond, num_radial]
            bond_basis_ag (Tensor): the bond basis in BondGraph [n_undirected_bond, num_radial]
            bond_vectors (Tensor): normalized bond vectors, for tracking the bond
                directions [n_directed_bond, 3]
        """
        neighbor = neighbor + image @ lattice # [1272, 3], translate neighbor atoms by their periodic image into the cell of the center atoms
        bond_vectors = center - neighbor # [1272, 3], vector from each atom to its neighbor
        bond_lengths = torch.norm(bond_vectors, dim=1) # [1272], distance from each atom to its neighbor
        # Normalize the bond vectors
        assert (bond_lengths == 0).sum() == 0
        bond_vectors = bond_vectors / bond_lengths[:, None] # [1272, 3], normalize each atom-to-neighbor vector

        # We create bond features only for undirected bonds
        # atom1 -> atom2 and atom2 -> atom1 should share same bond_basis
        undirected_bond_lengths = torch.index_select(
            bond_lengths, 0, undirected2directed
        ) # [1264], length of each undirected bond
        bond_basis_ag, smooth_factor_ag = self.rbf_expansion_ag(undirected_bond_lengths, return_smooth_factor=True)
        bond_basis_bg, smooth_factor_bg = self.rbf_expansion_bg(undirected_bond_lengths, return_smooth_factor=True)
        return bond_basis_ag, bond_basis_bg, bond_vectors, smooth_factor_ag.squeeze(-1), smooth_factor_bg.squeeze(-1) # bond_vectors is twice as long as the basis tensors


class AngleEncoder(nn.Module):
    """Encode an angle given the two bond vectors using Fourier Expansion."""

    def __init__(self, *, num_angular: int = 9, learnable: bool = True) -> None:
        """Initialize the angle encoder.

        Args:
            num_angular (int): number of angular basis to use. Must be an odd integer.
            learnable (bool): whether to set the frequencies of the Fourier expansion
                as learnable parameters. Default = False
        """
        super().__init__()
        if num_angular % 2 != 1:
            raise ValueError(f"{num_angular=} must be an odd integer")
        circular_harmonics_order = (num_angular - 1) // 2
        self.fourier_expansion = Fourier(
            order=circular_harmonics_order, learnable=learnable
        )

    def forward(self, bond_i: Tensor, bond_j: Tensor) -> Tensor:
        """Compute the angles between normalized vectors.

        Args:
            bond_i (Tensor): normalized left bond vector [n_angle, 3]
            bond_j (Tensor): normalized right bond vector [n_angle, 3]

        Returns:
            angle_fea (Tensor):  expanded cos_ij [n_angle, angle_feature_dim]
        """
        # 1 - 1e-6 for torch.acos stability
        cosine_ij = torch.sum(bond_i * bond_j, dim=1) * (1 - 1e-6)
        angle = torch.acos(cosine_ij)
        return self.fourier_expansion(angle)

class DihedralEncoder(nn.Module):
    """Encode a dihedral angle given the four bond vectors using Fourier Expansion."""

    def __init__(self, *, num_angular: int = 9, learnable: bool = True) -> None:
        """Initialize the dihedral encoder.

        Args:
            num_angular (int): number of angular basis to use. Must be an odd integer.
            learnable (bool): whether to set the frequencies of the Fourier expansion
                as learnable parameters. Default = False
        """
        super().__init__()
        if num_angular % 2 != 1:
            raise ValueError(f"{num_angular=} must be an odd integer")
        circular_harmonics_order = (num_angular - 1) // 2
        self.fourier_expansion = Fourier(
            order=circular_harmonics_order, learnable=learnable
        )

        self.eps = 1e-9

    def forward(self, bond_left: Tensor, bond_center: Tensor, bond_right: Tensor) -> Tensor:
        return self.forward_gemnet(bond_left, bond_center, bond_right)

    def forward_debug(self, bond_left: Tensor, bond_center: Tensor, bond_right: Tensor) -> Tensor:
        """Debug version that uses a fixed 30-degree angle for all dihedrals.

        Args:
            bond_left (Tensor): normalized left bond vector [n_dihedral, 3]  # B->A
            bond_center (Tensor): normalized center bond vector [n_dihedral, 3]  # B->C
            bond_right (Tensor): normalized right bond vector [n_dihedral, 3]  # C->D

        Returns:
            dihedral_fea (Tensor): expanded dihedral angles [n_dihedral, dihedral_feature_dim]
        """
        # Create a tensor of fixed 30-degree angles (π/6 radians)
        n_dihedral = bond_left.shape[0]
        fixed_angle = torch.full((n_dihedral,), torch.pi / 6, device=bond_left.device)

        # Apply Fourier expansion to the fixed angle
        return self.fourier_expansion(fixed_angle)

    def forward_v3(self, bond_left: Tensor, bond_center: Tensor, bond_right: Tensor) -> Tensor:
        """
        Args:
            bond_left (Tensor): normalized left bond vector [n_dihedral, 3]  # B->A
            bond_center (Tensor): normalized center bond vector [n_dihedral, 3]  # B->C
            bond_right (Tensor): normalized right bond vector [n_dihedral, 3]  # C->D

        Returns:
            dihedral_fea (Tensor): expanded dihedral angles [n_dihedral, dihedral_feature_dim]
        """
        bond_ab = -bond_left   # A->B = -(B->A)
        bond_bc = bond_center  # B->C
        bond_cd = bond_right   # C->D

        # GemNet-style: use vector rejection to project AB and CD onto plane orthogonal to BC
        proj_ab = self.vector_rejection(bond_ab, bond_bc)  # A-B projected onto BC plane
        proj_cd = self.vector_rejection(bond_cd, bond_bc)  # D-C projected onto BC plane

        # atan2(sin, cos) formulation for angle between proj_ab and proj_cd
        # Clamp x = <u, v> to (-1+eps, 1-eps) so floating-point error cannot push cos(theta) slightly outside [-1, 1]
        x = torch.sum(proj_ab * proj_cd, dim=1).clamp(-1+1e-6, 1-1e-6)  # dot product (cosine part)
        cross = torch.cross(proj_ab, proj_cd, dim=1)  # compute cross product
        sin_signed = torch.sum(cross * bond_bc, dim=1) \
                 / (torch.norm(bond_bc, dim=1) * 1.0)  # optional normalization

        # Final signed dihedral angle
        angle = torch.atan2(sin_signed, x)                     # [-π, π]

        return self.fourier_expansion(angle)

    def forward_v2_stabilized(self, bond_left: Tensor, bond_center: Tensor, bond_right: Tensor) -> Tensor:
        """ Stabilized version of v2 using safe normalization. Computes angle in [0, pi].
        Args:
            bond_left (Tensor): normalized left bond vector [n_dihedral, 3]  # B->A
            bond_center (Tensor): normalized center bond vector [n_dihedral, 3]  # B->C
            bond_right (Tensor): normalized right bond vector [n_dihedral, 3]  # C->D

        Returns:
            dihedral_fea (Tensor): expanded dihedral angles [n_dihedral, num_angular]
        """
        bond_ab = -bond_left   # A->B = -(B->A)
        bond_bc = bond_center  # B->C
        bond_cd = bond_right   # C->D

        # GemNet-style: use vector rejection to project AB and CD onto plane orthogonal to BC
        proj_ab = self.vector_rejection_v2(bond_ab, bond_bc, eps=self.eps)
        proj_cd = self.vector_rejection_v2(bond_cd, bond_bc, eps=self.eps)

        # --- Stabilization Start ---
        # Calculate norms of projections
        norm_proj_ab = torch.norm(proj_ab, dim=-1, keepdim=True)
        norm_proj_cd = torch.norm(proj_cd, dim=-1, keepdim=True)

        # Normalize the projected vectors safely, clamping denominator away from zero
        # This prevents division by zero if proj_ab or proj_cd are zero vectors (collinear case)
        safe_proj_ab = proj_ab / norm_proj_ab.clamp(min=self.eps)
        safe_proj_cd = proj_cd / norm_proj_cd.clamp(min=self.eps)

        # Calculate cosine (x) from normalized vectors
        # Clamp the cosine value to [-1+eps, 1-eps] to avoid numerical issues at +/-1
        x = torch.sum(safe_proj_ab * safe_proj_cd, dim=1).clamp(-1 + self.eps, 1 - self.eps)

        # Calculate sine magnitude (y) from normalized vectors
        # The norm of the cross product of *unit* vectors gives the sine magnitude
        # Since safe_proj_* are normalized, the input to norm here should be non-zero unless they are perfectly collinear
        y = torch.norm(torch.cross(safe_proj_ab, safe_proj_cd, dim=1), dim=1)
        # y = y.clamp(min=self.eps) # Optional: clamp y as well, although less critical now

        # --- Stabilization End ---

        # Calculate angle using atan2(y, x) -> [0, pi]
        # With normalized inputs leading to x=cos, y=sin_mag, x^2+y^2 should be close to 1,
        # avoiding the denominator=0 issue in atan2 gradient.
        angle = torch.atan2(y, x)

        # Handle cases where original projections were zero (collinear atoms)
        # If the original norm was near zero, the angle is ill-defined. Set to 0.
        # This prevents potential NaNs from 0/0 during safe_proj_* calculation propagating
        # Although clamping should handle division, setting angle explicitly might be safer.
        mask = (norm_proj_ab < self.eps) | (norm_proj_cd < self.eps)
        angle = torch.where(mask.squeeze(-1), torch.zeros_like(angle), angle)

        return self.fourier_expansion(angle)


    def forward_v2(self, bond_left: Tensor, bond_center: Tensor, bond_right: Tensor) -> Tensor:
        """
        Args:
            bond_left (Tensor): normalized left bond vector [n_dihedral, 3]  # B->A
            bond_center (Tensor): normalized center bond vector [n_dihedral, 3]  # B->C
            bond_right (Tensor): normalized right bond vector [n_dihedral, 3]  # C->D

        Returns:
            dihedral_fea (Tensor): expanded dihedral angles [n_dihedral, dihedral_feature_dim]
        """
        bond_ab = -bond_left   # A->B = -(B->A)
        bond_bc = bond_center  # B->C
        bond_cd = bond_right   # C->D

        # GemNet-style: use vector rejection to project AB and CD onto plane orthogonal to BC
        proj_ab = self.vector_rejection(bond_ab, bond_bc)  # A-B projected onto BC plane
        proj_cd = self.vector_rejection(bond_cd, bond_bc)  # D-C projected onto BC plane

        # atan2(sin, cos) formulation for angle between proj_ab and proj_cd
        x = torch.sum(proj_ab * proj_cd, dim=1)  # dot product (cosine part)
        y = torch.norm(torch.cross(proj_ab, proj_cd), dim=1)  # sine part (norm of cross)

        # Avoid NaNs by clamping y (sine norm) to small epsilon
        y = torch.clamp(y, min=1e-9)

        angle = torch.atan2(y, x)  # angle in [0, pi]
        return self.fourier_expansion(angle)

    def forward_gemnet(self, bond_left: Tensor, bond_center: Tensor, bond_right: Tensor) -> Tensor:
        """
        Args:
            bond_left (Tensor): normalized left bond vector [n_dihedral, 3]  # B->A
            bond_center (Tensor): normalized center bond vector [n_dihedral, 3]  # B->C
            bond_right (Tensor): normalized right bond vector [n_dihedral, 3]  # C->D
        """
        bond_ba = bond_left
        bond_bc = bond_center
        ba_proj = self.vector_rejection_gemnet(bond_ba, bond_bc)

        bond_cd = bond_right
        bond_cb = -bond_bc
        cd_proj = self.vector_rejection_gemnet(bond_cb, bond_cd)

        x = torch.sum(ba_proj * cd_proj, dim=-1)
        y = torch.cross(ba_proj, cd_proj).norm(dim=-1)

        y = torch.max(y, torch.tensor(1e-9, device=bond_left.device))
        angle = torch.atan2(y, x)
        return self.fourier_expansion(angle)

    @staticmethod
    def vector_rejection_v2(v: Tensor, n: Tensor, eps: float = 1e-9) -> Tensor:
        """Project vector v onto the plane orthogonal to n."""
        # Ensure n is not zero vector for projection calculation
        norm_sq_n = torch.sum(n * n, dim=-1, keepdim=True) # [N, 1]
        # Clamp denominator away from zero *before* division
        safe_norm_sq_n = norm_sq_n.clamp(min=eps)          # [N, 1]
        dot_vn = torch.sum(v * n, dim=-1, keepdim=True)    # [N, 1]
        # Calculate projection of v onto n
        proj_v_on_n = (dot_vn / safe_norm_sq_n) * n
        # Subtract projection to get rejection
        rejection = v - proj_v_on_n
        return rejection

    @staticmethod
    def vector_rejection(v: Tensor, n: Tensor, eps: float = 1e-9) -> Tensor:
        """Project vector v onto the plane orthogonal to n."""
        dot = torch.sum(v * n, dim=-1, keepdim=True)  # [N, 1]
        norm_sq = torch.sum(n * n, dim=-1, keepdim=True).clamp(min=eps)  # [N, 1]
        return v - (dot / norm_sq) * n

    @staticmethod
    def vector_rejection_gemnet(R_ab: Tensor, P_n: Tensor, eps: float = 1e-9) -> Tensor:
        """
        Project the vector R_ab onto a plane with normal vector P_n.

        Parameters
        ----------
            R_ab: Tensor, shape = (N,3)
                Vector from atom a to b.
            P_n: Tensor, shape = (N,3)
                Normal vector of a plane onto which to project R_ab.

        Returns
        -------
            R_ab_proj: Tensor, shape = (N,3)
                Projected vector (orthogonal to P_n).
        """
        a_x_b = torch.sum(R_ab * P_n, dim=-1)
        b_x_b = torch.sum(P_n * P_n, dim=-1)
        return R_ab - (a_x_b / b_x_b)[:, None] * P_n

    def forward_v0(self, bond_left: Tensor, bond_center: Tensor, bond_right: Tensor) -> Tensor:
        """Compute the dihedral angles between normalized vectors.

        Args:
            bond_left (Tensor): normalized left bond vector [n_dihedral, 3] # B->A
            bond_center (Tensor): normalized center bond vector [n_dihedral, 3] # B->C
            bond_right (Tensor): normalized right bond vector [n_dihedral, 3] # C->D

        Returns:
            dihedral_fea (Tensor): expanded dihedral angles [n_dihedral, dihedral_feature_dim]
        """
        # Since bond_left is B->A, we need to reverse it to get A->B
        bond_ab = -bond_left  # A->B = -(B->A)
        bond_bc = bond_center  # B->C
        bond_cd = bond_right  # C->D

        # Calculate normal vectors of the two planes
        n1 = torch.cross(bond_ab, bond_bc)  # normal vector of A-B-C plane
        n2 = torch.cross(bond_bc, bond_cd)  # normal vector of B-C-D plane

        # Normalize the normal vectors
        n1 = n1 / torch.norm(n1, dim=1, keepdim=True) # can produce NaN when n1/n2 are zero vectors, i.e. ab and bc are nearly parallel
        n2 = n2 / torch.norm(n2, dim=1, keepdim=True)

        # Calculate the dihedral angle using the dot product of normal vectors
        # and the sign of the triple product to determine the sign of the angle
        cos_angle = torch.sum(n1 * n2, dim=1)
        sign = torch.sign(torch.sum(bond_bc * torch.cross(n1, n2), dim=1))

        # Calculate the dihedral angle in radians
        # 1 - 1e-6 for torch.acos stability
        dihedral = sign * torch.acos(cos_angle * (1 - 1e-6))

        return self.fourier_expansion(dihedral)

MULTI_BODY_ENCODER_VERSION=3 # 1: per-order dot-product loop; 2: multiply m=0 components; 3: TensorProduct

class ThreeBodyEncoder(torch.nn.Module):
    def __init__(self, num_angular=9, learnable=False):
        super().__init__()

        if MULTI_BODY_ENCODER_VERSION == 1 or MULTI_BODY_ENCODER_VERSION == 2:
            max_l = 3
        elif MULTI_BODY_ENCODER_VERSION == 3:
            max_l = 3 # fixed; expansion is handled by the tensor product
        self.max_l = max_l

        # Spherical harmonics calculator
        self.spherical_harmonics = o3.SphericalHarmonics(
            list(range(max_l + 1)),
            normalize=True,
        )

        if MULTI_BODY_ENCODER_VERSION == 2:
            # Precompute the indices of all m=0 components
            self.m0_indices = []
            idx = 0
            for l in range(max_l + 1):
                self.m0_indices.append(idx + l)
                idx += 2 * l + 1
            # Convert the indices to a tensor and store them once
            self.register_buffer('m0_indices_tensor',
                                torch.tensor(self.m0_indices, dtype=torch.long))
        elif MULTI_BODY_ENCODER_VERSION == 3:
            self.tp = o3.FullyConnectedTensorProduct(
                irreps_in1=o3.Irreps.spherical_harmonics(max_l),
                irreps_in2=o3.Irreps.spherical_harmonics(max_l),
                irreps_out=o3.Irreps(f"{num_angular}x0e"),  # output invariants only (0e denotes scalars)
                shared_weights=True,
                internal_weights=True,
            )

        if MULTI_BODY_ENCODER_VERSION == 1 or MULTI_BODY_ENCODER_VERSION == 2:
            self.linear = nn.Linear(max_l + 1, num_angular)

    def _compute_invariants(self, v1, v2):
        """Compute invariants from two vectors using spherical harmonics.

        Args:
            v1 (Tensor): First vector [batch_size, 3]
            v2 (Tensor): Second vector [batch_size, 3]

        Returns:
            Tensor: Invariant features [batch_size, feature_dim]
        """
        # 1. Compute spherical harmonics
        sh1 = self.spherical_harmonics(v1)  # [batch_size, (l_max+1)^2]
        sh2 = self.spherical_harmonics(v2)  # [batch_size, (l_max+1)^2]

        # 2. Compute invariants
        if MULTI_BODY_ENCODER_VERSION == 1:
            features = []
            start_idx = 0
            for l in range(self.max_l + 1):
                # Extract the order-l component from the spherical harmonics coefficients
                end_idx = start_idx + 2*l + 1
                sh1_l = sh1[:, start_idx:end_idx]  # [batch_size, 2l+1]
                sh2_l = sh2[:, start_idx:end_idx]  # [batch_size, 2l+1]

                # Compute the invariant (dot product of order-l spherical harmonics)
                invariant = torch.sum(sh1_l * sh2_l, dim=-1, keepdim=True)  # [batch_size, 1]
                features.append(invariant)

                start_idx = end_idx
            # Concatenate all invariants
            features = torch.cat(features, dim=-1)  # [batch_size, max_l+1]
            features = self.linear(features)
        elif MULTI_BODY_ENCODER_VERSION == 2:
            # Extract all m=0 components at once
            sh1_m0 = sh1[:, self.m0_indices_tensor]  # [batch_size, max_l+1]
            sh2_m0 = sh2[:, self.m0_indices_tensor]  # [batch_size, max_l+1]

            # Product of the m=0 components
            features = sh1_m0 * sh2_m0  # [batch_size, max_l+1]
            features = self.linear(features)
        elif MULTI_BODY_ENCODER_VERSION == 3:
            features = self.tp(sh1, sh2)

        return features

    def forward(self, v1, v2):
        """
        Args:
            v1: first vector [batch_size, 3]
            v2: second vector [batch_size, 3]
        Returns:
            invariant features [batch_size, feature_dim]
        """
        return self._compute_invariants(v1, v2)

class FourBodyEncoder(ThreeBodyEncoder):
    def forward(self, v1, v2, v3):
        """
        Args:
            v1, v2, v3: three vectors [batch_size, 3]
            e.g. v1=BA, v2=BC, v3=CD
        Returns:
            invariant features [batch_size, feature_dim]
        """

        v1_norm = torch.nn.functional.normalize(v1, dim=-1)
        v2_norm = torch.nn.functional.normalize(v2, dim=-1)
        v3_norm = torch.nn.functional.normalize(v3, dim=-1)

        # 2. Compute normal vectors (defining the two planes)
        n1 = torch.cross(v1_norm, v2_norm)  # normal vector of plane 1
        n2 = torch.cross(v2_norm, v3_norm)  # normal vector of plane 2

        # cross product norms
        n1_norm = torch.norm(n1, dim=-1)
        n2_norm = torch.norm(n2, dim=-1)

        USE_ORIGINAL_VECTOR_WHEN_PARALLEL = False
        if USE_ORIGINAL_VECTOR_WHEN_PARALLEL:
            # Detect parallel vectors (cross product close to zero)
            threshold = 1e-6
            n1_parallel = n1_norm < threshold
            n2_parallel = n2_norm < threshold

            # For parallel vectors, use v1 and v2 directly as the normal vectors
            n1 = torch.where(n1_parallel.unsqueeze(-1), v1_norm, n1)
            n2 = torch.where(n2_parallel.unsqueeze(-1), v2_norm, n2)
        else:
            # Small perturbation
            n1 = n1 + 1e-8 * v1_norm
            n2 = n2 + 1e-8 * v2_norm

        # 3. Compute invariants
        return self._compute_invariants(n1, n2)
