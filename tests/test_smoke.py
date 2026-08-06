"""Smoke tests: model construction and a full forward pass on a small crystal."""

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from meip.model import MEIP


@pytest.fixture
def structure() -> Structure:
    """A small NaCl rock-salt cell."""
    lattice = Lattice.cubic(5.69)
    species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
    coords = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def model() -> MEIP:
    torch.manual_seed(0)
    return MEIP(
        atom_fea_dim=8,
        bond_fea_dim=8,
        atom_fea_irreps_override="8x0e",
        bond_fea_irreps_override="8x0e+4x1o+2x2e",
        radial_MLP_dims=[8, 8],
        mlp_hidden_dims=[8, 8],
        n_conv=2,
        num_radial=7,
        composition_model="MPtrj",
        graph_converter_algorithm="legacy",
    )


def test_predict_structure(model: MEIP, structure: Structure) -> None:
    prediction = model.predict_structure(structure, task="efsm")

    n_atoms = len(structure)
    assert np.isfinite(float(prediction["e"]))
    assert prediction["f"].shape == (n_atoms, 3)
    assert prediction["s"].shape == (3, 3)
    assert prediction["m"].shape == (n_atoms,)
    assert (prediction["m"] >= 0).all()


def test_save_load_roundtrip(model: MEIP, structure: Structure, tmp_path) -> None:
    e_before = float(model.predict_structure(structure, task="e")["e"])

    ckpt = tmp_path / "model.pth.tar"
    torch.save({"model": model.as_dict()}, ckpt)
    reloaded = MEIP.from_file(str(ckpt))

    e_after = float(reloaded.predict_structure(structure, task="e")["e"])
    assert e_before == pytest.approx(e_after, abs=1e-6)
