"""Smoke tests for the paper baselines in ``meip_baselines``.

MACE and NequIP tests are skipped unless ``mace-torch`` / ``nequip`` are installed
(``WITH_BASELINES=1 bash setup_env.sh``).
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_config import TrainConfig  # noqa: E402

# Parameter counts of the baseline configurations compared in the paper
EXPECTED_PARAMS = {"chgnet": 317_879, "mace": 346_032, "nequip": 326_061}
REQUIRED_PACKAGE = {"chgnet": None, "mace": "mace", "nequip": "nequip"}


@pytest.fixture
def structure() -> Structure:
    """A small rock-salt FeO cell (a transition-metal oxide)."""
    lattice = Lattice.cubic(4.33)
    species = ["Fe", "Fe", "Fe", "Fe", "O", "O", "O", "O"]
    coords = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],
    ]
    return Structure(lattice, species, coords)


@pytest.mark.parametrize("name", list(EXPECTED_PARAMS))
def test_baseline_predict_and_roundtrip(name: str, structure: Structure, tmp_path) -> None:
    if REQUIRED_PACKAGE[name]:
        pytest.importorskip(REQUIRED_PACKAGE[name])

    torch.manual_seed(0)
    config = TrainConfig(model=name)
    model_class = config.model_class()
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_class(**config.model_kwargs())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == EXPECTED_PARAMS[name]

    prediction = model.predict_structure(structure, task="efsm")
    n_atoms = len(structure)
    assert np.isfinite(float(prediction["e"]))
    assert prediction["f"].shape == (n_atoms, 3)
    assert prediction["s"].shape == (3, 3)
    assert prediction["m"].shape == (n_atoms,)

    ckpt = tmp_path / f"{name}.pth.tar"
    torch.save({"model": model.as_dict()}, ckpt)
    with contextlib.redirect_stdout(io.StringIO()):
        reloaded = model_class.from_file(str(ckpt))
    e_after = float(reloaded.predict_structure(structure, task="e")["e"])
    assert float(prediction["e"]) == pytest.approx(e_after, abs=1e-6)
