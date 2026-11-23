#!/bin/bash
# Set up a complete MEIP environment.
#
# Installs PyTorch first (so that every subsequent package resolves against the
# correct torch build), then MEIP itself, then optional extras, and finally
# verifies that all imports and versions are consistent.
#
# Usage:
#   PYTHON=/path/to/python bash setup_env.sh     # defaults to `python`
#   SKIP_CUDA=1 bash setup_env.sh                # CPU-only (auto-detected on macOS)
#   WITH_CUEQ=1 bash setup_env.sh                # also install cuequivariance kernels
#   WITH_BASELINES=1 bash setup_env.sh           # also install MACE/NequIP baselines

set -euo pipefail

PYTHON=${PYTHON:-python}
PIP="$PYTHON -m pip"

TORCH_VERSION=2.8.0
E3NN_VERSION=0.5.6
MACE_VERSION=0.3.13
NEQUIP_VERSION=0.16.1
CUEQ_VERSION=0.4.0

# Install CUDA builds unless disabled or running on macOS
if [[ "${SKIP_CUDA:-0}" == "1" || "$(uname)" == "Darwin" ]]; then
    CUDA=0
else
    CUDA=1
fi

echo "== Python: $($PYTHON --version) @ $($PYTHON -c 'import sys; print(sys.executable)')"
echo "== CUDA packages: $([[ $CUDA == 1 ]] && echo yes || echo no)"

# ---- 1. PyTorch first: everything else must resolve against this build ----
if [[ $CUDA == 1 ]]; then
    $PIP install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128
else
    $PIP install "torch==${TORCH_VERSION}"
fi

# ---- 2. MEIP and its dependencies (ase, e3nn, pymatgen, ...) ----
$PIP install -e .

# ---- 3. Optional extras ----
# cuequivariance acceleration (CUDA only; MEIP falls back to pure e3nn without it)
if [[ "${WITH_CUEQ:-0}" == "1" && $CUDA == 1 ]]; then
    $PIP install \
        "cuequivariance==${CUEQ_VERSION}" \
        "cuequivariance-torch==${CUEQ_VERSION}" \
        "cuequivariance-ops-torch-cu12==${CUEQ_VERSION}"
fi

# Training logging
$PIP install wandb

# Baselines (meip_baselines.MACE / meip_baselines.NequIP). mace-torch and nequip
# declare pins that conflict with this environment (e.g. mace-torch requires
# e3nn==0.4.4, nequip pulls matscipy which requires numpy<2), although the code
# runs fine with the versions below. They cannot be resolved in a single pip call
# (mace-torch pins e3nn==0.4.4, nequip needs e3nn>=0.5.6), so install them one at a
# time with their dependencies, then restore the pinned versions. pip reports the
# remaining declared conflicts as warnings; they can be ignored.
if [[ "${WITH_BASELINES:-0}" == "1" ]]; then
    $PIP install "mace-torch==${MACE_VERSION}"
    $PIP install "nequip==${NEQUIP_VERSION}"
    if [[ $CUDA == 1 ]]; then
        $PIP install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128
    else
        $PIP install "torch==${TORCH_VERSION}"
    fi
    $PIP install "e3nn==${E3NN_VERSION}" "numpy==2.2.4" "scipy==1.15.2" "ase==3.24.0"
fi

# ---- 4. Verify ----
echo "== Verifying imports and versions..."
$PYTHON - <<'EOF'
import importlib

modules = ["torch", "e3nn", "numpy", "scipy", "ase", "pymatgen.core", "jsonlines", "meip"]
failed = []
for mod in modules:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<16} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {mod:<16} IMPORT FAILED: {e}")
        failed.append(mod)

import torch
print(f"  cuda available:  {torch.cuda.is_available()}")

# Construct the paper-default model as a smoke test
from meip.model import MEIP
model = MEIP(
    atom_fea_dim=32,
    bond_fea_dim=32,
    atom_fea_irreps_override="32x0e",
    bond_fea_irreps_override="32x0e+16x1o+8x2e",
    radial_MLP_dims=[32, 32, 32],
    mlp_hidden_dims=[32, 32, 32],
    n_conv=3,
    num_radial=31,
    dihedral_fea_dim=0,
    graph_converter_algorithm="legacy",
)
assert model.n_params == 327_326, f"unexpected parameter count: {model.n_params:,}"
print(f"  MEIP paper-default model: {model.n_params:,} parameters (expected 327,326)")

import os
if os.environ.get("WITH_BASELINES") == "1":
    import io, contextlib, sys
    sys.path.insert(0, "scripts")
    from train_config import TrainConfig
    expected = {"chgnet": 317_879, "mace": 346_032, "nequip": 326_061}
    for name, n_expected in expected.items():
        cfg = TrainConfig(model=name)
        with contextlib.redirect_stdout(io.StringIO()):
            bl = cfg.model_class()(**cfg.model_kwargs())
        n = sum(p.numel() for p in bl.parameters() if p.requires_grad)
        assert n == n_expected, f"{name}: unexpected parameter count {n:,}"
        print(f"  baseline {name:<7} {n:,} parameters (expected {n_expected:,})")

if failed:
    raise SystemExit(f"Import checks failed: {failed}")
print("== All checks passed.")
EOF
