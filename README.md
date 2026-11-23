# MEIP

**MEIP** (Magnetic-moment-supervised E(3)-Equivariant Interatomic Potential) is a machine
learning interatomic potential for transition-metal systems. It combines an
E(3)-equivariant dual-graph message-passing architecture with DFT-derived absolute
on-site magnetic moments as auxiliary training targets, and predicts:

- total energy (eV/atom)
- interatomic forces (eV/Å, energy-conserving via autograd)
- stress tensor (GPa)
- absolute on-site magnetic moments (μ<sub>B</sub>)

Key architectural ingredients (see the paper for details):

- **Dual-graph representation** — an atom graph for pairwise interactions and a bond
  graph connecting adjacent directed bonds for three-body angular correlations.
- **O(3)-equivariant bond features** built from spherical harmonics of directed bond
  orientations, refined by Clebsch–Gordan tensor-product convolutions.
- **Pyramidal channel allocation** — a 2:1:0.5 split over l = 0, 1, 2 irreps
  (default `32x0e+16x1o+8x2e`), which is more parameter-efficient than a uniform
  allocation at matched parameter count.
- **Magnetic-moment supervision** — a magmom head applied to the atomic representation
  from the final interaction block.

The default 327K-parameter configuration (3 interaction blocks) matches or outperforms
representative baselines at comparable parameter counts on a curated transition-metal
subset of MPtrj.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.4, and e3nn ≥ 0.5.1. MEIP uses the pure Python
graph converter (`algorithm="legacy"`), which builds the directed bond graph required by
the model.

## Quick start

### Predicting properties

```python
from pymatgen.core import Structure
from meip.model import MEIP

model = MEIP.load("path/to/checkpoint.pth.tar")
structure = Structure.from_file("my_structure.cif")

prediction = model.predict_structure(structure)
print(prediction["e"])  # energy per atom (eV/atom)
print(prediction["f"])  # forces (eV/A)
print(prediction["s"])  # stress (GPa)
print(prediction["m"])  # absolute on-site magnetic moments (mu_B)
```

### ASE calculator, relaxation, and MD

```python
from meip.model import MEIPCalculator, MolecularDynamics, StructOptimizer

calc = MEIPCalculator.from_file("path/to/checkpoint.pth.tar")

relaxer = StructOptimizer(model=calc)
result = relaxer.relax(structure)

md = MolecularDynamics(atoms=structure, model=calc, ensemble="nvt", temperature=300)
md.run(500)
```

## Training

### 1. Prepare the dataset

Download the [MPtrj dataset](https://figshare.com/articles/dataset/23713842) and run:

```bash
python tools/prepare_dataset.py convert \
    --json datasets/MPtrj_2022.9_full.json --jsonl datasets/mptrj_full.jsonl
python tools/prepare_dataset.py split \
    --jsonl datasets/mptrj_full.jsonl --output-dir datasets/mptrj_tm_raw
python tools/prepare_dataset.py clean \
    --input-dir datasets/mptrj_tm_raw --output-dir datasets/mptrj_tm_subset
```

This reproduces the curation described in the paper: frames containing at least one
d-block transition metal (Sc–Zn, Y–Cd, Hf–Hg) with non-null magnetic-moment labels,
isolated-atom configurations removed, split 80:10:10 by `mp_id` to prevent leakage
between splits. The `clean` step also writes a `<split>.jsonl.sizes` file next to each
split, caching per-structure graph sizes; the dynamic batch sampler used for training
requires these files.

### 2. Launch training

```bash
torchrun --nproc_per_node=<num_gpus> scripts/train.py
```

Hyperparameters live in `scripts/train_config.py`; the defaults reproduce the paper's
327K-parameter model (loss weights E:F:S:M = 1.9:0.6:0.1:0.25, Huber δ = 0.1, Adam at
7e-3 with cosine annealing, 30 epochs). Set the `SCALE_FACTOR` environment variable to
scale the dynamic batch size to your GPU memory (the learning rate is scaled by
√SCALE_FACTOR automatically).

To change the architecture, adjust `dim_everywhere`, `n_conv`, or the irreps overrides,
e.g. `bond_fea_irreps_override="32x0e+16x1o+8x2e"`.

The paper results were obtained with DDP on 2 GPUs with 24 GB of memory each
(`SCALE_FACTOR=1`).

## Baselines

The `meip_baselines` package contains the baselines compared against MEIP in the
paper. They consume the same crystal graphs and are trained with the same dataset,
dynamic batch sampler, and trainer as MEIP; MACE and NequIP get the same linear
magnetic-moment head so that all models predict E/F/S/M.

| `--model` | Class | Params | Loss weights E:F:S:M |
|---|---|---|---|
| `chgnet` | `meip_baselines.CHGNet` | 318K | 1.6:0.4:0.1:0.1 |
| `mace` | `meip_baselines.MACE` | 346K | 1.9:0.9:0.3:0.15 |
| `nequip` | `meip_baselines.NequIP` | 326K | 1.8:0.8:0.1:0.1 |

`CHGNet` needs no extra packages. `MACE` and `NequIP` require `mace-torch==0.3.13` and
`nequip==0.16.1`, whose declared dependency pins conflict with this environment
(although the code runs fine). Install them with the setup script, which installs in
two rounds and then restores the pinned versions:

```bash
WITH_BASELINES=1 bash setup_env.sh
```

Train a baseline with the paper configuration (see `BASELINE_PRESETS` in
`scripts/train_config.py`):

```bash
torchrun --nproc_per_node=<num_gpus> scripts/train.py --model nequip
```

Load trained baseline weights with the matching class:

```python
from meip_baselines import NequIP

model = NequIP.load("path/to/nequip.pth.tar")
prediction = model.predict_structure(structure)
```

Baseline models also work with `StructOptimizer`, `MolecularDynamics`, and
`MEIPCalculator` (pass the model instance).

## Checkpoint conversion

`tools/convert_checkpoint.py` turns a training checkpoint into a slim release
checkpoint (model arguments and weights only, without optimizer state or logging
metadata):

```bash
python tools/convert_checkpoint.py checkpoints/<run>/epoch29_*.pth.tar meip.pth.tar
```

## Repository layout

```
meip/
├── meip/
│   ├── graph/        # crystal graph construction (atom graph + bond graph)
│   ├── model/        # MEIP model, equivariant blocks, ASE interface
│   ├── data/         # JSONL streaming dataset, dynamic batch samplers, augmentation
│   ├── trainer/      # DDP trainer with multi-task (E/F/S/M) losses
│   └── utils/        # common utilities
├── meip_baselines/   # CHGNet, MACE, and NequIP baselines
├── scripts/          # training entry point and configuration
├── tools/            # dataset preparation, checkpoint conversion
├── examples/         # minimal usage examples
└── tests/            # smoke tests
```

## Citation

If you use MEIP in your research, please cite the MEIP paper (reference to be added
upon publication).

## License

MEIP is released under the BSD 3-Clause license. The dual-graph message-passing
framework builds on [CHGNet](https://github.com/CederGroupHub/chgnet) (BSD-3, © 2023
The Regents of the University of California); see `LICENSE` for the full notices.
