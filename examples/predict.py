"""Minimal example: predict energy, forces, stress, and magnetic moments.

Usage:
    python examples/predict.py path/to/checkpoint.pth.tar path/to/structure.cif
"""

import sys

from pymatgen.core import Structure

from meip.model import MEIP


def main() -> None:
    ckpt_path, structure_path = sys.argv[1], sys.argv[2]

    model = MEIP.load(ckpt_path)
    structure = Structure.from_file(structure_path)

    prediction = model.predict_structure(structure)
    print(f"energy (eV/atom): {prediction['e']}")
    print(f"forces (eV/A):\n{prediction['f']}")
    print(f"stress (GPa):\n{prediction['s']}")
    print(f"|magmom| (mu_B): {prediction['m']}")


if __name__ == "__main__":
    main()
