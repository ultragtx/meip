"""Prepare the curated transition-metal subset of MPtrj used to train MEIP.

Pipeline (matching the dataset curation described in the paper):

1. ``convert``: flatten the hierarchical MPtrj JSON into a streaming JSONL file.
2. ``split``: keep frames that contain at least one transition-metal element and
   have non-null magnetic-moment labels, then split into train/val/test
   (80:10:10) grouped by ``mp_id`` so no material leaks across splits.
3. ``clean``: drop frames whose graphs are invalid (e.g. isolated atoms without
   any neighbor within the atom-graph cutoff) and cache graph sizes for the
   dynamic batch sampler.

Example:

    python tools/prepare_dataset.py convert \
        --json datasets/MPtrj_2022.9_full.json --jsonl datasets/mptrj_full.jsonl
    python tools/prepare_dataset.py split \
        --jsonl datasets/mptrj_full.jsonl --output-dir datasets/mptrj_tm_raw
    python tools/prepare_dataset.py clean \
        --input-dir datasets/mptrj_tm_raw --output-dir datasets/mptrj_tm_subset
"""

import argparse
import json
import os

import jsonlines

from meip.data.dataset_jsonl import StructureJsonlDataset
from meip.data.dataset_utils import (
    compute_and_save_graph_sizes,
    save_to_jsonl_with_graph_filter,
)
from meip.graph.converter import CrystalGraphConverter

# d-block metals of groups 3--12: 3d (Sc--Zn), 4d (Y--Cd), and 5d (Hf--Hg)
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}


def convert(json_path: str, jsonl_path: str) -> None:
    """Flatten the hierarchical MPtrj JSON into one frame per JSONL line."""
    with open(json_path) as f:
        data = json.load(f)

    with jsonlines.open(jsonl_path, "w") as writer:
        for mp_id, graphs in data.items():
            for graph_id, content in graphs.items():
                writer.write({"mp_id": mp_id, "graph_id": graph_id, **content})

    print(f"Converted {json_path} -> {jsonl_path}")


def split(jsonl_path: str, output_dir: str) -> None:
    """Filter TM frames with magmom labels and split by mp_id (80:10:10)."""
    os.makedirs(output_dir, exist_ok=True)
    dataset = StructureJsonlDataset(
        jsonl_path,
        CrystalGraphConverter(),
        shuffle=True,
        filter_out_null_magmom=True,
        filter_in_elements=TRANSITION_METALS,
        apply_filter=True,
    )
    dataset.get_train_val_test_loader(save_jsonl=True, output_dir=output_dir)
    print(f"Split written to {output_dir}")


def clean(input_dir: str, output_dir: str) -> None:
    """Remove frames with invalid graphs and cache graph sizes."""
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ("train", "val", "test"):
        dataset = StructureJsonlDataset(
            os.path.join(input_dir, f"{split_name}.jsonl"),
            CrystalGraphConverter(algorithm="legacy", enable_dihedral=False),
            shuffle=False,
            apply_filter=False,
        )
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        save_to_jsonl_with_graph_filter(dataset=dataset, output_path=output_path)

        # Cache graph sizes (<split>.jsonl.sizes) required by the dynamic batch sampler
        cleaned = StructureJsonlDataset(
            output_path,
            CrystalGraphConverter(algorithm="legacy", enable_dihedral=False),
            shuffle=False,
            apply_filter=False,
        )
        compute_and_save_graph_sizes(dataset=cleaned)
    print(f"Cleaned dataset written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="MPtrj JSON -> JSONL")
    p_convert.add_argument("--json", required=True)
    p_convert.add_argument("--jsonl", required=True)

    p_split = sub.add_parser("split", help="filter TM frames and split by mp_id")
    p_split.add_argument("--jsonl", required=True)
    p_split.add_argument("--output-dir", required=True)

    p_clean = sub.add_parser("clean", help="drop invalid graphs, cache graph sizes")
    p_clean.add_argument("--input-dir", required=True)
    p_clean.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "convert":
        convert(args.json, args.jsonl)
    elif args.command == "split":
        split(args.jsonl, args.output_dir)
    elif args.command == "clean":
        clean(args.input_dir, args.output_dir)
