"""Convert a training checkpoint into a slim, release-ready model checkpoint.

The output keeps only ``{"model": {"model_args": ..., "state_dict": ...}}`` and drops
optimizer/scheduler state, training history, and logging metadata. Checkpoints
written by the original CHGNet-based research code are also supported: pickled
references to ``chgnet.*`` modules are remapped to ``meip.*`` so that they load
without that code installed.

The converted file loads with the matching class, e.g.:

    MEIP.from_file(out)                       # meip.model.MEIP
    CHGNet.from_file(out)                     # meip_baselines.CHGNet
    MACE.from_file(out), NequIP.from_file(out)

Usage:
    python tools/convert_checkpoint.py input.pth.tar output.pth.tar
"""

import argparse
import pickle

import torch

# Derived objects that some model wrappers stored in model_args; they are rebuilt
# from the remaining constructor arguments and need not be serialized.
DERIVED_MODEL_ARGS = ("mace_config", "atom_ref")


class _RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "chgnet" or module.startswith("chgnet."):
            module = "meip" + module[len("chgnet") :]
        return super().find_class(module, name)


class _remapping_pickle:  # noqa: N801 - mimics the ``pickle`` module interface
    Unpickler = _RemappingUnpickler
    load = staticmethod(lambda f, **kw: _RemappingUnpickler(f, **kw).load())


def convert(input_path: str, output_path: str) -> None:
    ckpt = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_remapping_pickle,
    )
    model = ckpt["model"]
    model_args = {
        k: v for k, v in model["model_args"].items() if k not in DERIVED_MODEL_ARGS
    }
    torch.save(
        {"model": {"model_args": model_args, "state_dict": model["state_dict"]}},
        output_path,
    )
    n_params = sum(v.numel() for v in model["state_dict"].values())
    print(f"Wrote {output_path} ({n_params:,} tensor elements in state_dict)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    convert(args.input, args.output)
