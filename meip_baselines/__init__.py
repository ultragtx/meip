"""Baseline models used for comparison with MEIP in the paper.

All baselines consume MEIP crystal graphs and are trained with the same data
pipeline and ``DDPTrainer`` as MEIP, predicting energy, forces, stress, and absolute
on-site magnetic moments:

- ``CHGNet``: the original (invariant) CHGNet architecture.
- ``MACE``: MACE (``mace-torch``) with an added linear magnetic-moment head.
- ``NequIP``: NequIP (``nequip``) with an added linear magnetic-moment head.

``MACE`` and ``NequIP`` need their upstream packages; install them with
``WITH_BASELINES=1 bash setup_env.sh`` (see README). Imports are lazy, so ``CHGNet``
works without them.
"""

from __future__ import annotations

from importlib import import_module

_MODELS = {
    "CHGNet": "meip_baselines.chgnet_model",
    "MACE": "meip_baselines.mace_model",
    "NequIP": "meip_baselines.nequip_model",
}

__all__ = list(_MODELS)


def __getattr__(name: str):
    if name in _MODELS:
        return getattr(import_module(_MODELS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
