"""DDP training entry point for MEIP.

Launch with torchrun, e.g.:

    torchrun --nproc_per_node=4 scripts/train.py

The dataset directory (``TrainConfig.dataset_path``) must contain
``train.jsonl``, ``val.jsonl``, and ``test.jsonl`` produced by the tools in
``tools/`` (see README).
"""

import argparse
import os
import time

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group

from meip.data.dataset_jsonl import StructureJsonlDataset
from meip.graph.converter import CrystalGraphConverter
from meip.model import MEIP
from meip.trainer.ddp_trainer import DDPTrainer

from train_config import TrainConfig


def setup(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    destroy_process_group()


def make_dataloader(dataset, config: TrainConfig, world_size: int, rank: int, *, shuffle: bool):
    if config.use_ddp_dynamic_batch:
        return dataset.get_ddp_dynamic_loader(
            world_size=world_size,
            rank=rank,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            total_size=config.dynamic_total_size,
            atom_weight=config.dynamic_atom_weight,
            bond_weight=config.dynamic_bond_weight,
            utilization_threshold=config.dynamic_utilization_threshold,
            max_search_attempts=config.dynamic_max_search_attempts,
        )
    return dataset.get_distributed_loader(
        shuffle=shuffle,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )


def train(rank: int, world_size: int, load_ckpt: str | None = None) -> None:
    # Reduce file-descriptor pressure from DataLoader workers
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    setup(rank, world_size)

    config = TrainConfig()

    graph_converter = CrystalGraphConverter(
        algorithm=config.algorithm,
        enable_dihedral=config.enable_dihedral,
        enable_directed_bond_graph=config.enable_directed_bond_graph,
    )

    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = StructureJsonlDataset(
            jsonl_path=f"{config.dataset_path}/{split}.jsonl",
            graph_converter=graph_converter,
            subset_ratio=config.subset_ratio,
            return_none_if_invalid_graph=False,
            # Augmentation only applies to the training split
            enable_augmentation=config.enable_augmentation if split == "train" else False,
            augmentation_prob=config.augmentation_prob if split == "train" else 0.0,
            use_o3=config.use_o3,
            augmentation_seed=config.augmentation_seed,
        )

    train_loader = make_dataloader(datasets["train"], config, world_size, rank, shuffle=True)
    val_loader = make_dataloader(datasets["val"], config, world_size, rank, shuffle=False)
    test_loader = make_dataloader(datasets["test"], config, world_size, rank, shuffle=False)

    timestamp_str = time.strftime("%Y%m%d-%H%M", time.localtime())
    save_dir = f"checkpoints/meip_{timestamp_str}"
    if rank == 0:
        print(f"Saving model to {save_dir}")

    if not load_ckpt:
        model = MEIP(**config.model_kwargs())

        trainer = DDPTrainer(
            model=model,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            find_unused_parameters=config.find_unused_parameters,
            loss_type=config.loss_type,
            dwa_window_size=config.dwa_window_size,
            targets="efsm",
            energy_loss_ratio=config.energy_loss_ratio,
            force_loss_ratio=config.force_loss_ratio,
            stress_loss_ratio=config.stress_loss_ratio,
            mag_loss_ratio=config.mag_loss_ratio,
            optimizer=config.optimizer,
            weight_decay=config.weight_decay,
            scheduler=config.scheduler,
            scheduler_params=config.scheduler_params,
            criterion="Huber",
            delta=0.1,
            epochs=config.max_epochs,
            learning_rate=1e-3 * config.lr_scale,
            torch_seed=config.torch_seed,
            data_seed=config.data_seed,
            wandb_path=f"meip/{timestamp_str}",
        )
    else:
        if rank == 0:
            print(f"Loading checkpoint from {load_ckpt}")
        trainer = DDPTrainer.load(
            path=load_ckpt,
            model_class=MEIP,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
        )

    start_time = time.time()
    trainer.train(
        train_loader,
        val_loader,
        test_loader,
        save_dir=save_dir,
        save_test_result=True,
        train_composition_model=config.enable_composition_model
        and config.enable_composition_model_train,
    )
    if rank == 0:
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f} seconds ({elapsed / 3600:.2f} hours)")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEIP DDP training entry")
    parser.add_argument("--load-ckpt", type=str, default=None, help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    train(rank, world_size, args.load_ckpt)
