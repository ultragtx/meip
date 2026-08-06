import json
import torch
import random
import os
from typing import Any, List, Dict, Tuple, Set, Optional
from collections import deque

from torch.utils.data import Dataset, DataLoader, Sampler
from pymatgen.core.structure import Structure
from meip.data.dataset import get_loader, collate_graphs
from torch.utils.data.distributed import DistributedSampler
from meip.data.dataset_utils import make_storage_key
from meip.data.augmentation import O3Augmentation

from meip.data.samplers import DynamicBatchSampler, DDPDynamicBatchSampler
from meip.data.dataset_utils import (
    compute_and_save_graph_sizes,
    save_to_jsonl_with_graph_filter,
    save_to_jsonl_as_graph,
    save_to_parquet,
    analyze_graph_sizes,
    convert_targets_to_serializable
)

class StructureJsonlDataset(Dataset):
    """Stream large dataset from a JSONL file, with filtering and train/val/test split."""

    def __init__(
        self,
        jsonl_path: str,
        graph_converter: Any,
        targets: str = "efsm",
        energy_key: str = "energy_per_atom",
        force_key: str = "force",
        stress_key: str = "stress",
        magmom_key: str = "magmom",
        max_atomic_number: int = 100,  # Filter by atomic number
        filter_out_null_magmom: bool = True,  # Filter out entries whose magmom is None
        filter_out_elements: Set[str] = None,  # Elements to exclude
        filter_in_elements: Set[str] = None,  # Keep only structures containing these elements
        apply_filter: bool = False,  # Whether to apply filtering
        shuffle: bool = False,
        keys: Optional[List[Tuple[str, str]]] = None,
        offset_map: Optional[Dict[Tuple[str, str], int]] = None,
        subset_ratio: float = 1.0,  # Fraction of the data to use
        return_none_if_invalid_graph: bool = True,
        size_cache_path: Optional[str] = None,  # Path to size cache file
        # Data augmentation parameters
        enable_augmentation: bool = False,  # Enable data augmentation
        augmentation_prob: float = 0.5,    # Augmentation probability
        use_o3: bool = True,               # True for O(3), False for SO(3)
        augmentation_seed: Optional[int] = None,  # Random seed for augmentation
    ):
        """
        Args:
            apply_filter (bool): Whether to apply filtering rules (default True)
            subset_ratio (float): Fraction of the data to use (0.0 to 1.0), default 1.0 (use all data)
            size_cache_path (str): Path to the size cache file. If None, will use jsonl_path + '.sizes'
            enable_augmentation (bool): Whether to enable data augmentation (default False)
            augmentation_prob (float): Probability of applying augmentation (0.0 to 1.0), default 0.5
            use_o3 (bool): True for O(3) (rotations + reflections), False for SO(3) (rotations only)
            augmentation_seed (int): Random seed for data augmentation, for reproducibility
        """
        self.jsonl_path = jsonl_path
        self.graph_converter = graph_converter
        self.targets = targets
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.magmom_key = magmom_key
        self.max_atomic_number = max_atomic_number
        self.filter_out_null_magmom = filter_out_null_magmom
        self.filter_out_elements = filter_out_elements
        self.filter_in_elements = filter_in_elements
        self.apply_filter = apply_filter  # Whether to apply filtering
        self.subset_ratio = subset_ratio  # Fraction of the data to use

        self.return_none_if_invalid_graph = return_none_if_invalid_graph

        # Initialize data augmentation
        self.augmentation = O3Augmentation(
            enable_augmentation=enable_augmentation,
            augmentation_prob=augmentation_prob,
            use_o3=use_o3,
            random_seed=augmentation_seed
        )

        if keys is not None and offset_map is not None:
            self.keys = keys
            self.offset_map = offset_map
        else:
            self.keys = []
            self.offset_map = {}
            with open(jsonl_path, "r") as f:
                offset = 0
                for line in f:
                    data = json.loads(line)
                    if not apply_filter or self._filter_data(data):  # Filtering only runs when apply_filter=True
                        mp_id = data["mp_id"]
                        graph_id = data["graph_id"]
                        self.keys.append((mp_id, graph_id))
                        self.offset_map[(mp_id, graph_id)] = offset
                    offset += len(line.encode("utf-8"))

            if shuffle:
                random.seed(42)  # Fix the random seed for reproducibility
                random.shuffle(self.keys)

        # Compute the actual number of samples to use based on subset_ratio
        self.actual_num_keys = int(len(self.keys) * self.subset_ratio)
        self.keys = self.keys[:self.actual_num_keys]  # Keep only the required subset

        self.unique_mp_ids = list(set(mp_id for mp_id, _ in self.keys))

        # Load or initialize size cache
        self.size_cache_path = size_cache_path or (jsonl_path + '.sizes')
        self.size_cache = self._load_or_init_size_cache()

        print(f"Dataset initialized with {len(self.unique_mp_ids)} mp_ids, {self.actual_num_keys} structures (subset ratio: {subset_ratio}), (apply_filter={apply_filter}), (augmentation={'O(3)' if use_o3 else 'SO(3)'} @ {augmentation_prob:.2f} prob, enabled={enable_augmentation}).")

    def _load_or_init_size_cache(self) -> Dict[Tuple[str, str], Tuple[int, int]]:
        """Load size cache from file or initialize empty cache."""
        if os.path.exists(self.size_cache_path):
            print(f"Loading size cache from {self.size_cache_path}")
            with open(self.size_cache_path, 'r') as f:
                cache = json.load(f)
                return cache
        else:
            print(f"No size cache found at {self.size_cache_path}, initializing empty cache")
            return {}

    def _save_size_cache(self):
        """Save size cache to file."""
        # Convert tuple keys to strings for JSON serialization
        cache = {'_'.join(k): list(v) for k, v in self.size_cache.items()}
        with open(self.size_cache_path, 'w') as f:
            json.dump(cache, f)

    def _get_graph_sizes_from_jsonl(self, idx: int) -> Tuple[int, int]:
        """Get graph sizes from cache or compute and cache them."""
        mp_id, graph_id = self.keys[idx]
        key = make_storage_key(mp_id, graph_id)

        # Check cache first
        if key in self.size_cache:
            cached_sizes = self.size_cache[key]
            return (cached_sizes['atom_graph_size'], cached_sizes['bond_graph_size'])

        assert False, f"Should not happen {key} not in size cache, {self.size_cache_path}"

        # If not in cache, compute and cache
        offset = self.offset_map[key]
        with open(self.jsonl_path, "r") as f:
            f.seek(offset)
            line = f.readline()
            data = json.loads(line)

        struct = Structure.from_dict(data["structure"])
        crystal_graph = self.graph_converter(struct, graph_id=graph_id, mp_id=mp_id)

        # Cache the sizes
        sizes = (len(crystal_graph.atom_graph), len(crystal_graph.bond_graph))
        self.size_cache[key] = sizes
        # self._save_size_cache()  # Save after each new computation

        return sizes

    def _filter_data(self, data: dict) -> bool:
        """Filter out data that does not meet the criteria."""
        if self.filter_out_null_magmom and (data[self.magmom_key] is None):
            return False

        if self._contains_filtered_elements(data, self.filter_out_elements):
            return False

        if not self._contains_filtered_elements(data, self.filter_in_elements):
            return False

        return True

    def _contains_filtered_elements(self, data: dict, filter_elements: Set[str]) -> bool:
        """Check whether the data contains any of the given elements (without creating a Structure instance)."""
        if filter_elements is None:
            return False
        for site in data["structure"]["sites"]:
            element = site["species"][0]["element"]
            if element in filter_elements:
                return True
        return False

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        """Fetches a single item from the dataset by index."""
        mp_id, graph_id = self.keys[idx]
        offset = self.offset_map[(mp_id, graph_id)]

        with open(self.jsonl_path, "r") as f:
            f.seek(offset)
            line = f.readline()
            data = json.loads(line)

        try:
            struct = Structure.from_dict(data["structure"])

            # Prepare targets first
            targets = {}
            for key in self.targets:
                if key == "e":
                    targets["e"] = torch.tensor(data[self.energy_key], dtype=torch.float32)
                elif key == "f":
                    targets["f"] = torch.tensor(data[self.force_key], dtype=torch.float32)
                elif key == "s":
                    targets["s"] = torch.tensor(data[self.stress_key], dtype=torch.float32) * (-0.1)
                elif key == "m":
                    mag = data[self.magmom_key]
                    targets["m"] = None if mag is None else torch.abs(torch.tensor(mag, dtype=torch.float32))

            # Apply data augmentation if enabled
            struct, targets = self.augmentation.augment_data(struct, targets)

            # Convert structure to graph after augmentation
            crystal_graph = self.graph_converter(struct, graph_id=graph_id, mp_id=mp_id)

            return crystal_graph, targets

        except Exception as e:
            if self.return_none_if_invalid_graph:
                print(f"Skipping {make_storage_key(mp_id, graph_id)} because it has invalid graph: {str(e)}")
                return None, None
            else:
                print(f"Skipping {make_storage_key(mp_id, graph_id)} because it has invalid graph: {str(e)}")
                print(f"Return a random item")
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)

    def save_to_jsonl(self, output_path: str):
        # create dir if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        """Save the filtered dataset as JSONL."""
        with open(self.jsonl_path, "r") as infile, open(output_path, "w") as outfile:
            for mp_id, graph_id in self.keys:
                offset = self.offset_map[(mp_id, graph_id)]
                infile.seek(offset)
                line = infile.readline()
                outfile.write(line)

        print(f"Filtered dataset saved to {output_path}")

    def get_train_val_test_loader(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        save_jsonl: bool = False,
        output_dir: str = "./",
    ):
        """Splits the dataset by mp_id and returns train/val/test loaders."""
        unique_mp_ids = self.unique_mp_ids
        if shuffle:
            random.seed(42)
            random.shuffle(unique_mp_ids)

        n_train = int(train_ratio * len(unique_mp_ids))
        n_val = int(val_ratio * len(unique_mp_ids))

        train_mp_ids = unique_mp_ids[:n_train]
        val_mp_ids = unique_mp_ids[n_train:n_train + n_val]
        test_mp_ids = unique_mp_ids[n_train + n_val:]

        def filter_keys(mp_ids):
            return [(mp_id, graph_id) for mp_id, graph_id in self.keys if mp_id in mp_ids]

        train_keys = filter_keys(train_mp_ids)
        val_keys = filter_keys(val_mp_ids)
        test_keys = filter_keys(test_mp_ids)

        train_dataset = StructureJsonlDataset(self.jsonl_path, self.graph_converter, self.targets,
                                              self.energy_key, self.force_key, self.stress_key, self.magmom_key,
                                              keys=train_keys, offset_map=self.offset_map,
                                              return_none_if_invalid_graph=True)

        val_dataset = StructureJsonlDataset(self.jsonl_path, self.graph_converter, self.targets,
                                            self.energy_key, self.force_key, self.stress_key, self.magmom_key,
                                            keys=val_keys, offset_map=self.offset_map,
                                            return_none_if_invalid_graph=True)

        test_dataset = StructureJsonlDataset(self.jsonl_path, self.graph_converter, self.targets,
                                             self.energy_key, self.force_key, self.stress_key, self.magmom_key,
                                             keys=test_keys, offset_map=self.offset_map,
                                             return_none_if_invalid_graph=True)

        if save_jsonl:
            train_dataset.save_to_jsonl(os.path.join(output_dir, "train.jsonl"))
            val_dataset.save_to_jsonl(os.path.join(output_dir, "val.jsonl"))
            test_dataset.save_to_jsonl(os.path.join(output_dir, "test.jsonl"))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    def get_loader(
        self,
        batch_size=40,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=1,
    ):
        random.seed(42)
        """Creates a data loader for the dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=(num_workers > 0),
            drop_last=True,
        )

    def get_distributed_loader(
        self,
        batch_size=24,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=1,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=DistributedSampler(self, shuffle=shuffle),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=(num_workers > 0),
            drop_last=True,
        )

    def get_dynamic_loader(
        self,
        batch_size: int = 40,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        prefetch_factor: int = 1,
        total_size: int = 50000,
        atom_weight: float = 1.0,
        bond_weight: float = 1.0,
        utilization_threshold: float = 0.8,
        max_search_attempts: int = 50,
    ):
        """Creates a dynamic data loader that adjusts batch size based on weighted sum of graph sizes."""
        random.seed(42)
        sampler = DynamicBatchSampler(
            dataset=self,
            target_batch_size=batch_size,
            total_size=total_size,
            atom_weight=atom_weight,
            bond_weight=bond_weight,
            shuffle=shuffle,
            drop_last=True,
            utilization_threshold=utilization_threshold,
            max_search_attempts=max_search_attempts,
        )

        return DataLoader(
            self,
            batch_sampler=sampler,
            collate_fn=collate_graphs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=(num_workers > 0),
        )

    def get_dynamic_distributed_loader(
        self,
        batch_size: int = 24,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = False,
        prefetch_factor: int = 1,
        total_size: int = 50000,
        atom_weight: float = 1.0,
        bond_weight: float = 1.0,
        utilization_threshold: float = 0.8,
        max_search_attempts: int = 50,
    ):
        """Creates a dynamic distributed data loader using DistributedSampler + DynamicBatchSampler."""
        # Create DistributedSampler for data distribution across ranks
        distributed_sampler = DistributedSampler(self, shuffle=shuffle)

        # Create DynamicBatchSampler that works with DistributedSampler
        dynamic_sampler = DynamicBatchSampler(
            dataset=self,
            target_batch_size=batch_size,
            total_size=total_size,
            atom_weight=atom_weight,
            bond_weight=bond_weight,
            sampler=distributed_sampler,
            drop_last=True,
            utilization_threshold=utilization_threshold,
            max_search_attempts=max_search_attempts,
        )

        return DataLoader(
            self,
            batch_sampler=dynamic_sampler,
            collate_fn=collate_graphs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=(num_workers > 0),
        )

    def get_ddp_dynamic_loader(
        self,
        world_size: int,
        rank: int,
        batch_size: int = 24,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 1,
        total_size: int = 50000,
        atom_weight: float = 1.0,
        bond_weight: float = 1.0,
        utilization_threshold: float = 0.8,
        max_search_attempts: int = 50,
    ):
        """Creates a DDP dynamic data loader with global progress tracking."""
        # Create DDPDynamicBatchSampler
        ddp_dynamic_sampler = DDPDynamicBatchSampler(
            dataset=self,
            world_size=world_size,
            rank=rank,
            target_batch_size=batch_size,
            total_size=total_size,
            atom_weight=atom_weight,
            bond_weight=bond_weight,
            utilization_threshold=utilization_threshold,
            max_search_attempts=max_search_attempts,
            drop_last=True,
        )

        return DataLoader(
            self,
            batch_sampler=ddp_dynamic_sampler,
            collate_fn=collate_graphs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=(num_workers > 0),
        )

    def compute_avg_num_neighbors(self, num_samples: int = 1000) -> float:
        """
        Computes the average number of neighbors across a subset of graphs.
        This follows the same logic as MACE's compute_statistics function.

        Args:
            num_samples: Number of samples to use for computing statistics

        Returns:
            Average number of neighbors per atom
        """
        print(f"Computing avg_num_neighbors from {num_samples} samples...")

        random.seed(42)
        indices = random.sample(range(len(self)), min(num_samples, len(self)))

        all_neighbor_counts = []

        for i, idx in enumerate(indices):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{len(indices)}")

            crystal_graph, targets = self.__getitem__(idx)
            if crystal_graph is None:  # Skip invalid graphs
                continue

            # atom_graph contains directed edges: [num_directed_edges, 2]
            # where each row is [center_atom_idx, neighbor_atom_idx]
            if len(crystal_graph.atom_graph) > 0:
                _, receivers = crystal_graph.atom_graph[:, 0], crystal_graph.atom_graph[:, 1]
                # Count how many times each atom appears as a receiver (target of an edge)
                unique_receivers, counts = torch.unique(receivers, return_counts=True)
                all_neighbor_counts.extend(counts.tolist())

        if not all_neighbor_counts:
            print("Warning: No valid neighbor counts found!")
            return 0.0

        avg_num_neighbors = torch.mean(torch.tensor(all_neighbor_counts, dtype=torch.float32))

        print(f"Computed avg_num_neighbors = {avg_num_neighbors.item():.6f}")
        return avg_num_neighbors.item()
