from collections import deque
import random
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import time

if TYPE_CHECKING:
    from meip.data.dataset_jsonl import StructureJsonlDataset

class DynamicBatchSampler(Sampler):
    """Dynamic batch sampler that adjusts batch size based on weighted sum of graph sizes.

    Uses pre-cached graph sizes from JSONL data for optimal performance.
    The total size of each batch is controlled by weighted sum: atom_weight * atom_size + bond_weight * bond_size <= total_size

    This sampler can work with any underlying sampler, including:
    - None: for single GPU training
    - DistributedSampler: for distributed training
    """

    def __init__(
        self,
        dataset: "StructureJsonlDataset",
        target_batch_size: int = 100000,
        total_size: int = 50000,
        atom_weight: float = 1.0,
        bond_weight: float = 1.0,
        sampler: Optional[Sampler] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        utilization_threshold: float = 0.8,  # Utilization threshold; keep searching while below this value
        max_search_attempts: int = 50,  # Maximum number of failed search attempts
    ):
        """
        Args:
            dataset: The dataset to sample from
            target_batch_size: Target batch size
            total_size: Maximum total weighted graph size per batch
            atom_weight: Weight for atom graph size in weighted sum
            bond_weight: Weight for bond graph size in weighted sum
            sampler: Optional underlying sampler (e.g., DistributedSampler for distributed training)
            shuffle: Whether to shuffle the data (only used if sampler is None)
            drop_last: Whether to drop the last incomplete batch
            utilization_threshold: Minimum utilization ratio before submitting batch (0.0-1.0)
            max_search_attempts: Maximum number of failed search attempts before submitting batch
        """
        self.dataset = dataset
        self.target_batch_size = target_batch_size
        self.total_size = total_size
        self.atom_weight = atom_weight
        self.bond_weight = bond_weight
        self.sampler = sampler
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.utilization_threshold = utilization_threshold
        self.max_search_attempts = max_search_attempts

    def _calculate_weighted_size(self, atom_size: int, bond_size: int) -> float:
        """Calculate weighted sum of atom and bond graph sizes."""
        return self.atom_weight * atom_size + self.bond_weight * bond_size

    def __iter__(self):
        """Generate dynamic batches using a queue-based approach for better total_size utilization."""
        # Get indices from sampler or create our own
        if self.sampler is not None:
            indices_iter = self.sampler
        else:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            indices_iter = iter(indices)

        # Initialize queue for storing samples that couldn't fit in current batch
        pending_queue = deque()  # stores (idx, weighted_size) tuples

        while True:
            current_batch = []
            current_total_size = 0
            failed_attempts = 0

            # First, try to use samples from the pending queue
            temp_queue = deque()
            while pending_queue and len(current_batch) < self.target_batch_size:
                idx, weighted_size = pending_queue.popleft()

                if current_total_size + weighted_size <= self.total_size:
                    current_batch.append(idx)
                    current_total_size += weighted_size
                else:
                    temp_queue.append((idx, weighted_size))

            # Put back unused samples from queue
            pending_queue = temp_queue + pending_queue

            # Then, try to add new samples from the main iterator
            try:
                while len(current_batch) < self.target_batch_size:
                    idx = next(indices_iter)
                    atom_size, bond_size = self.dataset._get_graph_sizes_from_jsonl(idx)
                    weighted_size = self._calculate_weighted_size(atom_size, bond_size)

                    # Skip samples that exceed total_size individually
                    if weighted_size > self.total_size:
                        continue

                    # Try to add to current batch
                    if current_total_size + weighted_size <= self.total_size:
                        current_batch.append(idx)
                        current_total_size += weighted_size
                        failed_attempts = 0  # Reset failed attempts on successful addition
                    else:
                        # Add to pending queue and check if we should submit current batch
                        pending_queue.append((idx, weighted_size))
                        failed_attempts += 1

                        # Check if we should submit the current batch
                        current_utilization = current_total_size / self.total_size
                        if (current_utilization >= self.utilization_threshold or
                            failed_attempts >= self.max_search_attempts):
                            break

            except StopIteration:
                # No more samples available from main iterator
                if current_batch:
                    yield current_batch

                # Check if there are samples left in queue for final batch
                if pending_queue and not self.drop_last:
                    final_batch = []
                    final_total_size = 0

                    while pending_queue and len(final_batch) < self.target_batch_size:
                        idx, weighted_size = pending_queue.popleft()
                        if final_total_size + weighted_size <= self.total_size:
                            final_batch.append(idx)
                            final_total_size += weighted_size

                    if final_batch:
                        yield final_batch

                break

            # Yield the current batch if it's not empty
            if current_batch:
                yield current_batch
            else:
                # If we couldn't make any progress, break to avoid infinite loop
                break

    def __len__(self):
        """Estimate the number of batches."""
        # This is an approximation since batch sizes are dynamic
        dataset_len = len(self.sampler) if self.sampler is not None else len(self.dataset)
        return (dataset_len + self.target_batch_size - 1) // self.target_batch_size

class DDPDynamicBatchSampler(Sampler):
    """Dynamic batch sampler for DDP training with global progress tracking.

    This sampler:
    1. Maintains different data randomization each epoch across all ranks
    2. Balances batch counts across GPUs
    3. Provides global sample count tracking for wandb logging
    4. Tracks skipped samples due to large graph sizes
    """

    def __init__(
        self,
        dataset: "StructureJsonlDataset",
        world_size: int,
        rank: int,
        target_batch_size: int = 100,
        total_size: int = 50000,
        atom_weight: float = 1.0,
        bond_weight: float = 1.0,
        utilization_threshold: float = 0.8,
        max_search_attempts: int = 50,
        drop_last: bool = True,
    ):
        """
        Args:
            dataset: The dataset to sample from
            world_size: Total number of processes in DDP
            rank: Current process rank
            target_batch_size: Target batch size
            total_size: Maximum total weighted graph size per batch
            atom_weight: Weight for atom graph size in weighted sum
            bond_weight: Weight for bond graph size in weighted sum
            utilization_threshold: Minimum utilization ratio before submitting batch (0.0-1.0)
            max_search_attempts: Maximum number of failed search attempts before submitting batch
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.target_batch_size = target_batch_size
        self.total_size = total_size
        self.atom_weight = atom_weight
        self.bond_weight = bond_weight
        self.utilization_threshold = utilization_threshold
        self.max_search_attempts = max_search_attempts
        self.drop_last = drop_last

        # Epoch-specific data
        self.epoch = 0
        self.epoch_batch_plan = None  # Will be computed in set_epoch
        self.my_batches = None  # This rank's batches for current epoch
        self.skipped_samples = 0  # Track number of skipped samples
        self.total_samples = len(dataset)  # Total samples in dataset

    def set_epoch(self, epoch: int):
        """Set epoch and precompute all batch assignments for all ranks."""
        self.epoch = epoch
        self.epoch_batch_plan, self.my_batches = self._precompute_epoch_batches(epoch)

        return self.epoch_batch_plan, self.my_batches

    def _calculate_weighted_size(self, atom_size: int, bond_size: int) -> float:
        """Calculate weighted sum of atom and bond graph sizes."""
        return self.atom_weight * atom_size + self.bond_weight * bond_size

    def _precompute_epoch_batches(self, epoch: int):
        """Precompute batch assignments for all ranks for this epoch."""

        # Step 1: Generate global sample order (same across all ranks)
        random.seed(42 + epoch)  # Different seed for each epoch but same across ranks
        global_indices = list(range(len(self.dataset)))
        random.shuffle(global_indices)

        # Step 2: Generate global batches using queue-based approach
        global_batches, skipped = self._generate_global_batches(global_indices)
        self.skipped_samples = skipped

        # Step 3: Handle remainder batches to ensure equal distribution
        remainder = len(global_batches) % self.world_size
        if remainder > 0:
            # Collect samples from remainder batches
            remainder_samples = []
            for _ in range(remainder):
                batch_indices, _ = global_batches.pop()
                remainder_samples.extend(batch_indices)

            # Only try to redistribute if we have enough samples
            if len(remainder_samples) >= self.world_size:
                # Sort by size (large first)
                sample_sizes = [(idx, self._calculate_weighted_size(*self.dataset._get_graph_sizes_from_jsonl(idx)))
                               for idx in remainder_samples]
                sample_sizes.sort(key=lambda x: x[1], reverse=True)  # Large first!

                # Greedy assignment
                new_batches = [[] for _ in range(self.world_size)]
                batch_sizes = [0] * self.world_size

                success = True
                for idx, size in sample_sizes:
                    # Find the least loaded batch that can fit this item
                    best_batch = None
                    min_load = float('inf')

                    for i in range(self.world_size):
                        if batch_sizes[i] + size <= self.total_size and batch_sizes[i] < min_load:
                            best_batch = i
                            min_load = batch_sizes[i]

                    if best_batch is None:
                        # No batch can fit this item
                        success = False
                        break

                    new_batches[best_batch].append(idx)
                    batch_sizes[best_batch] += size

                if success:
                    for batch in new_batches:
                        global_batches.append((batch, len(batch)))
                else:
                    # If redistribution failed, count these samples as skipped
                    self.skipped_samples += len(remainder_samples)
            else:
                # Not enough samples to redistribute, count as skipped
                self.skipped_samples += len(remainder_samples)

        # Step 4: Distribute batches across ranks in round-robin fashion
        rank_batches = {rank: [] for rank in range(self.world_size)}
        for batch_idx, batch_data in enumerate(global_batches):
            assigned_rank = batch_idx % self.world_size
            rank_batches[assigned_rank].append(batch_data)

        # Step 5: Build epoch plan with cumulative sample counts
        epoch_plan = {}
        for rank in range(self.world_size):
            rank_plan = []
            for batch_indices, batch_size in rank_batches[rank]:
                rank_plan.append((batch_indices, batch_size))
            epoch_plan[rank] = rank_plan

        # Get this rank's batches
        my_batches = rank_batches[self.rank]

        return epoch_plan, my_batches

    def _generate_global_batches(self, global_indices):
        """Generate global batches using the queue-based dynamic batching algorithm."""
        from collections import deque

        indices_iter = iter(global_indices)
        pending_queue = deque()  # stores (idx, weighted_size) tuples
        global_batches = []
        skipped_count = 0  # Track skipped samples

        while True:
            current_batch = []
            current_total_size = 0
            failed_attempts = 0

            # First, try to use samples from the pending queue
            temp_queue = deque()
            while pending_queue and len(current_batch) < self.target_batch_size:
                idx, weighted_size = pending_queue.popleft()

                if current_total_size + weighted_size <= self.total_size:
                    current_batch.append(idx)
                    current_total_size += weighted_size
                else:
                    temp_queue.append((idx, weighted_size))

            # Put back unused samples from queue
            pending_queue = temp_queue + pending_queue

            # Then, try to add new samples from the main iterator
            try:
                while len(current_batch) < self.target_batch_size:
                    idx = next(indices_iter)

                    atom_size, bond_size = self.dataset._get_graph_sizes_from_jsonl(idx)

                    weighted_size = self._calculate_weighted_size(atom_size, bond_size)

                    # Skip samples that exceed total_size individually
                    if weighted_size > self.total_size:
                        skipped_count += 1
                        continue

                    # Try to add to current batch
                    if current_total_size + weighted_size <= self.total_size:
                        current_batch.append(idx)
                        current_total_size += weighted_size
                        failed_attempts = 0  # Reset failed attempts on successful addition
                    else:
                        # Add to pending queue and check if we should submit current batch
                        pending_queue.append((idx, weighted_size))
                        failed_attempts += 1

                        # Check if we should submit the current batch
                        current_utilization = current_total_size / self.total_size
                        if (current_utilization >= self.utilization_threshold or
                            failed_attempts >= self.max_search_attempts):
                            break

            except StopIteration:
                # No more samples available from main iterator
                if current_batch:
                    global_batches.append((current_batch, len(current_batch)))

                # Process remaining samples in queue for final batches
                while pending_queue and not self.drop_last:
                    final_batch = []
                    final_total_size = 0

                    temp_queue = deque()
                    while pending_queue and len(final_batch) < self.target_batch_size:
                        idx, weighted_size = pending_queue.popleft()
                        if final_total_size + weighted_size <= self.total_size:
                            final_batch.append(idx)
                            final_total_size += weighted_size
                        else:
                            temp_queue.append((idx, weighted_size))

                    pending_queue = temp_queue

                    if final_batch:
                        global_batches.append((final_batch, len(final_batch)))
                    else:
                        break  # Can't make progress

                break

            # Add the current batch if it's not empty
            if current_batch:
                global_batches.append((current_batch, len(current_batch)))
            else:
                # If we couldn't make any progress, break to avoid infinite loop
                break

        return global_batches, skipped_count

    def __iter__(self):
        """Iterate over this rank's batches."""
        if self.my_batches is None:
            raise RuntimeError("Must call set_epoch() before iterating")

        for batch_indices, batch_size in self.my_batches:
            yield batch_indices

    def __len__(self):
        """Return the number of batches for this rank."""
        if self.my_batches is None:
            # Estimate based on dataset size and world size
            return (len(self.dataset) + self.target_batch_size * self.world_size - 1) // (self.target_batch_size * self.world_size)
        return len(self.my_batches)

    def get_cumulative_samples_at_batch(self, batch_idx: int) -> int:
        """Get cumulative total samples across all ranks up to batch_idx.

        Args:
            batch_idx: Current batch index (0-based)

        Returns:
            Cumulative total number of samples processed across all ranks up to batch_idx,
            excluding skipped samples
        """
        if self.epoch_batch_plan is None:
            raise RuntimeError("Must call set_epoch() before getting global samples")

        total_samples = 0
        for rank in range(self.world_size):
            rank_batches = self.epoch_batch_plan[rank]

            # Sum samples up to and including batch_idx
            rank_samples = 0
            for i in range(min(batch_idx + 1, len(rank_batches))):
                _, batch_size = rank_batches[i]
                rank_samples += batch_size

            total_samples += rank_samples

        return total_samples

    def get_epoch_samples(self) -> int:
        """Get total number of samples for current epoch across all ranks.

        Returns:
            Total number of samples in current epoch across all ranks,
            excluding skipped samples
        """
        if self.epoch_batch_plan is None:
            raise RuntimeError("Must call set_epoch() before getting epoch samples")

        effective_samples = self.total_samples - self.skipped_samples

        return effective_samples

    def get_rank_samples_at_batch(self, rank: int, batch_idx: int) -> int:
        """Get samples for a specific rank up to batch_idx."""
        if self.epoch_batch_plan is None:
            raise RuntimeError("Must call set_epoch() before getting rank samples")

        if rank >= self.world_size:
            raise ValueError(f"Invalid rank {rank}, world_size is {self.world_size}")

        rank_batches = self.epoch_batch_plan[rank]
        rank_samples = 0

        for i in range(min(batch_idx + 1, len(rank_batches))):
            _, batch_size = rank_batches[i]
            rank_samples += batch_size

        return rank_samples

    def get_epoch_info(self):
        """Get information about the current epoch's batch distribution."""
        if self.epoch_batch_plan is None:
            return None

        info = {
            'epoch': self.epoch,
            'total_ranks': self.world_size,
            'batches_per_rank': {rank: len(batches) for rank, batches in self.epoch_batch_plan.items()},
            'samples_per_rank': {
                rank: sum(batch_size for _, batch_size in batches)
                for rank, batches in self.epoch_batch_plan.items()
            },
            'skipped_samples': self.skipped_samples,
            'total_samples': self.total_samples,
            'effective_samples': self.total_samples - self.skipped_samples
        }

        return info