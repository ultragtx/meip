from __future__ import annotations
import torch
# DDP imports
import torch.distributed as dist # Added
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler # Added for type hint

import datetime
import inspect
import os
import random
import shutil
import time
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

from meip.model.model import MEIP
from meip.utils import AverageMeter, mae, write_json
from meip.trainer.loss import CombinedLoss, DWALoss, UncertaintyLoss
from meip.trainer.scheduler import BypassLR

try:
    import wandb
except ImportError:
    wandb = None


if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Self

    from meip import TrainTask

LogFreq = Literal["epoch", "batch"]
LogEachEpoch, LogEachBatch = get_args(LogFreq)

# Helper function for metric synchronization
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

# Helper for synchronizing AverageMeters across ranks
def sync_metrics(meters: dict[str, AverageMeter], device) -> dict[str, float]:
    """Synchronizes AverageMeter stats across DDP ranks and returns global averages."""
    global_metrics = {}
    for name, meter in meters.items():
        # Sum up the total value and count across all ranks
        total_val = torch.tensor(meter.sum, device=device) # Ensure tensor is on device
        total_count = torch.tensor(meter.count, device=device)

        dist.all_reduce(total_val, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

        # Calculate global average (handle division by zero if count is zero)
        global_avg = (total_val / total_count) if total_count > 0 else torch.tensor(0.0, device=device)
        global_metrics[name] = global_avg.item() # Convert back to float
    return global_metrics



class DDPTrainer:
    """A trainer to train MEIP using energy, force, stress and magmom with DDP."""

    def __init__(
        self,
        model: MEIP,
        rank: int,
        local_rank: int,
        world_size: int,
        *,
        targets: TrainTask = "ef",
        energy_loss_ratio: float = 1,
        force_loss_ratio: float = 1,
        stress_loss_ratio: float = 0.1,
        mag_loss_ratio: float = 0.1,
        allow_missing_labels: bool = True,
        loss_type: str = "static",  # "static", "uncertainty", or "dwa"
        min_log_var: float = -7.0,  # Min log variance for uncertainty weighting
        max_log_var: float = 7.0,   # Max log variance for uncertainty weighting
        dwa_window_size: int = 1,    # Window size for DWA loss
        optimizer: str = "Adam",
        scheduler: str = "CosLR",
        criterion: str = "MSE",
        epochs: int = 50,
        starting_epoch: int = 0,
        learning_rate: float = 1e-3,
        print_freq: int = 100,
        empty_cache_freq: int = 5,
        torch_seed: int | None = None,
        data_seed: int | None = None,
        find_unused_parameters: bool = True,
        wandb_path: str | None = None,
        wandb_init_kwargs: dict | None = None,
        extra_run_config: dict | None = None,
        max_norm: float | None = None,  # Add max_norm parameter for gradient clipping
        warmup_epochs: int = 0,  # Add warmup epochs parameter
        warmup_start_lr: float = 1e-6,  # Add warmup start learning rate
        mae_unit: str = "eV",  # Display unit for MAE: "eV" or "meV"
        **kwargs,
    ) -> None:
        """Initialize all hyper-parameters for trainer.

        Args:
            model (nn.Module): a MEIP model
            targets ("ef" | "efs" | "efsm"): The training targets. Default = "ef"
            energy_loss_ratio (float): energy loss ratio in loss function
                Default = 1
            force_loss_ratio (float): force loss ratio in loss function
                Default = 1
            stress_loss_ratio (float): stress loss ratio in loss function
                Default = 0.1
            mag_loss_ratio (float): magmom loss ratio in loss function
                Default = 0.1
            allow_missing_labels (bool): whether to allow missing labels in the dataset,
                missed target will not contribute to loss and MAEs
                Default = True
            loss_type (str): type of loss function to use. Can be "static", "uncertainty", or "dwa"
            min_log_var (float): minimum value for log variance to avoid numerical issues
                Default = -7.0
            max_log_var (float): maximum value for log variance to avoid numerical issues
                Default = 7.0
            dwa_window_size (int): window size for DWA loss
            optimizer (str): optimizer to update model. Can be "Adam", "SGD", "AdamW",
                "RAdam". Default = 'Adam'
            scheduler (str): learning rate scheduler. Can be "CosLR", "ExponentialLR",
                "CosRestartLR", "BypassLR". Default = 'CosLR'
            criterion (str): loss function criterion. Can be "MSE", "Huber", "MAE"
                Default = 'MSE'
            epochs (int): number of epochs for training
                Default = 50
            starting_epoch (int): The epoch number to start training at.
            learning_rate (float): initial learning rate
                Default = 1e-3
            print_freq (int): frequency to print training output
                Default = 100
            empty_cache_freq (int): frequency to empty cache
                Default = 5
            torch_seed (int): random seed for torch
                Default = None
            data_seed (int): random seed for random
                Default = None
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use cuda with most available memory
                Default = False
            wandb_path (str | None): The project and run name separated by a slash:
                "project/run_name". If None, wandb logging is not used.
                Default = None
            wandb_init_kwargs (dict): Additional kwargs to pass to wandb.init.
                Default = None
            extra_run_config (dict): Additional hyper-params to be recorded by wandb
                that are not included in the trainer_args. Default = None
            max_norm (float | None): Maximum norm for gradient clipping
                Default = None
            warmup_epochs (int): number of warmup epochs
                Default = 0
            warmup_start_lr (float): initial learning rate for warmup
                Default = 1e-6
            mae_unit (str): Unit for displaying MAE values. "eV" or "meV".
                Use "meV" for small molecules (MD17, QM9) for better precision.
                Default = "eV"

            **kwargs (dict): additional hyper-params for optimizer, scheduler, etc.

        Raises:
            NotImplementedError: If the optimizer or scheduler is not implemented
            ImportError: If wandb_path is specified but wandb is not installed
            ValueError: If wandb_path is specified but not in the format
                'project/run_name'
        """
        # Store DDP info
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}") # Assign device based on local rank

        # Store trainer args for reproducibility (only rank 0 needs full args for logging)
        if self.rank == 0:
            self.trainer_args = {
                k: v
                for k, v in locals().items()
                if k not in {"self", "__class__", "model", "kwargs", "rank", "local_rank"}
            } | kwargs
        else:
            self.trainer_args = {} # Non-rank 0 doesn't need to store this

        self.targets = targets

        # MAE display unit (eV or meV)
        self.mae_unit = mae_unit.lower()
        self.mae_scale = 1000.0 if self.mae_unit == "mev" else 1.0
        self.mae_unit_str = "meV" if self.mae_unit == "mev" else "eV"

        # Seed setting: Ensure consistent initial weights but different data shuffling per epoch
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        if data_seed:
            random.seed(data_seed)
            np.random.seed(data_seed)

        model = model.to(self.device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # Convert to sync batch norm if needed
        self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=find_unused_parameters)

        # Define loss criterion
        if loss_type == "uncertainty":
            if self.rank == 0:
                print(f"Using UncertaintyLoss with min_log_var={min_log_var}, max_log_var={max_log_var}")

            self.criterion = UncertaintyLoss(
                target_str=self.targets,
                criterion=criterion,
                energy_loss_ratio=energy_loss_ratio,
                force_loss_ratio=force_loss_ratio,
                stress_loss_ratio=stress_loss_ratio,
                mag_loss_ratio=mag_loss_ratio,
                delta=kwargs.pop("delta", 0.1),
                allow_missing_labels=allow_missing_labels,
                min_log_var=min_log_var,
                max_log_var=max_log_var,
            )
        elif loss_type == "dwa":
            if self.rank == 0:
                print(f"Using DWALoss with window_size={dwa_window_size}")

            self.criterion = DWALoss(
                target_str=self.targets,
                criterion=criterion,
                energy_loss_ratio=energy_loss_ratio,
                force_loss_ratio=force_loss_ratio,
                stress_loss_ratio=stress_loss_ratio,
                mag_loss_ratio=mag_loss_ratio,
                delta=kwargs.pop("delta", 0.1),
                allow_missing_labels=allow_missing_labels,
                window_size=dwa_window_size,
            )
        else:
            if self.rank == 0:
                print(f"Using CombinedLoss with static weights")

            self.criterion = CombinedLoss(
                target_str=self.targets,
                criterion=criterion,
                energy_loss_ratio=energy_loss_ratio,
                force_loss_ratio=force_loss_ratio,
                stress_loss_ratio=stress_loss_ratio,
                mag_loss_ratio=mag_loss_ratio,
                delta=kwargs.pop("delta", 0.1),
                allow_missing_labels=allow_missing_labels,
            )

        # Gather all parameters that need to be optimized
        params_to_optimize = list(self.model.parameters())

        # Add loss function parameters if using learnable weights
        if loss_type == "uncertainty":
            loss_params = list(self.criterion.parameters())
            if loss_params:
                params_to_optimize.extend(loss_params)
                if self.rank == 0:
                    print(f"Added {len(loss_params)} loss weighting parameters to optimizer")

        # Redefine optimizer with all parameters
        if optimizer == "SGD":
            momentum = kwargs.pop("momentum", 0.9)
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.SGD(
                params_to_optimize,
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == "Adam":
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.Adam(
                params_to_optimize, learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "AdamW":
            weight_decay = kwargs.pop("weight_decay", 1e-2)
            self.optimizer = torch.optim.AdamW(
                params_to_optimize, learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "RAdam":
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.RAdam(
                params_to_optimize, learning_rate, weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented.")


        # Define learning rate scheduler
        default_decay_frac = 1e-2
        if scheduler in {"MultiStepLR", "multistep"}:
            scheduler_params = kwargs.pop(
                "scheduler_params",
                {
                    "milestones": [4 * epochs, 6 * epochs, 8 * epochs, 9 * epochs],
                    "gamma": 0.3,
                },
            )
            self.scheduler = MultiStepLR(self.optimizer, **scheduler_params)
            self.scheduler_type = "multistep"
        elif scheduler in {"ExponentialLR", "Exp", "Exponential"}:
            scheduler_params = kwargs.pop("scheduler_params", {"gamma": 0.98})
            self.scheduler = ExponentialLR(self.optimizer, **scheduler_params)
            self.scheduler_type = "exp"
        elif scheduler in {"CosineAnnealingLR", "CosLR", "Cos", "cos"}:
            scheduler_params = kwargs.pop(
                "scheduler_params", {"decay_fraction": default_decay_frac}
            )
            decay_fraction = scheduler_params.pop("decay_fraction", default_decay_frac)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,  # Maximum number of iterations.
                eta_min=decay_fraction * learning_rate,
            )
            self.scheduler_type = "cos"
        elif scheduler == "CosRestartLR":
            scheduler_params = kwargs.pop(
                "scheduler_params",
                {"decay_fraction": default_decay_frac, "T_0": 10, "T_mult": 2},
            )
            decay_fraction = scheduler_params.pop("decay_fraction", default_decay_frac)
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                eta_min=decay_fraction * learning_rate,
                **scheduler_params,
            )
            self.scheduler_type = "cosrestart"
        elif scheduler in {"BypassLR", "bypass", "none", "None"}:
            self.scheduler = BypassLR(self.optimizer)
            self.scheduler_type = "bypass"
            if self.rank == 0:
                print("Using BypassLR scheduler - learning rate will remain constant")
        elif scheduler in {"ReduceLROnPlateau", "Plateau", "Plat", "plat"}:
            scheduler_params = kwargs.pop(
                "scheduler_params",
                {"factor": 0.8, "patience": 20, "min_lr": 1e-6},
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                **scheduler_params,
            )
            self.scheduler_type = "plateau"
            if self.rank == 0:
                print(f"Using ReduceLROnPlateau scheduler with params: {scheduler_params}")
        else:
            raise NotImplementedError(f"Scheduler {scheduler} not implemented.")

        # Add warmup scheduler if warmup_epochs > 0
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr

        if self.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.warmup_start_lr / learning_rate,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            # Combine warmup with main scheduler
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[self.warmup_scheduler, self.scheduler],
                milestones=[self.warmup_epochs]
            )

        self.epochs = epochs
        self.starting_epoch = starting_epoch

        self.print_freq = print_freq
        self.empty_cache_freq = empty_cache_freq

        # Training history only needs to be accurate on rank 0 for saving best model logic
        self.training_history: dict[
            str, dict[Literal["train", "val", "test"], list[float]]
        ] = {key: {"train": [], "val": [], "test": []} for key in self.targets}

        # --- WandB Initialization (Rank 0 Only) ---
        self.use_wandb = False
        if self.rank == 0 and wandb_path:
            if wandb is None:
                raise ImportError("WandB requested but not installed.")
            if wandb_path.count("/") >= 1: # Allow subdirs like project/group/run_name
                project = wandb_path.split('/')[0]
                run_name = wandb_path.split('/')[-1]
            else:
                raise ValueError(f"{wandb_path=} should be 'project/run_name' or 'project/group/run_name'")

            print(f"Initializing WandB for project='{project}', name='{run_name}'")
            wandb.init(
                project=project,
                name=run_name,
                # group="DDP", # Optional: Group runs if needed
                config=self.trainer_args | (extra_run_config or {}),
                **(wandb_init_kwargs or {}),
            )
            self.use_wandb = True

        # --- End WandB ---

        # Check for leftover kwargs
        if kwargs and self.rank == 0:
             print(f"Warning: Unused kwargs provided: {kwargs}")

        self.max_norm = max_norm  # Store max_norm for gradient clipping

        # Add step tracking for wandb
        self.last_wandb_step = 0

        # Initialize checkpoint state (used for resume metrics calculation)
        self._checkpoint_state = None


    def _calculate_resume_metrics(self, train_loader: DataLoader, checkpoint_state: dict | None = None) -> tuple[int, int]:
        """Calculate cumulative samples seen and last wandb step when resuming from checkpoint.

        Args:
            train_loader: Training data loader
            checkpoint_state: Checkpoint state dictionary containing saved progress

        Returns:
            tuple: (cumulative_samples_seen, last_wandb_step)
        """
        if self.starting_epoch == 0:
            return 0, 0

        # Try to get exact values from checkpoint first
        if checkpoint_state is not None:
            saved_cumulative_samples = checkpoint_state.get("cumulative_samples_seen")
            saved_last_wandb_step = checkpoint_state.get("last_wandb_step")

            if saved_cumulative_samples is not None and saved_last_wandb_step is not None:
                if self.rank == 0:
                    print(f"Resuming from epoch {self.starting_epoch}")
                    print(f"Using saved cumulative samples seen: {saved_cumulative_samples}")
                    print(f"Using saved last wandb step: {saved_last_wandb_step}")
                return saved_cumulative_samples, saved_last_wandb_step

        # Fallback to estimation if saved values are not available (backward compatibility)
        if self.rank == 0:
            print(f"Resuming from epoch {self.starting_epoch}")
            print("Warning: No saved training progress found in checkpoint, using estimation")

        # Calculate dataset info for seen samples tracking
        total_dataset_size = len(train_loader.dataset)
        use_ddp_dynamic_batch = (hasattr(train_loader, 'batch_sampler') and
                                hasattr(train_loader.batch_sampler, 'get_cumulative_samples_at_batch'))

        if use_ddp_dynamic_batch:
            # For dynamic batch sampler, we need to estimate based on training history
            # This is an approximation since we don't have exact batch sizes from previous training
            avg_samples_per_epoch = total_dataset_size  # Rough estimate
            cumulative_samples_seen = self.starting_epoch * avg_samples_per_epoch
        else:
            # Calculate estimated samples per epoch for regular batch sampling
            estimated_samples_per_epoch = len(train_loader) * train_loader.batch_size if hasattr(train_loader, 'batch_size') else total_dataset_size // self.world_size
            cumulative_samples_seen = self.starting_epoch * estimated_samples_per_epoch * self.world_size

        # Set last wandb step to the cumulative samples seen
        last_wandb_step = cumulative_samples_seen

        if self.rank == 0:
            print(f"Estimated cumulative samples seen: {cumulative_samples_seen}")
            print(f"Setting last wandb step to: {last_wandb_step}")

        return cumulative_samples_seen, last_wandb_step

    def _check_wandb_step(self, current_step: int, context: str) -> None:
        """Check if wandb step is monotonically increasing.

        Args:
            current_step: Current step number to check
            context: Context string for debugging (e.g. "train_batch", "val_epoch")

        Raises:
            RuntimeError: If step number is not monotonically increasing
        """
        if current_step < self.last_wandb_step:
            raise RuntimeError(
                f"Non-monotonic wandb step detected in {context}\n"
                f"Current step {current_step} is less than last step {self.last_wandb_step}\n"
                f"Difference: {self.last_wandb_step - current_step}"
            )
        self.last_wandb_step = current_step

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        *,
        save_dir: str | None = None,
        save_test_result: bool = False,
        train_composition_model: bool = False,
        wandb_log_freq: LogFreq = LogEachBatch,
    ) -> None:
        """Train the model using torch data_loaders.

        Args:
            train_loader (DataLoader): train loader to update MEIP weights
            val_loader (DataLoader): val loader to test accuracy after each epoch
            test_loader (DataLoader):  test loader to test accuracy at end of training.
                Can be None.
                Default = None
            save_dir (str): the dir name to save the trained weights
                Default = None
            save_test_result (bool): Whether to save the test set prediction in a JSON
                file. Default = False
            train_composition_model (bool): whether to train the composition model
                (AtomRef), this is suggested when the fine-tuning dataset has large
                elemental energy shift from the pretrained MEIP, which typically comes
                from different DFT pseudo-potentials.
                Default = False
            wandb_log_freq ("epoch" | "batch"): Frequency of logging to wandb.
                'epoch' logs once per epoch, 'batch' logs after every batch.
                Default = "batch"

        Raises:
            AssertionError: If DataLoaders don't use DistributedSampler.
        """
        # Ensure DistributedSampler is used (check if sampler name contains "distributed" or "ddp")
        def has_distributed_sampler(loader):
            # Check batch_sampler first
            if hasattr(loader, 'batch_sampler') and loader.batch_sampler is not None:
                batch_sampler_name = loader.batch_sampler.__class__.__name__.lower()
                if 'distributed' in batch_sampler_name or 'ddp' in batch_sampler_name:
                    return True
                # Also check batch_sampler's internal sampler if it exists
                if hasattr(loader.batch_sampler, 'sampler') and loader.batch_sampler.sampler is not None:
                    sampler_name = loader.batch_sampler.sampler.__class__.__name__.lower()
                    if 'distributed' in sampler_name or 'ddp' in sampler_name:
                        return True

            # Check regular sampler
            if hasattr(loader, 'sampler') and loader.sampler is not None:
                sampler_name = loader.sampler.__class__.__name__.lower()
                if 'distributed' in sampler_name or 'ddp' in sampler_name:
                    return True

            return False

        assert has_distributed_sampler(train_loader), "train_loader requires a sampler with 'distributed' or 'ddp' in its name"
        assert has_distributed_sampler(val_loader), "val_loader requires a sampler with 'distributed' or 'ddp' in its name"
        if test_loader:
            assert has_distributed_sampler(test_loader), "test_loader requires a sampler with 'distributed' or 'ddp' in its name"

        # Synchronize all processes to ensure directory exists before proceeding
        dist.barrier()

        # Calculate dataset info for seen samples tracking
        total_dataset_size = len(train_loader.dataset)
        # Detect if using DDPDynamicBatchSampler
        use_ddp_dynamic_batch = (hasattr(train_loader, 'batch_sampler') and
                                hasattr(train_loader.batch_sampler, 'get_cumulative_samples_at_batch'))

        if use_ddp_dynamic_batch:
            if self.rank == 0:
                print(f"Begin Training: {self.world_size} GPUs, Using device {self.device} on Rank 0")
                print(f"Training targets: {self.targets}")
                print(f"Total dataset size: {total_dataset_size}")
                print(f"Using DDP dynamic batch sampler - global sample counting enabled")
        else:
            # Calculate estimated samples per epoch for regular batch sampling
            estimated_samples_per_epoch = len(train_loader) * train_loader.batch_size if hasattr(train_loader, 'batch_size') else len(train_loader.dataset) // self.world_size

            if self.rank == 0:
                print(f"Begin Training: {self.world_size} GPUs, Using device {self.device} on Rank 0")
                print(f"Training targets: {self.targets}")
                print(f"Total dataset size: {total_dataset_size}")
                print(f"Estimated samples per epoch per rank: {estimated_samples_per_epoch}")
                print(f"Estimated total samples per epoch: {estimated_samples_per_epoch * self.world_size}")
                print(f"Estimated batches per epoch: {len(train_loader)}")

        # Initialize cumulative sample counter and wandb step tracking
        # Calculate resume metrics if starting from checkpoint
        checkpoint_state = getattr(self, '_checkpoint_state', None)
        cumulative_samples_seen, resume_last_wandb_step = self._calculate_resume_metrics(train_loader, checkpoint_state)

        # Update wandb step tracking for resume
        if resume_last_wandb_step > 0:
            self.last_wandb_step = resume_last_wandb_step

        # Clear checkpoint state after initialization to save memory
        self._checkpoint_state = None

        # Print additional info for rank 0
        if self.rank == 0:
            print(f"model device: {next(self.model.parameters()).device}")
            if self.model.module.composition_model:
                print(f"composition model device: {next(self.model.module.composition_model.parameters()).device}")

            # Print if using uncertainty weighting
            if isinstance(self.criterion, UncertaintyLoss):
                print("Using uncertainty weighting for loss balancing")

        # Turn composition model training on / off
        # Access the original model through .module to set requires_grad
        if self.model.module.composition_model:
            for param in self.model.module.composition_model.parameters():
                param.requires_grad = train_composition_model

        for epoch in range(self.starting_epoch, self.epochs):
            # Initialize epoch timing trackers
            epoch_train_time = 0.0
            epoch_val_time = 0.0
            epoch_test_time = None
            # Print learning rate for current epoch
            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"\n=== Epoch {epoch} - Learning Rate: {current_lr:.7f} ===")

            # Set epoch for sampler - crucial for shuffling across epochs in DDP
            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
                train_loader.batch_sampler.set_epoch(epoch)
                if self.rank == 0:
                    print(train_loader.batch_sampler.get_epoch_info())
            elif hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            if hasattr(val_loader, 'batch_sampler') and hasattr(val_loader.batch_sampler, 'set_epoch'):
                val_loader.batch_sampler.set_epoch(epoch)
                if self.rank == 0:
                    print(val_loader.batch_sampler.get_epoch_info())
            elif hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'set_epoch'):
                val_loader.sampler.set_epoch(epoch)

            if test_loader:
                if hasattr(test_loader, 'batch_sampler') and hasattr(test_loader.batch_sampler, 'set_epoch'):
                    test_loader.batch_sampler.set_epoch(epoch)
                    if self.rank == 0:
                        print(test_loader.batch_sampler.get_epoch_info())
                elif hasattr(test_loader, 'sampler') and hasattr(test_loader.sampler, 'set_epoch'):
                    test_loader.sampler.set_epoch(epoch)



            # --- Train one epoch ---
            # For both regular and dynamic batch, use _train with appropriate sample counting
            _t0 = time.perf_counter()
            train_metrics = self._train(
                train_loader, epoch, wandb_log_freq,
                seen_samples_start_epoch=cumulative_samples_seen,
                samples_per_epoch=estimated_samples_per_epoch if not use_ddp_dynamic_batch else None,
                use_ddp_dynamic_batch=use_ddp_dynamic_batch
            )
            epoch_train_time = time.perf_counter() - _t0

            # Update cumulative samples based on batch sampler type
            if use_ddp_dynamic_batch:
                # For dynamic batch, get samples for current epoch
                epoch_samples = train_loader.batch_sampler.get_epoch_samples()
                cumulative_samples_seen += epoch_samples
            else:
                # For regular batch, use estimated samples
                cumulative_samples_seen += estimated_samples_per_epoch * self.world_size

            seen_samples_end_epoch = cumulative_samples_seen

            # Synchronize metrics across all ranks
            global_train_mae = sync_metrics(train_metrics['mae_meters'], self.device)
            global_train_loss = sync_metrics({'loss': train_metrics['loss_meter']}, self.device)['loss']

            # Synchronize uncertainty weights if available
            global_train_uncertainty = None
            if train_metrics.get('uncertainty_weights'):
                global_train_uncertainty = sync_metrics(train_metrics['uncertainty_weights'], self.device)

            # Synchronize DWA weights if available
            global_train_dwa = None
            if train_metrics.get('dwa_weights'):
                global_train_dwa = sync_metrics(train_metrics['dwa_weights'], self.device)

            # Check for NaN on rank 0 after sync
            if self.rank == 0 and ("e" in global_train_mae and np.isnan(global_train_mae["e"])):
                print("NaN detected in training MAE. Exiting.")
                break

            # --- Validate ---
            _v0 = time.perf_counter()
            val_metrics = self._validate(val_loader, seen_samples_end_epoch, is_test=False, wandb_log_freq=LogEachEpoch)
            epoch_val_time = time.perf_counter() - _v0
            global_val_mae = sync_metrics(val_metrics['mae_meters'], self.device)
            global_val_loss = sync_metrics({'loss': val_metrics['loss_meter']}, self.device)['loss']

            # Synchronize uncertainty weights for validation if available
            global_val_uncertainty = None
            if val_metrics.get('uncertainty_weights'):
                global_val_uncertainty = sync_metrics(val_metrics['uncertainty_weights'], self.device)

            # Synchronize DWA weights for validation if available
            global_val_dwa = None
            if val_metrics.get('dwa_weights'):
                global_val_dwa = sync_metrics(val_metrics['dwa_weights'], self.device)

            # --- Logging and Checkpointing (Rank 0 Only) ---
            if self.rank == 0:
                # Store history (using globally averaged metrics)
                for key in self.targets:
                    self.training_history[key]["train"].append(global_train_mae[key])
                    self.training_history[key]["val"].append(global_val_mae[key])

                # NaN check for validation
                if "e" in global_val_mae and np.isnan(global_val_mae["e"]):
                    print("NaN detected in validation MAE. Exiting.")
                    break

                print(f"--- Epoch {epoch} Summary ---")
                train_msg = f"Train Avg MAE({self.mae_unit_str}): " + " | ".join([f"{k}: {v * self.mae_scale:.4f}" for k, v in global_train_mae.items()]) + f" | Loss: {global_train_loss:.4f}"
                print(train_msg)
                val_msg = f"Val   Avg MAE({self.mae_unit_str}): " + " | ".join([f"{k}: {v * self.mae_scale:.4f}" for k, v in global_val_mae.items()]) + f" | Loss: {global_val_loss:.4f}"
                print(val_msg)

                # Print uncertainty weights if available
                if global_train_uncertainty:
                    unc_msg = "Uncertainty weights: " + " | ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in global_train_uncertainty.items()])
                    print(unc_msg)

                # Print DWA weights if available
                if global_train_dwa:
                    dwa_msg = "DWA weights: " + " | ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in global_train_dwa.items()])
                    print(dwa_msg)

                print("-" * 25)

                # Log epoch metrics to wandb
                if self.use_wandb:
                    log_data = {
                        f"train_global/{k}_mae": v for k, v in global_train_mae.items()
                    } | {
                        f"val_global/{k}_mae": v for k, v in global_val_mae.items()
                    } | {
                        "train_global/loss": global_train_loss,
                        "val_global/loss": global_val_loss,
                    }

                    # Add uncertainty weights to global logs
                    if global_train_uncertainty:
                        for k, v in global_train_uncertainty.items():
                            task = k.split('_')[1]  # Extract task name from "weight_e", etc.
                            log_data[f"train_global/uncertainty_{task}"] = v

                    if global_val_uncertainty:
                        for k, v in global_val_uncertainty.items():
                            task = k.split('_')[1]  # Extract task name
                            log_data[f"val_global/uncertainty_{task}"] = v

                    # Add DWA weights to global logs
                    if global_train_dwa:
                        for k, v in global_train_dwa.items():
                            task = k.split('_')[1]  # Extract task name from "dwa_e", etc.
                            log_data[f"train_global/dwa_{task}"] = v

                    if global_val_dwa:
                        for k, v in global_val_dwa.items():
                            task = k.split('_')[1]  # Extract task name
                            log_data[f"val_global/dwa_{task}"] = v

                    # Check step monotonicity for epoch logging
                    self._check_wandb_step(seen_samples_end_epoch, f"epoch_{epoch}")

                    print(f"Logging epoch {epoch} metrics at step {seen_samples_end_epoch}")
                    wandb.log(log_data, step=seen_samples_end_epoch)

            # Step the scheduler BEFORE saving checkpoint
            # Stepping 10 times per epoch as in the original implementation would require
            # extra synchronization, so we step once per epoch here instead
            if self.scheduler_type == "plateau":
                # ReduceLROnPlateau needs the validation metric
                self.scheduler.step(global_val_loss)
            else:
                self.scheduler.step()

            # --- Logging and Checkpointing AFTER scheduler step (Rank 0 Only) ---
            if self.rank == 0:
                # Save checkpoint logic - now includes updated scheduler state
                if save_dir:
                    self.save_checkpoint(epoch, global_val_mae, save_dir, seen_samples_end_epoch)
            # --- End Rank 0 block ---

            # Run test set evaluation every 10 epochs (all ranks participate)
            if test_loader and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.epochs - 1):
                # Synchronize all processes before test evaluation
                dist.barrier()

                if self.rank == 0:
                    print("\nRunning test set evaluation...")

                # Run test evaluation on all ranks
                _te0 = time.perf_counter()
                test_metrics = self._validate(
                    test_loader,
                    seen_samples=seen_samples_end_epoch,
                    is_test=True,
                    test_result_save_path=save_dir if save_test_result else None,
                    wandb_log_freq=LogEachEpoch,
                )
                epoch_test_time = time.perf_counter() - _te0
                global_test_mae = sync_metrics(test_metrics['mae_meters'], self.device)
                global_test_loss = sync_metrics({'loss': test_metrics['loss_meter']}, self.device)['loss']

                # Synchronize uncertainty weights for test if available
                global_test_uncertainty = None
                if test_metrics.get('uncertainty_weights'):
                    global_test_uncertainty = sync_metrics(test_metrics['uncertainty_weights'], self.device)

                # Synchronize DWA weights for test if available
                global_test_dwa = None
                if test_metrics.get('dwa_weights'):
                    global_test_dwa = sync_metrics(test_metrics['dwa_weights'], self.device)

                # Only rank 0 prints and logs
                if self.rank == 0:
                    test_msg = f"Test Avg MAE({self.mae_unit_str}): " + " | ".join([f"{k}: {v * self.mae_scale:.4f}" for k, v in global_test_mae.items()]) + f" | Loss: {global_test_loss:.4f}"
                    print(test_msg)

                    # Print uncertainty weights if available
                    if global_test_uncertainty:
                        unc_msg = "Test uncertainty weights: " + " | ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in global_test_uncertainty.items()])
                        print(unc_msg)

                    # Print DWA weights if available
                    if global_test_dwa:
                        dwa_msg = "Test DWA weights: " + " | ".join([f"{k.split('_')[1]}: {v:.4f}" for k, v in global_test_dwa.items()])
                        print(dwa_msg)

                    if self.use_wandb:
                        log_data = {
                            f"test_global/{k}_mae": v for k, v in global_test_mae.items()
                        } | {
                            "test_global/loss": global_test_loss,
                        }

                        # Add uncertainty weights to global test logs
                        if global_test_uncertainty:
                            for k, v in global_test_uncertainty.items():
                                task = k.split('_')[1]  # Extract task name
                                log_data[f"test_global/uncertainty_{task}"] = v

                        # Add DWA weights to global test logs
                        if global_test_dwa:
                            for k, v in global_test_dwa.items():
                                task = k.split('_')[1]  # Extract task name
                                log_data[f"test_global/dwa_{task}"] = v

                        # Check step monotonicity for test logging
                        self._check_wandb_step(seen_samples_end_epoch, f"test_epoch_{epoch}")

                        wandb.log(log_data, step=seen_samples_end_epoch)

                # Synchronize all processes after test evaluation
                dist.barrier()

            # Update DWALoss weights if applicable
            if isinstance(self.criterion, DWALoss):
                self.criterion.update_weights(device=self.device)
                if self.rank == 0:
                    weights = self.criterion.get_current_weights()
                    weight_msg = "DWA weights: " + " | ".join([f"{k}: {v:.4f}" for k, v in weights.items()])
                    print(weight_msg)

                    if self.use_wandb:
                        wandb.log({f"dwa_weights/{k}": v for k, v in weights.items()},
                                  step=seen_samples_end_epoch)

            # Log current learning rate (after scheduler step)
            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.use_wandb:
                    # Log LR and timing
                    wandb.log({"learning_rate": current_lr}, step=seen_samples_end_epoch)
                    time_log = {
                        "time/epoch_total_s": epoch_train_time + epoch_val_time + (epoch_test_time if epoch_test_time is not None else 0.0),
                        "time/train_s": epoch_train_time,
                        "time/val_s": epoch_val_time,
                    }
                    if epoch_test_time is not None:
                        time_log["time/test_s"] = epoch_test_time
                    wandb.log(time_log, step=seen_samples_end_epoch)
                # Print epoch timing summary
                epoch_total_time = epoch_train_time + epoch_val_time + (epoch_test_time if epoch_test_time is not None else 0.0)
                timing_msg = (
                    f"Epoch {epoch} timing (s): total {epoch_total_time:.2f} | "
                    f"train {epoch_train_time:.2f} | val {epoch_val_time:.2f}"
                )
                if epoch_test_time is not None:
                    timing_msg += f" | test {epoch_test_time:.2f}"
                print(timing_msg)
                print(f"Current learning rate: {current_lr:.7f}")

            # Barrier to ensure all processes finish epoch before starting next
            torch.cuda.empty_cache()
            dist.barrier()

        # Final barrier
        dist.barrier()
        if self.rank == 0:
             print("Training Finished.")

    def _train(
        self,
        train_loader: DataLoader,
        current_epoch: int,
        wandb_log_freq: LogFreq = LogEachBatch,
        seen_samples_start_epoch: int = 0,
        samples_per_epoch: int = 0,
        use_ddp_dynamic_batch: bool = False,
    ) -> dict: # Return local meters for aggregation
        """Train all data for one epoch.

        Args:
            train_loader (DataLoader): train loader to update MEIP weights
            current_epoch (int): used for resume unfinished training
            wandb_log_freq ("epoch" | "batch"): Frequency of logging to wandb
            seen_samples_start_epoch (int): Number of samples seen at start of this epoch
            samples_per_epoch (int): Total samples per epoch across all GPUs
            use_ddp_dynamic_batch (bool): Whether to use DDPDynamicBatchSampler

        Returns:
            dictionary of training errors
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # Use dicts for meters for easier synchronization later
        losses = AverageMeter()
        mae_errors = {target: AverageMeter() for target in self.targets}

        # Add uncertainty weight meters if using uncertainty weighting
        uncertainty_weights = {}
        if isinstance(self.criterion, UncertaintyLoss):
            # Get current weights once for consistent reporting
            current_weights = self.criterion.get_current_weights()
            uncertainty_weights = {
                f"weight_{k}": AverageMeter()
                for k in self.targets
            }
            # Initialize meters with current weights
            for k, v in current_weights.items():
                if k in self.targets:
                    uncertainty_weights[f"weight_{k}"].update(v, 1)

        # Add DWA weight meters if using DWA
        dwa_weights = {}
        if isinstance(self.criterion, DWALoss):
            current_weights = self.criterion.get_current_weights()
            dwa_weights = {
                f"dwa_{k}": AverageMeter()
                for k in current_weights
            }
            # Initialize meters with current weights
            for k, v in current_weights.items():
                dwa_weights[f"dwa_{k}"].update(v, 1)

        # switch to train mode (affects dropout, batchnorm etc.)
        self.model.train()

        batch_size = train_loader.batch_size

        start = time.perf_counter()  # start timer
        for idx, (graphs, targets) in enumerate(train_loader):
            # print(f"Rank {self.rank} is training batch...,graphs={len(graphs)} {idx + 1}/{len(train_loader)}")
            # measure data loading time
            data_time.update(time.perf_counter() - start)

            # get input
            for g in graphs:
                requires_force = "f" in self.targets
                g.atom_frac_coord.requires_grad = requires_force
            graphs = [g.to(self.device) for g in graphs]
            targets = {k: self.move_to(v, self.device) for k, v in targets.items()}

            # compute output
            prediction = self.model(graphs, task=self.targets) # DDP handles gather/scatter internally
            combined_loss = self.criterion(targets, prediction)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            combined_loss["loss"].backward()

            # Add gradient clipping if max_norm is specified
            if self.max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            self.optimizer.step()

            # Skip all_reduce for now and accumulate locally to avoid synchronization overhead
            losses.update(combined_loss["loss"].data.cpu().item(), len(graphs))
            for key in self.targets:
                mae_errors[key].update(
                    combined_loss[f"{key}_MAE"].cpu().item(),
                    combined_loss[f"{key}_MAE_size"],
                )

            # Track uncertainty weights if using uncertainty weighting
            if uncertainty_weights and isinstance(self.criterion, UncertaintyLoss):
                for key in self.targets:
                    weight_key = f"uncertainty_weight_{key}"
                    if weight_key in combined_loss:
                        uncertainty_weights[f"weight_{key}"].update(combined_loss[weight_key], 1)

            # Track DWA weights if using DWA
            if dwa_weights and isinstance(self.criterion, DWALoss):
                for key in self.targets:
                    dwa_key = f"dwa_weight_{key}"
                    if dwa_key in combined_loss:
                        dwa_weights[f"dwa_{key}"].update(combined_loss[dwa_key], 1)

            # free memory
            del graphs, targets
            del prediction, combined_loss
            torch.cuda.empty_cache()

            # measure elapsed time
            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter() # Reset timer for next batch

            # current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.rank == 0 and (idx == 0 or (idx + 1) % self.print_freq == 0 or idx == len(train_loader) - 1):
                # Note: meters accessed here contain local (rank 0) data for instantaneous print
                message = (
                    f"Epoch: [{current_epoch}][{idx + 1}/{len(train_loader)}] | "
                    f"LR {current_lr:.7f} | "
                    f"Time ({batch_time.avg:.3f})({data_time.avg:.3f}) | "
                    f"Loss {losses.val:.4f}({losses.avg:.4f}) | "
                    f"Batch Size {batch_size} | "  # Add batch size to output
                    f"MAE({self.mae_unit_str}) "
                )
                for key in self.targets:
                    if key in mae_errors: # Check if key exists
                        val_scaled = mae_errors[key].val * self.mae_scale
                        avg_scaled = mae_errors[key].avg * self.mae_scale
                        message += f"{key} {val_scaled:.4f}({avg_scaled:.4f})  "

                # Add uncertainty weights to print message if using uncertainty weighting
                if uncertainty_weights:
                    message += "| Weights "
                    for key in self.targets:
                        weight_key = f"weight_{key}"
                        if weight_key in uncertainty_weights and uncertainty_weights[weight_key].count > 0:
                            message += f"{key} {uncertainty_weights[weight_key].val:.3f}  "

                # Add DWA weights to print message
                if dwa_weights:
                    message += "| DWA "
                    for key in self.targets:
                        dwa_key = f"dwa_{key}"
                        if dwa_key in dwa_weights and dwa_weights[dwa_key].count > 0:
                            message += f"{key} {dwa_weights[dwa_key].val:.3f}  "

                print(message)

            # Log train batch metrics to wandb (based on rank 0's current averages)
            if self.rank == 0 and self.use_wandb and wandb_log_freq == LogEachBatch:
                log_data = {f"train/{k}_mae": v.avg for k, v in mae_errors.items() if v.count > 0} # Log avg up to this batch
                log_data.update({
                    "train/loss": losses.avg,
                    "train/learning_rate": current_lr,
                    "train/epoch": current_epoch,
                })

                # Add uncertainty weights to wandb logs
                if uncertainty_weights:
                    for key in self.targets:
                        weight_key = f"weight_{key}"
                        if weight_key in uncertainty_weights and uncertainty_weights[weight_key].count > 0:
                            log_data[f"train/uncertainty_{key}"] = uncertainty_weights[weight_key].avg

                # Add DWA weights to wandb logs
                if dwa_weights:
                    for key in self.targets:
                        dwa_key = f"dwa_{key}"
                        if dwa_key in dwa_weights and dwa_weights[dwa_key].count > 0:
                            log_data[f"train/dwa_{key}"] = dwa_weights[dwa_key].avg

                # Calculate seen samples up to this batch using batch_sampler if available
                if use_ddp_dynamic_batch:
                    seen_samples_current_batch = seen_samples_start_epoch + train_loader.batch_sampler.get_cumulative_samples_at_batch(idx)
                else:
                    # Fallback to old calculation method
                    seen_samples_current_batch = seen_samples_start_epoch + (idx + 1) * batch_size * self.world_size

                # Check step monotonicity
                self._check_wandb_step(seen_samples_current_batch, f"train_batch_{idx}")

                wandb.log(log_data, step=seen_samples_current_batch)

            # print(f"====Rank {self.rank} done training batch..., {idx + 1}")

        # print(f">>>>>>>>>>>>>>>Rank {self.rank} done training...")

        # Return the meters containing local sums and counts for aggregation
        return {
            "loss_meter": losses,
            "mae_meters": mae_errors,
            "uncertainty_weights": uncertainty_weights if uncertainty_weights else None,
            "dwa_weights": dwa_weights if dwa_weights else None
        }

    def _validate(
        self,
        val_loader: DataLoader,
        seen_samples: int = 0,
        *,
        is_test: bool = False,
        test_result_save_path: str | None = None,
        wandb_log_freq: LogFreq = LogEachEpoch,
    ) -> dict:
        """Validation or test step.

        Args:
            val_loader (DataLoader): val loader to test accuracy after each epoch
            seen_samples (int): Number of samples seen so far (for wandb x-axis)
            is_test (bool): whether it's test step
            test_result_save_path (str): path to save test_result
            wandb_log_freq ("epoch" | "batch"): Frequency of logging to wandb.
                'epoch' logs once per epoch, 'batch' logs after every batch.

        Returns:
            dictionary of training errors
        """

        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = {key: AverageMeter() for key in self.targets}

        # Add uncertainty weight meters if using uncertainty weighting
        uncertainty_weights = {}
        if isinstance(self.criterion, UncertaintyLoss):
            # Get current weights once for consistent reporting
            current_weights = self.criterion.get_current_weights()
            uncertainty_weights = {
                f"weight_{k}": AverageMeter()
                for k in self.targets
            }
            # Initialize meters with current weights
            for k, v in current_weights.items():
                if k in self.targets:
                    uncertainty_weights[f"weight_{k}"].update(v, 1)

        # Add DWA weight meters if using DWA
        dwa_weights = {}
        if isinstance(self.criterion, DWALoss):
            current_weights = self.criterion.get_current_weights()
            dwa_weights = {
                f"dwa_{k}": AverageMeter()
                for k in current_weights
            }
            # Initialize meters with current weights
            for k, v in current_weights.items():
                dwa_weights[f"dwa_{k}"].update(v, 1)

        # switch to evaluate mode
        self.model.eval()

        test_pred = []

        end = time.perf_counter()

        for ii, (graphs, targets) in enumerate(val_loader):
            # We need gradients for computing forces even in validation
            if "f" in self.targets or "s" in self.targets:
                for graph in graphs:
                    requires_force = "f" in self.targets
                    graph.atom_frac_coord.requires_grad = requires_force
                graphs = [g.to(self.device) for g in graphs]
                targets = {k: self.move_to(v, self.device) for k, v in targets.items()}
            else:
                with torch.no_grad():
                    graphs = [g.to(self.device) for g in graphs]
                    targets = {
                        k: self.move_to(v, self.device) for k, v in targets.items()
                    }

            # Compute output using eval_forward to prevent updating uncertainty weights
            prediction = self.model(graphs, task=self.targets)

            # Use eval_forward for loss computation if using uncertainty weighting
            if isinstance(self.criterion, UncertaintyLoss):
                combined_loss = self.criterion.eval_forward(targets, prediction)
            elif isinstance(self.criterion, DWALoss):
                combined_loss = self.criterion.eval_forward(targets, prediction)
            else:
                combined_loss = self.criterion(targets, prediction)

            losses.update(combined_loss["loss"].data.cpu().item(), len(graphs))
            for key in self.targets:
                mae_errors[key].update(
                    combined_loss[f"{key}_MAE"].cpu().item(),
                    combined_loss[f"{key}_MAE_size"],
                )

            # Track uncertainty weights if using uncertainty weighting
            if uncertainty_weights and isinstance(self.criterion, UncertaintyLoss):
                for key in self.targets:
                    weight_key = f"uncertainty_weight_{key}"
                    if weight_key in combined_loss:
                        uncertainty_weights[f"weight_{key}"].update(combined_loss[weight_key], 1)

            # Track DWA weights if using DWA
            if dwa_weights and isinstance(self.criterion, DWALoss):
                for key in self.targets:
                    dwa_key = f"dwa_weight_{key}"
                    if dwa_key in combined_loss:
                        dwa_weights[f"dwa_{key}"].update(combined_loss[dwa_key], 1)

            if is_test and test_result_save_path:
                # Create prediction dicts for this batch's data on this rank
                batch_preds = []
                for jj, graph_i in enumerate(graphs):
                    tmp = {
                        "mp_id": graph_i.mp_id,
                        "graph_id": graph_i.graph_id,
                        "energy": {
                            "ground_truth": targets["e"][jj].cpu().detach().tolist(),
                            "prediction": prediction["e"][jj].cpu().detach().tolist(),
                        },
                    }
                    if "f" in self.targets:
                        tmp["force"] = {
                            "ground_truth": targets["f"][jj].cpu().detach().tolist(),
                            "prediction": prediction["f"][jj].cpu().detach().tolist(),
                        }
                    if "s" in self.targets:
                        tmp["stress"] = {
                            "ground_truth": targets["s"][jj].cpu().detach().tolist(),
                            "prediction": prediction["s"][jj].cpu().detach().tolist(),
                        }
                    if "m" in self.targets:
                        if targets["m"][jj] is not None:
                            m_ground_truth = targets["m"][jj].cpu().detach().tolist()
                        else:
                            m_ground_truth = None
                        tmp["mag"] = {
                            "ground_truth": m_ground_truth,
                            "prediction": prediction["m"][jj].cpu().detach().tolist(),
                        }
                    batch_preds.append(tmp)

                # Gather predictions from all ranks onto rank 0
                gathered_preds = [None] * self.world_size
                dist.gather_object(batch_preds, gathered_preds if self.rank == 0 else None, dst=0)

                if self.rank == 0:
                    # Flatten the gathered list (list of lists)
                    for rank_preds in gathered_preds:
                        test_pred.extend(rank_preds)

            # free memory
            del graphs, targets
            del prediction, combined_loss
            torch.cuda.empty_cache()

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # --- Logging (Rank 0 Only) ---
            if self.rank == 0 and (ii + 1) % self.print_freq == 0:
                name = "Test" if is_test else "Val"
                message = (
                    f"{name}: [{ii + 1}/{len(val_loader)}] | "
                    f"Time ({batch_time.avg:.3f}) | "
                    f"Loss {losses.val:.4f}({losses.avg:.4f}) | MAE({self.mae_unit_str}) "
                )
                for key in self.targets:
                    val_scaled = mae_errors[key].val * self.mae_scale
                    avg_scaled = mae_errors[key].avg * self.mae_scale
                    message += f"{key} {val_scaled:.4f}({avg_scaled:.4f})  "

                # Add uncertainty weights to print message if using uncertainty weighting
                if uncertainty_weights:
                    message += "| Weights "
                    for key in self.targets:
                        weight_key = f"weight_{key}"
                        if weight_key in uncertainty_weights and uncertainty_weights[weight_key].count > 0:
                            message += f"{key} {uncertainty_weights[weight_key].val:.3f}  "

                # Add DWA weights to print message
                if dwa_weights:
                    message += "| DWA "
                    for key in self.targets:
                        dwa_key = f"dwa_{key}"
                        if dwa_key in dwa_weights and dwa_weights[dwa_key].count > 0:
                            message += f"{key} {dwa_weights[dwa_key].val:.3f}  "

                print(message)

        if self.rank == 0:
            if is_test:
                message = "**  "
                if test_result_save_path:
                    write_json(
                        test_pred, os.path.join(test_result_save_path, "test_result.json")
                    )
            else:
                message = "*   "
            for key in self.targets:
                avg_scaled = mae_errors[key].avg * self.mae_scale
                message += f"{key}_MAE({self.mae_unit_str}) ({avg_scaled:.4f}) \t"

            # Add final uncertainty weights to message if available
            if uncertainty_weights:
                message += "| Weights "
                for key in self.targets:
                    weight_key = f"weight_{key}"
                    if weight_key in uncertainty_weights and uncertainty_weights[weight_key].count > 0:
                        message += f"{key} {uncertainty_weights[weight_key].avg:.3f}  "

            # Add DWA weights to message if available
            if dwa_weights:
                message += "| DWA "
                for key in self.targets:
                    dwa_key = f"dwa_{key}"
                    if dwa_key in dwa_weights and dwa_weights[dwa_key].count > 0:
                        message += f"{key} {dwa_weights[dwa_key].avg:.3f}  "

            print(message)

        return {
            "loss_meter": losses,
            "mae_meters": mae_errors,
            "uncertainty_weights": uncertainty_weights if uncertainty_weights else None,
            "dwa_weights": dwa_weights if dwa_weights else None
        }

    @property
    def _init_keys(self) -> list[str]:
        # Updated to reflect new DDP params and removed device
        sig_params = list(inspect.signature(DDPTrainer.__init__).parameters)
        return [
            key for key in sig_params
            if key not in {"self", "model", "kwargs", "rank", "local_rank", "world_size"} # Exclude DDP params too
        ]

    def save(self, filename: str = "training_result.pth.tar", cumulative_samples_seen: int = 0) -> None:
        """Save the model, optimizer, scheduler, etc. (Rank 0 Only).

        Args:
            filename (str): Path to save the checkpoint file.
            cumulative_samples_seen (int): Total number of samples seen so far.
        """
        assert self.rank == 0, "Checkpoint saving is only supported on rank 0."

        state = {
            # Save the underlying model's state dict
            "model": self.model.module.as_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "training_history": self.training_history, # Rank 0 has the history
            "trainer_args": self.trainer_args, # Rank 0 has the args
            "world_size": self.world_size,
            # Save training progress for accurate resume
            "cumulative_samples_seen": cumulative_samples_seen,
            "last_wandb_step": self.last_wandb_step,
        }

        # Save wandb run ID if available
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    state["wandb_run_id"] = wandb.run.id
                    state["wandb_run_name"] = wandb.run.name
                    state["wandb_project"] = wandb.run.project
                    print(f"Rank 0: Saving wandb run ID: {wandb.run.id}")
            except Exception as e:
                print(f"Rank 0: Warning - Could not save wandb run ID: {e}")

        print(f"Rank 0: Saving checkpoint to {filename}")
        print(f"Rank 0: Cumulative samples seen: {cumulative_samples_seen}")
        print(f"Rank 0: Last wandb step: {self.last_wandb_step}")
        torch.save(state, filename)

    def save_checkpoint(
        self,
        epoch: int,
        mae_error: dict,
        save_dir: str,
        cumulative_samples_seen: int = 0,
        remove_old_ckpt: bool = False,
    ) -> None:
        """Save checkpoint (Rank 0 Only).

        Args:
            epoch (int): Current epoch number.
            mae_error (dict): Globally averaged MAE errors for checkpoint naming.
            save_dir (str): Directory to save checkpoints.
            cumulative_samples_seen (int): Total number of samples seen so far.
            remove_old_ckpt (bool): Whether to remove old checkpoints.
        """
        if self.rank != 0:
            return # Only rank 0 saves checkpoints

        os.makedirs(save_dir, exist_ok=True)

        if remove_old_ckpt:
            for fname in os.listdir(save_dir):
                if fname.startswith("epoch"):
                    os.remove(os.path.join(save_dir, fname))

        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        lr_str = f"lr{int(current_lr * 1e6)}"  # Convert to integer and multiply by 1e6

        # Create filename with epoch, learning rate and metrics
        err_str = "_".join(
            f"{key}{f'{mae_error[key] * 1000:.0f}' if key in mae_error else 'NA'}"
            for key in "efsm"
        )
        filename = os.path.join(save_dir, f"epoch{epoch}_{lr_str}_{err_str}.pth.tar")

        # Save the current state
        self.save(filename=filename, cumulative_samples_seen=cumulative_samples_seen)
        print(f"Rank 0: Saved checkpoint: {os.path.basename(filename)}")

        # TODO: get the best model, best e and best f

    @classmethod
    def load(
        cls,
        *,
        path: str,
        model_class,
        rank: int,
        local_rank: int,
        world_size: int,
        resume_wandb: bool = False,
        wandb_id: str | None = None,
    ) -> Self:
        """Load model and trainer args from a saved checkpoint (for inference or transfer learning).
           Does NOT restore DDP state or full trainer object.

        Args:
            path (str): Path to the saved .pth.tar file.
            model_class: The model class to instantiate.
            rank (int): Current process rank.
            local_rank (int): Local rank within the node.
            world_size (int): Total number of processes.
            resume_wandb (bool): Whether to resume wandb logging from the checkpoint.
                If True, wandb will be initialized with resume="allow" or "must" if wandb_id is provided.
                Default = False
            wandb_id (str | None): Specific wandb run ID to resume. If provided, wandb will be initialized
                with resume="must" and the specified run ID. Default = None

        Returns:
            DDPTrainer: A new DDPTrainer instance with the loaded model and args.
        """

        state = torch.load(path, map_location=torch.device("cpu"))

        trainer_args = state["trainer_args"]
        # remove model from trainer_args if present
        trainer_args.pop("model", None)
        # remove DDP specific keys from trainer_args
        trainer_args.pop("rank", None)
        trainer_args.pop("local_rank", None)
        trainer_args.pop("world_size", None)

        print(f"[Rank {rank}] Loading checkpoint from: {path}")
        if rank == 0:
            print(f"[Rank {rank}] State dict keys: {state.keys()}")
            print(f"[Rank {rank}] Trainer args: {trainer_args}")
            # start epoch
            start_epoch = len(state["training_history"]["e"]["train"])
            print(f"[Rank {rank}] Starting epoch: {start_epoch}")

        # assert trainer_args["world_size"] == world_size, f"Checkpoint world size {trainer_args['world_size']} does not match current world size {world_size}."

        model = model_class.from_dict(state["model"])

        print(f"[Rank {rank}] Loaded model params = {sum(p.numel() for p in model.parameters()):,}")

        # Handle wandb resume logic
        if resume_wandb and trainer_args.get("wandb_path"):
            if rank == 0:
                print(f"[Rank {rank}] Configuring wandb for resume...")

            # Modify wandb_init_kwargs to include resume parameters
            # Handle case where wandb_init_kwargs might be None
            wandb_init_kwargs = trainer_args.get("wandb_init_kwargs") or {}

            if wandb_id:
                # Resume specific run
                wandb_init_kwargs["resume"] = "must"
                wandb_init_kwargs["id"] = wandb_id
                if rank == 0:
                    print(f"[Rank {rank}] Will resume wandb run with ID: {wandb_id}")
            elif "wandb_run_id" in state:
                # Use saved wandb run ID from checkpoint
                wandb_init_kwargs["resume"] = "must"
                wandb_init_kwargs["id"] = state["wandb_run_id"]
                if rank == 0:
                    print(f"[Rank {rank}] Will resume wandb run with saved ID: {state['wandb_run_id']}")
                    if "wandb_run_name" in state:
                        print(f"[Rank {rank}] Previous run name: {state['wandb_run_name']}")
                    if "wandb_project" in state:
                        print(f"[Rank {rank}] Previous project: {state['wandb_project']}")
            else:
                # Allow resume if run exists
                wandb_init_kwargs["resume"] = "allow"
                if rank == 0:
                    print(f"[Rank {rank}] Will resume wandb run if it exists (no saved run ID)")

            trainer_args["wandb_init_kwargs"] = wandb_init_kwargs
        elif resume_wandb and not trainer_args.get("wandb_path"):
            if rank == 0:
                print(f"[Rank {rank}] Warning: resume_wandb=True but no wandb_path found in checkpoint")

        trainer = cls(
            model=model,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            **trainer_args,
        )

        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.scheduler.load_state_dict(state["scheduler"])
        trainer.training_history = state["training_history"]
        trainer.starting_epoch = len(state["training_history"]["e"]["train"])

        # Save checkpoint state for resume metrics calculation
        trainer._checkpoint_state = state

        return trainer

    @staticmethod
    def move_to(
        obj: Tensor | list[Tensor], device: torch.device
    ) -> Tensor | list[Tensor]:
        """Move object to device.

        Args:
            obj (Tensor | list[Tensor]): object(s) to move to device
            device (torch.device): device to move object to

        Raises:
            TypeError: if obj is not a tensor or list of tensors

        Returns:
            Tensor | list[Tensor]: moved object(s)
        """
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, list):
            out = []
            for tensor in obj:
                if tensor is not None:
                    out.append(tensor.to(device))
                else:
                    out.append(None)
            return out
        raise TypeError("Invalid type for move_to")
