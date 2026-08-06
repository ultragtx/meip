from torch import nn
import torch
from torch import Tensor
from meip.utils import mae

class CombinedLoss(nn.Module):
    """A combined loss function of energy, force, stress and magmom with static weighting."""

    def __init__(
        self,
        *,
        target_str: str = "ef",
        criterion: str = "MSE",
        energy_loss_ratio: float = 1,
        force_loss_ratio: float = 1,
        stress_loss_ratio: float = 0.1,
        mag_loss_ratio: float = 0.1,
        delta: float = 0.1,
        allow_missing_labels: bool = True,
    ) -> None:
        """Initialize the combined loss.

        Args:
            target_str: the training target label. Can be "e", "ef", "efs", "efsm" etc.
                Default = "ef"
            criterion: loss criterion to use
                Default = "MSE"
            energy_loss_ratio (float): energy loss ratio in loss function
                Default = 1
            force_loss_ratio (float): force loss ratio in loss function
                Default = 1
            stress_loss_ratio (float): stress loss ratio in loss function
                Default = 0.1
            mag_loss_ratio (float): magmom loss ratio in loss function
                Default = 0.1
            delta (float): delta for torch.nn.HuberLoss. Default = 0.1
            allow_missing_labels (bool): whether to allow missing labels in the dataset,
                missed target will not contribute to loss and MAEs
        """
        super().__init__()
        # Define loss criterion
        if criterion in {"MSE", "mse"}:
            self.criterion = nn.MSELoss()
        elif criterion in {"MAE", "mae", "l1"}:
            self.criterion = nn.L1Loss()
        elif criterion == "Huber":
            self.criterion = nn.HuberLoss(delta=delta)
        else:
            raise NotImplementedError
        self.target_str = target_str
        self.energy_loss_ratio = energy_loss_ratio
        if "f" not in self.target_str:
            self.force_loss_ratio = 0
        else:
            self.force_loss_ratio = force_loss_ratio
        if "s" not in self.target_str:
            self.stress_loss_ratio = 0
        else:
            self.stress_loss_ratio = stress_loss_ratio
        if "m" not in self.target_str:
            self.mag_loss_ratio = 0
        else:
            self.mag_loss_ratio = mag_loss_ratio
        self.allow_missing_labels = allow_missing_labels

    def get_current_weights(self) -> dict:
        """Returns the current loss weights.

        Returns:
            dict: Dictionary containing the current weights
        """
        return {
            "e": self.energy_loss_ratio if "e" in self.target_str else 0,
            "f": self.force_loss_ratio if "f" in self.target_str else 0,
            "s": self.stress_loss_ratio if "s" in self.target_str else 0,
            "m": self.mag_loss_ratio if "m" in self.target_str else 0,
        }

    def eval_forward(
        self,
        targets: dict[str, Tensor],
        prediction: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute loss without special evaluation handling (for base class).

        Args:
            targets (dict): DFT labels
            prediction (dict): MEIP prediction

        Returns:
            dictionary of all the loss, MAE and MAE_size
        """
        return self.forward(targets, prediction)

    def forward(
        self,
        targets: dict[str, Tensor],
        prediction: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute the combined loss using MEIP prediction and labels
        this function can automatically mask out magmom loss contribution of
        data points without magmom labels.

        Args:
            targets (dict): DFT labels
            prediction (dict): MEIP prediction

        Returns:
            dictionary of all the loss, MAE and MAE_size
        """
        out = {"loss": 0.0}
        task_losses = {}

        # Energy
        if "e" in self.target_str:
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(targets["e"])
                valid_e_target = targets["e"][valid_value_indices]
                valid_e_pred = prediction["e"][valid_value_indices]
                if valid_e_pred.shape == torch.Size([]):
                    valid_e_pred = valid_e_pred.view(1)
            else:
                valid_e_target = targets["e"]
                valid_e_pred = prediction["e"]

            e_loss = self.criterion(valid_e_target, valid_e_pred)
            task_losses["e"] = e_loss
            out["e_MAE"] = mae(valid_e_target, valid_e_pred)
            out["e_MAE_size"] = prediction["e"].shape[0]

        # Force
        if "f" in self.target_str:
            forces_pred = torch.cat(prediction["f"], dim=0)
            forces_target = torch.cat(targets["f"], dim=0)
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(forces_target)
                forces_target = forces_target[valid_value_indices]
                forces_pred = forces_pred[valid_value_indices]
            f_loss = self.criterion(forces_target, forces_pred)
            task_losses["f"] = f_loss
            out["f_MAE"] = mae(forces_target, forces_pred)
            out["f_MAE_size"] = forces_target.shape[0]

        # Stress
        if "s" in self.target_str:
            stress_pred = torch.cat(prediction["s"], dim=0)
            stress_target = torch.cat(targets["s"], dim=0)
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(stress_target)
                stress_target = stress_target[valid_value_indices]
                stress_pred = stress_pred[valid_value_indices]
            s_loss = self.criterion(stress_target, stress_pred)
            task_losses["s"] = s_loss
            out["s_MAE"] = mae(stress_target, stress_pred)
            out["s_MAE_size"] = stress_target.shape[0]

        # Mag
        if "m" in self.target_str:
            mag_preds, mag_targets = [], []
            m_mae_size = 0
            for mag_pred, mag_target in zip(prediction["m"], targets["m"], strict=True):
                # exclude structures without magmom labels
                if self.allow_missing_labels:
                    if mag_target is not None and not torch.isnan(mag_target).any():
                        mag_preds.append(mag_pred)
                        mag_targets.append(mag_target)
                        m_mae_size += mag_target.shape[0]
                else:
                    mag_preds.append(mag_pred)
                    mag_targets.append(mag_target)
                    m_mae_size += mag_target.shape[0]
            if mag_targets:
                mag_preds = torch.cat(mag_preds, dim=0)
                mag_targets = torch.cat(mag_targets, dim=0)
                m_loss = self.criterion(mag_targets, mag_preds)
                task_losses["m"] = m_loss
                out["m_MAE"] = mae(mag_targets, mag_preds)
                out["m_MAE_size"] = m_mae_size
            else:
                out["m_MAE"] = torch.zeros([1])
                out["m_MAE_size"] = m_mae_size

        # Use static weights
        if "e" in task_losses:
            out["loss"] += self.energy_loss_ratio * task_losses["e"]
        if "f" in task_losses:
            out["loss"] += self.force_loss_ratio * task_losses["f"]
        if "s" in task_losses:
            out["loss"] += self.stress_loss_ratio * task_losses["s"]
        if "m" in task_losses:
            out["loss"] += self.mag_loss_ratio * task_losses["m"]

        return out


class UncertaintyLoss(CombinedLoss):
    """A loss function using learned uncertainty weighting to balance multiple tasks."""

    def __init__(
        self,
        *,
        target_str: str = "ef",
        criterion: str = "MSE",
        energy_loss_ratio: float = 1,
        force_loss_ratio: float = 1,
        stress_loss_ratio: float = 0.1,
        mag_loss_ratio: float = 0.1,
        delta: float = 0.1,
        allow_missing_labels: bool = True,
        min_log_var: float = -7.0,  # Min log variance for uncertainty weighting
        max_log_var: float = 7.0,   # Max log variance for uncertainty weighting
    ) -> None:
        """Initialize the uncertainty weighted loss.

        Args:
            target_str: the training target label. Can be "e", "ef", "efs", "efsm" etc.
                Default = "ef"
            criterion: loss criterion to use
                Default = "MSE"
            energy_loss_ratio (float): initial energy loss ratio (used to initialize log variances)
                Default = 1
            force_loss_ratio (float): initial force loss ratio (used to initialize log variances)
                Default = 1
            stress_loss_ratio (float): initial stress loss ratio (used to initialize log variances)
                Default = 0.1
            mag_loss_ratio (float): initial magmom loss ratio (used to initialize log variances)
                Default = 0.1
            delta (float): delta for torch.nn.HuberLoss. Default = 0.1
            allow_missing_labels (bool): whether to allow missing labels in the dataset,
                missed target will not contribute to loss and MAEs
            min_log_var (float): minimum value for log variance to avoid numerical issues
                Default = -7.0
            max_log_var (float): maximum value for log variance to avoid numerical issues
                Default = 7.0
        """
        super().__init__(
            target_str=target_str,
            criterion=criterion,
            energy_loss_ratio=energy_loss_ratio,
            force_loss_ratio=force_loss_ratio,
            stress_loss_ratio=stress_loss_ratio,
            mag_loss_ratio=mag_loss_ratio,
            delta=delta,
            allow_missing_labels=allow_missing_labels,
        )

        # Initialize uncertainty weighting parameters
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        # Initialize log variances as learnable parameters
        # Starting with values corresponding to the static ratios provided
        self.log_var_e = nn.Parameter(torch.tensor(-2.0 if "e" in target_str else 0.0))
        self.log_var_f = nn.Parameter(torch.tensor(-2.0 if "f" in target_str else 0.0))
        self.log_var_s = nn.Parameter(torch.tensor(-2.0 if "s" in target_str else 0.0))
        self.log_var_m = nn.Parameter(torch.tensor(-2.0 if "m" in target_str else 0.0))

    def get_current_weights(self) -> dict:
        """Returns the current uncertainty weights without affecting gradients.

        Returns:
            dict: Dictionary containing the current uncertainty weights
        """
        with torch.no_grad():
            weights = {}
            if hasattr(self, 'log_var_e'):
                log_var_e = torch.clamp(self.log_var_e, self.min_log_var, self.max_log_var)
                weights["e"] = torch.exp(-log_var_e).item()
            if hasattr(self, 'log_var_f'):
                log_var_f = torch.clamp(self.log_var_f, self.min_log_var, self.max_log_var)
                weights["f"] = torch.exp(-log_var_f).item()
            if hasattr(self, 'log_var_s'):
                log_var_s = torch.clamp(self.log_var_s, self.min_log_var, self.max_log_var)
                weights["s"] = torch.exp(-log_var_s).item()
            if hasattr(self, 'log_var_m'):
                log_var_m = torch.clamp(self.log_var_m, self.min_log_var, self.max_log_var)
                weights["m"] = torch.exp(-log_var_m).item()
            return weights

    def eval_forward(
        self,
        targets: dict[str, Tensor],
        prediction: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute loss without updating uncertainty weights (for evaluation only).

        This method temporarily disables gradient tracking for uncertainty weight
        parameters while computing the loss.

        Args:
            targets (dict): DFT labels
            prediction (dict): MEIP prediction

        Returns:
            dictionary of all the loss, MAE and MAE_size
        """
        # Store original requires_grad states
        original_states = {}
        if hasattr(self, 'log_var_e'):
            original_states['e'] = self.log_var_e.requires_grad
            self.log_var_e.requires_grad_(False)
        if hasattr(self, 'log_var_f'):
            original_states['f'] = self.log_var_f.requires_grad
            self.log_var_f.requires_grad_(False)
        if hasattr(self, 'log_var_s'):
            original_states['s'] = self.log_var_s.requires_grad
            self.log_var_s.requires_grad_(False)
        if hasattr(self, 'log_var_m'):
            original_states['m'] = self.log_var_m.requires_grad
            self.log_var_m.requires_grad_(False)

        try:
            # Compute the loss
            result = self.forward(targets, prediction)
            return result
        finally:
            # Restore original requires_grad states
            if hasattr(self, 'log_var_e') and 'e' in original_states:
                self.log_var_e.requires_grad_(original_states['e'])
            if hasattr(self, 'log_var_f') and 'f' in original_states:
                self.log_var_f.requires_grad_(original_states['f'])
            if hasattr(self, 'log_var_s') and 's' in original_states:
                self.log_var_s.requires_grad_(original_states['s'])
            if hasattr(self, 'log_var_m') and 'm' in original_states:
                self.log_var_m.requires_grad_(original_states['m'])

    def forward(
        self,
        targets: dict[str, Tensor],
        prediction: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute the loss with uncertainty weighting.

        Args:
            targets (dict): DFT labels
            prediction (dict): MEIP prediction

        Returns:
            dictionary of all the loss, MAE and MAE_size
        """
        # First compute task losses and MAEs using parent class
        out = super().forward(targets, prediction)

        # Extract the individual task losses
        task_losses = {}
        if "e" in self.target_str:
            # Re-extract losses from the criterion directly
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(targets["e"])
                valid_e_target = targets["e"][valid_value_indices]
                valid_e_pred = prediction["e"][valid_value_indices]
                if valid_e_pred.shape == torch.Size([]):
                    valid_e_pred = valid_e_pred.view(1)
            else:
                valid_e_target = targets["e"]
                valid_e_pred = prediction["e"]
            task_losses["e"] = self.criterion(valid_e_target, valid_e_pred)

        if "f" in self.target_str:
            forces_pred = torch.cat(prediction["f"], dim=0)
            forces_target = torch.cat(targets["f"], dim=0)
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(forces_target)
                forces_target = forces_target[valid_value_indices]
                forces_pred = forces_pred[valid_value_indices]
            task_losses["f"] = self.criterion(forces_target, forces_pred)

        if "s" in self.target_str:
            stress_pred = torch.cat(prediction["s"], dim=0)
            stress_target = torch.cat(targets["s"], dim=0)
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(stress_target)
                stress_target = stress_target[valid_value_indices]
                stress_pred = stress_pred[valid_value_indices]
            task_losses["s"] = self.criterion(stress_target, stress_pred)

        if "m" in self.target_str:
            mag_preds, mag_targets = [], []
            for mag_pred, mag_target in zip(prediction["m"], targets["m"], strict=True):
                # exclude structures without magmom labels
                if self.allow_missing_labels:
                    if mag_target is not None and not torch.isnan(mag_target).any():
                        mag_preds.append(mag_pred)
                        mag_targets.append(mag_target)
                else:
                    mag_preds.append(mag_pred)
                    mag_targets.append(mag_target)

            if mag_targets:
                mag_preds = torch.cat(mag_preds, dim=0)
                mag_targets = torch.cat(mag_targets, dim=0)
                task_losses["m"] = self.criterion(mag_targets, mag_preds)

        # Reset loss value for uncertainty weighting
        out["loss"] = 0.0

        # Apply uncertainty weighting
        # Get clamped log vars for consistent loss calculation
        log_var_e_clamped = torch.clamp(self.log_var_e, self.min_log_var, self.max_log_var)
        log_var_f_clamped = torch.clamp(self.log_var_f, self.min_log_var, self.max_log_var)
        log_var_s_clamped = torch.clamp(self.log_var_s, self.min_log_var, self.max_log_var)
        log_var_m_clamped = torch.clamp(self.log_var_m, self.min_log_var, self.max_log_var)

        # Calculate precision (1/variance) terms
        precision_e = torch.exp(-log_var_e_clamped)
        precision_f = torch.exp(-log_var_f_clamped)
        precision_s = torch.exp(-log_var_s_clamped)
        precision_m = torch.exp(-log_var_m_clamped)

        # Calculate the total loss with uncertainty weighting
        if "e" in task_losses:
            out["loss"] += 0.5 * precision_e * task_losses["e"] + 0.5 * log_var_e_clamped
        if "f" in task_losses:
            out["loss"] += 0.5 * precision_f * task_losses["f"] + 0.5 * log_var_f_clamped
        if "s" in task_losses:
            out["loss"] += 0.5 * precision_s * task_losses["s"] + 0.5 * log_var_s_clamped
        if "m" in task_losses:
            out["loss"] += 0.5 * precision_m * task_losses["m"] + 0.5 * log_var_m_clamped

        # Store the current uncertainty weights for logging
        out["uncertainty_weight_e"] = precision_e.item() if "e" in task_losses else 0.0
        out["uncertainty_weight_f"] = precision_f.item() if "f" in task_losses else 0.0
        out["uncertainty_weight_s"] = precision_s.item() if "s" in task_losses else 0.0
        out["uncertainty_weight_m"] = precision_m.item() if "m" in task_losses else 0.0

        return out


class DWALoss(nn.Module):
    """Dynamic Weight Averaging (DWA) loss function for multi-task learning with sliding window support."""

    def __init__(
        self,
        *,
        target_str: str = "ef",
        criterion: str = "MSE",
        energy_loss_ratio: float = 1,
        force_loss_ratio: float = 1,
        stress_loss_ratio: float = 0.1,
        mag_loss_ratio: float = 0.1,
        delta: float = 0.1,
        allow_missing_labels: bool = True,
        temperature: float = 2.0,  # Temperature parameter for softmax
        window_size: int = 1,      # Size of sliding window (1 = original behavior)
    ) -> None:
        """Initialize the DWA loss.

        Args:
            target_str: the training target label. Can be "e", "ef", "efs", "efsm" etc.
                Default = "ef"
            criterion: loss criterion to use
                Default = "MSE"
            energy_loss_ratio: initial energy loss ratio (for first epoch)
                Default = 1
            force_loss_ratio: initial force loss ratio (for first epoch)
                Default = 1
            stress_loss_ratio: initial stress loss ratio (for first epoch)
                Default = 0.1
            mag_loss_ratio: initial magmom loss ratio (for first epoch)
                Default = 0.1
            delta: delta for torch.nn.HuberLoss. Default = 0.1
            allow_missing_labels: whether to allow missing labels in the dataset,
                missed target will not contribute to loss and MAEs
            temperature: temperature parameter for softmax in DWA
                Default = 2.0 (higher = more uniform weights)
            window_size: number of previous epochs to use for loss ratio calculation
                Default = 1 (original behavior, only use previous epoch)
        """
        super().__init__()
        # Define loss criterion
        if criterion in {"MSE", "mse"}:
            self.criterion = nn.MSELoss()
        elif criterion in {"MAE", "mae", "l1"}:
            self.criterion = nn.L1Loss()
        elif criterion == "Huber":
            self.criterion = nn.HuberLoss(delta=delta)
        else:
            raise NotImplementedError

        self.target_str = target_str
        self.allow_missing_labels = allow_missing_labels
        self.temperature = temperature

        # Initialize window size (must be at least 1)
        self.window_size = max(1, window_size)

        # Initialize tracking variables
        self.loss_history = {}  # Task -> list of past losses
        self.curr_losses = {}   # Current epoch losses
        self.epoch_counter = 0  # Track number of epochs

        # Initialize weights with static values for first epochs
        self.task_weights = {}
        if "e" in self.target_str:
            self.task_weights["e"] = energy_loss_ratio
        if "f" in self.target_str:
            self.task_weights["f"] = force_loss_ratio
        if "s" in self.target_str:
            self.task_weights["s"] = stress_loss_ratio
        if "m" in self.target_str:
            self.task_weights["m"] = mag_loss_ratio

    def update_weights(self, device=None):
        """Update task weights using DWA algorithm with sliding window at end of epoch.

        Args:
            device: Device to use for tensor operations
        """
        # Store current epoch's losses in history
        for task, loss in self.curr_losses.items():
            if task not in self.loss_history:
                self.loss_history[task] = []
            self.loss_history[task].append(loss)

            # Keep only the most recent window_size+1 losses
            # (+1 because we need the current and window_size previous)
            if len(self.loss_history[task]) > self.window_size + 1:
                self.loss_history[task].pop(0)

        self.epoch_counter += 1

        # Only update weights once we have enough history
        if self.epoch_counter <= self.window_size:
            # Reset for next epoch
            self.curr_losses = {}
            return

        # Calculate loss ratios using sliding window average
        ratios = {}
        for task in self.loss_history:
            if len(self.loss_history[task]) > self.window_size:
                # Use average of window_size previous epochs as denominator
                # and current epoch as numerator
                prev_avg = sum(self.loss_history[task][-(self.window_size+1):-1]) / self.window_size
                current = self.loss_history[task][-1]
                if prev_avg > 0:
                    ratios[task] = prev_avg / (current + 1e-6)
                else:
                    ratios[task] = 1.0

        # Apply softmax with temperature to get weights
        if ratios:
            tasks = list(ratios.keys())
            ratio_values = [ratios[task] for task in tasks]

            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            w_values = torch.tensor(ratio_values, device=device)
            w_softmax = torch.softmax(w_values / self.temperature, dim=0)

            # Update weights (normalize by number of tasks)
            num_tasks = len(tasks)
            for i, task in enumerate(tasks):
                self.task_weights[task] = w_softmax[i].item() * num_tasks

        # Reset for next epoch
        self.curr_losses = {}

    def get_current_weights(self) -> dict:
        """Return current task weights."""
        return self.task_weights

    def forward(
        self,
        targets: dict[str, Tensor],
        prediction: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute combined loss with DWA weighting."""
        out = {"loss": 0.0}
        task_losses = {}

        # Energy
        if "e" in self.target_str:
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(targets["e"])
                valid_e_target = targets["e"][valid_value_indices]
                valid_e_pred = prediction["e"][valid_value_indices]
                if valid_e_pred.shape == torch.Size([]):
                    valid_e_pred = valid_e_pred.view(1)
            else:
                valid_e_target = targets["e"]
                valid_e_pred = prediction["e"]

            e_loss = self.criterion(valid_e_target, valid_e_pred)
            task_losses["e"] = e_loss
            out["e_MAE"] = mae(valid_e_target, valid_e_pred)
            out["e_MAE_size"] = prediction["e"].shape[0]

            # Track raw loss for DWA (detach to avoid gradient tracking)
            self.curr_losses["e"] = self.curr_losses.get("e", 0) + e_loss.detach().item()

        # Force
        if "f" in self.target_str:
            forces_pred = torch.cat(prediction["f"], dim=0)
            forces_target = torch.cat(targets["f"], dim=0)
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(forces_target)
                forces_target = forces_target[valid_value_indices]
                forces_pred = forces_pred[valid_value_indices]
            f_loss = self.criterion(forces_target, forces_pred)
            task_losses["f"] = f_loss
            out["f_MAE"] = mae(forces_target, forces_pred)
            out["f_MAE_size"] = forces_target.shape[0]

            # Track loss for DWA
            self.curr_losses["f"] = self.curr_losses.get("f", 0) + f_loss.detach().item()

        # Stress
        if "s" in self.target_str:
            stress_pred = torch.cat(prediction["s"], dim=0)
            stress_target = torch.cat(targets["s"], dim=0)
            if self.allow_missing_labels:
                valid_value_indices = ~torch.isnan(stress_target)
                stress_target = stress_target[valid_value_indices]
                stress_pred = stress_pred[valid_value_indices]
            s_loss = self.criterion(stress_target, stress_pred)
            task_losses["s"] = s_loss
            out["s_MAE"] = mae(stress_target, stress_pred)
            out["s_MAE_size"] = stress_target.shape[0]

            # Track loss for DWA
            self.curr_losses["s"] = self.curr_losses.get("s", 0) + s_loss.detach().item()

        # Mag
        if "m" in self.target_str:
            mag_preds, mag_targets = [], []
            m_mae_size = 0
            for mag_pred, mag_target in zip(prediction["m"], targets["m"], strict=True):
                # exclude structures without magmom labels
                if self.allow_missing_labels:
                    if mag_target is not None and not torch.isnan(mag_target).any():
                        mag_preds.append(mag_pred)
                        mag_targets.append(mag_target)
                        m_mae_size += mag_target.shape[0]
                else:
                    mag_preds.append(mag_pred)
                    mag_targets.append(mag_target)
                    m_mae_size += mag_target.shape[0]
            if mag_targets:
                mag_preds = torch.cat(mag_preds, dim=0)
                mag_targets = torch.cat(mag_targets, dim=0)
                m_loss = self.criterion(mag_targets, mag_preds)
                task_losses["m"] = m_loss
                out["m_MAE"] = mae(mag_targets, mag_preds)
                out["m_MAE_size"] = m_mae_size

                # Track loss for DWA
                self.curr_losses["m"] = self.curr_losses.get("m", 0) + m_loss.detach().item()
            else:
                out["m_MAE"] = torch.zeros([1])
                out["m_MAE_size"] = m_mae_size

        # Apply DWA weights to calculate total loss
        for task, loss in task_losses.items():
            weight = self.task_weights.get(task, 1.0)
            out["loss"] += weight * loss
            # Store weights for logging
            out[f"dwa_weight_{task}"] = weight

        return out

    def eval_forward(
        self,
        targets: dict[str, Tensor],
        prediction: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """For evaluation: use weights but don't update tracking."""
        # Save current state
        saved_curr_losses = self.curr_losses.copy()
        self.curr_losses = {}

        try:
            # Forward pass without updating tracking
            result = self.forward(targets, prediction)
            return result
        finally:
            # Restore state
            self.curr_losses = saved_curr_losses

