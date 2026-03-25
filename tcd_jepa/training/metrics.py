"""Training metrics for TCD-JEPA."""

from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Accumulates training metrics over an epoch."""

    loss_sum: float = 0.0
    num_steps: int = 0
    lr: float = 0.0
    wd: float = 0.0
    momentum: float = 0.0

    # Gradient health stats
    grad_norm_sum: float = 0.0
    grad_norm_max: float = 0.0
    param_norm: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    loss_spikes: int = 0
    skipped_steps: int = 0

    def update(
        self,
        loss: float,
        lr: float = 0.0,
        wd: float = 0.0,
        momentum: float = 0.0,
        grad_norm: float = 0.0,
        param_norm: float = 0.0,
        nan_count: int = 0,
        inf_count: int = 0,
        is_spike: bool = False,
        skipped: bool = False,
    ) -> None:
        self.loss_sum += loss
        self.num_steps += 1
        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.grad_norm_sum += grad_norm
        self.grad_norm_max = max(self.grad_norm_max, grad_norm)
        self.param_norm = param_norm
        self.nan_count += nan_count
        self.inf_count += inf_count
        if is_spike:
            self.loss_spikes += 1
        if skipped:
            self.skipped_steps += 1

    @property
    def avg_loss(self) -> float:
        if self.num_steps == 0:
            return 0.0
        return self.loss_sum / self.num_steps

    @property
    def avg_grad_norm(self) -> float:
        if self.num_steps == 0:
            return 0.0
        return self.grad_norm_sum / self.num_steps

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.avg_loss,
            "lr": self.lr,
            "wd": self.wd,
            "momentum": self.momentum,
            "grad_norm_avg": self.avg_grad_norm,
            "grad_norm_max": self.grad_norm_max,
            "param_norm": self.param_norm,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "loss_spikes": self.loss_spikes,
            "skipped_steps": self.skipped_steps,
        }

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.num_steps = 0
        self.grad_norm_sum = 0.0
        self.grad_norm_max = 0.0
        self.nan_count = 0
        self.inf_count = 0
        self.loss_spikes = 0
        self.skipped_steps = 0
