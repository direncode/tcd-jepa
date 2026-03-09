"""Training metrics for TCD-JEPA."""

from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Accumulates training metrics over an epoch."""

    loss_sum: float = 0.0
    num_steps: int = 0
    lr: float = 0.0
    wd: float = 0.0
    momentum: float = 0.0

    def update(
        self,
        loss: float,
        lr: float = 0.0,
        wd: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        self.loss_sum += loss
        self.num_steps += 1
        self.lr = lr
        self.wd = wd
        self.momentum = momentum

    @property
    def avg_loss(self) -> float:
        if self.num_steps == 0:
            return 0.0
        return self.loss_sum / self.num_steps

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.avg_loss,
            "lr": self.lr,
            "wd": self.wd,
            "momentum": self.momentum,
        }

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.num_steps = 0
