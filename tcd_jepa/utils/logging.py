"""Experiment logging with optional wandb support and local CSV fallback."""

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("tcd_jepa")


class StepTimer:
    """Context manager for timing code sections."""

    def __init__(self) -> None:
        self._timers: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        elapsed = (time.perf_counter() - self._starts.pop(name, time.perf_counter())) * 1000
        self._timers[name] = elapsed
        return elapsed

    def get_timings(self) -> dict[str, float]:
        """Return all recorded timings in milliseconds."""
        return {f"time_{k}_ms": v for k, v in self._timers.items()}

    def reset(self) -> None:
        self._timers.clear()
        self._starts.clear()


class MetricLogger:
    """Tracks and logs training metrics to CSV and optionally to wandb."""

    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[dict[str, Any]] = None,
        step_log_freq: int = 50,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self.json_path = self.log_dir / "metrics.jsonl"
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self._fields_written = False
        self._wandb_run = None
        self._step_log_freq = step_log_freq

        if use_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=wandb_project or "tcd-jepa",
                    config=wandb_config or {},
                )
            except ImportError:
                logger.warning("wandb not installed, falling back to local logging only")

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics."""
        if step is not None:
            metrics["step"] = step

        # JSON lines (append)
        with open(self.json_path, "a") as f:
            f.write(json.dumps(metrics, default=str) + "\n")

        # CSV
        if not self._fields_written:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(metrics.keys()))
            self._csv_writer.writeheader()
            self._fields_written = True
        if self._csv_writer is not None:
            self._csv_writer.writerow(metrics)
            self._csv_file.flush()

        # wandb
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def log_step(self, metrics: dict[str, Any], step: int) -> None:
        """Log per-step metrics (throttled to every step_log_freq steps)."""
        if step % self._step_log_freq != 0:
            return
        self.log(metrics, step=step)

    def close(self) -> None:
        """Clean up resources."""
        if self._csv_file is not None:
            self._csv_file.close()
        if self._wandb_run is not None:
            import wandb

            wandb.finish()
