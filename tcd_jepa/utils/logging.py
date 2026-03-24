"""Experiment logging with optional wandb support and local CSV fallback."""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("tcd_jepa")


class MetricLogger:
    """Tracks and logs training metrics to CSV and optionally to wandb."""

    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[dict[str, Any]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self.json_path = self.log_dir / "metrics.jsonl"
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self._fields_written = False
        self._wandb_run = None

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
            f.write(json.dumps(metrics) + "\n")

        # CSV — rebuild writer when new fields appear
        fields = list(metrics.keys())
        if not self._fields_written:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_fieldnames = fields
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fields, extrasaction="ignore")
            self._csv_writer.writeheader()
            self._fields_written = True
        elif not set(fields).issubset(self._csv_fieldnames):
            # New columns appeared — rewrite CSV with expanded header
            self._csv_file.close()
            self._csv_fieldnames = list(dict.fromkeys(self._csv_fieldnames + fields))
            self._csv_file = open(self.csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_fieldnames, extrasaction="ignore")
        if self._csv_writer is not None:
            self._csv_writer.writerow(metrics)
            self._csv_file.flush()

        # wandb
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def close(self) -> None:
        """Clean up resources."""
        if self._csv_file is not None:
            self._csv_file.close()
        if self._wandb_run is not None:
            import wandb

            wandb.finish()
