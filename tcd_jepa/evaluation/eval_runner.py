"""Evaluation runner that orchestrates linear probe and k-NN evaluation."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tcd_jepa.evaluation.knn_evaluator import KNNEvaluator
from tcd_jepa.evaluation.linear_probe import LinearProbeEvaluator

logger = logging.getLogger("tcd_jepa")


class EvaluationRunner:
    """Orchestrates model evaluation during and after training.

    Runs linear probe and k-NN evaluation on frozen encoder features.
    """

    def __init__(
        self,
        encoder: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        num_classes: int = 10,
        eval_cfg: Optional[dict] = None,
    ) -> None:
        self.encoder = encoder
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

        eval_cfg = eval_cfg or {}
        self.linear_probe = LinearProbeEvaluator(
            num_classes=num_classes,
            epochs=eval_cfg.get("linear_probe_epochs", 100),
        )
        self.knn = KNNEvaluator(
            k_values=eval_cfg.get("knn_k", [1, 5, 20]),
        )

    @torch.no_grad()
    def _extract_all_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from train and test loaders."""
        train_features, train_labels = self.linear_probe.extract_features(
            self.encoder, self.train_loader, self.device,
        )
        test_features, test_labels = self.linear_probe.extract_features(
            self.encoder, self.test_loader, self.device,
        )
        return train_features, train_labels, test_features, test_labels

    def run_evaluation(self, epoch: int) -> dict[str, float]:
        """Run full evaluation suite.

        Args:
            epoch: Current training epoch (for logging).

        Returns:
            Dict with all evaluation metrics.
        """
        logger.info(f"Running evaluation at epoch {epoch}...")

        # Extract features once
        train_feats, train_labels, test_feats, test_labels = self._extract_all_features()

        results = {}

        # Linear probe
        try:
            lp_results = self.linear_probe.evaluate(
                train_feats, train_labels, test_feats, test_labels,
                device=self.device,
            )
            results.update(lp_results)
            logger.info(
                f"  Linear probe: train_acc={lp_results['linear_probe_train_acc']:.4f} "
                f"test_acc={lp_results['linear_probe_test_acc']:.4f}"
            )
        except Exception as e:
            logger.warning(f"  Linear probe failed: {e}")

        # k-NN
        try:
            knn_results = self.knn.evaluate(
                train_feats, train_labels, test_feats, test_labels,
                num_classes=self.num_classes,
            )
            results.update(knn_results)
            for k, acc in knn_results.items():
                logger.info(f"  {k}={acc:.4f}")
        except Exception as e:
            logger.warning(f"  k-NN evaluation failed: {e}")

        return results
