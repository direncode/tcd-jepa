"""Linear probe evaluation for self-supervised representations.

Trains a linear classifier on frozen encoder features to measure
representation quality.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("tcd_jepa")


class LinearProbeEvaluator:
    """Evaluates representations via linear probe classification.

    Freezes the encoder, extracts features from train/test data,
    and trains a linear classifier on top.
    """

    def __init__(
        self,
        num_classes: int = 10,
        epochs: int = 100,
        lr: float = 0.1,
        batch_size: int = 256,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay

    @torch.no_grad()
    def extract_features(
        self,
        encoder: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract frozen encoder features from a labeled dataset.

        Returns:
            Tuple of (features [N, D], labels [N]).
        """
        encoder.eval()
        all_features = []
        all_labels = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images = batch[0]
                    labels = torch.zeros(images.shape[0], dtype=torch.long)
            else:
                images = batch
                labels = torch.zeros(images.shape[0], dtype=torch.long)

            images = images.to(device)
            features = encoder(images)

            # Average pool if 3D (patch tokens)
            if features.dim() == 3:
                features = features.mean(dim=1)

            # Normalize features
            features = F.normalize(features, dim=-1)
            all_features.append(features.cpu())
            all_labels.append(labels)

        return torch.cat(all_features), torch.cat(all_labels)

    def evaluate(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> dict[str, float]:
        """Train and evaluate a linear probe.

        Returns:
            Dict with train_acc, test_acc.
        """
        device = device or torch.device("cpu")
        embed_dim = train_features.shape[1]

        # Build linear classifier
        classifier = nn.Linear(embed_dim, self.num_classes).to(device)
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # DataLoaders
        train_ds = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Train
        classifier.train()
        for epoch in range(self.epochs):
            for feats, labels in train_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = classifier(feats)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            train_logits = classifier(train_features.to(device))
            train_acc = (train_logits.argmax(dim=-1) == train_labels.to(device)).float().mean().item()

            test_logits = classifier(test_features.to(device))
            test_acc = (test_logits.argmax(dim=-1) == test_labels.to(device)).float().mean().item()

        return {
            "linear_probe_train_acc": train_acc,
            "linear_probe_test_acc": test_acc,
        }
