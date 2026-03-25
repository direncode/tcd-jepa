"""k-NN evaluation for self-supervised representations.

Measures representation quality by classifying test samples using
their nearest neighbors in the training set (cosine similarity).
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("tcd_jepa")


class KNNEvaluator:
    """Evaluates representations via k-nearest neighbor classification."""

    def __init__(self, k_values: list[int] | None = None, chunk_size: int = 256) -> None:
        self.k_values = k_values or [1, 5, 20]
        self.chunk_size = chunk_size

    @torch.no_grad()
    def evaluate(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        num_classes: int = 10,
    ) -> dict[str, float]:
        """Run k-NN classification.

        Uses chunked cosine similarity to avoid memory issues on large datasets.

        Returns:
            Dict with knn_k{k}_acc for each k.
        """
        # Normalize features
        train_features = F.normalize(train_features, dim=-1)
        test_features = F.normalize(test_features, dim=-1)

        max_k = max(self.k_values)
        results = {}

        # Process test features in chunks for memory efficiency
        all_preds = {k: [] for k in self.k_values}

        for start in range(0, test_features.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, test_features.shape[0])
            chunk = test_features[start:end]

            # Cosine similarity [chunk_size, N_train]
            sim = chunk @ train_features.T

            # Top-k neighbors
            topk_sim, topk_idx = sim.topk(max_k, dim=-1)
            topk_labels = train_labels[topk_idx]  # [chunk_size, max_k]

            for k in self.k_values:
                # Majority vote among k nearest neighbors
                knn_labels = topk_labels[:, :k]
                # One-hot vote counting
                votes = torch.zeros(knn_labels.shape[0], num_classes, device=knn_labels.device)
                votes.scatter_add_(1, knn_labels.long(), torch.ones_like(knn_labels, dtype=votes.dtype))
                preds = votes.argmax(dim=-1)
                all_preds[k].append(preds)

        for k in self.k_values:
            preds = torch.cat(all_preds[k])
            acc = (preds == test_labels).float().mean().item()
            results[f"knn_k{k}_acc"] = acc

        return results
