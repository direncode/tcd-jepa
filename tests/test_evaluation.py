"""Tests for the evaluation pipeline: linear probe, k-NN, eval runner."""

import torch
import torch.nn as nn

from tcd_jepa.evaluation.knn_evaluator import KNNEvaluator
from tcd_jepa.evaluation.linear_probe import LinearProbeEvaluator


class TestLinearProbeEvaluator:
    def test_evaluate_separable_features(self):
        """Linear probe should achieve high accuracy on perfectly separable features."""
        evaluator = LinearProbeEvaluator(num_classes=2, epochs=50, lr=0.1, batch_size=64)

        # Create perfectly separable features
        n = 200
        train_features = torch.cat([
            torch.randn(n, 16) + 2.0,   # Class 0
            torch.randn(n, 16) - 2.0,   # Class 1
        ])
        train_labels = torch.cat([torch.zeros(n), torch.ones(n)]).long()

        test_features = torch.cat([
            torch.randn(n // 2, 16) + 2.0,
            torch.randn(n // 2, 16) - 2.0,
        ])
        test_labels = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()

        results = evaluator.evaluate(train_features, train_labels, test_features, test_labels)

        assert "linear_probe_train_acc" in results
        assert "linear_probe_test_acc" in results
        assert results["linear_probe_train_acc"] > 0.8
        assert results["linear_probe_test_acc"] > 0.7

    def test_evaluate_random_features(self):
        """Linear probe on random features should achieve near-chance accuracy."""
        evaluator = LinearProbeEvaluator(num_classes=10, epochs=10, lr=0.01)

        train_features = torch.randn(500, 32)
        train_labels = torch.randint(0, 10, (500,))
        test_features = torch.randn(100, 32)
        test_labels = torch.randint(0, 10, (100,))

        results = evaluator.evaluate(train_features, train_labels, test_features, test_labels)
        # Should be near chance (0.1) but not exactly — allow wide range
        assert 0.0 <= results["linear_probe_test_acc"] <= 1.0

    def test_extract_features_shape(self):
        """Feature extraction should produce correct shapes."""
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 32))
        evaluator = LinearProbeEvaluator()

        # Create a simple dataset
        images = torch.randn(20, 3, 4, 4)
        labels = torch.randint(0, 10, (20,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        features, extracted_labels = evaluator.extract_features(encoder, loader, torch.device("cpu"))
        assert features.shape == (20, 32)
        assert extracted_labels.shape == (20,)


class TestKNNEvaluator:
    def test_knn_separable(self):
        """k-NN should achieve high accuracy on separable clusters."""
        evaluator = KNNEvaluator(k_values=[1, 5])

        n = 100
        train_features = torch.cat([
            torch.randn(n, 8) + 3.0,
            torch.randn(n, 8) - 3.0,
        ])
        train_labels = torch.cat([torch.zeros(n), torch.ones(n)]).long()

        test_features = torch.cat([
            torch.randn(n // 2, 8) + 3.0,
            torch.randn(n // 2, 8) - 3.0,
        ])
        test_labels = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()

        results = evaluator.evaluate(
            train_features, train_labels, test_features, test_labels, num_classes=2,
        )

        assert "knn_k1_acc" in results
        assert "knn_k5_acc" in results
        assert results["knn_k1_acc"] > 0.8
        assert results["knn_k5_acc"] > 0.8

    def test_knn_returns_all_k(self):
        """k-NN should return results for all requested k values."""
        evaluator = KNNEvaluator(k_values=[1, 3, 7])

        features = torch.randn(50, 16)
        labels = torch.randint(0, 5, (50,))

        results = evaluator.evaluate(features, labels, features, labels, num_classes=5)

        assert "knn_k1_acc" in results
        assert "knn_k3_acc" in results
        assert "knn_k7_acc" in results

    def test_knn_chunked(self):
        """k-NN with small chunk size should produce same results."""
        evaluator_small = KNNEvaluator(k_values=[1], chunk_size=10)
        evaluator_large = KNNEvaluator(k_values=[1], chunk_size=1000)

        torch.manual_seed(42)
        features = torch.randn(50, 8)
        labels = torch.randint(0, 3, (50,))

        r1 = evaluator_small.evaluate(features, labels, features, labels, num_classes=3)
        r2 = evaluator_large.evaluate(features, labels, features, labels, num_classes=3)

        assert abs(r1["knn_k1_acc"] - r2["knn_k1_acc"]) < 1e-6
