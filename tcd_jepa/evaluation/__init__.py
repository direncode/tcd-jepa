"""Evaluation pipeline for TCD-JEPA: linear probe, k-NN, and evaluation runner."""

from tcd_jepa.evaluation.eval_runner import EvaluationRunner
from tcd_jepa.evaluation.knn_evaluator import KNNEvaluator
from tcd_jepa.evaluation.linear_probe import LinearProbeEvaluator

__all__ = ["LinearProbeEvaluator", "KNNEvaluator", "EvaluationRunner"]
