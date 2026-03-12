"""End-to-end NL intelligence pipeline: documents → TCD-JEPA → insights.

Orchestrates the full commercial intelligence workflow:
1. Document processing (chunking, embedding, manifold placement)
2. Model training (JEPA + TCD crystallization)
3. Feature extraction and evaluation
4. Insight generation (topological analysis → NL report)

Usage:
    pipeline = NLIntelligencePipeline(config)
    report = pipeline.analyze_documents(documents)
    print(report.to_nl())
"""

import logging
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tcd_jepa.manifold.document_processor import DocumentProcessor, ProcessedCorpus
from tcd_jepa.manifold.insight_engine import InsightEngine, InsightReport
from tcd_jepa.manifold.masking import ManifoldMaskCollator
from tcd_jepa.manifold.model import build_manifold_jepa, ManifoldJEPAModel
from tcd_jepa.manifold.evaluation import (
    extract_manifold_features,
    compute_latent_ocean_kpis,
    linear_probe,
    knn_evaluate,
    compute_representation_metrics,
)
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.core.recursive_loop import RecursiveLoop

logger = logging.getLogger("tcd_jepa.nl_pipeline")


class DocumentManifoldDataset:
    """Wraps ProcessedCorpus as a dataset compatible with ManifoldTrainer."""

    def __init__(self, corpus: ProcessedCorpus, num_tokens: int = 64, num_samples: int = 500):
        from tcd_jepa.manifold.dataset import CausalManifoldDataset

        self.corpus = corpus
        self.num_entities = corpus.fingerprints.shape[0]
        self.num_clusters = int(corpus.entity_labels.max().item()) + 1 if len(corpus.entity_labels) > 0 else 1
        self.entity_labels = corpus.entity_labels
        self.adjacency = corpus.adjacency
        self.coords = corpus.coords

        self._inner = CausalManifoldDataset(
            fingerprints=corpus.fingerprints,
            coords=corpus.coords,
            adjacency=corpus.adjacency,
            velocity=corpus.velocity,
            entity_labels=corpus.entity_labels,
            num_tokens=min(num_tokens, self.num_entities),
            num_samples=num_samples,
            window_mode="geodesic" if (corpus.adjacency > 0).any() else "random",
        )

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


class NLIntelligencePipeline:
    """Full pipeline: documents → TCD-JEPA training → commercial intelligence."""

    DEFAULT_CONFIG = {
        "model": {
            "encoder": {
                "fingerprint_dim": 384,
                "num_tokens": 64,
                "coord_dim": 3,
                "embed_dim": 192,
                "depth": 6,
                "num_heads": 3,
                "mlp_ratio": 4.0,
                "use_causal_encoding": True,
                "use_velocity_encoding": True,
                "sphere_radius": 4.5,
            },
            "predictor": {
                "predictor_embed_dim": 96,
                "predictor_depth": 4,
                "num_heads": 3,
            },
            "ema": {"start": 0.996, "end": 1.0},
        },
        "training": {
            "epochs": 30,
            "batch_size": 16,
            "learning_rate": 0.001,
            "start_lr": 0.0001,
            "weight_decay": 0.05,
            "warmup_epochs": 3,
            "seed": 42,
        },
        "masking": {
            "context_ratio": [0.3, 0.5],
            "target_ratio": [0.15, 0.3],
            "num_context_masks": 4,
            "num_target_masks": 1,
            "min_keep": 4,
            "use_geodesic": True,
            "sphere_radius": 4.5,
        },
        "tcd": {
            "use_dynamic_predictor": True,
            "explore_every": 5,
            "crystallize_every": 10,
            "langevin_steps": 50,
            "langevin_step_size": 0.01,
            "persistence_threshold": 0.3,
            "max_modules": 10,
        },
        "document": {
            "chunk_size": 256,
            "chunk_overlap": 64,
            "semantic_threshold": 0.65,
            "embedding_model": "all-MiniLM-L6-v2",
            "min_chunks": 16,
        },
        "insight": {
            "top_k_per_category": 5,
            "anomaly_threshold": 2.0,
            "opportunity_sim_threshold": 0.7,
            "risk_boundary_ratio": 0.7,
        },
    }

    def __init__(self, config: Optional[dict] = None, device: Optional[torch.device] = None):
        self.cfg = self._merge_config(config or {})
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[ManifoldJEPAModel] = None
        self.recursive_loop: Optional[RecursiveLoop] = None
        self.corpus: Optional[ProcessedCorpus] = None

        doc_cfg = self.cfg["document"]
        enc_cfg = self.cfg["model"]["encoder"]
        self.processor = DocumentProcessor(
            fingerprint_dim=enc_cfg["fingerprint_dim"],
            sphere_radius=enc_cfg["sphere_radius"],
            chunk_size=doc_cfg["chunk_size"],
            chunk_overlap=doc_cfg["chunk_overlap"],
            semantic_threshold=doc_cfg["semantic_threshold"],
            embedding_model=doc_cfg["embedding_model"],
            min_chunks=doc_cfg["min_chunks"],
        )

        insight_cfg = self.cfg["insight"]
        self.insight_engine = InsightEngine(
            top_k_per_category=insight_cfg["top_k_per_category"],
            anomaly_threshold=insight_cfg["anomaly_threshold"],
            opportunity_sim_threshold=insight_cfg["opportunity_sim_threshold"],
            risk_boundary_ratio=insight_cfg["risk_boundary_ratio"],
        )

    def _merge_config(self, user_cfg: dict) -> dict:
        """Deep merge user config over defaults (recursive)."""
        return self._deep_merge(self.DEFAULT_CONFIG, user_cfg)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base."""
        result = {}
        for key in set(list(base.keys()) + list(override.keys())):
            if key in override and key in base:
                if isinstance(base[key], dict) and isinstance(override[key], dict):
                    result[key] = NLIntelligencePipeline._deep_merge(base[key], override[key])
                else:
                    result[key] = override[key]
            elif key in override:
                result[key] = override[key]
            else:
                result[key] = base[key]
        return result

    def analyze_documents(
        self,
        documents: list[dict],
        use_tcd: bool = True,
        num_epochs: Optional[int] = None,
    ) -> InsightReport:
        """Full pipeline: documents → intelligence report.

        Args:
            documents: List of dicts with 'text' (required), 'id', 'title', 'timestamp'.
            use_tcd: Enable TCD crystallization for topological insights.
            num_epochs: Override training epochs (None = use config).

        Returns:
            InsightReport with all insights and KPIs.
        """
        t0 = time.time()

        # 1. Process documents
        logger.info("Step 1: Processing documents...")
        self.corpus = self.processor.process_documents(documents)
        logger.info(f"  → {len(self.corpus.chunks)} chunks, "
                     f"{(self.corpus.adjacency > 0).sum().item()} links")

        # 2. Build model and train
        logger.info("Step 2: Training TCD-JEPA model...")
        epochs = num_epochs or self.cfg["training"]["epochs"]
        self._train_model(use_tcd=use_tcd, num_epochs=epochs)

        # 3. Extract features and compute KPIs
        logger.info("Step 3: Extracting features and computing KPIs...")
        dataset = DocumentManifoldDataset(
            self.corpus,
            num_tokens=self.cfg["model"]["encoder"]["num_tokens"],
        )
        features, labels, indices = extract_manifold_features(
            self.model, dataset, self.device, max_samples=min(500, len(dataset)),
        )

        kpis = compute_latent_ocean_kpis(
            self.model, dataset, self.device,
            recursive_loop=self.recursive_loop, max_samples=200,
        )

        # Aggregate features back to corpus-level entities
        N_corpus = len(self.corpus.chunks)
        entity_features = torch.zeros(N_corpus, features.shape[1])
        entity_counts = torch.zeros(N_corpus)
        for i in range(len(indices)):
            eid = indices[i].item()
            if eid < N_corpus:
                entity_features[eid] += features[i]
                entity_counts[eid] += 1
        # Fill in entities that weren't sampled
        valid = entity_counts > 0
        entity_features[valid] /= entity_counts[valid].unsqueeze(1)
        # For unsampled entities, use raw fingerprints projected through a mean
        if not valid.all() and valid.any():
            mean_feat = entity_features[valid].mean(0)
            entity_features[~valid] = mean_feat

        entity_labels = self.corpus.entity_labels[:N_corpus]

        # Compute prediction errors per chunk
        prediction_errors = self._compute_prediction_errors(dataset)
        # Trim/pad prediction errors to corpus size
        if len(prediction_errors) > N_corpus:
            prediction_errors = prediction_errors[:N_corpus]
        elif len(prediction_errors) < N_corpus:
            pad = prediction_errors.mean().expand(N_corpus - len(prediction_errors))
            prediction_errors = torch.cat([prediction_errors, pad])

        # 4. Generate insights
        logger.info("Step 4: Generating intelligence report...")
        report = self.insight_engine.generate_report(
            features=entity_features,
            labels=entity_labels,
            chunk_texts=self.corpus.chunk_texts,
            chunks=self.corpus.chunks,
            adjacency=self.corpus.adjacency,
            kpis=kpis,
            prediction_errors=prediction_errors,
            recursive_loop=self.recursive_loop,
        )

        elapsed = time.time() - t0
        logger.info(f"Pipeline complete in {elapsed:.1f}s — "
                     f"{len(report.insights)} insights generated")

        return report

    def analyze_text(self, text: str, **kwargs) -> InsightReport:
        """Convenience: analyze a single text document."""
        return self.analyze_documents([{"text": text, "id": "input"}], **kwargs)

    def analyze_files(self, file_paths: list[str], **kwargs) -> InsightReport:
        """Analyze document files."""
        documents = []
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            documents.append({"text": text, "id": path.stem, "title": path.name})
        return self.analyze_documents(documents, **kwargs)

    def _train_model(self, use_tcd: bool = True, num_epochs: int = 30) -> None:
        """Build and train the manifold JEPA model on processed documents."""
        enc_cfg = self.cfg["model"]["encoder"]
        pred_cfg = self.cfg["model"]["predictor"]
        train_cfg = self.cfg["training"]
        mask_cfg = self.cfg["masking"]
        tcd_cfg = self.cfg["tcd"]

        torch.manual_seed(train_cfg.get("seed", 42))
        np.random.seed(train_cfg.get("seed", 42))

        # Adjust num_tokens to corpus size
        num_tokens = min(enc_cfg["num_tokens"], self.corpus.fingerprints.shape[0])

        # Build model
        self.model = build_manifold_jepa(
            fingerprint_dim=enc_cfg["fingerprint_dim"],
            num_tokens=num_tokens,
            coord_dim=enc_cfg["coord_dim"],
            embed_dim=enc_cfg["embed_dim"],
            depth=enc_cfg["depth"],
            num_heads=enc_cfg["num_heads"],
            mlp_ratio=enc_cfg.get("mlp_ratio", 4.0),
            predictor_embed_dim=pred_cfg["predictor_embed_dim"],
            predictor_depth=pred_cfg["predictor_depth"],
            predictor_num_heads=pred_cfg["num_heads"],
            use_dynamic_predictor=use_tcd,
            use_causal_encoding=enc_cfg.get("use_causal_encoding", True),
            use_velocity_encoding=enc_cfg.get("use_velocity_encoding", True),
            sphere_radius=enc_cfg.get("sphere_radius", 4.5),
        ).to(self.device)

        # Build dataset
        dataset = DocumentManifoldDataset(
            self.corpus,
            num_tokens=num_tokens,
            num_samples=max(200, len(self.corpus.chunks) * 3),
        )

        # Build mask collator
        mask_collator = ManifoldMaskCollator(
            num_tokens=num_tokens,
            context_ratio=tuple(mask_cfg.get("context_ratio", [0.3, 0.5])),
            target_ratio=tuple(mask_cfg.get("target_ratio", [0.15, 0.3])),
            num_context_masks=mask_cfg.get("num_context_masks", 4),
            num_target_masks=mask_cfg.get("num_target_masks", 1),
            min_keep=mask_cfg.get("min_keep", 4),
            use_geodesic=mask_cfg.get("use_geodesic", True),
            sphere_radius=mask_cfg.get("sphere_radius", 4.5),
        )

        batch_size = min(train_cfg.get("batch_size", 16), len(dataset))
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=mask_collator, drop_last=True,
        )

        # Optimizer and schedulers
        steps_per_epoch = max(len(dataloader), 1)
        total_steps = num_epochs * steps_per_epoch

        optimizer = build_optimizer(
            self.model, lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        lr_scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=train_cfg.get("warmup_epochs", 3) * steps_per_epoch,
            start_lr=train_cfg.get("start_lr", 1e-4),
            ref_lr=train_cfg["learning_rate"],
            T_max=total_steps,
        )
        wd_scheduler = CosineWDSchedule(
            optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps,
        )
        ema_schedule = momentum_schedule(
            self.cfg["model"]["ema"]["start"],
            self.cfg["model"]["ema"]["end"],
            total_steps,
        )

        # TCD recursive loop
        self.recursive_loop = None
        if use_tcd:
            self.recursive_loop = RecursiveLoop(
                embed_dim=enc_cfg["embed_dim"],
                explore_every=tcd_cfg.get("explore_every", 5),
                crystallize_every=tcd_cfg.get("crystallize_every", 10),
                langevin_steps=tcd_cfg.get("langevin_steps", 50),
                langevin_step_size=tcd_cfg.get("langevin_step_size", 0.01),
                persistence_threshold=tcd_cfg.get("persistence_threshold", 0.3),
                max_modules=tcd_cfg.get("max_modules", 10),
                device=self.device,
            )
            self.model.set_module_registry(self.recursive_loop.crystallizer.registry)

        # Training loop
        from tcd_jepa.manifold.trainer import ManifoldTrainer
        trainer = ManifoldTrainer(
            model=self.model, optimizer=optimizer,
            lr_scheduler=lr_scheduler, wd_scheduler=wd_scheduler,
            momentum_schedule=ema_schedule, train_loader=dataloader,
            device=self.device, cfg=self.cfg,
            recursive_loop=self.recursive_loop,
        )

        trainer.train(num_epochs)

    def _compute_prediction_errors(self, dataset) -> torch.Tensor:
        """Compute per-chunk JEPA prediction error."""
        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for i in range(min(200, len(dataset))):
                sample = dataset[i]
                fp = sample["fingerprints"].unsqueeze(0).to(self.device)
                coords = sample.get("coords")
                if coords is not None:
                    coords = coords.unsqueeze(0).to(self.device)
                adj = sample.get("adjacency")
                if adj is not None:
                    adj = adj.unsqueeze(0).to(self.device)
                vel = sample.get("velocity")
                if vel is not None:
                    vel = vel.unsqueeze(0).to(self.device)

                z = self.model.context_encoder(fp, coords=coords, adjacency=adj, velocity=vel)
                t = self.model.target_encoder(fp, coords=coords, adjacency=adj, velocity=vel)

                # Per-token error, then mean per sample
                token_errors = (z - t).pow(2).sum(dim=-1).mean(dim=0)  # [N_tokens]
                all_errors.append(token_errors.cpu())

        if not all_errors:
            return torch.zeros(1)

        # Flatten to per-chunk approximation
        errors = torch.cat(all_errors, dim=0)
        return errors

    def save_model(self, path: str) -> None:
        """Save trained model for reuse."""
        if self.model is None:
            raise RuntimeError("No trained model to save")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.cfg,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a previously trained model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.cfg = checkpoint["config"]
        enc_cfg = self.cfg["model"]["encoder"]
        pred_cfg = self.cfg["model"]["predictor"]

        self.model = build_manifold_jepa(
            fingerprint_dim=enc_cfg["fingerprint_dim"],
            num_tokens=enc_cfg["num_tokens"],
            embed_dim=enc_cfg["embed_dim"],
            depth=enc_cfg["depth"],
            num_heads=enc_cfg["num_heads"],
            predictor_embed_dim=pred_cfg["predictor_embed_dim"],
            predictor_depth=pred_cfg["predictor_depth"],
            predictor_num_heads=pred_cfg["num_heads"],
            use_dynamic_predictor=self.cfg["tcd"].get("use_dynamic_predictor", True),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Model loaded from {path}")
