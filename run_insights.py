"""CLI entry point for TCD-JEPA Document Intelligence.

Processes natural language documents through the full TCD-JEPA pipeline
and generates commercial intelligence insights.

Usage:
    # Analyze text files
    python run_insights.py --files doc1.txt doc2.txt --tcd

    # Analyze a directory of documents
    python run_insights.py --dir ./documents/ --tcd --epochs 50

    # Analyze inline text
    python run_insights.py --text "Your document content here..." --tcd

    # Load pre-processed corpus
    python run_insights.py --corpus ./data/processed/ --tcd

    # Custom config
    python run_insights.py --config configs/nl_pipeline.yaml --files docs/*.txt

    # Save results
    python run_insights.py --files docs/*.txt --tcd --output ./results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("tcd_jepa.run_insights")


def main():
    parser = argparse.ArgumentParser(
        description="TCD-JEPA Document Intelligence — analyze documents for commercial insights",
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--files", nargs="+", help="Document files to analyze")
    input_group.add_argument("--dir", help="Directory of documents to analyze")
    input_group.add_argument("--text", help="Inline text to analyze")
    input_group.add_argument("--corpus", help="Pre-processed corpus directory")

    # Model settings
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--config", help="YAML config file")
    model_group.add_argument("--tcd", action="store_true", help="Enable TCD crystallization")
    model_group.add_argument("--epochs", type=int, default=30, help="Training epochs")
    model_group.add_argument("--embed-dim", type=int, default=192, help="Model embedding dimension")
    model_group.add_argument("--depth", type=int, default=6, help="Transformer depth")
    model_group.add_argument("--batch-size", type=int, default=16, help="Training batch size")

    # Document processing
    doc_group = parser.add_argument_group("Document Processing")
    doc_group.add_argument("--chunk-size", type=int, default=256, help="Words per chunk")
    doc_group.add_argument("--chunk-overlap", type=int, default=64, help="Overlap between chunks")
    doc_group.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                           help="Sentence-transformer model name")
    doc_group.add_argument("--semantic-threshold", type=float, default=0.65,
                           help="Cosine similarity threshold for semantic links")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", help="Output directory for results")
    output_group.add_argument("--save-corpus", help="Save processed corpus to directory")
    output_group.add_argument("--save-model", help="Save trained model to file")
    output_group.add_argument("--json", action="store_true", help="Output JSON instead of NL")
    output_group.add_argument("--quiet", action="store_true", help="Minimal output")
    output_group.add_argument("--lineage", action="store_true",
                              help="Include full lineage IDs in JSON output")
    output_group.add_argument("--export-lineage", help="Save lineage graph JSON to file")
    output_group.add_argument("--no-divergent", action="store_true",
                              help="Disable divergent analysis (sensitivity, contradictions)")
    output_group.add_argument("--deep-signals", action="store_true", default=None,
                              help="Enable deep signal harvesting (default: on when --tcd)")
    output_group.add_argument("--no-deep-signals", action="store_true",
                              help="Disable deep signal harvesting for faster runs")

    # Oracle intelligence mode
    oracle_group = parser.add_argument_group("Oracle Intelligence")
    oracle_group.add_argument("--oracle", action="store_true",
                              help="Enable Oracle mode — core 7-lens operational intelligence briefing")
    oracle_group.add_argument("--oracle-all", action="store_true",
                              help="Oracle mode with ALL 30+ available lenses")
    oracle_group.add_argument("--oracle-auto", action="store_true",
                              help="Oracle mode with auto-generated lenses from data")
    oracle_group.add_argument("--lens", nargs="+",
                              help="Specific lens(es) to render (e.g. --lens revenue fraud clinical)")
    oracle_group.add_argument("--list-lenses", action="store_true",
                              help="Print all available intelligence lenses and exit")
    oracle_group.add_argument("--domain", default="business",
                              help="Business domain (e.g. retail, healthcare, fintech)")
    oracle_group.add_argument("--suggest-lenses", action="store_true",
                              help="Suggest relevant lenses for the specified --domain")
    oracle_group.add_argument("--entity-type", default="segment",
                              help="What clusters represent (e.g. 'customer segment', 'product line')")
    oracle_group.add_argument("--actor", default="your team",
                              help="Who should act (e.g. 'sales team', 'product team')")
    oracle_group.add_argument("--resource-noun", default="resources",
                              help="What to allocate (e.g. 'budget', 'headcount')")
    oracle_group.add_argument("--currency", default="value",
                              help="Value metric (e.g. 'revenue', 'ARR', 'deal count')")
    oracle_group.add_argument("--cluster-labels", default=None,
                              help="JSON mapping of cluster IDs to names (e.g. '{\"0\":\"Enterprise\"}')")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("tcd_jepa").setLevel(logging.WARNING)

    # Build config
    config = {}
    if args.config:
        from tcd_jepa.utils.config import load_config_with_overrides
        config = load_config_with_overrides(args.config, [])

    # Override config with CLI args
    if args.embed_dim != 192 or args.depth != 6:
        config.setdefault("model", {}).setdefault("encoder", {})
        config["model"]["encoder"]["embed_dim"] = args.embed_dim
        config["model"]["encoder"]["depth"] = args.depth
    if args.batch_size != 16:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    config.setdefault("document", {}).update({
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_model": args.embedding_model,
        "semantic_threshold": args.semantic_threshold,
    })

    # Divergent analysis config
    if args.no_divergent:
        config.setdefault("divergent", {})["enabled"] = False

    # Deep signals config
    if args.no_deep_signals:
        config.setdefault("deep_signals", {})["enabled"] = False
    elif args.deep_signals or args.tcd:
        config.setdefault("deep_signals", {})["enabled"] = True

    # List/suggest lenses (no data needed)
    if args.list_lenses:
        from tcd_jepa.manifold.oracle_intelligence import OracleIntelligenceEngine
        print(OracleIntelligenceEngine().render_lens_catalog())
        sys.exit(0)
    if args.suggest_lenses:
        from tcd_jepa.manifold.oracle_intelligence import OracleIntelligenceEngine
        engine = OracleIntelligenceEngine()
        suggested = engine.list_lenses_for_domain(args.domain)
        print(f"Suggested lenses for '{args.domain}':")
        for name in suggested:
            lens = engine.get_lens(name)
            if lens:
                print(f"  {name:<20} {lens.title}")
        sys.exit(0)

    # Oracle context
    oracle_context = None
    oracle_mode = None
    if args.oracle or args.oracle_all or args.oracle_auto or args.lens:
        from tcd_jepa.manifold.oracle_intelligence import OracleContext
        cluster_labels = {}
        if args.cluster_labels:
            cluster_labels = {int(k): v for k, v in json.loads(args.cluster_labels).items()}
        oracle_context = OracleContext(
            domain=args.domain,
            entity_type=args.entity_type,
            actor=args.actor,
            resource_noun=args.resource_noun,
            currency=args.currency,
            cluster_labels=cluster_labels,
        )
        if args.oracle_all:
            oracle_mode = "oracle:all"
        elif args.oracle_auto:
            oracle_mode = "oracle:auto"
        elif args.lens:
            oracle_mode = ",".join(args.lens) if len(args.lens) > 1 else args.lens[0]
        else:
            oracle_mode = "oracle"

    # Initialize pipeline
    from tcd_jepa.manifold.nl_pipeline import NLIntelligencePipeline
    pipeline = NLIntelligencePipeline(config=config)

    # Determine input
    if args.corpus:
        # Load pre-processed corpus
        logger.info(f"Loading pre-processed corpus from {args.corpus}")
        pipeline.corpus = pipeline.processor.load(args.corpus)
        documents = None
    elif args.text:
        documents = [{"text": args.text, "id": "input"}]
    elif args.files:
        documents = []
        for path_str in args.files:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            documents.append({"text": text, "id": path.stem, "title": path.name})
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logger.error(f"Not a directory: {dir_path}")
            sys.exit(1)
        documents = []
        for ext in (".txt", ".md", ".rst", ".html"):
            for path in sorted(dir_path.glob(f"**/*{ext}")):
                text = path.read_text(encoding="utf-8", errors="replace")
                documents.append({
                    "text": text,
                    "id": path.stem,
                    "title": str(path.relative_to(dir_path)),
                })
    else:
        # Read from stdin
        logger.info("Reading from stdin (pipe documents or use --files/--dir/--text)...")
        text = sys.stdin.read()
        if not text.strip():
            logger.error("No input provided. Use --files, --dir, --text, or pipe to stdin.")
            sys.exit(1)
        documents = [{"text": text, "id": "stdin"}]

    if documents is not None:
        if not documents:
            logger.error("No documents to analyze.")
            sys.exit(1)
        logger.info(f"Analyzing {len(documents)} documents...")

    # Run pipeline
    if documents is not None:
        report = pipeline.analyze_documents(
            documents, use_tcd=args.tcd, num_epochs=args.epochs,
            oracle_context=oracle_context,
        )
    else:
        # Corpus already loaded, train and generate insights
        report = pipeline.analyze_documents(
            [{"text": c.text, "id": c.doc_id} for c in pipeline.corpus.chunks[:50]],
            use_tcd=args.tcd, num_epochs=args.epochs,
            oracle_context=oracle_context,
        )

    # Output
    if args.json:
        insight_list = []
        for i in report.insights:
            entry = {
                "insight_id": i.insight_id,
                "category": i.category,
                "severity": i.severity,
                "title": i.title,
                "description": i.description,
                "entities": i.entities,
                "confidence": i.confidence,
                "sensitivity": i.sensitivity,
                "evidence": i.evidence[:2],
                "source_chunk_ids": i.source_chunk_ids,
                "source_link_ids": i.source_link_ids,
                "source_cluster_ids": i.source_cluster_ids,
                "derivation_steps": i.derivation_steps,
                "confidence_breakdown": i.confidence_breakdown,
                "contradicts": i.contradicts,
                "supports": i.supports,
            }
            if args.lineage and report.lineage_graph:
                entry["lineage"] = report.export_lineage(i.insight_id)
            insight_list.append(entry)

        output = {
            "summary": report.summary,
            "kpis": report.kpis,
            "num_documents": report.num_documents,
            "num_chunks": report.num_chunks,
            "num_clusters": report.num_clusters,
            "num_links": report.num_links,
            "contradictions": [list(p) for p in report.get_contradictions()],
            "insights": insight_list,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(report.to_nl(mode=oracle_mode or "technical"))

    # Save outputs
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save NL report
        report_mode = oracle_mode or "technical"
        filename = "oracle_briefing.txt" if oracle_mode else "intelligence_report.txt"
        with open(out_dir / filename, "w") as f:
            f.write(report.to_nl(mode=report_mode))

        # Save JSON
        with open(out_dir / "intelligence_report.json", "w") as f:
            json.dump({
                "summary": report.summary,
                "kpis": report.kpis,
                "contradictions": [list(p) for p in report.get_contradictions()],
                "insights": [
                    {
                        "insight_id": i.insight_id,
                        "category": i.category,
                        "severity": i.severity,
                        "title": i.title,
                        "description": i.description,
                        "entities": i.entities,
                        "confidence": i.confidence,
                        "sensitivity": i.sensitivity,
                        "source_chunk_ids": i.source_chunk_ids,
                        "derivation_steps": i.derivation_steps,
                        "confidence_breakdown": i.confidence_breakdown,
                        "contradicts": i.contradicts,
                        "supports": i.supports,
                    }
                    for i in report.insights
                ],
            }, f, indent=2, default=str)

        logger.info(f"Results saved to {out_dir}")

    # Export lineage graph
    if args.export_lineage and report.lineage_graph:
        lineage_path = Path(args.export_lineage)
        lineage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lineage_path, "w") as f:
            f.write(report.lineage_graph.to_json())
        logger.info(f"Lineage graph saved to {lineage_path}")

    if args.save_corpus and pipeline.corpus:
        pipeline.processor.save(pipeline.corpus, args.save_corpus)

    if args.save_model:
        pipeline.save_model(args.save_model)

    # Print high-priority summary
    high_priority = report.get_high_priority()
    if high_priority and not args.quiet and not args.json:
        print(f"\n{'='*70}")
        print(f"HIGH PRIORITY: {len(high_priority)} critical findings")
        for hp in high_priority[:5]:
            print(f"  - [{hp.category.upper()}] {hp.title}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
