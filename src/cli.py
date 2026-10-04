"""Unified Command Line Interface for FEVER-X."""

import argparse
import json
from pathlib import Path
import sys
import subprocess

from src.pipeline.full_pipeline import FactVerificationPipeline
from src.utils.logger import get_logger

logger = get_logger("feverx_cli")


def cmd_verify(args) -> None:
    """Verifies a claim and outputs formatted results or JSON."""
    pipeline = FactVerificationPipeline(use_fallback_models=args.fallback)
    result = pipeline.verify(
        claim=args.claim,
        top_k=args.top_k,
        include_graph=not args.no_graph,
        aggregation_strategy=args.strategy,
    )

    if args.json:
        print(json.dumps(result.model_dump(), indent=2))
        return

    print("=" * 70)
    print(f"CLAIM: {result.claim}")
    print(f"VERDICT: {result.verdict.value}  (Confidence: {result.confidence * 100:.1f}%)")
    print("=" * 70)
    print(f"Summary: {result.explanation.get('summary')}")
    print("\nEntities Identified:")
    for ent in result.entities:
        print(f"  - {ent.text} [{ent.label}]")

    if result.subclaims:
        print(f"\nDecomposed Subclaims ({len(result.subclaims)}):")
        for sc in result.subclaims:
            print(f"  - {sc.text}")

    print(f"\nTop Evidence Sentences ({len(result.top_evidence)}):")
    for idx, ev in enumerate(result.top_evidence, start=1):
        print(f"  {idx}. [{ev.doc_id}:S{ev.sentence_id}] (Score: {ev.rerank_score:.3f})")
        print(f"     \"{ev.text}\"")

    print("\nReasoning Audit Steps:")
    for step in result.explanation.get("reasoning_steps", []):
        print(f"  {step}")
    print("=" * 70)


def cmd_retrieve(args) -> None:
    """Retrieves candidates for a query."""
    pipeline = FactVerificationPipeline(use_fallback_models=args.fallback)
    methods = [m.strip() for m in args.methods.split(",")]
    candidates = pipeline.retrieve(
        query=args.query,
        top_k=args.top_k,
        methods=methods,
        use_entity_boost=not args.no_entity_boost,
        use_rrf=not args.no_rrf,
    )

    if args.json:
        print(json.dumps([c.model_dump() for c in candidates], indent=2))
        return

    print(f"Retrieved {len(candidates)} candidates for: '{args.query}'")
    for idx, c in enumerate(candidates, start=1):
        print(f"{idx}. [{c.doc_id}:S{c.sentence_id}] ({c.retriever}, Score: {c.score:.4f})")
        print(f"   \"{c.text}\"")


def cmd_evaluate(args) -> None:
    """Runs ablation evaluation benchmark."""
    from scripts.run_evaluation import main as eval_main
    sys.argv = ["run_evaluation.py", "--k", str(args.k), "--output_dir", args.output_dir]
    eval_main()


def cmd_api(args) -> None:
    """Launches the FastAPI backend microservice."""
    import uvicorn
    print(f"Starting FEVER-X FastAPI service on {args.host}:{args.port}...")
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)


def cmd_ui(args) -> None:
    """Launches the Streamlit dashboard."""
    print(f"Starting FEVER-X Streamlit UI on port {args.port}...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py",
        "--server.port",
        str(args.port),
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        prog="feverx",
        description="FEVER-X: Explainable Multi-Stage Fact Verification CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify a factual claim")
    p_verify.add_argument("claim", type=str, help="Claim statement to verify")
    p_verify.add_argument("--top_k", type=int, default=3, help="Number of evidence sentences")
    p_verify.add_argument("--strategy", type=str, default="weighted_prob", choices=["weighted_prob", "max_confidence", "majority_vote"])
    p_verify.add_argument("--no_graph", action="store_true", help="Disable knowledge graph construction")
    p_verify.add_argument("--fallback", action="store_true", default=True, help="Use fallback models for CPU speed")
    p_verify.add_argument("--json", action="store_true", help="Output raw JSON")
    p_verify.set_defaults(func=cmd_verify)

    # retrieve
    p_ret = subparsers.add_parser("retrieve", help="Retrieve candidate evidence sentences")
    p_ret.add_argument("query", type=str, help="Search query")
    p_ret.add_argument("--top_k", type=int, default=5, help="Number of candidates")
    p_ret.add_argument("--methods", type=str, default="bm25,sentence_transformer,tfidf")
    p_ret.add_argument("--no_entity_boost", action="store_true", help="Disable entity boosting")
    p_ret.add_argument("--no_rrf", action="store_true", help="Disable RRF fusion")
    p_ret.add_argument("--fallback", action="store_true", default=True)
    p_ret.add_argument("--json", action="store_true", help="Output raw JSON")
    p_ret.set_defaults(func=cmd_retrieve)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run 9-stage ablation study")
    p_eval.add_argument("--k", type=int, default=10, help="Cutoff K for metrics")
    p_eval.add_argument("--output_dir", type=str, default="data/evaluation", help="Output directory")
    p_eval.set_defaults(func=cmd_evaluate)

    # api
    p_api = subparsers.add_parser("api", help="Launch FastAPI REST backend")
    p_api.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    p_api.add_argument("--port", type=int, default=8000, help="Port number")
    p_api.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    p_api.set_defaults(func=cmd_api)

    # ui
    p_ui = subparsers.add_parser("ui", help="Launch Streamlit Web UI")
    p_ui.add_argument("--port", type=int, default=8501, help="Port number")
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
