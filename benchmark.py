"""Latency benchmark harness.

Runs a fixed set of real queries through the pipeline multiple times and
reports P50 / P70 / P100 latency, broken out per stage. Two paths are
measured separately because they have fundamentally different latency
characteristics:

  * "local"       - query embedding + vector search + guardrail check.
                    Everything here is in-process, no network call, and is
                    the piece that can realistically be held under 200ms.
  * "end_to_end"  - local path + the network-bound Gemini generation call
                    (and, in --voice mode, the Deepgram STT call first).
                    This is reported honestly rather than only measuring
                    the part that makes the number look good.

Usage:
    python benchmark.py                 # text queries, local + end-to-end
    python benchmark.py --local-only    # skip the network LLM leg
    python benchmark.py --runs 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from backend import guardrails
from backend.pipeline.rag import RAGPipeline

QUERIES = [
    "What is FastAPI used for?",
    "How does vector search find relevant documents?",
    "Why does chunk size matter for retrieval quality?",
    "What role do guardrails play in a RAG pipeline?",
    "Does FastAPI support asynchronous request handlers?",
    "What is the capital of France?",  # expected to be refused: out of corpus
    "How do overlapping chunks help retrieval?",
    "What makes FastAPI good for calling external APIs?",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if pct >= 100:
        return ordered[-1]
    index = (pct / 100) * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def summarize(label: str, samples_ms: list[float]) -> dict:
    return {
        "stage": label,
        "n": len(samples_ms),
        "p50_ms": round(percentile(samples_ms, 50), 2),
        "p70_ms": round(percentile(samples_ms, 70), 2),
        "p100_ms": round(percentile(samples_ms, 100), 2),
        "mean_ms": round(statistics.mean(samples_ms), 2) if samples_ms else None,
    }


async def run_benchmark(runs: int, local_only: bool) -> dict:
    pipeline = RAGPipeline()

    local_ms: list[float] = []
    end_to_end_ms: list[float] = []
    generation_ms: list[float] = []
    refused = 0
    answered = 0

    for _ in range(runs):
        for query in QUERIES:
            local_start = time.perf_counter()
            retrieval = pipeline.retriever.retrieve(query=query, top_k=5)
            guardrail_result = guardrails.check(query, retrieval.results)
            local_elapsed_ms = (time.perf_counter() - local_start) * 1000
            local_ms.append(local_elapsed_ms)

            if local_only:
                continue

            result = await pipeline.run(query=query, top_k=5)
            end_to_end_ms.append(result.latencies.total_ms)
            if result.latencies.generation_ms is not None:
                generation_ms.append(result.latencies.generation_ms)
            if result.answered:
                answered += 1
            else:
                refused += 1

    report = {
        "runs": runs,
        "queries_per_run": len(QUERIES),
        "total_samples": runs * len(QUERIES),
        "stages": [summarize("local (embed + retrieve + guardrail)", local_ms)],
    }

    if not local_only:
        report["stages"].append(summarize("generation (LLM, network-bound)", generation_ms))
        report["stages"].append(summarize("end_to_end (full pipeline)", end_to_end_ms))
        report["guardrail_outcomes"] = {"answered": answered, "refused": refused}

    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark the RAG pipeline")
    parser.add_argument("--runs", type=int, default=10, help="Passes over the query set")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip the network-bound LLM generation leg",
    )
    parser.add_argument("--output", default="benchmark_report.json")
    args = parser.parse_args()

    report = asyncio.run(run_benchmark(runs=args.runs, local_only=args.local_only))

    print(json.dumps(report, indent=2))

    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
