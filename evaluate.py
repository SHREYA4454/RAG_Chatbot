"""Evaluate retrieval using a JSON list of {"query": ..., "relevant": [filename, ...]}."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def evaluate(cases: list[dict[str, Any]], base_url: str, top_k: int) -> dict[str, Any]:
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    for case in cases:
        relevant = set(case["relevant"])
        if not relevant:
            raise ValueError("Each benchmark case must include at least one relevant filename")
        started = time.perf_counter()
        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"question": case["query"], "top_k": top_k},
            timeout=60,
        )
        response.raise_for_status()
        latencies.append((time.perf_counter() - started) * 1000)
        filenames = [source["filename"] for source in response.json()["sources"]]
        hits = [index for index, filename in enumerate(filenames, start=1) if filename in relevant]
        recall_values.append(len(set(filenames) & relevant) / len(relevant))
        reciprocal_ranks.append(1 / hits[0] if hits else 0.0)
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
    return {
        "cases": len(cases),
        "recall_at_k": round(statistics.mean(recall_values), 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "latency_mean_ms": round(statistics.mean(latencies), 1),
        "latency_p95_ms": round(p95, 1),
        "top_k": top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="JSON file containing benchmark cases")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    with args.benchmark.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    print(json.dumps(evaluate(cases, args.url, args.top_k), indent=2))


if __name__ == "__main__":
    main()
