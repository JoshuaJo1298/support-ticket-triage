"""Compare routers against held-out synthetic tickets and write JSON results."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean, median

from routing import LABELS, MODEL, REVIEW, keyword_router, llm_router


def score(cases, router):
    results = []
    confusion = {label: {choice: 0 for choice in LABELS} for label in LABELS}
    for expected, ticket in cases:
        prediction = router(ticket)
        confusion[expected][prediction.category] += 1
        results.append({"ticket": ticket, "expected": expected, **vars(prediction)})
    f1 = {}
    for label in LABELS:
        true_positives = confusion[label][label]
        predicted = sum(row[label] for row in confusion.values())
        actual = sum(confusion[label].values())
        f1[label] = 2 * true_positives / (predicted + actual)
    return {
        "macro_f1": mean(f1.values()),
        "per_category_f1": f1,
        "manual_review_rate": sum(row["category"] == REVIEW for row in results) / len(results),
        "median_latency_ms": median(row["latency_ms"] for row in results),
        "estimated_api_cost_usd_per_ticket": mean(row["cost_usd"] for row in results),
        "model_errors": sum(bool(row["error"]) for row in results),
        "confusion_matrix": confusion,
        "predictions": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    data = Path(__file__).parent / "data" / "held_out.tsv"
    cases = [line.split("\t", 1) for line in data.read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 60 and all(sum(row[0] == label for row in cases) == 12 for label in LABELS)
    result = {"cases": len(cases), "keyword": score(cases, keyword_router)}
    if not args.baseline_only:
        warmup = llm_router("Can I have a copy of my receipt?")
        if warmup.error:
            parser.exit(1, f"Ollama warmup failed: {warmup.error}\n")
        result["model"] = os.getenv("OLLAMA_MODEL", MODEL)
        result["llm"] = score(cases, llm_router)
        if result["llm"]["model_errors"] == len(cases):
            parser.exit(1, "All model calls failed. No model comparison was produced.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
