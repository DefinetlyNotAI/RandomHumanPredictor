"""Chronological prequential (predict-then-train) evaluation."""
from __future__ import annotations
from math import log
from .model import OnlinePredictor, validate_number


def evaluate(numbers: list[int]) -> dict:
    model = OnlinePredictor()
    results = []
    for actual in numbers:
        validate_number(actual)
        prediction = model.predict(top_k=10)
        rank = sorted(range(101), key=lambda x: (-prediction.probabilities[x], x)).index(actual) + 1
        results.append({"actual": actual, "guess": prediction.choice, "rank": rank,
                        "probability": prediction.probabilities[actual]})
        model.learn(actual)
    n = len(results)
    if not n:
        return {"rounds": 0, "top1": None, "top5": None, "top10": None,
                "log_loss": None, "baseline_log_loss": log(101), "results": []}
    return {"rounds": n,
            "top1": sum(r["rank"] == 1 for r in results) / n,
            "top5": sum(r["rank"] <= 5 for r in results) / n,
            "top10": sum(r["rank"] <= 10 for r in results) / n,
            "log_loss": -sum(log(r["probability"]) for r in results) / n,
            "baseline_log_loss": log(101), "results": results}
