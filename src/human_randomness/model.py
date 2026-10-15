"""Pure-Python online probabilistic ensemble. No external ML dependency.

All recommendations are produced *before* the next observation is added.
Mixture weights are updated using the probability each expert assigned to the
observed number. This is a proper probabilistic scoring rule (log loss).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp, log
from typing import Sequence

LOW, HIGH, N = 0, 100, 101
EXPERTS = ("uniform", "frequency", "recency", "digits", "transition", "distance", "nonrepeat")


def normalize(values: Sequence[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("distribution must have positive mass")
    return [v / total for v in values]


def validate_number(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not LOW <= value <= HIGH:
        raise ValueError("number must be an integer from 0 through 100")
    return value


@dataclass(frozen=True)
class Prediction:
    choice: int
    top: tuple[int, ...]
    probabilities: tuple[float, ...]
    weights: dict[str, float]


class OnlinePredictor:
    """Adaptive mixture of smoothed experts using discounted historical likelihoods."""

    def __init__(self, history: Sequence[int] | None = None, *, learning_rate: float = 0.6,
                 weight_decay: float = 0.995):
        if not 0 < learning_rate <= 2:
            raise ValueError("learning_rate must be in (0, 2]")
        if not 0 < weight_decay <= 1:
            raise ValueError("weight_decay must be in (0, 1]")
        self.history: list[int] = []
        self.log_weights = {name: 0.0 for name in EXPERTS}
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        for x in history or []:
            self.learn(x)

    def _expert_distributions(self) -> dict[str, list[float]]:
        h = self.history
        result = {"uniform": [1.0 / N] * N}
        counts = Counter(h)
        length = len(h)

        # Dirichlet smoothing makes every number possible.
        result["frequency"] = normalize([counts[x] + 2.0 for x in range(N)])

        # Recent choices carry decaying mass, with a uniform safety floor.
        recent = [0.5] * N
        for age, x in enumerate(reversed(h[-100:])):
            recent[x] += exp(-age / 15)
        result["recency"] = normalize(recent)

        # Digit and tens-group preferences, with joint smoothing.
        last_digit = Counter(x % 10 for x in h)
        decade = Counter(min(x // 10, 9) for x in h)
        result["digits"] = normalize([
            1.0 + 2.0 * last_digit[x % 10] / (length + 10)
            + 2.0 * decade[min(x // 10, 9)] / (length + 10)
            for x in range(N)
        ])

        # Exact previous-choice Markov transitions backed off to nearby states.
        transitions = [0.5] * N
        if h:
            last = h[-1]
            for prev, nxt in zip(h, h[1:]):
                closeness = exp(-abs(prev - last) / 12)
                transitions[nxt] += closeness
        result["transition"] = normalize(transitions)

        # Learn the person's preferred magnitude of jumps, not just direction.
        jumps = Counter(abs(b - a) for a, b in zip(h, h[1:]))
        if h:
            last = h[-1]
            result["distance"] = normalize([
                1.0 + sum(count * exp(-abs(abs(x - last) - gap) / 4)
                          for gap, count in jumps.items()) / (length + 10)
                for x in range(N)
            ])
        else:
            result["distance"] = [1.0 / N] * N

        # Favors numbers not recently selected. Competes with recency expert.
        banned = set(h[-5:])
        result["nonrepeat"] = normalize([0.5 if x in banned else 1.0 for x in range(N)])
        return result

    def expert_weights(self) -> dict[str, float]:
        max_log = max(self.log_weights.values())
        raw = {name: exp(weight - max_log) for name, weight in self.log_weights.items()}
        total = sum(raw.values())
        return {name: weight / total for name, weight in raw.items()}

    def predict(self, top_k: int = 5) -> Prediction:
        if not isinstance(top_k, int) or not 1 <= top_k <= N:
            raise ValueError("top_k must be between 1 and 101")
        experts = self._expert_distributions()
        weights = self.expert_weights()
        probabilities = [sum(weights[name] * experts[name][x] for name in EXPERTS)
                         for x in range(N)]
        ranked = sorted(range(N), key=lambda x: (-probabilities[x], x))
        return Prediction(ranked[0], tuple(ranked[:top_k]), tuple(probabilities), weights)

    def learn(self, actual: int) -> None:
        """Update each expert using its pre-observation probability for `actual`."""
        validate_number(actual)
        experts = self._expert_distributions()
        for name in EXPERTS:
            self.log_weights[name] = (self.weight_decay * self.log_weights[name]
                                      + self.learning_rate * log(experts[name][actual]))
        # Renormalization prevents log-score drift over long-running sessions.
        peak = max(self.log_weights.values())
        for name in EXPERTS:
            self.log_weights[name] -= peak
        self.history.append(actual)
