"""
SĀRTHI V3-7 — High-Resolution Decision Latency Instrumentation.

Provides non-intrusive, secret-free, sub-millisecond precision latency tracking
and benchmark statistical analysis for the SĀRTHI V3 cognitive-to-physical pipeline.

Latency Markers:
  - tau_nemotron: Human instruction -> Nemotron task understanding response
  - tau_context: WorldState -> DecisionContext synthesis
  - tau_jev: DecisionQuestion -> Local Laya bounded JevDecision
  - tau_validation: CandidateAction -> SarthiDecisionEngine deterministic evaluation
  - tau_ik: Decision -> Differential IK solve duration
  - tau_sim: Simulation physics stepping / joint trajectory execution
  - tau_verify: Physical outcome state verification
  - tau_total: Outer end-to-end transaction wall-clock duration (measured independently)

Strict Guarantees:
  - Uses monotonic, high-resolution time (time.perf_counter).
  - Strictly non-authoritative: cannot command robot or bypass validation.
  - Zero secrets: no API keys, tokens, or credentials stored or logged.
  - Mathematically sound statistics: count, min, max, mean, median, p95.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import logging
import math
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

LATENCY_MARKER_NAMES = (
    "tau_nemotron_ms",
    "tau_context_ms",
    "tau_jev_ms",
    "tau_validation_ms",
    "tau_ik_ms",
    "tau_sim_ms",
    "tau_verify_ms",
    "tau_total_ms",
)


@dataclass
class V3LatencyBreakdown:
    """Sub-millisecond latency profile for a V3 execution transaction or step."""
    tau_nemotron_ms: float = 0.0
    tau_context_ms: float = 0.0
    tau_jev_ms: float = 0.0
    tau_validation_ms: float = 0.0
    tau_ik_ms: float = 0.0
    tau_sim_ms: float = 0.0
    tau_verify_ms: float = 0.0
    tau_total_ms: float = 0.0

    def __post_init__(self) -> None:
        # Enforce non-negative invariant across all latency markers
        for marker in LATENCY_MARKER_NAMES:
            val = getattr(self, marker, 0.0)
            if val is None or math.isnan(val) or val < 0.0:
                setattr(self, marker, 0.0)

    def to_dict(self) -> Dict[str, float]:
        """Serializes latency breakdown to a dictionary with 3-decimal precision."""
        return {
            "tau_nemotron_ms": round(float(self.tau_nemotron_ms), 3),
            "tau_context_ms": round(float(self.tau_context_ms), 3),
            "tau_jev_ms": round(float(self.tau_jev_ms), 3),
            "tau_validation_ms": round(float(self.tau_validation_ms), 3),
            "tau_ik_ms": round(float(self.tau_ik_ms), 3),
            "tau_sim_ms": round(float(self.tau_sim_ms), 3),
            "tau_verify_ms": round(float(self.tau_verify_ms), 3),
            "tau_total_ms": round(float(self.tau_total_ms), 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "V3LatencyBreakdown":
        """Constructs an instance safely from a dictionary."""
        kwargs = {}
        for m in LATENCY_MARKER_NAMES:
            kwargs[m] = float(data.get(m, 0.0) or 0.0)
        return cls(**kwargs)


class LatencyTracker:
    """
    High-resolution monotonic timer tracking each stage of the V3 decision cycle.
    Independent, thread-safe for sequential calls, zero-secret.
    """

    def __init__(self) -> None:
        self._timers: Dict[str, float] = {}
        self._measurements: Dict[str, float] = {m: 0.0 for m in LATENCY_MARKER_NAMES}
        self._total_start: Optional[float] = None

    def start_total(self) -> None:
        """Starts the outer transaction timer for tau_total_ms."""
        self._total_start = time.perf_counter()

    def stop_total(self) -> float:
        """Stops the outer transaction timer and records tau_total_ms independently."""
        if self._total_start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._total_start) * 1000.0
        self._measurements["tau_total_ms"] = max(0.0, elapsed_ms)
        self._total_start = None
        return self._measurements["tau_total_ms"]

    def start(self, marker: str) -> None:
        """Starts timing a specific latency marker."""
        self._timers[marker] = time.perf_counter()

    def stop(self, marker: str) -> float:
        """Stops timing a specific latency marker and adds to its measurement."""
        start_t = self._timers.pop(marker, None)
        if start_t is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        curr = self._measurements.get(marker, 0.0)
        self._measurements[marker] = max(0.0, curr + elapsed_ms)
        return elapsed_ms

    def record(self, marker: str, duration_ms: float) -> None:
        """Directly records or updates an explicit duration measurement."""
        val = max(0.0, float(duration_ms)) if duration_ms is not None else 0.0
        self._measurements[marker] = val

    def accumulate(self, marker: str, duration_ms: float) -> None:
        """Accumulates duration onto an existing measurement (e.g. multi-step sum)."""
        val = max(0.0, float(duration_ms)) if duration_ms is not None else 0.0
        curr = self._measurements.get(marker, 0.0)
        self._measurements[marker] = curr + val

    @contextmanager
    def measure(self, marker: str) -> Iterator[None]:
        """Context manager for timing a block with sub-millisecond precision."""
        self.start(marker)
        try:
            yield
        finally:
            self.stop(marker)

    def get_breakdown(self) -> V3LatencyBreakdown:
        """Returns the current snapshot as a validated V3LatencyBreakdown."""
        return V3LatencyBreakdown(
            tau_nemotron_ms=self._measurements.get("tau_nemotron_ms", 0.0),
            tau_context_ms=self._measurements.get("tau_context_ms", 0.0),
            tau_jev_ms=self._measurements.get("tau_jev_ms", 0.0),
            tau_validation_ms=self._measurements.get("tau_validation_ms", 0.0),
            tau_ik_ms=self._measurements.get("tau_ik_ms", 0.0),
            tau_sim_ms=self._measurements.get("tau_sim_ms", 0.0),
            tau_verify_ms=self._measurements.get("tau_verify_ms", 0.0),
            tau_total_ms=self._measurements.get("tau_total_ms", 0.0),
        )

    def reset(self) -> None:
        """Resets all timers and stored measurements."""
        self._timers.clear()
        self._measurements = {m: 0.0 for m in LATENCY_MARKER_NAMES}
        self._total_start = None


# ---------------------------------------------------------------------------
# Benchmark Statistical Analysis
# ---------------------------------------------------------------------------

def compute_metric_statistics(values: Sequence[float]) -> Dict[str, float]:
    """
    Computes count, min, max, mean, median, and p95 for a sequence of numeric latencies.
    Returns 0.0 for all statistics if values is empty.
    """
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }

    clean_vals = [max(0.0, float(v)) for v in values if v is not None and not math.isnan(v)]
    if not clean_vals:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }

    arr = np.array(clean_vals, dtype=np.float64)
    return {
        "count": float(len(clean_vals)),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
    }


def aggregate_latency_samples(
    samples: Sequence[Union[V3LatencyBreakdown, Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """
    Calculates summary statistics (count, min, max, mean, median, p95)
    for each of the 8 standard V3 latency markers across all provided samples.
    """
    marker_series: Dict[str, List[float]] = {m: [] for m in LATENCY_MARKER_NAMES}

    for s in samples:
        if isinstance(s, V3LatencyBreakdown):
            d = s.to_dict()
        elif isinstance(s, dict):
            d = s
        else:
            continue

        for m in LATENCY_MARKER_NAMES:
            if m in d:
                marker_series[m].append(float(d[m]))

    stats: Dict[str, Dict[str, float]] = {}
    for m in LATENCY_MARKER_NAMES:
        stats[m] = compute_metric_statistics(marker_series[m])

    return stats


@dataclass
class BenchmarkSuiteResult:
    """Comprehensive benchmark execution report container."""
    benchmark_name: str
    num_runs: int
    raw_runs: List[Dict[str, float]]
    statistics: Dict[str, Dict[str, float]]
    timestamp_iso: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "num_runs": self.num_runs,
            "statistics": self.statistics,
            "raw_runs": self.raw_runs,
            "timestamp_iso": self.timestamp_iso,
            "metadata": self.metadata,
        }
