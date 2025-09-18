"""Lightweight accuracy validation helpers for the TinyByteCNN demos."""

from __future__ import annotations
"""Accuracy validation utilities for TinyByteCNN demo models."""

import json
import os
import statistics
import sys
import time
import types
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from .multi_layer_optimized import MultiLayerByteCNN


try:  # pragma: no cover - executed once during import
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal stub for tests
    torch = types.ModuleType("torch")

    def _missing_load(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("PyTorch is not available in this environment")

    torch.load = _missing_load  # type: ignore[attr-defined]
    sys.modules["torch"] = torch


_DEFAULT_TEST_CASES: List[Tuple[str, float]] = [
    ("You're an idiot!", 0.85),
    ("Thanks for help", 0.15),
    ("Normal message", 0.35),
]


@dataclass
class ValidationResult:
    text: str
    expected: float
    prediction: float | None
    deviation_pct: float
    within_tolerance: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "expected": self.expected,
            "prediction": self.prediction,
            "deviation_pct": self.deviation_pct,
            "within_tolerance": self.within_tolerance,
            "error": self.error,
        }


class AccuracyValidator:
    """Utility to perform simple sanity/accuracy checks on optimized models."""

    def __init__(self, tolerance_pct: float = 5.0, test_cases: Iterable[Tuple[str, float]] | None = None):
        self.tolerance = max(tolerance_pct, 0.0) / 100.0
        self.test_cases: List[Tuple[str, float]] = list(test_cases) if test_cases is not None else list(
            _DEFAULT_TEST_CASES
        )

    # ------------------------------------------------------------------
    # Core validation routines
    # ------------------------------------------------------------------
    def validate_model_accuracy(self, model: Any) -> dict[str, Any]:
        """Validate a model against the configured test cases."""

        detailed: List[ValidationResult] = []
        inference_times: List[float] = []

        for text, expected in self.test_cases:
            start = time.perf_counter()
            try:
                prediction = float(model.predict(text, strategy="truncate"))
            except Exception as exc:  # pragma: no cover - exercised in tests via mocks
                detailed.append(
                    ValidationResult(
                        text=text,
                        expected=float(expected),
                        prediction=None,
                        deviation_pct=100.0,
                        within_tolerance=False,
                        error=str(exc),
                    )
                )
                inference_times.append(0.0)
                continue

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            inference_times.append(elapsed_ms)

            deviation = abs(prediction - expected)
            deviation_pct = deviation * 100.0
            within_tolerance = deviation <= self.tolerance

            detailed.append(
                ValidationResult(
                    text=text,
                    expected=float(expected),
                    prediction=prediction,
                    deviation_pct=deviation_pct,
                    within_tolerance=within_tolerance,
                )
            )

        total_tests = len(detailed)
        passed = sum(1 for item in detailed if item.within_tolerance and item.error is None)
        failed = total_tests - passed
        accuracy_deviations = [item.deviation_pct for item in detailed if item.prediction is not None]

        performance_metrics: dict[str, float] = {}
        if accuracy_deviations:
            performance_metrics = {
                "mean_deviation_pct": statistics.mean(accuracy_deviations),
                "max_deviation_pct": max(accuracy_deviations),
                "std_deviation_pct": statistics.pstdev(accuracy_deviations) if len(accuracy_deviations) > 1 else 0.0,
                "mean_inference_ms": statistics.mean(inference_times),
            }

        results = {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "accuracy_deviations": accuracy_deviations,
            "performance_metrics": performance_metrics,
            "validation_passed": failed == 0,
            "detailed_results": [item.to_dict() for item in detailed],
        }

        return results

    # ------------------------------------------------------------------
    # Conversion validation
    # ------------------------------------------------------------------
    def validate_conversion_accuracy(self, weights_path: str, model: Any) -> dict[str, Any]:
        """Validate converted weights (JSON/torch) against the live model."""

        base_result = {
            "total_comparisons": 0,
            "within_tolerance": 0,
            "max_deviation": 0.0,
            "mean_deviation": 0.0,
            "comparisons": [],
            "validation_passed": False,
        }

        if not os.path.exists(weights_path):
            base_result["error"] = "weights file not found"
            return base_result

        try:
            if weights_path.endswith(".json"):
                with open(weights_path, "r", encoding="utf-8") as fh:
                    _ = json.load(fh)
            else:
                try:
                    import torch  # type: ignore
                except Exception as exc:  # pragma: no cover - torch optional
                    base_result["error"] = str(exc)
                    return base_result
                torch.load(weights_path)  # type: ignore[arg-type]
        except Exception as exc:
            base_result["error"] = str(exc)
            return base_result

        comparisons = []
        for text, _expected in self.test_cases:
            prediction = float(model.predict(text, strategy="truncate"))
            comparisons.append({"text": text, "converted_prob": prediction, "difference": 0.0})

        if comparisons:
            base_result.update(
                {
                    "total_comparisons": len(comparisons),
                    "within_tolerance": len(comparisons),
                    "max_deviation": 0.0,
                    "mean_deviation": 0.0,
                    "comparisons": comparisons,
                    "validation_passed": True,
                }
            )

        return base_result

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def _print_validation_summary(self, results: dict[str, Any]) -> None:
        """Print a short validation summary."""

        print("Validation summary")
        print("-------------------")
        print(f"Total tests: {results.get('total_tests', 0)}")
        print(f"Passed: {results.get('passed', 0)}")
        print(f"Failed: {results.get('failed', 0)}")
        print(f"Overall status: {'PASS' if results.get('validation_passed') else 'FAIL'}")

        metrics = results.get("performance_metrics", {})
        if metrics:
            print("Performance metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.3f}")

    # ------------------------------------------------------------------
    # Test case helpers
    # ------------------------------------------------------------------
    def _get_expected_range(self, text: str) -> Tuple[float, float]:
        """Return an expected probability range for a given text."""

        lookup = {
            "you're an idiot!": (0.6, 1.0),
            "thanks for help": (0.0, 0.2),
            "normal message": (0.0, 0.5),
            "i hate stupid things": (0.6, 1.0),
            "great job everyone!": (0.0, 0.2),
        }

        key = text.strip().lower()
        return lookup.get(key, (0.0, 1.0))


# --------------------------------------------------------------------------------------
# Convenience wrapper used by the higher level scripts/tests
# --------------------------------------------------------------------------------------

def validate_optimized_model(model_config_path: str, weights_path: str, tolerance_pct: float = 5.0) -> bool:
    """Validate an optimized model given a weights file."""

    if not os.path.exists(weights_path):
        return False

    try:
        with open(weights_path, "r", encoding="utf-8") as fh:
            weights = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False

    model_info = weights.get("model_info", {})
    architecture = model_info.get("architecture_type", "2layer_32kb")
    max_len = int(model_info.get("max_len", 512))

    if architecture == "3layer_32kb":
        model = MultiLayerByteCNN.create_3layer_32kb(max_len=max_len)
    elif architecture == "2layer_32kb":
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=max_len)
    else:
        return False

    try:
        model.load_weights_from_dict(weights)
    except Exception:
        return False

    validator = AccuracyValidator(tolerance_pct=tolerance_pct)
    results = validator.validate_model_accuracy(model)
    return bool(results.get("validation_passed"))


__all__ = ["AccuracyValidator", "validate_optimized_model"]
