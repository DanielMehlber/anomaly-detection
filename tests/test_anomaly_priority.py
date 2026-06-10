"""Tests for symptom suppression and spatial conflict resolution."""

from __future__ import annotations

from src.anomaly_priority import resolve_results_by_priority
from src.models.events import AnomalyClass, AnomalyType, FilterResult


def _result(anomaly_type: AnomalyType, metric: float, threshold: float) -> FilterResult:
    return FilterResult(
        filter_name="test",
        anomaly_class=AnomalyClass.SPATIAL,
        anomaly_type=anomaly_type,
        metric_value=metric,
        threshold=threshold,
        is_anomaly=True,
    )


def test_black_screen_suppresses_spatial_symptoms():
    rules = {
        "black_screen": ["missing_element", "geometric_distortion"],
    }
    results = [
        FilterResult(
            filter_name="temporal",
            anomaly_class=AnomalyClass.TEMPORAL,
            anomaly_type=AnomalyType.BLACK_SCREEN,
            metric_value=1.0,
            threshold=10.0,
            is_anomaly=True,
        ),
        _result(AnomalyType.MISSING_ELEMENT, 0.9, 0.3),
    ]

    resolved = resolve_results_by_priority(results, rules)
    missing = next(r for r in resolved if r.anomaly_type == AnomalyType.MISSING_ELEMENT)
    assert not missing.is_anomaly
    assert missing.metadata.get("suppressed_by") == "black_screen"


def test_spatial_conflict_keeps_stronger_distortion_signal():
    rules: dict[str, list[str]] = {}
    results = [
        _result(AnomalyType.MISSING_ELEMENT, 0.4, 0.3),
        _result(AnomalyType.GEOMETRIC_DISTORTION, 12.0, 5.0),
    ]

    resolved = resolve_results_by_priority(results, rules)
    missing = next(r for r in resolved if r.anomaly_type == AnomalyType.MISSING_ELEMENT)
    distortion = next(r for r in resolved if r.anomaly_type == AnomalyType.GEOMETRIC_DISTORTION)

    assert not missing.is_anomaly
    assert distortion.is_anomaly


def test_spatial_conflict_keeps_missing_when_distortion_is_weak():
    rules: dict[str, list[str]] = {}
    results = [
        _result(AnomalyType.MISSING_ELEMENT, 0.4, 0.05),
        _result(AnomalyType.GEOMETRIC_DISTORTION, 8.0, 5.0),
    ]

    resolved = resolve_results_by_priority(results, rules)
    missing = next(r for r in resolved if r.anomaly_type == AnomalyType.MISSING_ELEMENT)
    distortion = next(r for r in resolved if r.anomaly_type == AnomalyType.GEOMETRIC_DISTORTION)

    assert missing.is_anomaly
    assert not distortion.is_anomaly
