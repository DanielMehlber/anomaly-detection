"""
Anomaly priority and symptom suppression.

Not every co-occurring filter result should become an event. Some detections
are downstream symptoms of another root cause (e.g. keypoint loss during a
black screen). Suppression rules encode those relationships without forcing
unrelated anomaly types (such as flicker vs. spatial distortion) to compete.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from src.models.events import AnomalyType, FilterResult

# Maps a dominant anomaly to symptom types that must not open parallel events.
DEFAULT_SUPPRESSION_RULES: dict[str, list[str]] = {
    AnomalyType.BLACK_SCREEN.value: [
        AnomalyType.MISSING_ELEMENT.value,
        AnomalyType.KEYPOINT_LOSS.value,
        AnomalyType.GEOMETRIC_DISTORTION.value,
    ],
    AnomalyType.FLICKER.value: [
        AnomalyType.MISSING_ELEMENT.value,
        AnomalyType.KEYPOINT_LOSS.value,
        AnomalyType.GEOMETRIC_DISTORTION.value,
    ],
    AnomalyType.GEOMETRIC_DISTORTION.value: [
        AnomalyType.KEYPOINT_LOSS.value,
    ],
}

# Kept for configuration backwards compatibility; no longer used as a single winner list.
DEFAULT_ANOMALY_PRIORITY: tuple[str, ...] = (
    AnomalyType.BLACK_SCREEN.value,
    AnomalyType.GEOMETRIC_DISTORTION.value,
    AnomalyType.KEYPOINT_LOSS.value,
    AnomalyType.FLICKER.value,
)


def _normalize_rules(rules: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return a copy of ``rules`` with string keys and list values."""
    return {str(key): list(value) for key, value in rules.items()}


def suppressed_types_for_dominants(
    dominant_types: Iterable[AnomalyType],
    suppression_rules: dict[str, list[str]],
) -> set[AnomalyType]:
    """Collect all anomaly types that should be suppressed given active dominants."""
    rules = _normalize_rules(suppression_rules)
    suppressed: set[AnomalyType] = set()

    for dominant in dominant_types:
        for victim_name in rules.get(dominant.value, []):
            suppressed.add(AnomalyType(victim_name))

    return suppressed


def find_suppressor(
    victim: AnomalyType,
    results: list[FilterResult],
    suppression_rules: dict[str, list[str]],
) -> AnomalyType | None:
    """Return the dominant anomaly that suppresses ``victim`` on this frame."""
    rules = _normalize_rules(suppression_rules)
    for result in results:
        if not result.is_anomaly:
            continue
        if victim.value in rules.get(result.anomaly_type.value, []):
            return result.anomaly_type
    return None


def resolve_results_by_priority(
    results: list[FilterResult],
    suppression_rules: dict[str, list[str]] | Iterable[str],
) -> list[FilterResult]:
    """
    Apply symptom-suppression rules to a frame's filter results.

    ``suppression_rules`` accepts either a rule dictionary or a legacy priority
    list. Legacy lists are converted to pairwise suppression chains.
    """
    if isinstance(suppression_rules, dict):
        rules = _normalize_rules(suppression_rules)
    else:
        rules = _legacy_priority_to_rules(suppression_rules)

    dominants = [result.anomaly_type for result in results if result.is_anomaly]
    suppressed = suppressed_types_for_dominants(dominants, rules)

    resolved: list[FilterResult] = []
    for result in results:
        if suppressed and result.is_anomaly and result.anomaly_type in suppressed:
            metadata = dict(result.metadata)
            suppressor = find_suppressor(result.anomaly_type, results, rules)
            if suppressor is not None:
                metadata["suppressed_by"] = suppressor.value
            resolved.append(replace(result, is_anomaly=False, metadata=metadata))
        else:
            resolved.append(result)

    return _resolve_spatial_conflicts(resolved)


def _resolve_spatial_conflicts(results: list[FilterResult]) -> list[FilterResult]:
    """
    Choose between missing-element and distortion when both fire on one frame.

    The filter with the stronger normalized exceedance (metric / threshold) is
    kept; the other is treated as a secondary symptom.
    """
    missing = next(
        (result for result in results if result.anomaly_type == AnomalyType.MISSING_ELEMENT and result.is_anomaly),
        None,
    )
    distortion = next(
        (
            result
            for result in results
            if result.anomaly_type == AnomalyType.GEOMETRIC_DISTORTION and result.is_anomaly
        ),
        None,
    )
    if missing is None or distortion is None:
        return results

    missing_score = missing.metric_value / max(missing.threshold, 1e-6)
    distortion_score = distortion.metric_value / max(distortion.threshold, 1e-6)
    if distortion_score >= 2.0:
        loser = AnomalyType.MISSING_ELEMENT
        winner = AnomalyType.GEOMETRIC_DISTORTION
    elif missing_score > distortion_score:
        loser = AnomalyType.GEOMETRIC_DISTORTION
        winner = AnomalyType.MISSING_ELEMENT
    else:
        loser = AnomalyType.MISSING_ELEMENT
        winner = AnomalyType.GEOMETRIC_DISTORTION

    resolved: list[FilterResult] = []
    for result in results:
        if result.anomaly_type == loser and result.is_anomaly:
            metadata = dict(result.metadata)
            metadata["suppressed_by"] = winner.value
            resolved.append(replace(result, is_anomaly=False, metadata=metadata))
        else:
            resolved.append(result)
    return resolved


def is_suppressed_by_dominant(
    candidate: AnomalyType,
    dominant: AnomalyType,
    suppression_rules: dict[str, list[str]],
) -> bool:
    """Return True when ``dominant`` should suppress ``candidate`` events."""
    rules = _normalize_rules(suppression_rules)
    return candidate.value in rules.get(dominant.value, [])


def priority_rank(anomaly_type: AnomalyType, priority_order: Iterable[str]) -> int:
    """Legacy helper retained for metadata; lower rank means higher priority."""
    ranks = {name: index for index, name in enumerate(priority_order)}
    return ranks.get(anomaly_type.value, len(ranks))


def _legacy_priority_to_rules(priority_order: Iterable[str]) -> dict[str, list[str]]:
    """Convert the old single-winner priority list into a suppression chain."""
    order = list(priority_order)
    rules: dict[str, list[str]] = {}
    for index, dominant in enumerate(order):
        rules[dominant] = order[index + 1 :]
    return rules
