"""

Registration and matching of stable control elements from calibration.



Control elements are reference ORB features that were consistently visible

throughout the calibration window. Only these registered positions are used

for missing-element detection and highlighting — not every texture keypoint.

"""



from __future__ import annotations



import cv2

import numpy as np



# Minimum share of calibration frames in which a reference keypoint must match.

_DEFAULT_STABILITY_RATE = 0.75

# Minimum pixel distance between two registered control elements.

_DEFAULT_MIN_DISTANCE = 40.0

# Maximum number of control elements to register (avoids grid noise).

_DEFAULT_MAX_ELEMENTS = 5

# Radius used to map a registered position to a reference keypoint.

_DEFAULT_REGISTRATION_RADIUS = 35.0

# Tight radius for deciding whether a control is visible in the current frame.

_DEFAULT_PRESENCE_RADIUS = 14.0





def select_control_elements(

    reference_keypoints: list[cv2.KeyPoint],

    per_frame_matched_indices: list[set[int]],

    *,

    stability_rate: float = _DEFAULT_STABILITY_RATE,

    min_distance: float = _DEFAULT_MIN_DISTANCE,

    max_elements: int = _DEFAULT_MAX_ELEMENTS,

) -> list[tuple[float, float]]:

    """

    Pick stable, spatially separated reference positions as control elements.



    Args:

        reference_keypoints: ORB keypoints from the calibration reference frame.

        per_frame_matched_indices: For each calibration frame, ref indices that matched.



    Returns:

        Registered control element positions as (x, y) floats.

    """

    if not reference_keypoints or not per_frame_matched_indices:

        return []



    frame_count = len(per_frame_matched_indices)

    match_counts: dict[int, int] = {}

    for matched in per_frame_matched_indices:

        for index in matched:

            match_counts[index] = match_counts.get(index, 0) + 1



    candidates: list[tuple[float, int]] = []

    for index, count in match_counts.items():

        if index >= len(reference_keypoints):

            continue

        if count / frame_count <= stability_rate:

            continue

        response = float(reference_keypoints[index].response)

        candidates.append((response, index))



    candidates.sort(reverse=True)

    selected: list[tuple[float, float]] = []



    for _, index in candidates:

        x, y = reference_keypoints[index].pt

        if _is_too_close((x, y), selected, min_distance):

            continue

        selected.append((float(x), float(y)))

        if len(selected) >= max_elements:

            break



    return selected





def control_element_loss_samples(

    control_positions: list[tuple[float, float]],

    per_frame_keypoints: list[list[cv2.KeyPoint]],

    *,

    presence_radius: float = _DEFAULT_PRESENCE_RADIUS,

) -> list[float]:

    """Compute per-frame missing ratios for registered controls during calibration."""

    if not control_positions:

        return []



    samples: list[float] = []

    for keypoints in per_frame_keypoints:

        missing = detect_missing_controls(

            control_positions,

            keypoints,

            presence_radius=presence_radius,

        )

        samples.append(len(missing) / len(control_positions))

    return samples





def _indices_for_positions(

    reference_keypoints: list[cv2.KeyPoint],

    control_positions: list[tuple[float, float]],

    registration_radius: float = _DEFAULT_REGISTRATION_RADIUS,

) -> list[int]:

    """Map each control position to the nearest reference keypoint index."""

    indices: list[int] = []

    for cx, cy in control_positions:

        best_index = -1

        best_distance = registration_radius

        for index, keypoint in enumerate(reference_keypoints):

            distance = float(np.hypot(keypoint.pt[0] - cx, keypoint.pt[1] - cy))

            if distance <= best_distance:

                best_distance = distance

                best_index = index

        if best_index >= 0:

            indices.append(best_index)

    return indices





def detect_missing_controls(

    control_positions: list[tuple[float, float]],

    current_keypoints: list[cv2.KeyPoint],

    *,

    presence_radius: float = _DEFAULT_PRESENCE_RADIUS,

) -> list[tuple[int, int]]:

    """

    Return integer (x, y) positions of control elements absent from the frame.



    A control is considered present when any current ORB keypoint lies within

    ``presence_radius`` pixels of the registered position.

    """

    if not control_positions:

        return []



    missing: list[tuple[int, int]] = []

    for cx, cy in control_positions:

        found = False

        for keypoint in current_keypoints:

            if np.hypot(keypoint.pt[0] - cx, keypoint.pt[1] - cy) <= presence_radius:

                found = True

                break

        if not found:

            missing.append((int(round(cx)), int(round(cy))))



    return missing





def control_missing_ratio(

    control_positions: list[tuple[float, float]],

    current_keypoints: list[cv2.KeyPoint],

    *,

    presence_radius: float = _DEFAULT_PRESENCE_RADIUS,

) -> float:

    """Fraction of registered control elements missing from the current frame."""

    if not control_positions:

        return 0.0

    missing = detect_missing_controls(

        control_positions,

        current_keypoints,

        presence_radius=presence_radius,

    )

    return len(missing) / len(control_positions)





def _is_too_close(

    point: tuple[float, float],

    selected: list[tuple[float, float]],

    min_distance: float,

) -> bool:

    return any(np.hypot(point[0] - other[0], point[1] - other[1]) < min_distance for other in selected)


