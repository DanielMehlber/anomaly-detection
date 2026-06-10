"""Tests for registered control-element selection and matching."""



from __future__ import annotations



import cv2

import numpy as np



from src.calibration.control_elements import (

    detect_missing_controls,

    select_control_elements,

)





def _make_keypoints(points: list[tuple[float, float]]) -> list[cv2.KeyPoint]:

    return [cv2.KeyPoint(x, y, 20) for x, y in points]





def test_detect_missing_controls_finds_exact_keypoint_match():

    """Regression: distance must use (dx, dy), not (dx, y)."""

    keypoints = _make_keypoints([(100.0, 200.0), (300.0, 400.0)])

    controls = [(100.0, 200.0)]



    missing = detect_missing_controls(controls, keypoints)



    assert missing == []





def test_detect_missing_controls_ignores_distant_grid_keypoints():

    """Nearby but offset grid features must not mask a removed marker."""

    controls = [(100.0, 100.0)]

    keypoints = _make_keypoints([(118.0, 100.0)])



    missing = detect_missing_controls(controls, keypoints, presence_radius=14.0)



    assert missing == [(100, 100)]





def test_select_control_elements_prefers_stable_features():

    reference = _make_keypoints([(100, 100), (200, 200), (300, 300), (400, 400)])

    per_frame = [

        {0, 1, 2, 3},

        {0, 1, 2, 3},

        {0, 1, 2},

        {0, 1, 2, 3},

    ]



    controls = select_control_elements(reference, per_frame, stability_rate=0.75)



    assert len(controls) >= 3

    assert not any(abs(x - 400) < 5 and abs(y - 400) < 5 for x, y in controls)





def test_detect_missing_controls_uses_registered_positions_only(calibrated_baseline, partial_frame):

    from src.filters.missing_elements import MissingElementsFilter

    from src.models.frame import Frame



    missing_filter_positions = calibrated_baseline.control_element_positions

    assert len(missing_filter_positions) >= 3



    filt = MissingElementsFilter()

    filt.configure(calibrated_baseline, {"enabled": True})

    result = filt.process(Frame(image=partial_frame, timestamp_seconds=90.0, frame_index=2700))



    assert result.metric_value > 0.2

    assert result.is_anomaly

    assert result.highlight_regions



    for region in result.highlight_regions:

        assert region.center is not None

        nearest_registered = min(

            np.hypot(region.center[0] - rx, region.center[1] - ry)

            for rx, ry in missing_filter_positions

        )

        assert nearest_registered < 8.0, f"Marker at {region.center} is not on a registered control"





def test_missing_markers_not_placed_on_unregistered_grid(calibrated_baseline, partial_frame):

    """Missing markers must sit on registered controls, not random unmatched ORB points."""

    from src.filters.missing_elements import MissingElementsFilter

    from src.models.frame import Frame



    filt = MissingElementsFilter()

    filt.configure(calibrated_baseline, {"enabled": True})



    result = filt.process(

        Frame(image=partial_frame, timestamp_seconds=90.0, frame_index=2700),

    )



    assert result.is_anomaly

    assert result.highlight_regions



    registered = calibrated_baseline.control_element_positions

    for region in result.highlight_regions:

        assert region.center is not None

        cx, cy = region.center

        nearest = min(np.hypot(cx - rx, cy - ry) for rx, ry in registered)

        assert nearest < 8.0



    kept_corners = [(80, 80), (560, 80)]

    assert any(

        all(np.hypot(region.center[0] - px, region.center[1] - py) > 50 for px, py in kept_corners)

        for region in result.highlight_regions

        if region.center is not None

    )


