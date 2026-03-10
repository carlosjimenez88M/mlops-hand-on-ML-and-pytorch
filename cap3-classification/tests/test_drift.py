"""Tests for drift detection."""

from __future__ import annotations

import pandas as pd

from cap3_classification.monitoring.drift import generate_drift_report


def test_drift_report_flags_shift():
    reference_df = pd.DataFrame({"pixel_00": [0, 0, 1, 1], "target": [0, 1, 0, 1]})
    current_df = pd.DataFrame({"pixel_00": [9, 9, 10, 10], "target": [0, 1, 0, 1]})
    report = generate_drift_report(
        reference_df=reference_df,
        current_df=current_df,
        target_column="target",
        max_psi=0.2,
    )
    assert report.drift_detected is True
