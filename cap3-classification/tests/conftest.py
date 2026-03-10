"""Shared fixtures for project tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def toy_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pixel_00": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "pixel_01": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "pixel_02": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5],
            "pixel_03": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5],
            "target": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
        }
    )
