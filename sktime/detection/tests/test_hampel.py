"""Tests for HampelFilter detector."""

__author__ = ["priyanshuharshbodhi1"]

import pandas as pd
import pytest

from sktime.detection.hampel import HampelFilter
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(HampelFilter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,y_expected",
    [
        (
            pd.DataFrame([0, 0, 0, 100, 0, 0, 0, 0, 100, 0, 0]),
            pd.DataFrame({"ilocs": [3, 8]}),
        ),
    ],
)
def test_predict(X, y_expected):
    detector = HampelFilter(window_length=3)
    detector.fit(X)
    y_actual = detector.predict(X)
    pd.testing.assert_frame_equal(
        y_actual.reset_index(drop=True),
        y_expected.reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.skipif(
    not run_test_for_class(HampelFilter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_does_not_mutate_input():
    X = pd.DataFrame([0, 0, 0, 100, 0, 0, 0, 0, 100, 0, 0])
    X_original = X.copy(deep=True)

    detector = HampelFilter(window_length=3)
    detector.fit_transform(X)

    pd.testing.assert_frame_equal(X, X_original)
