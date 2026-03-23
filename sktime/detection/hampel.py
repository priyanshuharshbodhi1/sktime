"""Hampel filter for outlier detection."""

__author__ = ["priyanshuharshbodhi1"]

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector


class HampelFilter(BaseDetector):
    """Outlier detector based on the Hampel filter.

    Uses a sliding window centered on each point to compute the local median
    and median absolute deviation (MAD). Points where the deviation from the
    median exceeds ``n_sigma * k * MAD`` are flagged as outliers.

    Parameters
    ----------
    window_length : int, default=10
        Half-width of the sliding window. The window around each point spans
        ``window_length`` positions in total (``window_length // 2`` on each side).
    n_sigma : int, default=3
        Number of scaled MADs a point must exceed to be flagged as an outlier.
    k : float, default=1.4826
        Scale factor for the MAD. For Gaussian data, 1.4826 makes ``k * MAD``
        a consistent estimator of the standard deviation.

    References
    ----------
    .. [1] Hampel F. R., "The influence curve and its role in robust estimation",
       Journal of the American Statistical Association, 69, 382-393, 1974
    """

    _tags = {
        "authors": "priyanshuharshbodhi1",
        "maintainers": "priyanshuharshbodhi1",
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": False,
        "fit_is_empty": True,
        "tests:core": True,
    }

    def __init__(self, window_length=10, n_sigma=3, k=1.4826):
        self.window_length = window_length
        self.n_sigma = n_sigma
        self.k = k
        super().__init__()

    def _predict(self, X):
        """Find outlier positions in X.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.Series of int
            iloc positions of detected outliers.
        """
        Z = X.iloc[:, 0].to_numpy(dtype=float)
        n = len(Z)
        half = self.window_length // 2
        outliers = []

        for i in range(n):
            if np.isnan(Z[i]):
                continue
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window = Z[start:end]
            window = window[~np.isnan(window)]
            if len(window) == 0:
                continue
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            if np.abs(Z[i] - median) > self.n_sigma * self.k * mad:
                outliers.append(i)

        if not outliers:
            return self._empty_sparse()
        return pd.Series(outliers)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"window_length": 3}, {"window_length": 5, "n_sigma": 2}]
