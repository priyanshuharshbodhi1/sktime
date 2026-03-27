"""Implements Chronos-2 forecaster."""

__author__ = ["priyanshuharshbodhi1"]

__all__ = ["Chronos2Forecaster"]

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton

if _check_soft_dependencies("torch", severity="none"):
    import torch
else:

    class torch:
        """Dummy class if torch is unavailable."""

        bfloat16 = None


if _check_soft_dependencies("transformers", severity="none"):
    import transformers
else:

    class transformers:
        """Dummy class if transformers is unavailable."""

        @staticmethod
        def set_seed(seed):
            """Set random seed."""


class Chronos2Forecaster(BaseForecaster):
    """Interface to the Chronos-2 Zero-Shot Forecaster by Amazon Research.

    Chronos-2 is a pretrained encoder-only time series foundation model
    developed by Amazon for zero-shot forecasting. It supports univariate,
    multivariate, and covariate-informed forecasting tasks within a single
    architecture. The official code and technical report are given at [1]_ and [2]_.

    Unlike Chronos (v1), Chronos-2 natively handles multivariate targets,
    past-only covariates, and known-future covariates via a group attention
    mechanism described in [2]_.

    Parameters
    ----------
    model_path : str, default="amazon/chronos-2"
        Path to the Chronos-2 HuggingFace model.

    config : dict, optional, default=None
        Configuration overrides. Supported keys:

        - "limit_prediction_length" : bool, default=False
            If True, raises an error when prediction_length exceeds the model's
            maximum prediction length.
        - "torch_dtype" : torch.dtype, default=torch.bfloat16
            Data type for model weights and operations.
        - "device_map" : str, default="cpu"
            Device for inference, e.g., "cpu", "cuda", or "mps".
        - "batch_size" : int, default=256
            Number of time series per batch during prediction.
        - "context_length" : int or None, default=None
            Maximum context length for inference. Defaults to model's
            context length (8192 for amazon/chronos-2).
        - "cross_learning" : bool, default=False
            If True, enables cross-learning across all input series in a batch,
            sharing information via the group attention mechanism.

    seed : int or None, optional, default=None
        Random seed for reproducibility.

    ignore_deps : bool, optional, default=False
        If True, dependency checks are skipped.

    Attributes
    ----------
    model_pipeline : Chronos2Pipeline
        The underlying model pipeline used for forecasting.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting
    .. [2] Abdul Fatir Ansari and others (2025).
       Chronos-2: Towards a Universal, General-Purpose Forecasting Foundation Model.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.chronos2 import Chronos2Forecaster
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> forecaster = Chronos2Forecaster("amazon/chronos-2")  # doctest: +SKIP
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": ["priyanshuharshbodhi1"],
        "maintainers": ["priyanshuharshbodhi1"],
        "python_dependencies": ["chronos"],
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
        "capability:insample": False,
        "capability:global_forecasting": True,
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    }

    _default_config = {
        "limit_prediction_length": False,
        "torch_dtype": torch.bfloat16,
        "device_map": "cpu",
        "batch_size": 256,
        "context_length": None,
        "cross_learning": False,
    }

    def __init__(
        self,
        model_path: str = "amazon/chronos-2",
        config: dict = None,
        seed: int | None = None,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.seed = seed
        self._seed = np.random.randint(0, 2**31) if seed is None else seed
        self.config = config
        self.ignore_deps = ignore_deps

        self._config = self._default_config.copy()
        if config is not None:
            self._config.update(config)

        self.model_pipeline = None

        if ignore_deps:
            self.set_tags(python_dependencies=[])

        super().__init__()

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable model pipeline."""
        state = self.__dict__.copy()
        if hasattr(self, "model_pipeline"):
            state["model_pipeline"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    def _get_pipeline_kwargs(self):
        return {
            "pretrained_model_name_or_path": self.model_path,
            "torch_dtype": self._config["torch_dtype"],
            "device_map": self._config["device_map"],
        }

    def _get_unique_key(self):
        kwargs = self._get_pipeline_kwargs()
        return str(sorted(kwargs.items()))

    def _load_pipeline(self):
        return _CachedChronos2(
            key=self._get_unique_key(),
            chronos2_kwargs=self._get_pipeline_kwargs(),
        ).load_from_checkpoint()

    def _ensure_model_pipeline_loaded(self):
        """Reload model pipeline if needed after unpickling."""
        if not hasattr(self, "model_pipeline") or self.model_pipeline is None:
            if hasattr(self, "_is_fitted") and self._is_fitted:
                self.model_pipeline = self._load_pipeline()

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series.
        X : pd.DataFrame, optional
            Past exogenous covariates.
        fh : ForecastingHorizon, optional

        Returns
        -------
        self
        """
        self.model_pipeline = self._load_pipeline()
        return self


@_multiton
class _CachedChronos2:
    """Cached Chronos-2 model to ensure only one instance exists in memory.

    Chronos-2 is a zero-shot model and immutable, so sharing the same instance
    across multiple uses has no side effects.
    """

    def __init__(self, key, chronos2_kwargs):
        self.key = key
        self.chronos2_kwargs = chronos2_kwargs
        self.model_pipeline = None

    def load_from_checkpoint(self):
        """Load Chronos-2 pipeline from pretrained checkpoint."""
        if self.model_pipeline is not None:
            return self.model_pipeline

        from chronos import Chronos2Pipeline

        self.model_pipeline = Chronos2Pipeline.from_pretrained(**self.chronos2_kwargs)
        return self.model_pipeline
