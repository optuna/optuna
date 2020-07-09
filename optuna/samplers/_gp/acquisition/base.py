import abc

import numpy as np

from optuna.samplers._gp.model import BaseModel


class BaseAcquisitionFunction(object, metaclass=abc.ABCMeta):
    """Base class for acquisition functions"""

    @abc.abstractmethod
    def compute_acq(self, x: np.ndarray, model: BaseModel) -> np.ndarray:
        """Computes the acquisition function value given input point x, based on the given model.

        Args:
            x:
                Input point. The shape is `(n, input_dim)`, where `n` is the number of points and
                `input_dim` is the input dimension.
            model:
                Current model.

        Returns:
            Acquisition function value. The shape is `(n, output_dim)`, where `n` is the number of
            points and `output_dim` is the output dimension.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def compute_grad(self, x: np.ndarray, model: BaseModel) -> np.ndarray:
        """Computes the gradient of the acquisition function given input point x, based on the given model.

        Args:
            x:
                Input point. The shape is `(n, input_dim)`, where `n` is the number of points and
                `input_dim` is the input dimension.
            model:
                Current model.

        Returns:
            Gradient value of the acquisition function. The shape is (n, input_dim, output_dim),
            where `n` is the number of points, `input_dim` is the input dimension, and `output_dim`
            is the output dimension.
        """
        raise NotImplementedError

    @staticmethod
    def _verify_input(x: np.ndarray, model: BaseModel) -> None:
        if x.ndim != 2:
            raise ValueError(
                "In `BaseAcquisitionFunction.compute_[acq, grad]`, `x.ndim` should be 2, "
                "but {} is specified.".format(x.ndim)
            )

        if x.shape[1] != model.input_dim:
            raise ValueError(
                "In `BaseAcquisitionFunction.compute_[acq, grad]`, "
                "`x.shape[1]` should be `model.input_dim = {}`, "
                "but `x.shape[1] = {}` is specified.".format(model.input_dim, x.shape[1])
            )

    @staticmethod
    def _verify_output_acq(y: np.ndarray, model: BaseModel) -> None:
        if y.ndim != 2:
            raise ValueError(
                "In `y =  BaseAcquisitionFunction.compute_acq()`, `y.ndim` should be 2, "
                "but {} is specified.".format(y.ndim)
            )

        if y.shape[1] != model.output_dim:
            raise ValueError(
                "In `y =  BaseAcquisitionFunction.compute_acq()`, "
                "`y.shape[1]` should be `model.output_dim = {}`, "
                "but `y.shape[1] = {} is specified".format(model.output_dim, y.shape[1])
            )

    @staticmethod
    def _verify_output_grad(dy: np.ndarray, model: BaseModel) -> None:
        if dy.ndim != 3:
            raise ValueError(
                "`dy.ndim` for `dy =  BaseAcquisitionFunction.compute_grad()` should be 3, "
                "but {} is specified.".format(dy.ndim)
            )

        if dy.shape[1] != model.input_dim:
            raise ValueError(
                "In `dy =  BaseAcquisitionFunction.compute_grad()`, "
                "`dy.shape[1]` should be `model.input_dim = {}`, "
                "but `dy.shape[1] = {} is specified".format(model.input_dim, dy.shape[1])
            )

        if dy.shape[2] != model.output_dim:
            raise ValueError(
                "In `dy =  BaseAcquisitionFunction.compute_grad()`, "
                "`dy.shape[2]` should be `model.output_dim = {}`, "
                "but `dy.shape[2] = {} is specified".format(model.output_dim, dy.shape[2])
            )
