import abc
from typing import Optional
from typing import Tuple

import numpy as np


class BaseModel(object, metaclass=abc.ABCMeta):
    """Base class for Gaussian process models"""

    @property
    def input_dim(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def output_dim(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def n_mcmc_samples(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add observation data to the model.

        This method update the posterior distribution and model hyperparameters.

        Args:
            x:
                The input points in the domain space. The shape is `(n, input_dim)`, where `n` is
                the number of points and `input_dim` is the input dimension.
            y:
                The input points in the objective space. The shape is `(n, output_dim)`, where `n`
                is the number of points and `output_dim` is the output dimension.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the prediction.

        This function computes the posterior mean vector and the square root of the posterior
        covariance matrix.

        Args:
            x:
                Input points. The shape is `(n, input_dim)`, where `n` is the number of points and
                `input_dim` is the input dimension.

        Returns:
            The tuple of the computed posterior mean vector and the square root of the posterior
            covariance matrix. The shape of the posterior mean vector is `(n, output_dim)`. The
            shape of the square root of the posterior covariance matrix is
            `(n, output_dim, output_dim)`. Here, `n` is the number of points and `output_dim` is
            the output dimension.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def predict_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the gradient of the prediction.

        This function computes the posterior mean vector and the gradient of the square root of the
        posterior covariance matrix.

        Args:
            x:
                Input points. The shape is `(n, input_dim)`, where `n` is the number of points and
                `input_dim` is the input dimension.

        Returns:
            The tuple of the computed gradient of the posterior mean vector and the gradient of the
            square root of the posterior covariance matrix. The shape of the gradient of the
            posterior mean vector is `(n, input_dim, output_dim)`. The shape of the square root of
            the gradient of the posterior covariance matrix is
            `(n, input_dim, output_dim, output_dim)`. Here, `n` is the number of points,
            `input_dim` is the input dimension, and `output_dim` is the output dimension.
        """

        raise NotImplementedError

    def _verify_input(self, x: np.ndarray) -> None:
        assert self.input_dim is not None

        if x.ndim != 2:
            raise ValueError(
                "In `BaseModel.[predict, predict_gradient]`, `x.ndim` should be 2, "
                "but {} is specified.".format(x.ndim)
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                "In `BaseAcquisitionFunction.compute_[acq, grad]`, "
                "`x.shape[1]` should be `self.input_dim = {}`, "
                "but `x.shape[1] = {}` is specified.".format(self.input_dim, x.shape[1])
            )

    def _verify_output_predict(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        assert self.output_dim is not None

        if mu.ndim != 2:
            raise ValueError(
                "In `mu, sigma = BaseModel.predict()`, `mu.ndim` should be 2, "
                "but {} is specified.".format(mu.ndim)
            )

        if sigma.ndim != 3:
            raise ValueError(
                "In `mu, sigma = BaseModel.predict()`, `sigma.ndim` should be 3, "
                "but {} is specified.".format(sigma.ndim)
            )

        if mu.shape[1] != self.output_dim:
            raise ValueError(
                "In `mu, sigma = BaseModel.predict(), "
                "`mu.shape[1]` should be `output_dim = {}`, "
                "but `mu.shape[1] = {} is specified".format(self.output_dim, mu.shape[1])
            )

        if sigma.shape[1] != self.output_dim or sigma.shape[2] != self.output_dim:
            raise ValueError(
                "In `mu, sigma = BaseModel.predict(), "
                "`mu.shape[1:]` should be `(output_dim, output_dim) = {}`, "
                "but `mu.shape[1:] = {} is specified".format(
                    (self.output_dim, self.output_dim), mu.shape[1:]
                )
            )

    def _verify_output_grad(self, dmu: np.ndarray, dsigma: np.ndarray) -> None:
        assert self.input_dim is not None
        assert self.output_dim is not None

        if dmu.ndim != 3:
            raise ValueError(
                "In `dmu, dsigma = BaseModel.predict_gradient()`, `dmu.ndim` should be 3, "
                "but {} is specified.".format(dmu.ndim)
            )

        if dsigma.ndim != 4:
            raise ValueError(
                "In `dmu, dsigma = BaseModel.predict_gradient()`, `dsigma.ndim` should be 4, "
                "but {} is specified.".format(dsigma.ndim)
            )

        if dmu.shape[1:] != (self.input_dim, self.output_dim):
            raise ValueError(
                "In `dmu, dsigma = BaseModel.predict_gradient(), "
                "`dmu.shape[1:]` should be `(input_dim, output_dim) = {}`, "
                "but `dmu.shape[1:] = {} is specified".format(
                    (self.input_dim, self.output_dim), dmu.shape[1:]
                )
            )

        if dsigma.shape[1:] != (self.input_dim, self.output_dim, self.output_dim):
            raise ValueError(
                "In `dmu, dsigma = BaseModel.predict_gradient(), "
                "`dsigma.shape[1:]` should be `(input_dim, output_dim, output_dim) = {}`, "
                "but `dsigma.shape[1:] = {} is specified".format(
                    (self.input_dim, self.output_dim, self.output_dim), dsigma.shape[1:]
                )
            )
