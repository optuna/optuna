import numpy as np
from scipy import linalg
from scipy import stats

from optuna.samplers._gp.acquisition.base import BaseAcquisitionFunction
from optuna.samplers._gp.model import BaseModel


class EI(BaseAcquisitionFunction):
    """Expected Improvement acquisition function.

    For the detail of the algorithm, please see
    Jones, D., Schonlau, M., and Welch, W. Expensive global optimization of expensive black-box
    functions. Journal of Global Optimization, 13:455–492, 1998.
    """

    def __init__(self, sigma0: float = 1e-10):
        self._sigma0 = sigma0

    def compute_acq(self, x: np.ndarray, model: BaseModel) -> np.ndarray:

        x = np.atleast_2d(x)
        self._verify_input(x, model)
        n = x.shape[0]
        mus, sigmas = model.predict(x)

        def _compute(a: int) -> np.ndarray:
            mu, sigma = mus[a], sigmas[a]
            y_best = np.min(model.y, axis=0)
            inv_sigma = np.asarray(
                [linalg.inv(sigma[i] + self._sigma0 * np.eye(model.output_dim)) for i in range(n)]
            )
            gamma = np.einsum("ijk,ik->ij", inv_sigma, y_best - mu)

            _Phi = stats.norm.cdf(gamma)
            _phi = stats.norm.pdf(gamma)
            y = np.einsum("ijk,ik->ij", sigma, gamma * _Phi + _phi)
            return y

        y = (
            np.sum([_compute(a) for a in range(model.n_mcmc_samples)], axis=0)
            / model.n_mcmc_samples
        )

        self._verify_output_acq(y, model)

        return y

    def compute_grad(self, x: np.ndarray, model: BaseModel) -> np.ndarray:

        x = np.atleast_2d(x)
        self._verify_input(x, model)
        n = x.shape[0]
        mus, sigmas = model.predict(x)
        dmus, dsigmas = model.predict_gradient(x)

        def _compute(a: int) -> np.ndarray:
            mu, sigma, dmu, dsigma = mus[a], sigmas[a], dmus[a], dsigmas[a]
            y_best = np.min(model.y, axis=0)
            inv_sigma = np.asarray(
                [linalg.inv(sigma[i] + self._sigma0 * np.eye(model.output_dim)) for i in range(n)]
            )
            gamma = np.einsum("ijk,ik->ij", inv_sigma, y_best - mu)
            dgamma = -np.einsum("Ipr,Iirs,Is->Iip", inv_sigma, dsigma, gamma) + np.einsum(
                "Ipq,Iiq->Iip", inv_sigma, y_best - dmu
            )

            _Phi = stats.norm.cdf(gamma)
            _phi = stats.norm.pdf(gamma)
            z = gamma * _Phi + _phi
            dz = np.einsum("Iip,Ip->Iip", dgamma, _Phi)

            dy = np.einsum("Iijp,Ip->Iij", dsigma, z) + np.einsum("Ijp,Iip->Iij", sigma, dz)
            return dy

        dy = (
            np.sum([_compute(a) for a in range(model.n_mcmc_samples)], axis=0)
            / model.n_mcmc_samples
        )

        self._verify_output_grad(dy, model)

        return dy
