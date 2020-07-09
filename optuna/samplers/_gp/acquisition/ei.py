import numpy as np
from scipy import linalg
from scipy import stats

from optuna.samplers._gp.acquisition.base import BaseAcquisitionFunction
from optuna.samplers._gp.model import BaseModel


_SIGMA0_2 = 1e-10


class EI(BaseAcquisitionFunction):
    """Expected Improvement acquisition function.

    For the detail of the algorithm, please see
    Jones, D., Schonlau, M., and Welch, W. Expensive global optimization of expensive black-box
    functions. Journal of Global Optimization, 13:455–492, 1998.
    """

    def __init__(self):
        pass

    def compute_acq(self, x: np.ndarray, model: BaseModel) -> np.ndarray:

        self._verify_input(x, model)
        n = x.shape[0]

        def _compute():
            y_best = np.min(model.Y, axis=0)
            mu, sigma = model.predict(x)  # Predict after sampling hyperparameters by MCMC
            inv_sigma = np.asarray([linalg.inv(sigma[i] + _SIGMA0_2 * np.eye(model.output_dim)) for i in range(n)])
            gamma = np.einsum("ijk,ik->ij", inv_sigma, mu - y_best)

            _Phi = stats.norm.cdf(gamma)
            _phi = stats.norm.pdf(gamma)
            y = np.einsum("ijk,ik->ij", sigma, gamma * _Phi + _phi)
            return y

        y = np.sum([_compute() for _ in range(model.n_mcmc_samples)]) / model.n_mcmc_samples

        self._verify_output_acq(y, model)

        return y

    def compute_grad(self, x: np.ndarray, model: BaseModel) -> np.ndarray:

        self._verify_input(x, model)
        n = x.shape[0]

        def _compute():
            y_best = np.min(model.Y, axis=0)
            mu, sigma = model.predict(x)  # Predict after sampling hyperparameters by MCMC
            dmu, dsigma = model.predict_gradient(x)  # Same as above

            inv_sigma = np.asarray([linalg.inv(sigma[i] + _SIGMA0_2 * np.eye(model.output_dim)) for i in range(n)])
            gamma = np.einsum("ijk,ik->ij", inv_sigma, mu - y_best)
            dgamma = - np.einsum("Ipr,Iirs,Is->Iip", inv_sigma, dsigma, gamma) + np.einsum("Ipq,Iiq->Iip", inv_sigma, dmu - y_best)

            _Phi = stats.norm.cdf(gamma)
            _phi = stats.norm.pdf(gamma)
            z = gamma * _Phi + _phi
            dz = np.einsum("Iip,Ip->Iip", dgamma, _Phi)

            dy = np.einsum("Iijp,Ip->Iij", dsigma, z) + np.einsum("Ijp,Iip->Iij", sigma, dz)
            return dy

        dy = np.sum([_compute() for _ in range(model.n_mcmc_samples)]) / model.n_mcmc_samples

        self._verify_output_grad(dy, model)

        return dy
