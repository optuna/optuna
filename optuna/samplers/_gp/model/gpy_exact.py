from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from optuna._imports import try_import
from optuna.samplers._gp.model.base import BaseModel

with try_import() as _imports:
    import GPy


class GPyExact(BaseModel):
    """An exact Gaussian process model based on GPy library.

    .. note::
        We use GPy library. See https://github.com/SheffieldML/GPy.
    """

    def __init__(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        noise_var: Union[float, str] = 'gpy_default',
        kernel: str = 'Matern52',
        consider_ard: bool = True,
        gamma_prior_expectation: float = 2.,
        gamma_prior_variance: float = 4.,
        max_optimize_iters: int = 200,
        hmc_step_size: float = 0.1,
        hmc_burnin: int = 100,
        hmc_n_samples: int = 10,
        hmc_subsample_interval: int = 10,
        hmc_iters: int = 20,
    ):

        self._x = x
        self._y = y
        self._noise_var = noise_var
        self._kernel = kernel
        self._consider_ard = consider_ard
        self._gamma_prior_expectation = gamma_prior_expectation
        self._gamma_prior_variance = gamma_prior_variance
        self._max_optimize_iters = max_optimize_iters
        self._hmc_step_size = hmc_step_size
        self._hmc_burnin = hmc_burnin
        self._hmc_n_samples = hmc_n_samples
        self._hmc_subsample_interval = hmc_subsample_interval
        self._hmc_iters = hmc_iters

        self._input_dim = None
        self._output_dim = None
        self._gpy_model = None
        self._hmc = None
        self._hmc_samples = None
        self._cached_mus = {}
        self._cached_sigmas = {}
        self._cached_dmus = {}
        self._cached_dsigmas = {}

        if x is not None and y is not None:
            self._initialize()

    def input_dim(self) -> Optional[int]:

        return self._input_dim

    def output_dim(self) -> Optional[int]:

        return self._output_dim

    def hmc_n_samples(self) -> int:

        return self._hmc_n_samples

    def _initialize(self) -> None:

        self._verify_data(self._x, self._y)
        self._input_dim = self._x.shape[1]
        self._output_dim = self._y.shape[1]

        assert self._input_dim is not None
        assert self._output_dim is not None
        if self._kernel == 'SquaredExponential' or self._kernel == 'RBF':
            k = GPy.kern.RBF(input_dim=self._input_dim, variance=1., ARD=self._consider_ard)
        elif self._kernel == 'Matern52':
            k = GPy.kern.Matern52(input_dim=self._input_dim, variance=1., ARD=self._consider_ard)
        else:
            kernel_list = ['RBF', 'Squared Exponential', 'Matern52']
            raise NotImplementedError(
                'The kernel should be one of the {}. '
                'However, {} is specified.'.format(kernel_list, self._kernel)
            )

        self._gpy_model = GPy.models.GPRegression(self._x, self._y, kernel=k, noise_var=self._noise_var)
        self._gpy_model.kern.set_prior(GPy.priors.Gamma.from_EV(self._gamma_prior_expectation, self._gamma_prior_variance))
        self._gpy_model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(self._gamma_prior_expectation, self._gamma_prior_variance))

        self._update_model()

    @staticmethod
    def _verify_data(x: np.ndarray, y: np.ndarray) -> None:

        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(
                "The shape of the `x` and `y` should be `(n, input_dim)` and `(n, output_dim)`, "
                "but `x.shape = {}` and `y.shape = {}` are specified."
                .format(x.shape, y.shape)
            )

    def _update_model(self) -> None:

        self._gpy_model.optimize(max_iters=self._max_optimize_iters)
        self._gpy_model.param_array[:] = self._gpy_model.param_array * (1. + np.random.randn(self._gpy_model.param_array.size) * 0.01)
        self._hmc = GPy.inference.mcmc.HMC(self._gpy_model, stepsize=self._hmc_step_size)
        samples = self._hmc.sample(num_samples=self._hmc_burnin + self._hmc_n_samples * self._hmc_subsample_interval, hmc_iters=self._hmc_iters)
        self._hmc_samples = samples[self._hmc_burnin::self._hmc_subsample_interval]

    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:

        self._verify_data(x, y)

        if self._gpy_model is None:
            self._x = x
            self._y = y
        else:
            self._x = np.vstack((self._x, x))
            self._y = np.vstack((self._y, y))

        self._initialize()

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        mu, sigma = None, None
        if self._cached_mus[x] > 0:
            mu = self._cached_mus[x].pop()
        if self._cached_sigmas[x] > 0:
            sigma = self._cached_sigmas[x].pop()

        if mu is None or sigma is None:
            x = np.atleast_2d(x)
            ps = self._gpy_model.param_array.copy()
            _mus = []
            _sigmas = []
            for s in self._hmc_samples:
                if self._gpy_model._fixes_ is None:
                    self._gpy_model[:] = s
                else:
                    self._gpy_model[self._gpy_model._fixes_] = s
                self._gpy_model.trigger_params_changed()
                m, v = self._gpy_model.predict(x)
                _mus.append(m)
                _sigmas.append(np.sqrt(np.clip(v, 1e-10, np.inf)))
            self._gpy_model.param_array[:] = ps
            self._gpy_model.trigger_params_changed()
            self._cached_mus[x] = _mus
            self._cached_sigmas[x] = _sigmas
            mu = self._cached_mus[x].pop()
            sigma = self._cached_sigmas[x].pop()

        return mu, sigma

    def predict_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        dmu, dsigma = None, None
        if self._cached_dmus[x] > 0:
            dmu = self._cached_dmus[x].pop()
        if self._cached_dsigmas[x] > 0:
            dsigma = self._cached_dsigmas[x].pop()

        if dmu is None or dsigma is None:
            x = np.atleast_2d(x)
            ps = self._gpy_model.param_array.copy()
            _dmus = []
            _dsigmas = []
            for s in self._hmc_samples:
                if self._gpy_model._fixes_ is None:
                    self._gpy_model[:] = s
                else:
                    self._gpy_model[self._gpy_model._fixes_] = s
                self._gpy_model.trigger_params_changed()
                _, v = self._gpy_model.predict(x)
                std = np.sqrt(np.clip(v, 1e-10, np.inf))
                dm, dv = self._gpy_model.predictive_gradients(x)
                dm = dm[:, :, 0]
                ds = dv / (2 * std)
                _dmus.append(dm)
                _dsigmas.append(ds)
            self._gpy_model.param_array[:] = ps
            self._gpy_model.trigger_params_changed()
            self._cached_dmus[x] = _dmus
            self._cached_dsigmas[x] = _dsigmas
            dmu = self._cached_dmus[x].pop()
            dsigma = self._cached_dsigmas[x].pop()

        return dmu, dsigma
