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
        noise_var: Union[float, str] = "gpy_default",
        kernel: str = "Matern52",
        consider_ard: bool = True,
        gamma_prior_expectation: float = 2.0,
        gamma_prior_variance: float = 4.0,
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

        if x is not None and y is not None:
            self._initialize()

    @property
    def x(self) -> np.ndarray:

        assert self._x is not None
        return self._x

    @property
    def y(self) -> np.ndarray:

        assert self._y is not None
        return self._y

    @property
    def input_dim(self) -> Optional[int]:

        return self._input_dim

    @property
    def output_dim(self) -> Optional[int]:

        return self._output_dim

    @property
    def n_mcmc_samples(self) -> int:

        return self._hmc_n_samples

    def _initialize(self) -> None:

        assert self._x is not None
        assert self._y is not None
        self._verify_data(self._x, self._y)
        self._input_dim = self._x.shape[1]
        self._output_dim = self._y.shape[1]

        assert self._input_dim is not None
        assert self._output_dim is not None
        if self._kernel == "SquaredExponential" or self._kernel == "RBF":
            k = GPy.kern.RBF(input_dim=self._input_dim, variance=1.0, ARD=self._consider_ard)
        elif self._kernel == "Matern52":
            k = GPy.kern.Matern52(input_dim=self._input_dim, variance=1.0, ARD=self._consider_ard)
        else:
            kernel_list = ["RBF", "Squared Exponential", "Matern52"]
            raise NotImplementedError(
                "The kernel should be one of the {}. "
                "However, {} is specified.".format(kernel_list, self._kernel)
            )

        if self._noise_var == "gpy_default":
            noise_var = self._y.var() * 0.01
        elif isinstance(self._noise_var, float):
            noise_var = self._noise_var
        else:
            noise_vare_list = ["gpy_default"]
            raise ValueError(
                "The noise variance should float or one of the {}. "
                "However, {} is specified.".format(noise_vare_list, self._noise_var)
            )
        self._gpy_model = GPy.models.GPRegression(self._x, self._y, kernel=k, noise_var=noise_var)
        self._gpy_model.kern.set_prior(
            GPy.priors.Gamma.from_EV(self._gamma_prior_expectation, self._gamma_prior_variance),
            warning=False,
        )
        self._gpy_model.likelihood.variance.set_prior(
            GPy.priors.Gamma.from_EV(self._gamma_prior_expectation, self._gamma_prior_variance),
            warning=False,
        )

        self._update_model()

    @staticmethod
    def _verify_data(x: np.ndarray, y: np.ndarray) -> None:

        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(
                "The shape of the `x` and `y` should be `(n, input_dim)` and `(n, output_dim)`, "
                "but `x.shape = {}` and `y.shape = {}` are specified.".format(x.shape, y.shape)
            )

    def _update_model(self) -> None:

        assert self._gpy_model is not None
        self._gpy_model.optimize(max_iters=self._max_optimize_iters)
        self._gpy_model.param_array[:] = self._gpy_model.param_array * (
            1.0 + np.random.randn(self._gpy_model.param_array.size) * 0.01
        )
        self._hmc = GPy.inference.mcmc.HMC(self._gpy_model, stepsize=self._hmc_step_size)
        samples = self._hmc.sample(
            num_samples=self._hmc_burnin + self._hmc_n_samples * self._hmc_subsample_interval,
            hmc_iters=self._hmc_iters,
        )
        self._hmc_samples = samples[self._hmc_burnin :: self._hmc_subsample_interval]

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

        assert self._gpy_model is not None
        x = np.atleast_2d(x)
        ps = self._gpy_model.param_array.copy()
        _mus = []
        _sigmas = []
        for s in self._hmc_samples:
            if self._gpy_model._fixes_ is None:
                self._gpy_model[:] = s
            else:
                self._gpy_model[self._gpy_model._fixes_] = s
            self._gpy_model._trigger_params_changed()
            m, v = self._gpy_model.predict(x)
            _mus.append(m)
            _sigmas.append(np.sqrt(np.clip(v, 1e-10, np.inf)))
        self._gpy_model.param_array[:] = ps
        self._gpy_model._trigger_params_changed()
        _mus = np.asarray(_mus)
        _sigmas = np.asarray(_sigmas)

        while _mus.ndim < 3:
            _mus = _mus.reshape(_mus.shape + (1,))
        while _sigmas.ndim < 4:
            _sigmas = _sigmas.reshape(_sigmas.shape + (1,))

        if (
            _mus.shape[0] != self._hmc_n_samples
            or _sigmas.shape[0] != self._hmc_n_samples
            or _mus.shape[1] != _sigmas.shape[1]
            or _mus.shape[2] != self._output_dim
            or _sigmas.shape[2] != self._output_dim
            or _sigmas.shape[3] != self._output_dim
        ):
            raise ValueError(
                "In mus, sigmas = GPyExact.predict(), "
                "mus.shape should be (n_mcmc_samples, n, output_dim) and "
                "sigmas.shape should be (n_mcmc_samples, n, output_dim, output_dim), "
                "but {} and {} are specified.".format(_mus.shape, _sigmas.shape)
            )

        return _mus, _sigmas

    def predict_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        assert self._gpy_model is not None
        x = np.atleast_2d(x)
        ps = self._gpy_model.param_array.copy()
        _dmus = []
        _dsigmas = []
        for s in self._hmc_samples:
            if self._gpy_model._fixes_ is None:
                self._gpy_model[:] = s
            else:
                self._gpy_model[self._gpy_model._fixes_] = s
            self._gpy_model._trigger_params_changed()
            _, v = self._gpy_model.predict(x)
            std = np.sqrt(np.clip(v, 1e-10, np.inf))
            dm, dv = self._gpy_model.predictive_gradients(x)
            dm = dm[:, :, 0]
            ds = dv / (2 * std)
            _dmus.append(dm)
            _dsigmas.append(ds)
        self._gpy_model.param_array[:] = ps
        self._gpy_model._trigger_params_changed()
        _dmus = np.asarray(_dmus)
        _dsigmas = np.asarray(_dsigmas)

        while _dmus.ndim < 4:
            _dmus = _dmus.reshape(_dmus.shape + (1,))
        while _dsigmas.ndim < 5:
            _dsigmas = _dsigmas.reshape(_dsigmas.shape + (1,))

        if (
            _dmus.shape[0] != self._hmc_n_samples
            or _dsigmas.shape[0] != self._hmc_n_samples
            or _dmus.shape[1] != _dsigmas.shape[1]
            or _dmus.shape[2] != self._input_dim
            or _dmus.shape[3] != self._output_dim
            or _dsigmas.shape[2] != self._input_dim
            or _dsigmas.shape[3] != self._output_dim
            or _dsigmas.shape[4] != self._output_dim
        ):
            raise ValueError(
                "In dmus, dsigmas = GPyExact.predict_gradient(), "
                "dmus.shape should be (n_mcmc_samples, n, input_dim, output_dim) and "
                "dsigmas.shape should be (n_mcmc_samples, n, input_dim, output_dim, output_dim), "
                "but {} and {} are specified.".format(_dmus.shape, _dsigmas.shape)
            )

        return _dmus, _dsigmas
