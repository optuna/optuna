from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import warnings

import numpy as np
from scipy import optimize
from scipy.sparse import linalg
from scipy.sparse import spmatrix

from optuna import create_study
from optuna import create_trial
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial


class WrappedOptimizeResult(optimize.OptimizeResult):
    def __init__(self) -> None:
        self.study: Optional[Study] = None
        super().__init__()


def scipy_minimize(
    *,
    fun: Callable[[np.ndarray, Any], float],
    x0: np.ndarray,
    pruner: Optional[BasePruner] = None,
    storage: Optional[Union[str, BaseStorage]] = None,
    study_name: Optional[str] = None,
    load_if_exists: bool = False,
    search_space: Optional[Dict[str, BaseDistribution]] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
    args: Any = (),
    method: Optional[Union[str, BaseSampler]] = None,
    jac: Optional[Union[str, bool, Callable[[np.ndarray, Any], np.ndarray]]] = None,
    hess: Optional[
        Union[
            str,
            optimize.HessianUpdateStrategy,
            Callable[[np.ndarray, Any], Union[np.ndarray, linalg.LinearOperator, spmatrix]],
        ]
    ] = None,
    hessp: Optional[Union[Callable[[np.ndarray, np.ndarray, Any], np.ndarray]]] = None,
    constraints: Optional[
        Union[
            optimize.LinearConstraint,
            optimize.NonlinearConstraint,
            List[optimize.LinearConstraint],
            List[optimize.NonlinearConstraint],
            List[Dict[str, Any]],
        ]
    ] = (),
    tol: Optional[float] = None,
    callback: Optional[Callable[[np.ndarray], bool]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> WrappedOptimizeResult:
    """Wrapper function of `scipy.optimize.minimize` to use Optuna's sampling algorithms.

    Example:

        .. testcode::

            import numpy as np
            import optuna
            from optuna.integration import scipy_minimize


            # rosenbrock function
            def f(x: np.ndarray) -> float:
                return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


            search_space = {}
            for i in range(10):
                search_space[f"x{i}"] = optuna.distributions.UniformDistribution(-10, 10)

            x0 = [9 for _ in range(10)]

            method = None
            res = scipy_minimize(fun=f, x0=x0, method=method, search_space=search_space)

            method = optuna.samplers.TPESampler()
            res = scipy_minimize(
                fun=f, x0=x0, method=method, search_space=search_space, timeout=10
            )
    """
    if not isinstance(method, BaseSampler):
        if search_space is None:
            warnings.warn(
                "The search space is `None`. In this case, optuna yesterday is not available and "
                "an empty study will be saved. You will get the same result as if you had simply "
                "run `scipy.optimize.minimize`."
            )

        study = create_study(storage=storage, study_name=study_name, load_if_exists=load_if_exists)

        def _scipy_callback(xk: np.ndarray) -> bool:
            if search_space is None and callback is not None:
                return callback(xk)

            assert search_space is not None
            value = fun(xk, *args)
            params = {}
            for i, name in enumerate(search_space):
                params[name] = xk[i]

            study.add_trial(
                create_trial(
                    value=value,
                    params=params,
                    distributions=search_space,
                )
            )

            if callback is None:
                return False
            else:
                return callback(xk)

        bounds = _SearchSpaceTransform(search_space).bounds if search_space is not None else None

        result: WrappedOptimizeResult = optimize.minimize(
            fun=fun,
            x0=x0,
            args=args,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=_scipy_callback,
            options=options,
        )
        result.study = study

        return result
    else:
        if search_space is None:
            raise ValueError("Specify the search space if you want to use Optuna's samplers.")

        if any(
            [
                not isinstance(d, (UniformDistribution, LogUniformDistribution))
                for d in search_space.values()
            ]
        ):
            raise ValueError(
                "All distributions in the search space should be one of `UniformDistribution` or "
                f"`LogUniformDistribution`. It is {search_space}"
            )

        def _objective(trial: Trial) -> float:
            x = np.empty(len(x0), dtype=np.float64)
            assert search_space is not None
            for i, (name, distribution) in enumerate(search_space.items()):
                log = isinstance(distribution, LogUniformDistribution)
                assert isinstance(distribution, (UniformDistribution, LogUniformDistribution))
                x[i] = trial.suggest_float(name, distribution.low, distribution.high, log=log)

            return fun(x, *args)

        def _optuna_callback(study: Study, trial: FrozenTrial) -> None:
            if callback is None:
                return
            x = np.empty(len(trial.params), dtype=np.float64)
            for i, param_value in enumerate(trial.params.values()):
                x[i] = param_value
            callback(x)

        study = create_study(
            sampler=method,
            pruner=pruner,
            storage=storage,
            study_name=study_name,
            load_if_exists=load_if_exists,
        )

        study.optimize(
            _objective, callbacks=[_optuna_callback], n_trials=n_trials, timeout=timeout
        )

        result = WrappedOptimizeResult()
        result.study = study
        result.fun = study.best_value
        result.message = "Optuna optimization successfully finished"
        result.nfev = result.nit = len(study.trials)
        result.status = 0
        result.success = True
        result.x = np.asarray(list(study.best_params.values()), dtype=np.float64)

        return result
