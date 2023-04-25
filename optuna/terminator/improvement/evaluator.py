from __future__ import annotations

import abc

import numpy as np

from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import StudyDirection
from optuna.terminator import _distribution_is_log
from optuna.terminator._search_space.intersection import IntersectionSearchSpace
from optuna.terminator.improvement._preprocessing import BasePreprocessing
from optuna.terminator.improvement._preprocessing import OneToHot
from optuna.terminator.improvement._preprocessing import PreprocessingPipeline
from optuna.terminator.improvement._preprocessing import SelectTopTrials
from optuna.terminator.improvement._preprocessing import ToMinimize
from optuna.terminator.improvement._preprocessing import UnscaleLog
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    from botorch.acquisition.analytic import UpperConfidenceBound
    from botorch.fit import fit_gpytorch_model
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize
    from botorch.models.transforms import Standardize
    from botorch.optim import optimize_acqf
    import gpytorch
    import torch

DEFAULT_TOP_TRIALS_RATIO = 0.5
DEFAULT_MIN_N_TRIALS = 20


@experimental_class("3.2.0")
class BaseImprovementEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(
        self,
        trials: list[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:
        pass


@experimental_class("3.2.0")
class RegretBoundEvaluator(BaseImprovementEvaluator):
    def __init__(
        self,
        top_trials_ratio: float = DEFAULT_TOP_TRIALS_RATIO,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        self._top_trials_ratio = top_trials_ratio
        self._min_n_trials = min_n_trials

    def get_preprocessing(self) -> BasePreprocessing:
        processes = [
            SelectTopTrials(
                top_trials_ratio=self._top_trials_ratio,
                min_n_trials=self._min_n_trials,
            ),
            UnscaleLog(),
            ToMinimize(),
            OneToHot(),
        ]

        return PreprocessingPipeline(processes)

    def evaluate(
        self,
        trials: list[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:
        search_space = IntersectionSearchSpace().calculate(trials)
        self._validate_input(trials, search_space)

        one_to_hot = OneToHot()
        preprocessing = PreprocessingPipeline([
            SelectTopTrials(
                top_trials_ratio=self._top_trials_ratio,
                min_n_trials=self._min_n_trials,
            ),
            UnscaleLog(),
            ToMinimize(),
            one_to_hot,
        ])

        fit_trials = preprocessing.apply(trials, study_direction)

        x, bounds, y, categorical_indices = _convert_trials_to_tensors(
            fit_trials, one_to_hot.encoded_params
        )

        gp = _fit_gp(x, bounds, y)

        n_params = len(search_space)
        n_trials = len(fit_trials)
        beta = _get_beta(n_params=n_params, n_trials=n_trials)

        return _calculate_min_ucb(gp, beta, x) - _calculate_min_lcb(
            gp, beta, x, bounds, categorical_indices
        )

    @classmethod
    def _validate_input(
        cls, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> None:
        if len([t for t in trials if t.state == TrialState.COMPLETE]) == 0:
            raise ValueError(
                "Because no trial has been completed yet, the regret bound cannot be evaluated."
            )

        if len(search_space) == 0:
            raise ValueError(
                "The intersection search space is empty. This condition is not supported by "
                f"{cls.__name__}."
            )


def _convert_trials_to_tensors(
    trials: list[FrozenTrial],
    encoded_params: dict[str, list[str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Convert a list of FrozenTrial objects to tensors inputs and bounds.

    This function assumes the following condition for input trials:
    - any categorical param is converted to a float or int one;
    - log is unscaled for any float/int distribution;
    - the state is COMPLETE for any trial;
    - direction is MINIMIZE for any trial.
    """
    search_space = IntersectionSearchSpace().calculate(trials)
    sorted_params = sorted(search_space.keys())
    sorted_params_indices = {param: i for i, param in enumerate(sorted_params)}

    x = []
    for trial in trials:
        assert trial.state == TrialState.COMPLETE
        x_row = []
        for param in sorted_params:
            distribution = search_space[param]

            assert not _distribution_is_log(distribution)
            assert not isinstance(distribution, CategoricalDistribution)

            param_value = float(trial.params[param])
            x_row.append(param_value)

        x.append(x_row)

    min_bounds = []
    max_bounds = []
    for param, distribution in search_space.items():
        assert isinstance(distribution, (FloatDistribution, IntDistribution))
        min_bounds.append(distribution.low)
        max_bounds.append(distribution.high)
    bounds = [min_bounds, max_bounds]

    y = torch.tensor([trial.value for trial in trials], dtype=torch.float64)

    categorical_indices = [
        torch.tensor([sorted_params_indices[param] for param in encoded])
        for encoded in encoded_params.values()
    ]

    return torch.tensor(x, dtype=torch.float64), torch.tensor(bounds, dtype=torch.float64), y, categorical_indices


def _fit_gp(x: torch.Tensor, bounds: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    y = torch.unsqueeze(y, 1)
    gp = SingleTaskGP(
        x,
        y,
        input_transform=Normalize(d=x.shape[1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    return gp


def _calculate_min_lcb(
    gp: SingleTaskGP, beta: float, x: torch.Tensor, bounds: torch.Tensor, categorical_indices: list[torch.Tensor]
) -> float:
    neg_lcb_func = UpperConfidenceBound(gp, beta=beta, maximize=False)

    linear_constraints = [
        (indices, torch.ones(len(indices), dtype=torch.double), 1.0) 
        for indices in categorical_indices
    ]

    with gpytorch.settings.fast_pred_var():  # type: ignore[no-untyped-call]


        opt_x, lcb = optimize_acqf(
            neg_lcb_func,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=2048,
            sequential=True,
            # Applying linear constraints is slow, so we only apply it when necessary.
            equality_constraints=linear_constraints if linear_constraints else None,
            # options={"sample_around_best": True},
        )
        min_lcb = -lcb.item()
        print(list(gp.named_parameters()))
        print([opt_x[0, indices].detach().numpy() for indices in categorical_indices])
        print([opt_x[0, indices].sum().item() for indices in categorical_indices])
        print([(opt_x[0, indices] @ torch.log(opt_x[0, indices] + 1e-100)).item() for indices in categorical_indices])
        print(min_lcb)

        min_lcb_x = torch.min(-neg_lcb_func(x[:, None, :])).item()

        import itertools
        def onehot(indices: torch.Tensor) -> torch.Tensor:
            ret = torch.zeros(opt_x.shape[1], dtype=torch.double)
            ret[indices] = 1
            return ret
        all_xs = torch.vstack([onehot(torch.tensor(indices)) for indices in itertools.product(*categorical_indices)])

        lcbs = -neg_lcb_func(all_xs[:, None, :])
        ans = torch.argmin(lcbs)
        min_lcb_all_xs = lcbs[ans].item()
        real_opt_xs = all_xs[ans]
        print(min_lcb_all_xs)
        print([all_xs[ans, indices].detach().numpy() for indices in categorical_indices])

        # show_x = real_opt_xs.clone()
        # show_x[categorical_indices[1]] = opt_x[0, categorical_indices[1]]

        assert real_opt_xs[categorical_indices[1][4]] == 1.0

        base = real_opt_xs.clone()
        base[categorical_indices[1]] = 0.0
        ternary_vertex = [base + onehot(torch.tensor([categorical_indices[1][i]])) for i in (2, 3, 4)]


        ts = torch.linspace(0.0, 1.0, 101)
        t1, t2 = torch.meshgrid(ts, ts)
        mask = (t1 + t2 <= 1.0)
        t1 = t1[mask]
        t2 = t2[mask]
        t0 = torch.maximum(torch.tensor(0.0), 1.0 - t1 - t2)
        print(t0, t1, t2)

        xs = ternary_vertex[0][None, :] * t0[:, None] + ternary_vertex[1][None, :] * t1[:, None] + ternary_vertex[2][None, :] * t2[:, None]

        print(xs.shape)
        vs = -neg_lcb_func(xs[:, None, :])

        import plotly.figure_factory as ff
        fig = ff.create_ternary_contour(np.array([t0.detach().numpy(), t1.detach().numpy(), t2.detach().numpy()]), vs.detach().numpy(),
                                pole_labels=["2", "3", "4"],
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Viridis',
                                showscale=True,)
                
        # Set contour lines width to 0
        for trace in fig.data:
            if trace.type == 'scatterternary':
                trace.line.width = 0
        fig.show()
        # max_t = 1.0 / (1.0 - show_x[categorical_indices[1][4]])
        # print(show_x)
        # ts = torch.linspace(0.0, float(max_t), 100)
        # xs = real_opt_xs * (1 - ts[:, None]) + show_x * ts[:, None]
        # vs = -neg_lcb_func(xs[:, None, :])

        # import plotly.figure_factory as ff

        # plt.plot(ts.detach().numpy(), vs.detach().numpy())
        # plt.show()
        
        min_lcb = min(min_lcb, min_lcb_x)

    return min_lcb


def _calculate_min_ucb(gp: SingleTaskGP, beta: float, x: torch.Tensor) -> float:
    ucb_func = UpperConfidenceBound(gp, beta=beta, maximize=True)

    with gpytorch.settings.fast_pred_var():  # type: ignore[no-untyped-call]
        min_ucb = torch.min(ucb_func(x[:, None, :])).item()

    return min_ucb


def _get_beta(n_params: int, n_trials: int, delta: float = 0.1) -> float:
    beta = 2 * np.log(n_params * n_trials**2 * np.pi**2 / 6 / delta)

    # The following div is according to the original paper: "We then further scale it down
    # by a factor of 5 as defined in the experiments in Srinivas et al. (2010)"
    beta /= 5

    return beta
