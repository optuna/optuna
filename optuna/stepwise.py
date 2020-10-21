import copy
import functools
import operator
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np

import optuna
from optuna import distributions
from optuna import logging
from optuna import pruners
from optuna import samplers
from optuna.study import ObjectiveFuncType
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


SuggesterType = Callable[[optuna.Trial], Dict[str, Any]]
GridValueType = Union[str, float, int, bool, None]

MakeStepType = Callable[[Dict[str, Any]], "Step"]
StepType = Union["Step", MakeStepType]
StepListType = Sequence[Tuple[str, StepType]]
StepObjectiveType = Callable[[optuna.Trial, Dict[str, Any]], float]

_OptunaCallback = Callable[[optuna.Study, FrozenTrial], None]
_NumericType = TypeVar("_NumericType", int, float)


class _BaseBudget(Generic[_NumericType]):
    """Manage a resource budget, None represents infinite."""

    def __init__(self, unit_name: str, value: Optional[_NumericType]):
        self.unit_name = unit_name
        self.initial: Optional[_NumericType] = value
        self._remaining: Optional[_NumericType] = value

    @property
    def remaining(self) -> Optional[_NumericType]:
        return self._remaining

    @property
    def is_depleted(self) -> bool:
        return self.remaining is not None and self.remaining <= 0

    def try_spend(self, value: Optional[_NumericType]) -> Optional[_NumericType]:
        if self.remaining is None:
            return value

        if value is None:
            return self.remaining

        value = min(self.remaining, value)
        self._remaining = self.remaining - value
        return value


class _TrialBudget(_BaseBudget[int]):
    def __init__(self, timeout: Optional[int]):
        super().__init__("trial", timeout)


class _TimeBudget(_BaseBudget[float]):
    def __init__(self, timeout: Optional[float]):
        super().__init__("second", timeout)
        self._start: Optional[float] = None
        self._elapsed: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self._start is None:
            raise RuntimeError(f"Timer is not running. Use {type(self).__name__}.start() first.")
        return time.perf_counter() - self._start

    @property
    def remaining(self) -> Optional[float]:
        if self.initial is None:
            return None
        return self.initial - self.elapsed

    def start(self) -> None:
        if self._start is not None:
            raise RuntimeError(f"Timer is running. Use {type(self).__name__}.stop() first.")
        self._start = time.perf_counter()

    def stop(self) -> None:
        if self._start is None:
            raise RuntimeError(f"Timer is not running. Use {type(self).__name__}.start() first.")
        self._elapsed = time.perf_counter() - self._start
        self._start = None

    def __enter__(self) -> "_TimeBudget":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


class Step:
    """A Step encampsulates the specific parameters of one specific optimization task.

    It also defines which parameters to optimize and from which distributions to suggest
    their values.

    Common parameters for the sequence, such as the objective function are defined on a
    :class:`.StepwiseTuner`.

    Args:
        distributions:
            A dictionary whose key and value are a parameter name and the corresponding
            distributions to suggest from.
        sampler:
            A sampler object that implements background algorithm for value suggestion.
            If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used
            as the default. See also :class:`~optuna.samplers`.
        pruner:
            A pruner object that decides early stopping of unpromising trials. If :obj:`None`
            is specified, :class:`~optuna.pruners.MedianPruner` is used as the default. See
            also :class:`~optuna.pruners`.
        n_trials:
            The number of trials. If this argument is set to :obj:`None`, there is no
            limitation on the number of trials for this step.
        timeout:
            Stop study after the given number of second(s). If this argument is set to
            :obj:`None`, the step is executed without time limitation.

    """

    def __init__(
        self,
        distributions: Mapping[str, distributions.BaseDistribution],
        sampler: Optional[samplers.BaseSampler] = None,
        pruner: Optional[pruners.BasePruner] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        self.distributions = distributions
        self.sampler = sampler
        self.pruner = pruner
        self.n_trials = n_trials
        self.timeout = timeout

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest a value for each parameters."""
        return {name: trial._suggest(name, dist) for name, dist in self.distributions.items()}


class GridStep(Step):
    """A Step that samples using grid search.

    Args:
        search_space:
            A dictionary whose key and value are a parameter name and the corresponding candidates
            of values, respectively.
        pruner:
            A pruner object that decides early stopping of unpromising trials. If :obj:`None`
            is specified, :class:`~optuna.pruners.MedianPruner` is used as the default. See
            also :class:`~optuna.pruners`.
        n_trials:
            The number of trials. If this argument is set to :obj:`None`, there is no
            limitation on the number of trials for this step.
        timeout:
            Stop study after the given number of second(s). If this argument is set to
            :obj:`None`, the step is executed without time limitation.

    """

    def __init__(
        self,
        search_space: Mapping[str, List[GridValueType]],
        pruner: Optional[pruners.BasePruner] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        if not n_trials:
            n_trials = functools.reduce(operator.mul, map(len, search_space.values()))
        super().__init__(
            self._get_distributions(search_space),
            samplers.GridSampler(search_space),
            pruner,
            n_trials,
            timeout,
        )

    def _get_distributions(
        self, search_space: Mapping[str, List[GridValueType]]
    ) -> Mapping[str, distributions.BaseDistribution]:
        dists = {}
        for name, values in search_space.items():
            if isinstance(values[0], (str, bool)):
                dist = distributions.CategoricalDistribution(values)
            elif isinstance(values[0], int):
                dist = distributions.IntUniformDistribution(
                    min(values), max(values)  # type: ignore
                )
            else:
                # add 1 to high to include higher bound
                dist = distributions.UniformDistribution(
                    min(values), max(values) + 1  # type: ignore
                )
            dists[name] = dist
        return dists


class _NoOpStep(Step):
    def __init__(self, base_params: Dict[str, Any], timeout: Optional[int] = None):
        super().__init__(distributions={}, sampler=None, pruner=None, n_trials=1, timeout=timeout)
        self.base_params = base_params

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        return self.base_params


def is_better(
    direction: StudyDirection, val_score: Optional[float], best_score: Optional[float]
) -> bool:
    if val_score is None or best_score is None:
        return False

    if direction == StudyDirection.MINIMIZE:
        return val_score < best_score
    return val_score > best_score


class StepwiseTuner:
    """A StepwiseTuner corresponds to a sequence of optimization tasks.

    Args:
        objective:
             A callable that implements objective function. The callable must accept
             a class:`~optuna.Trial` object and a dictionary of parameters to optimize.
        steps:
            List of (step_name, :class:`~.Step` or Callable[[Dict[str, Any]], class:`~.Step`])
            tuples that will be optimized in the
            in the order in which they are listed.
        default_params:
            The parameters that will serve as a baseline for the optimization in order
            to avoid performance regression.
        study:
            The study that will hold the trials for the sequence of steps.  If this argument
            is set to :obj:`None`, a default study is created.

    Attributes:
        study:
            Return the study holding the trials.
        best_params:
            Return the dictionary of best parameters found.
    """

    def __init__(
        self,
        objective: StepObjectiveType,
        steps: Sequence[Tuple[str, StepType]],
        default_params: Optional[Dict[str, Any]] = None,
        study: Optional[optuna.Study] = None,
    ):
        self.steps = steps
        self.objective = objective
        self.default_params = copy.deepcopy(default_params or {})
        self.best_params = copy.deepcopy(self.default_params)
        self.study = study or optuna.create_study()

        self.logger = logging.get_logger(__name__)

    @property
    def step_name_key(self) -> str:
        """Return the trials'`system_attrs` key indicating the name of the step."""
        return f"{type(self).__name__}:step_name"

    @property
    def best_value(self) -> float:
        """Return the best objective value in the study.

        Returns:
            A float representing the best objective value.

        """
        try:
            return self.study.best_value
        except ValueError:
            # Return the default score because no trials have completed.
            return -np.inf if self.study.direction == "minimize" else np.inf

    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[_OptunaCallback]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        """Optimize the objective function by executing each step sequentially.

        A pre-step will be added in order to establish a baseline with default parameters
        and avoid performance regression. Therefore,

        Args:
            n_trials:
                The number of trials. If this argument is set to :obj:`None`, there is no
                limitation on the number of trials. If :obj:`timeout` is also set to :obj:`None`,
                the study continues to create trials until it receives a termination signal such
                as Ctrl+C or SIGTERM.
            timeout:
                Stop study after the given number of second(s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If :obj:`n_trials` is
                also set to :obj:`None`, the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            n_jobs:
                The number of parallel jobs. If this argument is set to :obj:`-1`, the number is
                set to CPU count.
            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e. the study will stop for any
                exception except for :class:`~optuna.exceptions.TrialPruned`.
            callbacks:
                List of callback functions that are invoked at the end of each trial. Each function
                must accept two parameters with the following types in this order:
                :class:`~optuna.study.Study` and :class:`~optuna.FrozenTrial`.
            gc_after_trial:
                Flag to determine whether to automatically run garbage collection after each trial.
                Set to :obj:`True` to run the garbage collection, :obj:`False` otherwise.
                When it runs, it runs a full collection by internally calling :func:`gc.collect`.
                If you see an increase in memory consumption over several trials, try setting this
                flag to :obj:`True`.

                .. seealso::

                    :ref:`out-of-memory-gc-collect`

            show_progress_bar:
                Flag to show progress bars or not. To disable progress bar, set this ``False``.
                Currently, progress bar is experimental feature and disabled
                when ``n_jobs`` :math:`\\ne 1`.

        Raises:
            RuntimeError:
                If nested invocation of this method occurs.

        """
        self._check_n_trials(n_trials)
        optimize_kwargs = {
            "n_jobs": n_jobs,
            "catch": catch,
            "callbacks": callbacks,
            "gc_after_trial": gc_after_trial,
            "show_progress_bar": show_progress_bar,
        }

        steps: Sequence[Tuple[str, StepType]] = self.steps
        if not self.study.trials:
            # Establish a baseline to avoid performance regression
            baseline_step = (f"{self.step_name_key}:baseline", _NoOpStep(self.best_params))
            steps = [baseline_step, *steps]
            if n_trials is not None:
                n_trials += 1

        with _TimeBudget(timeout) as time_budget:
            trial_budget = _TrialBudget(n_trials)

            for step_name, step in steps:
                if self._check_depleted_budgets(step_name, (trial_budget, time_budget)):
                    break

                if callable(step):
                    step = step(self.best_params)

                if not any((n_trials, timeout, step.n_trials, step.timeout)):
                    raise ValueError(
                        "When 'n_trials' and 'timeout' are both None, all steps should "
                        + "at least specify 'n_trials' or 'timeout'. "
                        + f"Found unconstrained step: {step_name}."
                    )

                self._optimize_step(
                    step_name,
                    step,
                    n_trials=trial_budget.try_spend(step.n_trials or n_trials),
                    timeout=time_budget.try_spend(step.timeout or timeout),
                    **optimize_kwargs,  # type: ignore
                )

        if self.study.trials:
            self.logger.info(
                f"Finished optimization. Best value is {self.study.best_trial.value} "
                + f"with parameters: {self.best_params}."
            )

    def _optimize_step(
        self,
        step_name: str,
        step: Step,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[_OptunaCallback]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        if show_progress_bar:
            callbacks = callbacks or []
            callbacks.append(self._progress_bar_callback(step_name))

        default_sampler = self.study.sampler
        default_pruner = self.study.pruner
        self.study.sampler = step.sampler or default_sampler
        self.study.pruner = step.pruner or default_pruner

        prev_best_value = self.best_value
        self.study.optimize(
            self._create_objective(self.best_params, step_name, step),
            n_trials=n_trials or step.n_trials,
            timeout=timeout or step.timeout,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )
        if is_better(self.study.direction, self.study.best_trial.value, prev_best_value):
            self.best_params.update(self.study.best_trial.params)

        self.study.sampler = default_sampler
        self.study.pruner = default_pruner

    def _create_objective(
        self, base_params: Dict[str, Any], step_name: str, step: Step
    ) -> ObjectiveFuncType:
        def objective(trial: optuna.Trial) -> float:
            trial.set_system_attr(self.step_name_key, step_name)

            params = copy.deepcopy(base_params)
            params.update(step.suggest(trial))
            return self.objective(trial, params)

        return objective

    def _progress_bar_callback(self, step_name: str) -> _OptunaCallback:
        def _callback(study: optuna.Study, trial: FrozenTrial) -> None:
            pbar = study._progress_bar  # type: ignore
            step_desc = f"{step_name}, validation score: {study.best_value:.6f}"
            pbar._progress_bar.set_description(step_desc)

        return _callback

    def _check_n_trials(self, n_trials: Optional[int]) -> None:
        if n_trials is None:
            return

        n_steps = len(self.steps)
        if self.study.trials:
            step_mg = f"{n_steps} steps"
        else:
            step_mg = (
                f"{n_steps} + 1 steps "
                + "(one extra trial is required for establishing a baseline)"
            )
            n_steps += 1

        if n_trials < n_steps:
            self.logger.warning(f"{n_trials} n_trials < {step_mg}.")

    def _check_depleted_budgets(self, step_name: str, budgets: Iterable[_BaseBudget]) -> bool:
        is_depleted = False
        for budget in budgets:
            if budget.is_depleted:
                is_depleted = True
                self.logger.warning(
                    f"Not enough {budget.unit_name}s remaining. "
                    + f"Skipping step '{step_name}' and following steps."
                )
        return is_depleted
