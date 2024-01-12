import optuna
from optuna._imports import try_import
from optuna._experimental import experimental_class
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
import numpy as np
import warnings

with try_import() as _imports:
    import scipy.stats as ss

@experimental_class("3.6.0")
class WilcoxonPruner(BasePruner):
    def __init__(
        self,
        *,
        p_threshold: float = 0.2,
        n_startup_steps: int = 0,
    ) -> None:
        if n_startup_steps < 0:
            raise ValueError(
                f"n_startup_steps must be nonnegative but got {n_startup_steps}."
            )
        if not 0.0 <= p_threshold <= 1.0:
            raise ValueError(
                f"p_threshold must be between 0 and 1 but got {p_threshold}."
            )
        
        self._n_startup_steps = n_startup_steps
        self._p_threshold = p_threshold
    
    def prune(self, study: "optuna.study.Study", trial: FrozenTrial) -> bool:

        def extract_step_values(t: FrozenTrial) -> np.ndarray | None:
            step = t.last_step
            if step is None:
                return np.empty((0,))
            
            all_steps = list(t.intermediate_values.keys())
            if all_steps != list(range(1, step+1)):
                return None
            
            intermediate_average_values = np.array([0] + [
                t.intermediate_values[step] for step in all_steps
            ])
            step_values = np.diff(intermediate_average_values * np.arange(step+1))
            return step_values
            
        step = trial.last_step
        if step is None or step < self._n_startup_steps:
            return False
        
        step_values = extract_step_values(trial)
        if step_values is None:
            raise ValueError(
                "WilcoxonPruner requires intermediate average values to be "
                "given at step [1, 2, ..., n], but got "
                f"{list(trial.intermediate_values.keys())} for trial {trial.number})."
            )
        
        if np.any(np.isnan(step_values)):
            warnings.warn(
                f"The intermediate values of the current trial (trial {trial.number}) "
                f"contain NaNs. WilcoxonPruner will not prune this trial."
            )
            return False

        try:
            best_trial = study.best_trial
        except ValueError:
            return False
        
        best_step_values = extract_step_values(best_trial)
        if best_step_values is None:
            warnings.warn(
                "WilcoxonPruner requires intermediate average values to be "
                "given at steps [1, 2, ..., n], but the best trial "
                f"(trial {best_trial.number}) has intermediate values at "
                f"steps {list(best_trial.intermediate_values.keys())}. "
                "WilcoxonPruner will not prune the current trial."
            )
            return False
        
        if len(best_step_values) < len(step_values):
            warnings.warn(
                f"The best trial (trial {best_trial.number}) has less "
                f"({len(best_step_values)}) steps than the current trial. Only the "
                f"first {len(best_step_values)} intermediate values will be compared."
            )
            step_values = step_values[:len(best_step_values)]
        elif len(best_step_values) > len(step_values):
            best_step_values = best_step_values[:len(step_values)]

        if np.any(np.isnan(best_step_values)):
            warnings.warn(
                f"The intermediate values of the best trial (trial {best_trial.number}) "
                f"contain NaNs. WilcoxonPruner will not prune the current trial."
            )
            return False

        diffs = step_values - best_step_values
        diffs[np.isnan(diffs)] = 0  # inf - inf or (-inf) - (-inf)
        if study.direction == StudyDirection.MAXIMIZE:
            diffs *= -1

        if len(diffs) == 0:
            return False
        
        p = ss.wilcoxon(diffs, alternative="greater", zero_method="zsplit").pvalue
        return p < self._p_threshold
