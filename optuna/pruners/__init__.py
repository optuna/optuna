import optuna
from optuna.pruners._base import BasePruner  # NOQA
from optuna.pruners._hyperband import HyperbandPruner  # NOQA
from optuna.pruners._median import MedianPruner  # NOQA
from optuna.pruners._nop import NopPruner  # NOQA
from optuna.pruners._percentile import PercentilePruner  # NOQA
from optuna.pruners._successive_halving import SuccessiveHalvingPruner  # NOQA
from optuna.pruners._threshold import ThresholdPruner  # NOQA


def _filter_study(
    study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
) -> "optuna.study.Study":
    if isinstance(study.pruner, HyperbandPruner):
        # Create `_BracketStudy` to use trials that have the same bracket id.
        pruner = study.pruner  # type: HyperbandPruner
        return pruner._create_bracket_study(study, pruner._get_bracket_id(study, trial))
    else:
        return study
