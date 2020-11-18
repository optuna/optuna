import warnings

from optuna import _study_direction
from optuna import _study_summary
from optuna import exceptions
from optuna import trial
from optuna._deprecated import deprecated


_message = (
    "`structs` is deprecated. Classes have moved to the following modules. "
    "`structs.StudyDirection`->`study.StudyDirection`, "
    "`structs.StudySummary`->`study.StudySummary`, "
    "`structs.FrozenTrial`->`trial.FrozenTrial`, "
    "`structs.TrialState`->`trial.TrialState`, "
    "`structs.TrialPruned`->`exceptions.TrialPruned`."
)
warnings.warn(_message, FutureWarning)

# The use of the structs.StudyDirection is deprecated and it is recommended that you use
# study.StudyDirection instead. See the API reference for more details.
StudyDirection = _study_direction.StudyDirection

# The use of the structs.TrialState is deprecated and it is recommended that you use
# trial.TrialState instead. See the API reference for more details.
TrialState = trial.TrialState


@deprecated(
    "1.4.0",
    text=(
        "This class was moved to :mod:`~optuna.trial`. Please use "
        ":class:`~optuna.trial.FrozenTrial` instead."
    ),
)
class FrozenTrial(trial.FrozenTrial):
    pass


@deprecated(
    "1.4.0",
    text=(
        "This class was moved to :mod:`~optuna.study`. Please use "
        ":class:`~optuna.study.StudySummary` instead."
    ),
)
class StudySummary(_study_summary.StudySummary):
    pass


@deprecated(
    "0.19.0",
    text=(
        "This class was moved to :mod:`~optuna.exceptions`. Please use "
        ":class:`~optuna.exceptions.TrialPruned` instead."
    ),
)
class TrialPruned(exceptions.TrialPruned):
    """Exception for pruned trials."""

    pass
