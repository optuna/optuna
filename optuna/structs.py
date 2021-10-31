import warnings

from optuna import exceptions
from optuna import trial
from optuna._deprecated import deprecated
from optuna.study import _study_direction
from optuna.study import _study_summary


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

