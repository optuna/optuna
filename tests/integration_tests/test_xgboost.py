import pytest
import xgboost as xgb

import pfnopt
from pfnopt.integration.xgboost import XGBoostPruningExtension


class FixedValuePruner(pfnopt.pruners.BasePruner):

    def __init__(self, is_pruning):
        # type: (bool) -> None

        self.is_pruning = is_pruning

    def prune(self, storage, study_id, trial_id, step):
        # type: (pfnopt.storages.BaseStorage, int, int, int) -> bool

        return self.is_pruning


def test_xgboost_prunint_extension_call():
    # type: () -> None

    env = xgb.core.CallbackEnv(model='test',
                               cvfolds=1,
                               begin_iteration=0,
                               end_iteration=1,
                               rank=1,
                               iteration=1,
                               evaluation_result_list=[['validation-error', 1.]])

    # The pruner is deactivated.
    study = pfnopt.create_study(pruner=FixedValuePruner(False))
    trial = study._run_trial(func=lambda _: 1.0, catch=(Exception,))
    extension = XGBoostPruningExtension(trial, 'validation-error')
    extension(env)

    # The pruner is activated.
    study = pfnopt.create_study(pruner=FixedValuePruner(True))
    trial = study._run_trial(func=lambda _: 1.0, catch=(Exception,))
    extension = XGBoostPruningExtension(trial, 'validation-error')
    with pytest.raises(pfnopt.structs.TrialPruned):
        extension(env)


def test_xgboost_pruning_extension():
    # type: () -> None

    # TODO(Yanase): This seems to be an integration test rather than a unit test.
    def objective(trial):
        # type: (pfnopt.trial.Trial) -> float

        dtrain = xgb.DMatrix([[1.]], label=[1.])
        dtest = xgb.DMatrix([[1.]], label=[1.])

        pruning_callback = XGBoostPruningExtension(trial, 'validation-error')
        xgb.train({'silent': 1, 'objective': 'binary:logistic'}, dtrain, 1,
                  evals=[(dtest, 'validation')], callbacks=[pruning_callback])
        return 1.0

    study = pfnopt.create_study(pruner=FixedValuePruner(True))
    study.run(objective, n_trials=1)
    assert study.trials[0].state == pfnopt.structs.TrialState.PRUNED

    study = pfnopt.create_study(pruner=FixedValuePruner(False))
    study.run(objective, n_trials=1)
    assert study.trials[0].state == pfnopt.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.
