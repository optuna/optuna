import optuna
import yaml

from optuna.facade import create_study_from_dict
from optuna.facade import study_optimize
from optuna.trial import Trial  # NOQA
from optuna import types

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA


def test_sampler():
    # type: () -> None

    yaml_string = '''
    study_foo_test_sampler:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: MedianPruner
      optuna_study_optimize:
        n_trials: 5
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return (args['x'] - 2) ** 2

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_string_config():
    # type: () -> None

    yaml_string = '''
    study_foo_test_string_config:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: MedianPruner
      optuna_study_optimize:
        n_trials: 5
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize('args')
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return (args['x'] - 2) ** 2

    study = objective(args=params_dict)
    assert isinstance(study, optuna.study.Study)


def test_return_study_false():
    # type: () -> None

    yaml_string = '''
    study_foo_test_return_study_false:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: MedianPruner
      optuna_study_optimize:
        n_trials: 5
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize(return_study=False)
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return (args['x'] - 2) ** 2

    metric = objective(params_dict)
    assert isinstance(metric, float)


def test_specify_study_name():
    # type: () -> None

    yaml_string = '''
    study_foo:
      optuna_create_study:
        direction: minimize
      optuna_study_optimize:
        n_trials: 5
    study_bar:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: MedianPruner
      optuna_study_optimize:
        n_trials: 5
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize(study_name='study_bar')
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return (args['x'] - 2) ** 2

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_simple():
    # type: () -> None

    yaml_string = '''
    study_foo_test_simple:
      optuna_create_study:
        direction: minimize
      optuna_study_optimize:
        n_trials: 5
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return (args['x'] - 2) ** 2

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_empty_sampler():
    # type: () -> None

    yaml_string = '''
    study_foo_test_empty_sampler:
      optuna_create_study:
        direction: minimize
        sampler:
        pruner:
      optuna_study_optimize:
        n_trials: 5
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return (args['x'] - 2) ** 2

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_nested():
    # type: () -> None

    yaml_string = '''
    study_foo_test_nested:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: MedianPruner
      optuna_study_optimize:
        n_trials: 5
    level_1:
      x1:
        optuna_suggest: uniform
        low: -10
        high: 10
      level_2:
        x2:
          optuna_suggest: uniform
          low: -10
          high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return args['level_1']['x1'] + args['level_1']['level_2']['x2']

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_various_suggests():
    # type: () -> None

    yaml_string = '''
    study_foo_test_various_suggests:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: MedianPruner
      optuna_study_optimize:
        n_trials: 5
    x_categorical:
      optuna_suggest: categorical
      choices:
        - 10
        - 20
        - 30
    x_discrete_uniform:
      optuna_suggest: discrete_uniform
      low: 1000
      high: 5000
      q: 1000
    x_int:
      optuna_suggest: int
      low: 1
      high: 5
    x_loguniform:
      optuna_suggest: loguniform
      low: 100
      high: 1000
    x_uniform:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        metric = (
            args['x_categorical']
            + args['x_discrete_uniform']
            + args['x_int']
            + args['x_loguniform']
            + args['x_uniform']
        )
        return metric

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_skopt_sampler():
    # type: () -> None

    yaml_string = '''
    study_foo_skopt_sampler:
      optuna_create_study:
        direction: minimize
        sampler:
          type: SkoptSampler
        pruner:
          type: MedianPruner
      optuna_study_optimize:
        n_trials: 5
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return (args['x'] - 2) ** 2

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_random_sampler():
    # type: () -> None

    yaml_string = '''
    study_foo_test_random_sampler:
      optuna_create_study:
        direction: minimize
        sampler:
          type: RandomSampler
        pruner:
          type: MedianPruner
      optuna_study_optimize:
        n_trials: 5
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float
        # Optionally, pruning can be set up using args['optuna_trial']
        return (args['x'] - 2) ** 2

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_pruner():
    # type: () -> None

    yaml_string = '''
    study_foo_test_pruner:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: MedianPruner
          n_startup_trials: 5
          n_warmup_steps: 0
      optuna_study_optimize:
        n_trials: 10
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float

        for i in range(10):
            metric = (args['x'] - 2) ** 2 - i
            trial = args.get('optuna_trial')
            if trial:
                trial.report(metric, i)
                if trial.should_prune():
                    raise optuna.structs.TrialPruned()
        return metric

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_successive_halving_pruner():
    # type: () -> None

    yaml_string = '''
    study_foo_test_successive_halving_pruner:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: SuccessiveHalvingPruner
          min_resource: 1
          reduction_factor: 4
          min_early_stopping_rate: 0
      optuna_study_optimize:
        n_trials: 10
    x:
      optuna_suggest: uniform
      low: -10
      high: 10
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float

        for i in range(10):
            metric = (args['x'] - 2) ** 2 - i
            trial = args.get('optuna_trial')
            if trial:
                trial.report(metric, i)
                if trial.should_prune():
                    raise optuna.structs.TrialPruned()
        return metric

    study = objective(params_dict)
    assert isinstance(study, optuna.study.Study)


def test_no_optuna():
    # type: () -> None

    yaml_string = '''
    x: 2.222
    '''

    params_dict = yaml.safe_load(yaml_string)

    @study_optimize()
    def objective(args):
        # type: (Dict[str, Any]) -> float
        return args['x'] - 2

    metric = objective(params_dict)
    assert abs(metric - 0.222) < 2.3e-06


def test_create_study_from_dict():
    # type: () -> None

    yaml_string = '''
    study_foo_test_create_study_from_dict:
      optuna_create_study:
        direction: minimize
        sampler:
          type: TPESampler
          seed: 123
        pruner:
          type: MedianPruner
    '''

    params_dict = yaml.safe_load(yaml_string)

    study = create_study_from_dict(params_dict)
    assert isinstance(study, optuna.study.Study)

    def objective(trial):
        # type: (Trial) -> float

        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2

    study.optimize(objective, n_trials=5)


def test_create_study_from_dict_simple():
    # type: () -> None

    yaml_string = '''
    direction: minimize
    sampler:
      type: TPESampler
      seed: 123
    pruner:
      type: MedianPruner
    '''

    params_dict = yaml.safe_load(yaml_string)

    study = create_study_from_dict(params_dict)
    assert isinstance(study, optuna.study.Study)

    def objective(trial):
        # type: (Trial) -> float

        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2

    study.optimize(objective, n_trials=5)


def test_create_study_from_dict_explicit_study_name():
    # type: () -> None

    yaml_string = '''
    study_name: study_foo_test_create_study_from_dict_explicit_study_name
    direction: minimize
    sampler:
      type: TPESampler
      seed: 123
    pruner:
      type: MedianPruner
    '''

    params_dict = yaml.safe_load(yaml_string)

    study = create_study_from_dict(params_dict)
    assert isinstance(study, optuna.study.Study)

    def objective(trial):
        # type: (Trial) -> float

        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2

    study.optimize(objective, n_trials=5)


def test_create_study_from_dict_with_key_only():
    # type: () -> None

    yaml_string = '''
    study_foo_test_create_study_from_dict_with_key_only:
      optuna_create_study:
    '''

    params_dict = yaml.safe_load(yaml_string)

    study = create_study_from_dict(params_dict)
    assert isinstance(study, optuna.study.Study)

    def objective(trial):
        # type: (Trial) -> float

        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2

    study.optimize(objective, n_trials=5)


def test_create_study_from_dict_default():
    # type: () -> None

    study = create_study_from_dict()
    assert isinstance(study, optuna.study.Study)

    def objective(trial):
        # type: (Trial) -> float

        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2

    study.optimize(objective, n_trials=5)
