from functools import wraps
from six import iteritems

from optuna.integration import SkoptSampler # NOQA
from optuna.pruners import MedianPruner # NOQA
from optuna.pruners import SuccessiveHalvingPruner # NOQA
from optuna.samplers import RandomSampler # NOQA
from optuna.samplers import TPESampler # NOQA
from optuna.study import create_study # NOQA
from optuna.study import Study  # NOQA
from optuna.trial import Trial  # NOQA
from optuna import types

if types.TYPE_CHECKING:
    from typing import Callable # NOQA
    from typing import Dict  # NOQA
    from typing import Union  # NOQA


def _get_suggested_values_recursively(
        parameters, # type: Dict
        trial=None, # type: Trial
):
    # type: (...) -> (Dict, Dict, str)

    assert isinstance(trial, (Trial, type(None),))
    assert isinstance(parameters, dict)

    params_out = dict()
    studies_out = dict()
    suggest_out = None

    for k, v in iteritems(parameters):
        if isinstance(v, dict):
            v = v.copy()

            create_study = 'optuna_create_study' in v
            if create_study:
                studies_out[k] = v

            suggest = v.pop('optuna_suggest', None)
            if suggest and trial:
                v = \
                    trial.suggest_categorical(k, **v) if "categorical" in suggest else \
                    trial.suggest_discrete_uniform(k, **v) if "discrete_uniform" in suggest else \
                    trial.suggest_int(k, **v) if "int" in suggest else \
                    trial.suggest_loguniform(k, **v) if "loguniform" in suggest else \
                    trial.suggest_uniform(k, **v) if "uniform" in suggest else \
                    v

            if not (create_study or suggest):
                v, studies, suggest = _get_suggested_values_recursively(v, trial)
                studies_out.update(studies)

            suggest_out = suggest_out or suggest
        params_out[k] = v
    return params_out, studies_out, suggest_out


def _get_suggested_values(
        parameters, # type: Dict
        trial=None, # type: Trial
):
    # type: (...) -> (Dict, Dict, str)
    parameters = {'_ROOT_': parameters}
    params, studies, suggest = _get_suggested_values_recursively(parameters, trial)
    return params['_ROOT_'], studies, suggest


def optuna_decorator(
        config=0, # type: Union[int, str, None]
        return_study=True, # type: bool
        study_name=0, # type: Union[int, str]
):
    # type: (...) ->  Callable[[Callable], Callable]
    """
    Decorator to tune hyperparameters using Optuna configured by a dictionary.
    
    Use `optuna_create_study` key to specify arguments for
    :func:`~optuna.study.create_study` method.
    If `optuna_create_study` key is missing,
    the wrapped function is run once without using Optuna

    Use `optuna_study_optimize` key to specify arguments for
    :func:`~optuna.study.Study.optimize` method.
    If `optuna_study_optimize` key is missing,
    :func:`~optuna.study.Study.optimize` method is not called.

    For each parameter to tune, add `optuna_suggest` key to specify
    which `suggest_...` method to use as well as its arguments except `name`.

    * :func:`~optuna.trial.Trial.suggest_categorical`
    * :func:`~optuna.trial.Trial.suggest_discrete_uniform`
    * :func:`~optuna.trial.Trial.suggest_int`
    * :func:`~optuna.trial.Trial.suggest_loguniform`
    * :func:`~optuna.trial.Trial.suggest_uniform`

    Optionally, the following objects can be accessed inside the wrapped
    function.

    * :class:`~optuna.trial.Trial` object by ``optuna_trial`` key \
    to use features such as pruning.

    * :class:`~optuna.study.Study` object by ``optuna_study`` key \
    to get information such as the best tuned parameters.

    Example:

        The following code snippet shows how to use optuna_decorator.

        .. code:: python

            from optuna import optuna_decorator
            import yaml

            yaml_string = u'''
            study_foo:
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

            @optuna_decorator()
            def objective(args):
                # Optionally, pruning can be set up using args['optuna_trial']
                return (args['x'] - 2) ** 2

            study = objective(params_dict)

            print('Best parameters:', study.best_params)
            print('Best value:', study.best_value)

    Args:
        config:
            Specify the parameters dictionary including Optuna configuration
            by:

            *  :obj:`int`: the index of the positional argument for the wrapped function
            *  :obj:`str`: the name of the keyword argument for the wrapped function
            *  :obj:`None`: the wrapped function is run once without using Optuna

            :obj:`0` in default, which means the first argument is used.
        return_study:
            If this is :obj:`True` (default), the wrapped function returns
            :class:`~optuna.study.Study` object.
            If this is :obj:`False` , the wrapped function returns
            the best value only.

        study_name:
            The name of study specified in the dictionary.

            *  :obj:`int`: the index of the Optuna study configuration in the dictionary.
            *  :obj:`str`: the name of the Optuna study configuration in the dictionary.

            :obj:`0` in default, which means the first Optuna study configuration
            in the dictionary is used.

    Returns:
        A decorator
    """
    def _optuna_decorator(func):
        # type: (Callable[..., float]) ->  Callable[..., float]
        @wraps(func)
        def wrapper(*args, **kwargs):
            params = \
                args[config] if isinstance(config, int) else \
                kwargs[config] if isinstance(config, str) else \
                dict()
            _, studies_params, _ = _get_suggested_values(params)
            if studies_params:
                study_name_str = \
                    study_name if isinstance(study_name, str) else \
                    list(studies_params.keys())[study_name] if isinstance(study_name, int) else \
                    list(studies_params.keys())[0]
                study_params = studies_params[study_name_str]
                optuna_create_study = study_params.get('optuna_create_study') or dict()

                sampler = optuna_create_study.get('sampler')
                if sampler:
                    sampler_type = sampler.pop('type', None)
                    optuna_create_study['sampler'] = \
                        RandomSampler(**sampler) if 'Random' in sampler_type else \
                        SkoptSampler(**sampler) if 'Skopt' in sampler_type else \
                        TPESampler(**sampler) # if 'TPE' in sampler_type else \

                pruner = optuna_create_study.get('pruner')
                if pruner:
                    pruner_type = pruner.pop('type', None)
                    optuna_create_study['pruner'] = \
                        SuccessiveHalvingPruner(**pruner) \
                            if 'SuccessiveHalving' in pruner_type else \
                        MedianPruner(**pruner) # if 'Median' in pruner_type else \

                optuna_create_study['study_name'] = \
                    optuna_create_study.get('study_name') or \
                    (study_name_str if study_name_str != '_ROOT_' else None)
                study = create_study(**optuna_create_study)

                if 'optuna_study_optimize' in study_params:
                    optuna_study_optimize = study_params.get('optuna_study_optimize') or dict()
                    args_list = list(args)

                    def objective(trial):

                        params_suggested, _, _ = _get_suggested_values(params, trial)
                        params_suggested['optuna_trial'] = trial
                        params_suggested['optuna_study'] = study

                        if isinstance(config, int):
                            args_list[config] = params_suggested
                        if isinstance(config, str):
                            kwargs[config] = params_suggested

                        args_tuple = tuple(args_list)

                        metric = func(*args_tuple, **kwargs)
                        return metric

                    study.optimize(objective, **optuna_study_optimize)

                output = \
                    study if return_study else \
                    study.best_value if optuna_study_optimize is not None else \
                    None
                return output

            metric = func(*args, **kwargs)
            return metric

        return wrapper
    return _optuna_decorator


def _find_key_recursively(
        dictionary, # type: Dict
        keyword, # type: str
):
    # type: (...) -> bool
    if keyword in dictionary:
        return True
    for k, v in iteritems(dictionary):
        if isinstance(v, dict) and _find_key_recursively(v, keyword):
            return True
    return False


def create_study_from_dict(
        params={}, # type: Dict
):
    # type: (...) -> Study
    """
    Create a :class:`~optuna.study.Study` object from a dictionary.

    Example:

        The following code snippet shows how to use create_study_from_dict.

        .. code:: python

            from optuna import create_study_from_dict
            import yaml

            yaml_string = u'''
            direction: minimize
            sampler:
              type: TPESampler
              seed: 123
            pruner:
              type: MedianPruner
            '''

            params_dict = yaml.safe_load(yaml_string)

            study = create_study_from_dict(params_dict)

            def objective(trial):
                x = trial.suggest_uniform('x', -10, 10)
                return (x - 2) ** 2

            study.optimize(objective, n_trials=5)

            print('Best parameters:', study.best_params)
            print('Best value:', study.best_value)

    Args:
        params:
            A dictionary including arguments for
            :func:`~optuna.study.create_study` method.

    Returns:
        A :class:`~optuna.study.Study` object.

    """

    params = \
        dict(optuna_create_study=params) \
        if not _find_key_recursively(params, 'optuna_create_study') else \
        params

    @optuna_decorator()
    def _create_study_from_dict(params):
        pass # NOQA

    return _create_study_from_dict(params)
