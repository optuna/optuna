import itertools
from typing import Tuple

import pytest

import optuna
from optuna._updated_trials_queue import UpdatedTrialsQueue
from optuna.trial import TrialState


@pytest.mark.parametrize(
    "states, deepcopy",
    itertools.product(
        (
            (TrialState.WAITING,),
            (TrialState.RUNNING,),
            (TrialState.COMPLETE,),
            (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
        ),
        (True, False),
    ),
)
def test_updated_trials_queue(states: Tuple[TrialState], deepcopy: bool) -> None:
    study = optuna.create_study()
    queue = UpdatedTrialsQueue(study, states)

    # Test empty queue.
    with pytest.raises(IndexError):
        queue.get(deepcopy=deepcopy)

    study.enqueue_trial({})
    if TrialState.WAITING in states:
        queue.get(deepcopy=deepcopy).state == TrialState.WAITING
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    trial = study.ask()
    if TrialState.RUNNING in states:
        queue.get(deepcopy=deepcopy).state == TrialState.RUNNING
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    study.tell(trial, 0)
    if TrialState.COMPLETE in states:
        queue.get(deepcopy=deepcopy).state == TrialState.COMPLETE
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    # The queue is empty.
    with pytest.raises(IndexError):
        queue.get(deepcopy=deepcopy)


@pytest.mark.parametrize(
    "states, deepcopy",
    itertools.product(
        (
            (TrialState.WAITING,),
            (TrialState.RUNNING,),
            (TrialState.COMPLETE,),
            (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
        ),
        (True, False),
    ),
)
def test_get_trials(states: Tuple[TrialState], deepcopy: bool) -> None:
    study = optuna.create_study()
    queue = UpdatedTrialsQueue(study, states)

    # Test empty queue.
    assert queue.get_trials(deepcopy=deepcopy) == []

    study.enqueue_trial({})
    if TrialState.WAITING in states:
        assert len(queue.get_trials(deepcopy=deepcopy)) == 1
    else:
        assert queue.get_trials(deepcopy=deepcopy) == []

    trial = study.ask()
    if TrialState.RUNNING in states:
        assert len(queue.get_trials(deepcopy=deepcopy)) == 1
    else:
        assert queue.get_trials(deepcopy=deepcopy) == []

    study.tell(trial, 0)
    if TrialState.COMPLETE in states:
        assert len(queue.get_trials(deepcopy=deepcopy)) == 1
    else:
        assert queue.get_trials(deepcopy=deepcopy) == []

    # The queue is empty.
    assert queue.get_trials(deepcopy=deepcopy) == []


@pytest.mark.parametrize(
    "states, deepcopy",
    itertools.product(
        (
            (TrialState.WAITING,),
            (TrialState.RUNNING,),
            (TrialState.COMPLETE,),
            (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
        ),
        (True, False),
    ),
)
def test_updated_trials_queue_with_multi_trials(states: Tuple[TrialState], deepcopy: bool) -> None:
    study = optuna.create_study()
    queue = UpdatedTrialsQueue(study, states)

    study.enqueue_trial({})
    study.enqueue_trial({})
    if TrialState.WAITING in states:
        queue.get(deepcopy=deepcopy).state == TrialState.WAITING
        queue.get(deepcopy=deepcopy).state == TrialState.WAITING
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    trial0 = study.ask()
    trial1 = study.ask()
    if TrialState.RUNNING in states:
        queue.get(deepcopy=deepcopy).state == TrialState.RUNNING
        queue.get(deepcopy=deepcopy).state == TrialState.RUNNING
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    study.tell(trial1, 0)
    if TrialState.COMPLETE in states:
        trial = queue.get(deepcopy=deepcopy)
        trial.state == TrialState.COMPLETE
        trial.number == 1
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    study.tell(trial0, 0)
    if TrialState.COMPLETE in states:
        trial = queue.get(deepcopy=deepcopy)
        trial.state == TrialState.COMPLETE
        trial.number == 0
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    # The queue is empty.
    with pytest.raises(IndexError):
        queue.get(deepcopy=deepcopy)


@pytest.mark.parametrize(
    "states, deepcopy",
    itertools.product(
        (
            (TrialState.WAITING,),
            (TrialState.RUNNING,),
            (TrialState.COMPLETE,),
            (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
        ),
        (True, False),
    ),
)
def test_get_trials_with_multi_trials(states: Tuple[TrialState], deepcopy: bool) -> None:
    study = optuna.create_study()
    queue = UpdatedTrialsQueue(study, states)

    study.enqueue_trial({})
    study.enqueue_trial({})
    if TrialState.WAITING in states:
        assert len(queue.get_trials(deepcopy=deepcopy)) == 2
    else:
        assert queue.get_trials(deepcopy=deepcopy) == []

    trial0 = study.ask()
    trial1 = study.ask()
    if TrialState.RUNNING in states:
        assert len(queue.get_trials(deepcopy=deepcopy)) == 2
    else:
        assert queue.get_trials(deepcopy=deepcopy) == []

    study.tell(trial1, 0)
    if TrialState.COMPLETE in states:
        trials = queue.get_trials(deepcopy=deepcopy)
        assert len(trials) == 1
        assert trials[0].number == 1
    else:
        assert queue.get_trials(deepcopy=deepcopy) == []

    study.tell(trial0, 0)
    if TrialState.COMPLETE in states:
        trials = queue.get_trials(deepcopy=deepcopy)
        assert len(trials) == 1
        assert trials[0].number == 0
    else:
        assert queue.get_trials(deepcopy=deepcopy) == []

    # The queue is empty.
    assert queue.get_trials(deepcopy=deepcopy) == []


@pytest.mark.parametrize(
    "states, deepcopy",
    itertools.product(
        (
            (TrialState.WAITING,),
            (TrialState.RUNNING,),
            (TrialState.COMPLETE,),
            (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
        ),
        (True, False),
    ),
)
def test_updated_trials_queue_with_partial_get(states: Tuple[TrialState], deepcopy: bool) -> None:
    study = optuna.create_study()
    queue = UpdatedTrialsQueue(study, states)

    study.enqueue_trial({})
    study.enqueue_trial({})
    if TrialState.WAITING in states:
        queue.get(deepcopy=deepcopy).state == TrialState.WAITING
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    trial0 = study.ask()
    trial1 = study.ask()
    if TrialState.RUNNING in states:
        queue.get(deepcopy=deepcopy).state == TrialState.RUNNING
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)

    study.tell(trial0, 0)
    study.tell(trial1, 0)
    if TrialState.COMPLETE in states:
        queue.get(deepcopy=deepcopy).state == TrialState.COMPLETE
    else:
        with pytest.raises(IndexError):
            queue.get(deepcopy=deepcopy)


@pytest.mark.parametrize(
    "states",
    (
        (TrialState.WAITING, TrialState.RUNNING),
        (TrialState.WAITING, TrialState.COMPLETE),
        (TrialState.RUNNING, TrialState.COMPLETE),
        (TrialState.WAITING, TrialState.RUNNING, TrialState.COMPLETE),
        (
            TrialState.WAITING,
            TrialState.RUNNING,
            TrialState.COMPLETE,
            TrialState.PRUNED,
            TrialState.FAIL,
        ),
    ),
)
def test_invalid_states_combination(states: Tuple[TrialState]) -> None:
    study = optuna.create_study()
    with pytest.raises(RuntimeError):
        UpdatedTrialsQueue(study, states)
