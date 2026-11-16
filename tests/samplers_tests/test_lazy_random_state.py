from copy import deepcopy

from optuna.samplers._lazy_random_state import LazyRandomState


def test_lazy_state() -> None:
    state = LazyRandomState()
    assert state._rng is None
    state.rng.seed(1)
    assert state._rng is not None


def test_lazy_state_deepcopy() -> None:
    state = LazyRandomState()
    copied_state = deepcopy(state)

    assert state._rng is None
    assert copied_state._rng is None

    assert state.rng is not copied_state.rng
