from optuna.samplers._lazy_random_state import LazyRandomState


def test_lazy_state() -> None:
    state = LazyRandomState()
    assert state._rng is None
    state.rng.seed(1)
    assert state._rng is not None
