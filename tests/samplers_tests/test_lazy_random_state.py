from optuna.samplers._lazy_random_sate import LazyRandomSate


def test_lazy_state() -> None:
    state = LazyRandomSate()
    assert state._rng is None
    state.rng.seed(1)
    assert state._rng is not None
