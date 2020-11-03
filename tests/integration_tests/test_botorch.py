import optuna
from optuna.integration import BoTorchSampler

# TODO(hvy): Test optional import.
# TODO(hvy): Test warp/unwarp and check bounds.
# TODO(hvy): Test 0 warmup trials.



def test_botorch_single() -> None:
    sampler = BoTorchSampler()

def test_botorch_infer_relative_search_space_1d() -> None:

    sampler = BoTorchSampler()
    study = optuna.create_study(sampler=sampler)

    # The distribution has only one candidate.
    study.optimize(lambda t: t.suggest_int("x", 1.0, 1.0), n_trials=1)
    assert sampler.infer_relative_search_space(study, study.best_trial) == {}
