from optuna.distributions import FloatDistribution
from optuna.terminator.improvement.gp.botorch import _BoTorchGaussianProcess
from optuna.trial import create_trial


def test_fit_predict() -> None:
    # A typical fit-predict scenario is being tested here, where there are more than one trials
    # and the Gram matrix is a regular one.
    trials = [
        create_trial(
            value=1.0,
            distributions={
                "bacon": FloatDistribution(-1.0, 1.0),
                "egg": FloatDistribution(-1.0, 1.0),
            },
            params={
                "bacon": 1.0,
                "egg": 0.0,
            },
        ),
        create_trial(
            value=-1.0,
            distributions={
                "bacon": FloatDistribution(-1.0, 1.0),
                "egg": FloatDistribution(-1.0, 1.0),
            },
            params={
                "bacon": 0.0,
                "egg": 1.0,
            },
        ),
    ]

    gp = _BoTorchGaussianProcess()
    gp.fit(trials)
    gp.predict_mean_std(trials)


def test_fit_predict_single_trial() -> None:
    trials = [
        create_trial(
            value=1.0,
            distributions={
                "bacon": FloatDistribution(-1.0, 1.0),
                "egg": FloatDistribution(-1.0, 1.0),
            },
            params={
                "bacon": 1.0,
                "egg": 0.0,
            },
        ),
    ]

    gp = _BoTorchGaussianProcess()
    gp.fit(trials)
    gp.predict_mean_std(trials)


def test_fit_predict_single_param() -> None:
    trials = [
        create_trial(
            value=1.0,
            distributions={
                "spam": FloatDistribution(-1.0, 1.0),
            },
            params={
                "spam": 1.0,
            },
        ),
    ]

    gp = _BoTorchGaussianProcess()
    gp.fit(trials)
    gp.predict_mean_std(trials)


def test_fit_predict_non_regular_gram_matrix() -> None:
    # This test case validates that the GP class works even when the Gram matrix is non-regular,
    # which typically orrcurs when multiple trials share the same parameters.

    trials = [
        create_trial(
            value=1.0,
            distributions={
                "bacon": FloatDistribution(-1.0, 1.0),
                "egg": FloatDistribution(-1.0, 1.0),
            },
            params={
                "bacon": 1.0,
                "egg": 0.0,
            },
        ),
        create_trial(
            value=1.0,
            distributions={
                "bacon": FloatDistribution(-1.0, 1.0),
                "egg": FloatDistribution(-1.0, 1.0),
            },
            params={
                "bacon": 1.0,
                "egg": 0.0,
            },
        ),
    ]

    gp = _BoTorchGaussianProcess()
    gp.fit(trials)
    gp.predict_mean_std(trials)
