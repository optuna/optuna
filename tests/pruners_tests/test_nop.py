import optuna


def test_nop_pruner() -> None:

    pruner = optuna.pruners.NopPruner()
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)

    # A NopPruner instance is always deactivated.
    assert not trial.should_prune()


def test_repr() -> None:
    pruner = optuna.pruners.NopPruner()
    from optuna.pruners import NopPruner

    restored_pruner = eval(repr(pruner))

    assert isinstance(restored_pruner, NopPruner)
