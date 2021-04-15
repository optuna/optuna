import os
from tempfile import TemporaryDirectory

import optuna


def test_save_study() -> None:
    src_study = optuna.create_study()

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "study.pkl")

        assert not os.path.exists(path)

        optuna.serialization.save_study(src_study, path)

        assert os.path.exists(path)


def test_load_study() -> None:
    # Depends on the save logic.
    src_study = optuna.create_study()
    src_study.set_user_attr("foo", "bar")
    src_study.set_system_attr("baz", "qux")
    src_study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "study.pkl")
        optuna.serialization.save_study(src_study, path)

        dst_study = optuna.create_study()
        optuna.serialization.load_study(dst_study, path)

        assert src_study.user_attrs == dst_study.user_attrs
        assert src_study.system_attrs == dst_study.system_attrs
        assert len(src_study.trials) == len(dst_study.trials)

    # Multiple objectives.
    src_study = optuna.create_study(directions=["minimize", "maximize"])
    src_study.optimize(
        lambda t: (t.suggest_float("x0", 0, 1), t.suggest_float("x1", 0, 1)), n_trials=3
    )

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "study.pkl")
        optuna.serialization.save_study(src_study, path)

        dst_study = optuna.create_study(directions=["minimize", "maximize"])
        optuna.serialization.load_study(dst_study, path)

        assert len(src_study.trials) == len(dst_study.trials)
