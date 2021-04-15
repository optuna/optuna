import os
from tempfile import TemporaryDirectory

import optuna


def test_save_study_and_load_study() -> None:
    src_study = optuna.create_study()

    src_study.set_user_attr("foo", "bar")
    src_study.set_system_attr("baz", "qux")
    src_study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "src_study")

        assert not os.path.exists(path)

        optuna.serialization.save_study(src_study, path)

        assert os.path.exists(path)

        dst_study = optuna.create_study()
        optuna.serialization.load_study(dst_study, path)

        assert src_study.user_attrs == dst_study.user_attrs
        assert src_study.system_attrs == dst_study.system_attrs
        assert len(src_study.trials) == len(dst_study.trials)
