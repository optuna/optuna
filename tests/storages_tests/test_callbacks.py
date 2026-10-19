import optuna
from optuna.storages import RetryHeartbeatStaleTrialCallback
from optuna.testing.storages import StorageSupplier
from optuna.trial import TrialState


def test_retry_history_with_success() -> None:
    # Trial that is not a retry should return empty list.
    with StorageSupplier("sqlite") as storage:
        study = optuna.create_study(storage=storage)
        trial = study.ask()
        study.tell(trial, 1.0)  # Complete the trial to get FrozenTrial.
        assert RetryHeartbeatStaleTrialCallback.retry_history(study.trials[0]) == []


def test_retry_history_with_failures() -> None:
    max_retry = 3
    callback = RetryHeartbeatStaleTrialCallback(max_retry=max_retry)
    with StorageSupplier(
        "sqlite", heartbeat_interval=1, grace_period=2, heartbeat_stale_trial_callback=callback
    ) as storage:
        study = optuna.create_study(storage=storage)
        for n_retries in range(1, max_retry + 1):
            # Create a trial and manually set it as failed.
            trial = study.ask({"x": optuna.distributions.FloatDistribution(-1, 1)})
            storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
            callback(study, study.trials[trial.number])  # Manually call callback.

            # Get all trials after another trial for retry is enqueued.
            trials = study.trials
            # Original failed trial + retried trials.
            assert len(trials) == n_retries + 1
            # The last trial before the retried trial must be failed.
            assert trials[trial.number].state == TrialState.FAIL
            # Retry should show the full history of the previous trials.
            assert RetryHeartbeatStaleTrialCallback.retry_history(
                trials[trial.number + 1]
            ) == list(range(n_retries))


def test_retry_history_with_more_than_max_retry() -> None:
    max_retry = 3
    callback = RetryHeartbeatStaleTrialCallback(max_retry=max_retry)
    with StorageSupplier(
        "sqlite", heartbeat_interval=1, grace_period=2, heartbeat_stale_trial_callback=callback
    ) as storage:
        study = optuna.create_study(storage=storage)
        for n_retries in range(max_retry + 2):
            trial = study.ask({"x": optuna.distributions.FloatDistribution(-1, 1)})
            storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
            callback(study, study.trials[trial.number])  # Manually call callback.

        # After max_retry retries, the parameter should be different from the original.
        assert study.trials[0].params["x"] != study.trials[-1].params["x"]


def test_retry_callback_system_attrs_filtering() -> None:
    callback = RetryHeartbeatStaleTrialCallback()
    with StorageSupplier("sqlite") as storage:
        study = optuna.create_study(storage=storage)
        trial = study.ask({"x": optuna.distributions.FloatDistribution(-1, 1)})
        trial_id = trial._trial_id

        # Set both specification system attrs and runtime execution system attrs.
        storage.set_trial_system_attr(trial_id, "fixed_params", {"x": 0.5})
        storage.set_trial_system_attr(trial_id, "grid_id", 42)
        storage.set_trial_system_attr(trial_id, "search_space", {"x": [0.5, 1.0]})
        storage.set_trial_system_attr(trial_id, "completed_rung_0", 1.5)
        storage.set_trial_system_attr(trial_id, "completed_rung_1", 2.0)
        storage.set_trial_system_attr(trial_id, "constraints:c", 5.0)
        storage.set_trial_system_attr(trial_id, "artifact:artifact_id", '{"id": 1}')
        storage.set_trial_system_attr(trial_id, "x_for_tell", [0.5])
        storage.set_trial_user_attr(trial_id, "user_key", "user_value")
        storage.set_trial_state_values(trial_id, state=TrialState.FAIL)

        frozen = storage.get_trial(trial_id)
        callback(study, frozen)

        retry_trial = study.trials[1]
        assert retry_trial.state == TrialState.WAITING
        assert retry_trial.user_attrs == {"user_key": "user_value"}

        # Specification attributes and retry metadata should be inherited.
        assert retry_trial.system_attrs["failed_trial"] == 0
        assert retry_trial.system_attrs["retry_history"] == [0]
        assert retry_trial.system_attrs["fixed_params"] == {"x": 0.5}
        assert retry_trial.system_attrs["grid_id"] == 42
        assert retry_trial.system_attrs["search_space"] == {"x": [0.5, 1.0]}

        # Runtime execution attributes must not be inherited.
        assert "completed_rung_0" not in retry_trial.system_attrs
        assert "completed_rung_1" not in retry_trial.system_attrs
        assert "constraints:c" not in retry_trial.system_attrs
        assert "artifact:artifact_id" not in retry_trial.system_attrs
        assert "x_for_tell" not in retry_trial.system_attrs


def test_retry_callback_pruner_and_constraint_isolation() -> None:
    callback = RetryHeartbeatStaleTrialCallback()
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2)
    with StorageSupplier("sqlite") as storage:
        study = optuna.create_study(storage=storage, pruner=pruner)

        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", 0, 10)
            trial.set_constraint("c", -1.0)
            for step in range(4):
                trial.report(x, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return x

        # Trial 0 sets constraint c=5.0 and passes rungs before failing.
        study.enqueue_trial({"x": 5.0})
        t0 = study.ask()
        t0.set_constraint("c", 5.0)
        for step in range(4):
            t0.report(5.0, step)
            t0.should_prune()

        # Competing trials (x=1.0) complete and establish better scores at rungs.
        for _ in range(2):
            study.enqueue_trial({"x": 1.0})
        study.optimize(objective, n_trials=2)

        # Fail trial 0 and enqueue its retry (trial 3).
        storage.set_trial_state_values(t0._trial_id, state=TrialState.FAIL)
        callback(study, study.trials[0])

        retry_trial = study.trials[3]
        assert retry_trial.state == TrialState.WAITING
        assert "completed_rung_0" not in retry_trial.system_attrs
        assert "constraints:c" not in retry_trial.system_attrs
        assert retry_trial.constraints == {}

        # Retried trial (x=5.0) should be evaluated freshly and pruned at step 1.
        study.optimize(objective, n_trials=1)

        retried_frozen = study.trials[3]
        assert retried_frozen.state == TrialState.PRUNED
        assert retried_frozen.constraints == {"c": -1.0}
