from threading import Event
from threading import Thread

from optuna.storages import BaseStorage
from optuna.storages import fail_stale_trials
from optuna.trial import Trial


class HeartBeatFactory:
    def __init__(self, trial: Trial):
        self.trial = trial

    def launch(self) -> None:
        fail_stale_trials(self.trial.study)

        def _record_heartbeat(trial_id: int, storage: BaseStorage, stop_event: Event) -> None:
            heartbeat_interval = storage.get_heartbeat_interval()
            assert heartbeat_interval is not None
            while True:
                storage.record_heartbeat(trial_id)
                if stop_event.wait(timeout=heartbeat_interval):
                    return

        stop_event = Event()
        thread = Thread(
            target=_record_heartbeat,
            args=(self.trial._trial_id, self.trial.study._storage, stop_event),
        )
        thread.start()
        self.stop_event = stop_event
        self.thread = thread

    def stop(self):
        assert self.stop_event is not None
        assert self.thread is not None

        self.stop_event.set()
        self.thread.join()
