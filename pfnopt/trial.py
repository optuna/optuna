import enum


class State(enum.Enum):

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3


class Trial(object):

    def __init__(self, trial_id):
        self.trial_id = trial_id
        self.state = State.RUNNING
        self.params = {}
        self.params_in_internal_repr = {}  # TODO(Akiba): eliminate me
        self.system_attrs = {}
        self.user_attrs = {}
        self.value = None
        self.intermediate_values = {}
