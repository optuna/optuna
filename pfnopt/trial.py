import enum


class State(enum.Enum):

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3


class SystemAttributes(object):

    def __init__(self):
        self.datetime_start = None
        self.datetime_complete = None


class Trial(object):

    def __init__(self, trial_id):
        self.trial_id = trial_id
        self.state = State.RUNNING
        self.params = {}
        self.system_attrs = SystemAttributes()
        self.user_attrs = {}
        self.value = None
        self.intermediate_values = {}
