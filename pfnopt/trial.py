class Trial(object):

    def __init__(self, trial_id):
        self.trial_id = trial_id
        self.params = {}
        self.info = {}
        self.result = None
        self.intermediate_results = {}
