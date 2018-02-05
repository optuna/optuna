# TODO: if appropriate, change to namedtuple
class Trial(object):

    # TODO: add meta data
    def __init__(self, trial_id):
        self.trial_id = trial_id
        self.params = {}
        self.info = {}
        self.result = None
        self.intermediate_results = {}
