# TODO: if appropriate, change to namedtuple
class Trial(object):

    # TODO: add meta data
    def __init__(self, trial_id, params, result, intermediate_results=None):
        self.trial_id = trial_id
        self.params = params
        self.result = result
        self.intermediate_results = intermediate_results or {}
