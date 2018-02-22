class BaseSampler(object):

    def sample(self, distribution, observation_pairs):
        raise NotImplementedError
