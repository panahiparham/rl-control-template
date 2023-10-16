import numpy as np
from PyRlEnvs.domains.RandomWalk import buildRandomWalk, invertedFeatures, tabularFeatures, dependentFeatures
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem
from PyFixedReps.BaseRepresentation import BaseRepresentation

from functools import partial

from utils.policies import Policy


class RandomWalk(BaseProblem):
    def _buildRepresentation(self, name, noise_ratio):
        m = None
        if name == 'tabular':
            m = tabularFeatures(self.states)

        if name == 'inverted':
            m = invertedFeatures(self.states)

        if name == 'dependent':
            m = dependentFeatures(self.states)

        assert m is not None

        if noise_ratio > 0:
            return NoisyMappedRepresentation(m, noise_ratio)
        else:
            return MappedRepresentation(m)

    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector, states: int, behavior: float, target: float, noise: float):
        super().__init__(exp, idx, collector)
        self.exp = exp
        self.idx = idx
        
        self.states = states

        mu_pl = behavior
        mu_probs = np.array([mu_pl, 1 - mu_pl])
        self.behavior = Policy(lambda s: mu_probs, np.random.default_rng(0))


        pi_pl = target
        pi_probs = np.array([pi_pl, 1 - pi_pl])
        self.target = Policy(lambda s: pi_probs, np.random.default_rng(0))

        self.env = buildRandomWalk(self.states)()

        # build representation
        representation = self.env_params['features']
        noise_ratio = noise
        self.rep = self._buildRepresentation(representation, noise_ratio)

        self.observations = (self.rep.features(), )
        self.actions = 2
        self.gamma = 0.99

# problem instances
LongRandChain = partial(RandomWalk, states=101, behavior=0.5, target=0.5, noise=0.0)
ShortRandChain = partial(RandomWalk, states=11, behavior=0.5, target=0.5, noise=0.0)
LongBiasedChain = partial(RandomWalk, states=101, behavior=0.75, target=0.25, noise=0.0)
ShortBiasedChain = partial(RandomWalk, states=11, behavior=0.75, target=0.25, noise=0.0)

NoisyLongRandChain = partial(RandomWalk, states=101, behavior=0.5, target=0.5, noise=0.1)
NoisyShortRandChain = partial(RandomWalk, states=11, behavior=0.5, target=0.5, noise=0.1)
NoisyLongBiasedChain = partial(RandomWalk, states=101, behavior=0.75, target=0.25, noise=0.1)
NoisyShortBiasedChain = partial(RandomWalk, states=11, behavior=0.75, target=0.25, noise=0.1)


class MappedRepresentation(BaseRepresentation):
    def __init__(self, m: np.ndarray):
        self.map = addDummyTerminalState(m)

    def encode(self, s: int):
        return self.map[s]

    def features(self):
        return self.map.shape[1]
    

class NoisyMappedRepresentation(MappedRepresentation):
    def __init__(self, m: np.ndarray, noise_ratio: float):
        super().__init__(m)
        self.noise_ratio = noise_ratio
        self.noise_size = np.floor(self.noise_ratio * super().features()).astype(int) + 1

    def encode(self, s: int):
        encoded = super().encode(s)
        noise = np.random.randint(2, size=(self.noise_size,))
        return np.concatenate([encoded, noise], axis=0)

    def features(self):
        return super().features() + self.noise_size


def addDummyTerminalState(m: np.ndarray):
    t = np.zeros((1, m.shape[1]))
    return np.concatenate([m, t], axis=0)