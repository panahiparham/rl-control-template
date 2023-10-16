import numpy as np

from PyRlEnvs.domains.RandomWalk import buildRandomWalk, invertedFeatures, tabularFeatures, dependentFeatures
from RlGlue.environment import BaseEnvironment
from PyFixedReps.BaseRepresentation import BaseRepresentation



class RandomWalk(BaseEnvironment):
    def __init__(self, states: int, features: str, noise_ratio: float):
        self.states = states
        self.features = features
        self.noise_ratio = noise_ratio

        self.env = buildRandomWalk(self.states)()
        self.rep = buildRepresentation(self.states, self.features, self.noise_ratio)

        self.observations = (self.rep.features(), )


    def start(self):
        s = self.env.start()
        phi = self.rep.encode(s).astype('float32')
        return phi


    def step(self, a: int):
        r, sp, t, _ = self.env.step(a)
        phip = self.rep.encode(sp).astype('float32')
        return (r, phip, t, {}) 
        


def buildRepresentation(states, features, noise_ratio):
    m = None
    if features == 'tabular':
        m = tabularFeatures(states)

    if features == 'inverted':
        m = invertedFeatures(states)

    if features == 'dependent':
        m = dependentFeatures(states)

    assert m is not None

    if noise_ratio > 0:
        return NoisyMappedRepresentation(m, noise_ratio)
    else:
        return MappedRepresentation(m)
    

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