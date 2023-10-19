import numpy as np

from PyRlEnvs.domains.RandomWalk import buildRandomWalk
from RlGlue.environment import BaseEnvironment
from PyFixedReps.BaseRepresentation import BaseRepresentation

from PyRlEnvs.utils.math import try2jit


class RandomWalk(BaseEnvironment):
    def __init__(self, states: int, features: str, noise_ratio: float, seed: int):
        self.states = states
        self.features = features
        self.noise_ratio = noise_ratio
        self.rng = np.random.default_rng(seed)

        self.env = buildRandomWalk(self.states)()
        self.rep = buildRepresentation(self.states, self.features, self.noise_ratio)

        self.observations = (self.rep.features(), )


    def start(self):
        s = self.env.start()
        self.rep.update_noise(self.rng)
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

    return MappedRepresentation(m, states, noise_ratio)
    

class MappedRepresentation(BaseRepresentation):
    def __init__(self, m: np.ndarray, states: int, noise_ratio: float = 0.0):
        self.map = addDummyTerminalState(m)
        self.noise_size = np.floor(noise_ratio * self.map.shape[1]).astype(int)

        if self.noise_size > 0:
            self.noise_bits = np.zeros((states, self.noise_size)).astype(int)


    def encode(self, s: int):
        feats = self.map[s]
        if self.noise_size > 0:
            m = np.concatenate([feats, self.noise_bits[s, :]], axis=0)
        else:
            m = feats
        return _normRows(m)

    def features(self):
        return self.map.shape[1] + self.noise_size
    
    def update_noise(self, rng):
        if self.noise_size > 0:
            self.noise_bits = rng.integers(0, 2, size=self.noise_bits.shape)


def addDummyTerminalState(m: np.ndarray):
    t = np.zeros((1, m.shape[1]))
    return np.concatenate([m, t], axis=0)


def tabularFeatures(n: int):
    return np.eye(n)


def invertedFeatures(n: int):
    # additive inverse of tabular (hence name)
    return 1 - tabularFeatures(n)


@try2jit
def dependentFeatures(n: int):
    nfeats = int(np.floor(n / 2) + 1)
    m = np.zeros((n, nfeats))

    idx = 0
    for i in range(nfeats):
        m[idx, 0: i + 1] = 1
        idx += 1

    for i in range(nfeats - 1, 0, -1):
        m[idx, -i:] = 1
        idx += 1

    return m


# some utility functions to encode other important parts of the problem spec
# not necessarily environment specific, but this is as good a place as any to store them
@try2jit
def _normRows(m: np.ndarray):
    sums = m.sum(axis=0)
    return m / sums