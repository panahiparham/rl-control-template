import numpy as np
from environments.RandomWalk import RandomWalk as Env
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem
from functools import partial


class RandomWalk(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector, states: int, behavior: float, target: float, noise: float):
        super().__init__(exp, idx, collector)
        self.exp = exp
        self.idx = idx

        self.states = states
        self.features = self.env_params['features']
        self.noise_ratio = noise

        self.behavior_probs = lambda s: np.array([behavior, 1 - behavior])
        self.target_probs = lambda s: np.array([target, 1 - target])

        self.env = Env(self.states, self.features, self.noise_ratio)
        self.observations = self.env.observations

        self.gamma = 0.99

        # computing msve preamble
        self.ds = self.env.env.computeStateDistribution(self.target_probs)
        R = self.env.env.constructRewardVector(self.target_probs)
        P_gamma = self.env.env.constructTransitionMatrix(self.target_probs, gamma = self.gamma)
        I = np.eye(self.ds.shape[0])

        self.v_pi = np.linalg.pinv(I - P_gamma).dot(R)
        self.all_states = np.array([self.env.rep.encode(i) for i in range(R.shape[0])])


    def evaluate(self):
        v_hat = np.squeeze(self.agent.values(self.all_states))
        return self.ds.dot(np.square(v_hat - self.v_pi))
        

# problem instances
TestChain = partial(RandomWalk, states=5, behavior=0.5, target=0.5, noise=0.0)

LongRandChain = partial(RandomWalk, states=101, behavior=0.5, target=0.5, noise=0.0)
ShortRandChain = partial(RandomWalk, states=11, behavior=0.5, target=0.5, noise=0.0)
LongBiasedChain = partial(RandomWalk, states=101, behavior=0.75, target=0.75, noise=0.0)
ShortBiasedChain = partial(RandomWalk, states=11, behavior=0.75, target=0.75, noise=0.0)

NoisyLongRandChain = partial(RandomWalk, states=101, behavior=0.5, target=0.5, noise=0.2)
NoisyShortRandChain = partial(RandomWalk, states=11, behavior=0.5, target=0.5, noise=0.2)
NoisyLongBiasedChain = partial(RandomWalk, states=101, behavior=0.75, target=0.75, noise=0.2)
NoisyShortBiasedChain = partial(RandomWalk, states=11, behavior=0.75, target=0.75, noise=0.2)

