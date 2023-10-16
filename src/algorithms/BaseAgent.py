import numpy as np
import RlGlue.agent

from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector


class BaseAgent(RlGlue.agent.BaseAgent):
    def __init__(self, observations: Tuple[int, ...], params: Dict, collector: Collector, seed: int):
        self.observations = observations
        self.params = params
        self.collector = collector

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.gamma = params.get('gamma', 1)
        self.n_step = params.get('n_step', 1)

        self.behavior_probs = params['behavior_probs']
        self.target_probs = params['target_probs']

    def cleanup(self):
        ...

    # -------------------
    # -- Checkpointing --
    # -------------------
    def __getstate__(self):
        return {
            '__args': (self.observations, self.params, self.collector, self.seed, self.behavior),
            'rng': self.rng,
        }

    def __setstate__(self, state):
        self.__init__(*state['__args'])
        self.rng = state['rng']
