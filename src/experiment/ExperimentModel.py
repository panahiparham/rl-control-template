import sys
import json
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

from typing import Optional
from PyExpUtils.utils.dict import merge, pick, hyphenatedStringify
from PyExpUtils.utils.arrays import unwrap
from PyExpUtils.utils.str import interpolate

class ExperimentModel(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path)
        self.agent = d['AGENT']
        self.environment = d['ENVIRONMENT']

        self.episode_cutoff = d.get('EPISODE_CUTOFF', -1)
        self.total_steps = d.get('TOTAL_TIMESTEPS')

    def interpolateSavePath(self, idx: int, key: Optional[str] = None):
        key = self._getSaveKey(key)

        permute = unwrap(self.getKeys())
        params = pick(self.getPermutation(idx), permute)
        param_string = hyphenatedStringify(params)

        run = self.getRun(idx)

        special_keys = {
            'params': param_string,
            'run': str(run),
            'name': self.getExperimentName(),
            'desc': self.path.split('/')[-1].split('.')[0]
        }
        d = merge(self.__dict__, special_keys)

        return interpolate(str(key), d)


def load(path=None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp
