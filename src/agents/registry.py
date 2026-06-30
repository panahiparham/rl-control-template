from typing import Any

from agents.bad_init_dqn import BadInitDQNConfig
from agents.bad_init_dqn import make_train as bad_init_dqn_make_train
from agents.dqn import DQNConfig
from agents.dqn import make_train as dqn_make_train


def agent_registry(name: str, params: dict[str, Any]):
    if name == 'DQN':
        return DQNConfig(**params), dqn_make_train
    if name == 'BadInitDQN':
        return BadInitDQNConfig(**params), bad_init_dqn_make_train
    raise Exception('Unknown algorithm')
