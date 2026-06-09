from typing import Any

from agents.dqn import DQNConfig
from agents.dqn import make_train as dqn_make_train


def agent_registry(name: str, params: dict[str, Any]):
    print(params)
    if name == 'DQN':
        return DQNConfig(**params), dqn_make_train
    else:
        raise Exception('Unknown algorithm')
