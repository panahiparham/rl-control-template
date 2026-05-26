from typing import Any
from rl_agents.dqn import DQNConfig, make_train

def agent_registry(name: str, params: dict[str, Any]):
    print(params)
    if name == 'DQN':
        return DQNConfig(**params), make_train

    raise Exception('Unknown algorithm')
