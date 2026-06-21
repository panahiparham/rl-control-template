import jax
import jax.numpy as jnp
import pytest

from agents.dqn import DQNConfig, LNQNetwork, QNetwork, _make_q_network


def test_qnetwork_output_shape():
    net = QNetwork(action_dim=4)
    x = jnp.zeros((8,))
    params = net.init(jax.random.PRNGKey(0), x)
    out = net.apply(params, x)
    assert out.shape == (4,)


def test_ln_qnetwork_output_shape():
    net = LNQNetwork(action_dim=4)
    x = jnp.zeros((8,))
    params = net.init(jax.random.PRNGKey(0), x)
    out = net.apply(params, x)
    assert out.shape == (4,)


def test_make_q_network_ln_mlp_returns_ln_qnetwork():
    cfg = DQNConfig(NETWORK_PRESET="ln-mlp")
    net = _make_q_network(cfg, action_dim=3)
    assert isinstance(net, LNQNetwork)


def test_make_q_network_mlp_returns_qnetwork():
    cfg = DQNConfig(NETWORK_PRESET="mlp")
    net = _make_q_network(cfg, action_dim=3)
    assert isinstance(net, QNetwork)


def test_dqn_config_ln_mlp_valid():
    cfg = DQNConfig(NETWORK_PRESET="ln-mlp")
    assert cfg.NETWORK_PRESET == "ln-mlp"


def test_make_q_network_invalid_preset_raises():
    cfg = DQNConfig.__new__(DQNConfig)
    object.__setattr__(cfg, "NETWORK_PRESET", "bad-preset")
    with pytest.raises(ValueError, match="Invalid NETWORK_PRESET"):
        _make_q_network(cfg, action_dim=3)
