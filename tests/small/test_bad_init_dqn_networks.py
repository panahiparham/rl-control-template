import jax
import jax.numpy as jnp
import pytest

from agents.bad_init_dqn import BadInitDQNConfig, LNNoAffineQNetwork, LNQNetwork, QNetwork, _make_q_network


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


def test_ln_no_affine_qnetwork_output_shape():
    net = LNNoAffineQNetwork(action_dim=4)
    x = jnp.zeros((8,))
    params = net.init(jax.random.PRNGKey(0), x)
    out = net.apply(params, x)
    assert out.shape == (4,)


def test_make_q_network_mlp_returns_qnetwork():
    cfg = BadInitDQNConfig(NETWORK_PRESET="mlp")
    net = _make_q_network(cfg, action_dim=3)
    assert isinstance(net, QNetwork)


def test_make_q_network_ln_mlp_returns_ln_qnetwork():
    cfg = BadInitDQNConfig(NETWORK_PRESET="ln-mlp")
    net = _make_q_network(cfg, action_dim=3)
    assert isinstance(net, LNQNetwork)


def test_make_q_network_ln_mlp_no_affine_returns_ln_no_affine_qnetwork():
    cfg = BadInitDQNConfig(NETWORK_PRESET="ln-mlp-no-affine")
    net = _make_q_network(cfg, action_dim=3)
    assert isinstance(net, LNNoAffineQNetwork)


def test_make_q_network_invalid_preset_raises():
    cfg = BadInitDQNConfig.__new__(BadInitDQNConfig)
    object.__setattr__(cfg, "NETWORK_PRESET", "bad-preset")
    with pytest.raises(ValueError, match="Invalid NETWORK_PRESET"):
        _make_q_network(cfg, action_dim=3)


def test_qnetwork_kernel_stddev_near_one():
    """Verify Dense kernels are initialized with stddev≈1, not scaled-down defaults."""
    net = QNetwork(action_dim=4)
    x = jnp.zeros((8,))
    params = net.init(jax.random.PRNGKey(0), x)
    kernel = params['params']['Dense_0']['kernel']
    assert float(jnp.std(kernel)) > 0.5
