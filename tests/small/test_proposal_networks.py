import jax
import jax.numpy as jnp
import pytest

from agents.proposal import LNNoAffineQNetwork, ProposalConfig, RunnerState, _make_q_network
from agents.registry import agent_registry


def test_ln_no_affine_qnetwork_output_shape():
    net = LNNoAffineQNetwork(action_dim=4)
    x = jnp.zeros((8,))
    params = net.init(jax.random.PRNGKey(0), x)
    out = net.apply(params, x)
    assert out.shape == (4,)


def test_make_q_network_default_returns_ln_no_affine_qnetwork():
    cfg = ProposalConfig()
    assert cfg.NETWORK_PRESET == "ln-mlp-no-affine"
    net = _make_q_network(cfg, action_dim=3)
    assert isinstance(net, LNNoAffineQNetwork)


def test_make_q_network_invalid_preset_raises():
    cfg = ProposalConfig.__new__(ProposalConfig)
    object.__setattr__(cfg, "NETWORK_PRESET", "mlp")
    with pytest.raises(ValueError, match="Invalid NETWORK_PRESET"):
        _make_q_network(cfg, action_dim=3)


def test_config_has_no_target_network_hypers():
    cfg = ProposalConfig()
    assert not hasattr(cfg, "TARGET_NETWORK_FREQUENCY")
    assert not hasattr(cfg, "TAU")


def test_runner_state_has_no_target_params():
    assert "target_params" not in RunnerState._fields


def test_registry_returns_proposal_config():
    cfg, make_train = agent_registry("Proposal", {})
    assert isinstance(cfg, ProposalConfig)
    assert callable(make_train)
