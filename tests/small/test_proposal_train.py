import gymnax
import gymnax.wrappers
import jax

from agents.proposal import ProposalConfig, make_train

_TOTAL = 20
_STARTS = 5
_ELIGIBLE = _TOTAL - _STARTS - 1  # t in {STARTS+1, ..., TOTAL-1}


def test_train_runs_and_updates_without_target_network():
    env, env_params = gymnax.make("CartPole-v1")
    env_params = env_params.replace(max_steps_in_episode=50)
    env = gymnax.wrappers.LogWrapper(env)
    config = ProposalConfig(
        TOTAL_TIMESTEPS=_TOTAL,
        LEARNING_STARTS=_STARTS,
        BATCH_SIZE=4,
        BUFFER_SIZE=200,
    )
    train_fn = make_train(config, env, env_params)
    output = jax.jit(train_fn)(jax.random.PRNGKey(0), None)
    assert int(output["runner_state"].train_state.step) == _ELIGIBLE
