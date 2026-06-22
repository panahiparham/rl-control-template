import jax
import gymnax
import gymnax.wrappers

from agents.dqn import DQNConfig, make_train

_TOTAL = 20
_STARTS = 5
_ELIGIBLE = _TOTAL - _STARTS - 1  # t in {STARTS+1, ..., TOTAL-1}


def _run(train_frequency: float) -> int:
    env, env_params = gymnax.make("CartPole-v1")
    env_params = env_params.replace(max_steps_in_episode=50)
    env = gymnax.wrappers.LogWrapper(env)
    config = DQNConfig(
        TOTAL_TIMESTEPS=_TOTAL,
        LEARNING_STARTS=_STARTS,
        BATCH_SIZE=4,
        BUFFER_SIZE=200,
    )
    train_fn = make_train(config, env, env_params)
    output = train_fn(jax.random.PRNGKey(0), {"TRAIN_FREQUENCY": train_frequency})
    return int(output["runner_state"].train_state.step)


def test_train_frequency_1():
    # One update per step: 14 eligible steps × 1 update = 14
    assert _run(1.0) == _ELIGIBLE


def test_train_frequency_2():
    # One update every 2 steps: even t in {6,...,19} → {6,8,10,12,14,16,18} = 7
    assert _run(2.0) == 7


def test_train_frequency_0_5():
    # Two updates per step: 14 eligible steps × 2 updates = 28
    assert _run(0.5) == _ELIGIBLE * 2
