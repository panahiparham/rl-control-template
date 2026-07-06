from typing import Callable, Literal, NamedTuple, Protocol, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.typed_module import TypedApply
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.gym_env import DiscreteActionSpace, GymEnv
from rl_components.structs import chex_struct


@chex_struct(frozen=True, kw_only=True)
class ProposalConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: float = 1.0
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_FRACTION: float = 0.5
    ENV_NAME: str = "MountainCar-v0"
    SEED: int = 42
    NETWORK_PRESET: Literal["ln-mlp-no-affine"] = "ln-mlp-no-affine"


class LNNoAffineQNetwork(TypedApply[jax.Array], nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(64)(x)
        x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class _HasNetworkPreset(Protocol):
    @property
    def NETWORK_PRESET(self) -> Literal["ln-mlp-no-affine"]: ...


def _make_q_network(
    config: _HasNetworkPreset,
    action_dim: int,
    observation_shape: tuple[int, ...] | None = None,
) -> LNNoAffineQNetwork:
    if config.NETWORK_PRESET == "ln-mlp-no-affine":
        return LNNoAffineQNetwork(action_dim)
    raise ValueError(
        f"Invalid NETWORK_PRESET {config.NETWORK_PRESET!r}. Expected 'ln-mlp-no-affine'."
    )


class RunnerState(NamedTuple):
    train_state: TrainState
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class ProposalTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(config: ProposalConfig, env: GymEnv[DiscreteActionSpace], env_params: object | None = None) -> Callable[[jax.Array, dict | None], ProposalTrainOutput]:
    def train(rng: jax.Array, hypers: dict | None = None) -> ProposalTrainOutput:
        # Merge hypers with config defaults
        if hypers is None:
            hypers = {}

        # Extract hyperparameters with fallbacks to config
        lr = hypers.get('LR', config.LR)
        buffer_size = hypers.get('BUFFER_SIZE', config.BUFFER_SIZE)
        batch_size = hypers.get('BATCH_SIZE', config.BATCH_SIZE)
        learning_starts = hypers.get('LEARNING_STARTS', config.LEARNING_STARTS)
        # TRAIN_FREQUENCY is structural (controls fori_loop iterations) so it must
        # be a static Python int — always read from config, never from traced hypers.
        _tf = float(config.TRAIN_FREQUENCY)
        if _tf >= 1.0:
            _step_period = int(round(_tf))
            _updates_per_step = 1
        else:
            _step_period = 1
            _updates_per_step = int(round(1.0 / _tf))
        gamma = hypers.get('GAMMA', config.GAMMA)
        epsilon_start = hypers.get('EPSILON_START', config.EPSILON_START)
        epsilon_end = hypers.get('EPSILON_END', config.EPSILON_END)
        epsilon_fraction = hypers.get('EPSILON_FRACTION', config.EPSILON_FRACTION)
        total_timesteps = hypers.get('TOTAL_TIMESTEPS', config.TOTAL_TIMESTEPS)

        # INIT NETWORK
        observation_shape = tuple(env.observation_space(env_params).shape)
        network = _make_q_network(config, env.action_space(env_params).n, observation_shape=observation_shape)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(observation_shape, dtype=env.observation_space(env_params).dtype)
        params = network.init(_rng, init_x)

        tx = optax.adam(lr)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        # INIT BUFFER
        buffer = ReplayBuffer(
            buffer_size,
            env.observation_space(env_params).shape,
            (),  # action_shape for discrete is empty
            jnp.int32,  # action_dtype
        )
        buffer_state = buffer.init()

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)  # gymnax JitWrapped

        def _update_step(runner_state: RunnerState, t: jax.Array) -> tuple[RunnerState, dict[str, jax.Array]]:
            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # EPSILON GREEDY
            epsilon = jnp.maximum(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end)
                * (t / (total_timesteps * epsilon_fraction)),
            )

            rng, _rng_action, _rng_step = jax.random.split(rng, 3)
            q_values = network.apply(train_state.params, last_obs)
            greedy_action = jnp.argmax(q_values)
            random_action = jax.random.randint(_rng_action, (), 0, env.action_space(env_params).n)
            chose_random = jax.random.uniform(_rng_action, ()) < epsilon
            action = jnp.where(chose_random, random_action, greedy_action)

            # STEP ENV
            obsv, env_state, reward, done, info = env.step(_rng_step, env_state, action, env_params)  # gymnax JitWrapped

            # Separate truncation (time limit) from true termination (task ended).
            # returned_episode_lengths equals max_steps iff the episode timed out.
            truncated = done & (info['returned_episode_lengths'] >= env_params.max_steps_in_episode)
            terminated = done & ~truncated

            # ADD TO BUFFER — skip truncated transitions: gymnax auto-resets after done=True,
            # so obsv is the first obs of a new episode, not a valid next state for bootstrapping.
            buffer_state = jax.lax.cond(
                ~truncated,
                lambda: buffer.add(
                    buffer_state,
                    last_obs[None, ...],
                    action[None, ...],
                    reward[None, ...],
                    obsv[None, ...],
                    terminated[None, ...],
                ),
                lambda: buffer_state,
            )

            # TRAIN
            def _do_train(train_state: TrainState, buffer_state: ReplayBufferState, rng: jax.Array) -> tuple[TrainState, jax.Array]:
                rng, _rng = jax.random.split(rng)
                obs, actions, rewards, next_obs, terminated = buffer.sample(buffer_state, _rng, batch_size)

                def _loss_fn(
                    params: VariableDict,
                    obs: jax.Array,
                    actions: jax.Array,
                    rewards: jax.Array,
                    next_obs: jax.Array,
                    terminated: jax.Array,
                ) -> jax.Array:
                    q_values = network.apply(params, obs)
                    q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()

                    # No target network: bootstrap from the online params.
                    next_q_values = network.apply(params, next_obs)
                    next_q_max = jnp.max(next_q_values, axis=-1)
                    target = rewards + gamma * next_q_max * (1.0 - terminated)

                    loss = jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))
                    return loss

                grad_fn = jax.value_and_grad(_loss_fn)
                loss, grads = grad_fn(train_state.params, obs, actions, rewards, next_obs, terminated)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss

            can_train = (t > learning_starts) & (t % _step_period == 0)

            def _do_multi_train(train_state: TrainState, buffer_state: ReplayBufferState, rng: jax.Array) -> tuple[TrainState, jax.Array]:
                def body(_, carry: tuple[TrainState, jax.Array, jax.Array]) -> tuple[TrainState, jax.Array, jax.Array]:
                    ts, loss, key = carry
                    key, step_key = jax.random.split(key)
                    ts, loss = _do_train(ts, buffer_state, step_key)
                    return ts, loss, key
                ts, loss, _ = jax.lax.fori_loop(0, _updates_per_step, body, (train_state, 0.0, rng))
                return ts, loss

            train_state, loss = jax.lax.cond(
                can_train,
                lambda: _do_multi_train(train_state, buffer_state, rng),
                lambda: (train_state, jnp.zeros(())),
            )

            runner_state = RunnerState(train_state=train_state, buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng)
            return runner_state, info

        # RUNNER
        runner_state = RunnerState(train_state=train_state, buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
