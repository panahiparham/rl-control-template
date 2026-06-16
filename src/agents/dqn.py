from typing import Callable, Literal, NamedTuple, Protocol, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.initializers import legacy_dqn_uniform
from jax_nn.layers import NatureCNN
from jax_nn.typed_module import TypedApply
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.gym_env import DiscreteActionSpace, GymEnv
from rl_components.structs import chex_struct


@chex_struct(frozen=True, kw_only=True)
class DQNConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    TARGET_NETWORK_FREQUENCY: int = 1_000
    GAMMA: float = 0.99
    TAU: float = 1.0  # Soft update
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_FRACTION: float = 0.5
    ENV_NAME: str = "MountainCar-v0"
    SEED: int = 42
    NETWORK_PRESET: Literal["mlp", "nature_cnn"] = "mlp"


class QNetwork(TypedApply[jax.Array], nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class _HasNetworkPreset(Protocol):
    @property
    def NETWORK_PRESET(self) -> Literal["mlp", "nature_cnn"]: ...


class NatureQNetwork(TypedApply[jax.Array], nn.Module):
    action_dim: int
    observation_layout: Literal["hwc", "fhwc"]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _prepare_nature_observations(x, self.observation_layout)
        x = NatureCNN()(x)
        input_units = x.shape[-1]
        x = nn.Dense(
            512,
            kernel_init=legacy_dqn_uniform(),
            bias_init=legacy_dqn_uniform(num_input_units=input_units),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=legacy_dqn_uniform(),
            bias_init=legacy_dqn_uniform(num_input_units=512),
        )(x)
        return x


def _prepare_nature_observations(
    x: jax.Array,
    observation_layout: Literal["hwc", "fhwc"],
) -> jax.Array:
    if observation_layout == "hwc":
        if x.ndim not in (3, 4):
            raise ValueError(
                "NETWORK_PRESET='nature_cnn' with HWC observations expects shape (height, width, channels) or (batch, height, width, channels)."
            )
        return x

    if x.ndim == 4:
        moved = jnp.moveaxis(x, 0, -2)
        return moved.reshape(moved.shape[:-2] + (moved.shape[-2] * moved.shape[-1],))

    if x.ndim == 5:
        moved = jnp.moveaxis(x, 1, -2)
        return moved.reshape(moved.shape[:-2] + (moved.shape[-2] * moved.shape[-1],))

    raise ValueError(
        "NETWORK_PRESET='nature_cnn' with Atari-style observations expects shape (frames, height, width, channels) or (batch, frames, height, width, channels)."
    )


def _infer_nature_observation_layout(observation_shape: tuple[int, ...]) -> Literal["hwc", "fhwc"]:
    if len(observation_shape) == 3:
        return "hwc"
    if len(observation_shape) == 4:
        return "fhwc"
    raise ValueError(
        "NETWORK_PRESET='nature_cnn' requires image observations shaped (height, width, channels) or Atari-style (frames, height, width, channels)."
    )


def _make_q_network(
    config: _HasNetworkPreset,
    action_dim: int,
    observation_shape: tuple[int, ...] | None = None,
) -> QNetwork | NatureQNetwork:
    if config.NETWORK_PRESET == "mlp":
        return QNetwork(action_dim)
    if config.NETWORK_PRESET == "nature_cnn":
        if observation_shape is None:
            raise ValueError("NETWORK_PRESET='nature_cnn' requires observation_shape to build the Q-network.")
        return NatureQNetwork(
            action_dim=action_dim,
            observation_layout=_infer_nature_observation_layout(observation_shape),
        )
    raise ValueError(
        f"Invalid NETWORK_PRESET {config.NETWORK_PRESET!r}. Expected one of: 'mlp', 'nature_cnn'."
    )


class RunnerState(NamedTuple):
    train_state: TrainState
    target_params: VariableDict
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class DQNTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(config: DQNConfig, env: GymEnv[DiscreteActionSpace], env_params: object | None = None) -> Callable[[jax.Array, dict | None], DQNTrainOutput]:
    def train(rng: jax.Array, hypers: dict | None = None) -> DQNTrainOutput:
        # Merge hypers with config defaults
        if hypers is None:
            hypers = {}

        # Extract hyperparameters with fallbacks to config
        lr = hypers.get('LR', config.LR)
        buffer_size = hypers.get('BUFFER_SIZE', config.BUFFER_SIZE)
        batch_size = hypers.get('BATCH_SIZE', config.BATCH_SIZE)
        learning_starts = hypers.get('LEARNING_STARTS', config.LEARNING_STARTS)
        train_frequency = hypers.get('TRAIN_FREQUENCY', config.TRAIN_FREQUENCY)
        target_network_frequency = hypers.get('TARGET_NETWORK_FREQUENCY', config.TARGET_NETWORK_FREQUENCY)
        gamma = hypers.get('GAMMA', config.GAMMA)
        tau = hypers.get('TAU', config.TAU)
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

        target_params = params

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
            train_state, target_params, buffer_state, env_state, last_obs, rng = runner_state

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
            def _do_train(train_state: TrainState, target_params: VariableDict, buffer_state: ReplayBufferState, rng: jax.Array) -> tuple[TrainState, jax.Array]:
                rng, _rng = jax.random.split(rng)
                obs, actions, rewards, next_obs, terminated = buffer.sample(buffer_state, _rng, batch_size)

                def _loss_fn(
                    params: VariableDict,
                    target_params: VariableDict,
                    obs: jax.Array,
                    actions: jax.Array,
                    rewards: jax.Array,
                    next_obs: jax.Array,
                    terminated: jax.Array,
                ) -> jax.Array:
                    q_values = network.apply(params, obs)
                    q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()

                    next_q_values = network.apply(target_params, next_obs)
                    next_q_max = jnp.max(next_q_values, axis=-1)
                    target = rewards + gamma * next_q_max * (1.0 - terminated)

                    loss = jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))
                    return loss

                grad_fn = jax.value_and_grad(_loss_fn)
                loss, grads = grad_fn(train_state.params, target_params, obs, actions, rewards, next_obs, terminated)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss

            can_train = (t > learning_starts) & (t % train_frequency == 0)
            train_state, loss = jax.lax.cond(
                can_train,
                lambda: _do_train(train_state, target_params, buffer_state, rng),
                lambda: (train_state, 0.0),
            )

            # UPDATE TARGET
            should_update_target = t % target_network_frequency == 0
            target_params = jax.lax.cond(
                should_update_target,
                lambda: jax.tree_util.tree_map(
                    lambda tp, p: tau * p + (1.0 - tau) * tp,
                    target_params,
                    train_state.params,
                ),
                lambda: target_params,
            )

            runner_state = RunnerState(train_state=train_state, target_params=target_params, buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng)
            return runner_state, info

        # RUNNER
        runner_state = RunnerState(train_state=train_state, target_params=target_params, buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
