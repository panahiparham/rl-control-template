import os
import sys
sys.path.append(os.getcwd())

import time
import socket
import logging
import argparse
import jax
import jax.numpy as jnp
import gymnax
import gymnax.wrappers
from experiment import ExperimentModel
from PyExpUtils.results.tools import getParamsAsDict

from ml_instrumentation.Collector import Collector
from ml_instrumentation.Sampler import Identity, Ignore
from ml_instrumentation.metadata import attach_metadata
from agents.registry import agent_registry

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--silent', action='store_true', default=False)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
device = 'gpu' if args.gpu else 'cpu'
jax.config.update('jax_platform_name', device)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('exp')
prod = 'cdr' in socket.gethostname() or args.silent
if not prod:
    logger.setLevel(logging.DEBUG)

def _buildEnvironment(environment_name, episode_cutoff):
    env, env_params = gymnax.make(environment_name)
    env_params = env_params.replace(max_steps_in_episode=episode_cutoff)
    env = gymnax.wrappers.LogWrapper(env)
    return env, env_params

def _buildAgent(agent_name, params, total_steps):
    config, make_train = agent_registry(agent_name, params | {'TOTAL_TIMESTEPS': total_steps})
    return config, make_train

def getTrainFunction(environment_name, agent_name, params, total_steps, episode_cutoff):
    env, env_params = _buildEnvironment(environment_name, episode_cutoff)
    config, make_train = _buildAgent(agent_name, params, total_steps)
    return make_train(
        config,
        env,
        env_params,
    )

# ----------------------
# -- Experiment Def'n --
# ----------------------

exp = ExperimentModel.load(args.exp)
indices = args.idxs

for idx in indices:
    run = exp.getRun(idx)
    params = exp.get_hypers(idx)
    rng = jax.random.key(idx)

    train_fn = getTrainFunction(exp.environment, exp.agent, params, exp.total_steps, exp.episode_cutoff)
    jitted_train = jax.jit(train_fn)

    start_time = time.time()
    out = jitted_train(rng)
    jax.block_until_ready(out)
    total_time = time.time() - start_time

    metrics = out["metrics"]

    # collect data
    returned_episode = jax.device_get(metrics["returned_episode"])
    returned_returns = jax.device_get(metrics["returned_episode_returns"])
    returned_lengths = jax.device_get(metrics["returned_episode_lengths"])

    # Indices (timesteps) where an episode ended
    timesteps = jnp.where(returned_episode, size=returned_episode.shape[0], fill_value=-1)[0]
    timesteps = timesteps[timesteps >= 0]

    # Pair timesteps with returns/lengths at those timesteps
    episode_returns = returned_returns[returned_episode]
    episode_lengths = returned_lengths[returned_episode]

    collector = Collector(
        config={
            'return': Identity(),
            'episode': Identity(),
            'steps': Identity(),
        },
        default=Ignore(),
    )
    collector.set_experiment_id(idx)
    for episode_num, (frame, ret, _len) in enumerate(zip(timesteps, episode_returns, episode_lengths, strict=True)):
        collector.set_frame(int(frame))
        collector.collect('return', float(ret))
        collector.collect('episode', episode_num)
        collector.collect('steps', int(_len))

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    context = exp.buildSaveContext(idx, base=args.save_path)
    save_path = context.resolve('results.db')
    meta = getParamsAsDict(exp, idx)
    meta |= {'seed': exp.getRun(idx)}
    attach_metadata(save_path, idx, meta)
    collector.merge(context.resolve('results.db'))
    collector.close()
