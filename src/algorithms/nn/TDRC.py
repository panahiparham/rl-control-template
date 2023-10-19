from functools import partial
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.interface import Batch

from algorithms.nn.NNAgent import NNAgent, AgentState
from representations.networks import NetworkBuilder

import jax
import jax.numpy as jnp
import chex
import optax
import numpy as np
import haiku as hk

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map

class TDRC(NNAgent):
    def __init__(self, observations: Tuple, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, params, collector, seed)
        self.beta = params.get('beta', 1.)
        self.stepsize = self.optimizer_params['alpha']

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        zero_init = hk.initializers.Constant(0)
        self.v = builder.addHead(lambda: hk.Linear(1, name='v', w_init=zero_init, b_init=zero_init))
        self.h = builder.addHead(lambda: hk.Linear(1, name='h', w_init=zero_init, b_init=zero_init), grad=False)

    # jit'ed internal value function approximator
    # considerable speedup, especially for larger networks (note: haiku networks are not jit'ed by default)
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):
        phi = self.phi(state.params, x).out
        return self.v(state.params, phi)

    def update(self):
        self.steps += 1

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # skip updates if the buffer isn't full yet
        if self.buffer.size() <= self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        self.state, metrics = self._computeUpdate(self.state, batch)

        metrics = jax.device_get(metrics)

        priorities = metrics['delta']
        priorities = np.abs(priorities)
        self.buffer.update_priorities(batch, priorities)

        for k, v in metrics.items():
            self.collector.collect(k, np.mean(v).item())

    # -------------
    # -- Updates --
    # -------------

    # compute the update and return the new parameter states
    # and optimizer state (i.e. ADAM moving averages)
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Batch):
        params = state.params
        grad, metrics = jax.grad(self._loss, has_aux=True)(params, batch)

        updates, new_optim = self.optimizer.update(grad, state.optim, params)
        new_params = optax.apply_updates(params, updates)

        new_state = AgentState(
            params=new_params,
            optim=new_optim,
        )

        return new_state, metrics

    # compute the total TDRC loss for both sets of parameters (value parameters and h parameters)
    def _loss(self, params, batch: Batch):
        batch_size = batch.x.shape[0]

        phi = self.phi(params, batch.x).out
        v = self.v(params, phi)
        chex.assert_shape(v, (batch_size, 1))
        v = v[:, 0]
        h = self.h(params, phi)
        chex.assert_shape(h, (batch_size, 1))
        h = h[:, 0]

        phi_p = self.phi(params, batch.xp).out
        vp = self.v(params, phi_p)
        chex.assert_shape(vp, (batch_size, 1))
        vp = vp[:, 0]

        v_loss, h_loss, metrics = tdc_loss(v, batch.r, batch.gamma, vp, h)
        regularizer = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params['h']))

        h_loss = h_loss.mean()
        v_loss = v_loss.mean()

        metrics |= {
            'v_loss': v_loss,
            'h_loss': h_loss,
        }

        return v_loss + h_loss + regularizer, metrics

# ---------------
# -- Utilities --
# ---------------

def tdc_loss(v, r, gamma, vtp1, h):
    target = r + gamma * vtp1
    target = jax.lax.stop_gradient(target)

    delta = target - v
    delta_hat = h

    v_loss = 0.5 * delta**2 + gamma * jax.lax.stop_gradient(delta_hat) * vtp1
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat)**2

    return v_loss, h_loss, {
        'delta': delta,
        'h': delta_hat,
    }
