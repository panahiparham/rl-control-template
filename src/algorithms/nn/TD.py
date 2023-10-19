from functools import partial
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.ReplayBuffer import Batch

from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.jax import mse_loss

import jax
import chex
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu

@cxu.dataclass
class AgentState:
    params: Any
    optim: optax.OptState


def v_loss(v, r, gamma, vp):
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta =  - v
    return mse_loss(v, target), {
        'delta': delta,
    }

class TD(NNAgent):
    def __init__(self, observations: Tuple, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, params, collector, seed)

        self.state = AgentState(
            params=self.state.params,
            optim=self.state.optim,
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        self.v = builder.addHead(lambda: hk.Linear(1, name='v'))

    # internal compiled version of the value function
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
        if self.buffer.size() < self.batch_size:
            return

        self.updates += 1

        batch = self.buffer.sample(self.batch_size)
        weights = self.buffer.isr_weights(batch.eid)
        self.state, metrics = self._computeUpdate(self.state, batch, weights)

        metrics = jax.device_get(metrics)

        priorities = metrics['delta']
        priorities = np.abs(priorities)
        self.buffer.update_priorities(batch, priorities)

        for k, v in metrics.items():
            self.collector.collect(k, np.mean(v).item())

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Batch, weights: jax.Array):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, batch, weights)

        updates, optim = self.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            optim=optim,
        )

        return new_state, metrics

    def _loss(self, params: hk.Params, batch: Batch, weights: jax.Array):
        phi = self.phi(params, batch.x).out
        phi_p = self.phi(params, batch.xp).out

        batch_size = batch.x.shape[0]
        
        v = self.v(params, phi)
        chex.assert_shape(v, (batch_size, 1))
        v = v[:, 0]

        vp = self.v(params, phi_p)
        chex.assert_shape(vp, (batch_size, 1))
        vp = vp[:, 0]

        batch_loss = jax.vmap(v_loss, in_axes=0)
        losses, metrics = batch_loss(v, batch.r, batch.gamma, vp)

        chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(weights * losses)

        return loss, metrics
