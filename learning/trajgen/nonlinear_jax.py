import jax
import jax.numpy as jnp

import numpy as np
from flax import linen as nn
import optax
import scipy.linalg as spl

from .trajutils import _diff_coeff, _facln, _cost_matrix
from flax.training import train_state


def init_coeff(key, coeff0):
    return coeff0

class MinJerkReg(nn.Module):
    ts: jnp.ndarray
    order: int
    rho: int
    num_steps: int
    regularizer: callable
    coeff0: jnp.ndarray
    p: int


    def setup(self):
        self.coeff = self.param('coeff', init_coeff, self.coeff0)


    @nn.compact
    def __call__(self, x0):
        # Compute cost matrices (note it's shared across dimensions)
        num_seg = len(self.ts) - 1
        #coeff0 = np.zeros(4, 3, 6)
        #coeff = self.param('coeff', lambda rng, shape: coeff0, coeff0.shape)
        durations = self.ts[1:] - self.ts[:-1]
        #cost_mat = jnp.block([
        #    _cost_matrix(self.order, 3, d) for d in durations
        #])
        cost_mat = spl.block_diag(*[_cost_matrix(self.order, 3, d) for d in durations])
        #import ipdb;
        #ipdb.set_trace()
        cost = 0
        for pp in range(self.p):
            cost += jnp.dot(self.coeff[pp].reshape(-1),
                              cost_mat @ self.coeff[pp].reshape(-1))
        #cost = jnp.dot(coeff.flatten(), jnp.dot(cost_mat, coeff.flatten()))
        if self.regularizer is not None:
            ref = coeff2traj(self.coeff, self.ts, self.num_steps)[1].T.flatten()
            cost += self.rho * self.regularizer.apply_fn(self.regularizer.params, jnp.concatenate([x0, ref]))[0]
        return cost


#@jax.jit
def coeff2traj(coeffs, ts, numsteps):
    p = coeffs.shape[0]
    ref = jnp.zeros((p, numsteps))
    times = np.linspace(ts[0], ts[-1], numsteps)
    k = 0
    for i, tt in enumerate(times):
        if tt > ts[k+1]:
            k += 1
        ref.at[:, i].set(jnp.dot(_diff_coeff(coeffs.shape[2] - 1, tt - ts[k], 0), coeffs[:, k, :].T))
    return times, ref


#@jax.jit
def back_prop(model_state, cost_and_grad_fn):
    cost, grad = cost_and_grad_fn(model_state.params)
    model_state.apply_gradients(grads=grad)
    return model_state


def generate(waypoints, ts, order, num_steps, p, rho, vf, coeff0, num_iter=100, lr=1e-3):

    #import ipdb;
    #ipdb.set_trace()
    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    costfn = MinJerkReg(ts, order, rho, num_steps, vf, coeff0, p)


    x0 = waypoints[:, 0]

    variables = costfn.init(init_rng, x0, mutable=['params'])

    optimizer = optax.adam(learning_rate=lr)

    model_state = train_state.TrainState.create(apply_fn=costfn.apply, params=variables, tx=optimizer)

    @jax.jit
    def loss(params):
        cost = model_state.apply_fn(params, x0)
        return cost


    cost_and_grad_fn = jax.value_and_grad(loss, has_aux=False)

    model_state = back_prop(model_state, cost_and_grad_fn)

    return model_state.params.pop('params')[1].pop('coeff')[1]
