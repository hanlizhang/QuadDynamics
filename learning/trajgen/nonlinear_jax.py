import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from flax import linen as nn
import jax.scipy.linalg as spl
from .trajutils import _diff_coeff, _facln, _cost_matrix
from jaxopt import ScipyMinimize


def _coeff_constr_A_wp(ts, n, num_coeffs):
    ''' Construct the matrix for the linear constraints on the coeffs.
    Assumes the coeffs are stacked as [c1, c2, ..., c_{#seg}].T
    Note: This is now applicable to min jerk only.
    '''
    #n = coeffs.shape[2]         # n := order of polynomial + 1
    num_seg = len(ts) - 1
    num_constraints = num_seg * 2
    #num_coeffs = np.prod(coeffs.shape[1:])
    A = np.zeros((num_constraints, num_coeffs))
    # Continuity constraints
    for i in range(num_seg):
        A[i*2, i*n:(i+1)*n] = _diff_coeff(n-1, 0, 0)
        A[i*2+1, i*n:(i+1)*n] = _diff_coeff(n-1, ts[i+1]-ts[i], 0)
    return A


def _coeff_constr_A(ts, n, num_coeffs):
    ''' Construct the matrix for the linear constraints on the coeffs.
    Assumes the coeffs are stacked as [c1, c2, ..., c_{#seg}].T
    Note: This is now applicable to min jerk only.
    '''

    num_seg = len(ts) - 1
    num_constraints = num_seg * 4 + 2

    A = np.zeros((num_constraints, num_coeffs))
    # Continuity constraints
    for i in range(num_seg):
        A[i*4, i*n:(i+1)*n] = _diff_coeff(n-1, 0, 0)
        A[i*4+1, i*n:(i+1)*n] = _diff_coeff(n-1, ts[i+1]-ts[i], 0)
        if i < num_seg-1:
            A[i*4+2, i*n:(i+2)*n] = np.concatenate([
                _diff_coeff(n-1, ts[i+1]-ts[i], 1), -_diff_coeff(n-1, 0, 1) ])
            A[i*4+3, i*n:(i+2)*n] = np.concatenate([
                _diff_coeff(n-1, ts[i+1]-ts[i], 2), -_diff_coeff(n-1, 0, 2) ])
    # Boundary constraints
    A[-4, :n] = _diff_coeff(n-1, 0, 1)
    A[-3, :n] = _diff_coeff(n-1, 0, 2)
    A[-2, -n:] = _diff_coeff(n-1, ts[-1]-ts[-2], 1)
    A[-1, -n:] = _diff_coeff(n-1, ts[-1]-ts[-2], 2)

    return A


def _coeff_constr_b_wp(wps, ts, n):
    ''' b of the linear constraints
    Input:
        - wps:      np.array(p, #segments+1)
        - coeffs:   np.array(p, #segments, order of polynomial)
    '''
    #n = coeffs.shape[2]         # n := order of polynomial + 1
    num_seg = len(ts) - 1
    num_constraints = num_seg * 2
    b = np.zeros((wps.shape[0], num_constraints))
    # Continuity constraints
    for i in range(num_seg):
        b[:, i*2] = wps[:, i]
        b[:, i*2+1] = wps[:, i+1]
    return b



def _coeff_constr_b(wps, ts, n):
    ''' b of the linear constraints
    Input:
        - wps:      np.array(p, #segments+1)
        - coeffs:   np.array(p, #segments, order of polynomial)
    '''

    num_seg = len(ts) - 1
    num_constraints = num_seg * 4 + 2
    b = np.zeros((wps.shape[0], num_constraints))
    # Continuity constraints
    for i in range(num_seg):
        b[:, i*4] = wps[:, i]
        b[:, i*4+1] = wps[:, i+1]
    return b


#@jax.jit
def coeff2traj(coeffs, ts, numsteps):
    p = coeffs.shape[0]
    ref = []
    times = np.linspace(ts[0], ts[-1], numsteps)
    k = 0
    for j in range(p):
        for i, tt in enumerate(times):
            if tt > ts[k+1]:
                k += 1
            ref.append(jnp.polyval(coeffs[j, :], tt - ts[k]))
    return jnp.hstack(ref)


#@jax.jit
def back_prop(model_state, cost_and_grad_fn):
    cost, grad = cost_and_grad_fn(model_state.params)
    model_state.apply_gradients(grads=grad)
    return model_state


def generate(waypoints, ts, order, num_steps, p, rho, vf, coeff0,  num_iter=100, lr=1e-3):

    x0 = waypoints[:, 0]

    n = coeff0.shape[2]
    num_coeffs = np.prod(coeff0.shape[1:])
    print("Num coeffs", num_coeffs)

    A_coeff = _coeff_constr_A(ts, n, num_coeffs)
    b_coeff = _coeff_constr_b(waypoints, ts, n)

    num_seg = len(ts) - 1
    durations = ts[1:] - ts[:-1]

    #cost_mat = spl.block_diag(*[_cost_matrix(order, num_seg, d) for d in durations])

    coeff = np.zeros((coeff0.shape[0], num_coeffs))
    for i in range(len(waypoints)-1):
        coeff[i, :] = coeff0[i].flatten()
    ref = coeff2traj(coeff, ts, num_steps)

    util_cost = jnp.sum(jnp.array([jnp.linalg.norm(A_coeff @ coeff[pp, :] - b_coeff[pp, :]) ** 2 for pp in range(p)]))
    def nn_reg_GD(coeff):

        #import ipdb;
        #ipdb.set_trace()
        #for pp in range(p):
        #    cost += jnp.dot(coeff[pp], cost_mat @ coeff[pp])
        #cost += vf.apply_fn(vf.params, jnp.concatenate([x0, ref]))[0]
        cost = jnp.exp(vf.apply_fn(vf.params, jnp.concatenate([x0, ref]))[0])
        t_cost = cost + util_cost
        return t_cost

    print("Constraint matrix", _coeff_constr_A(ts, n, num_coeffs).shape)
    print("b", _coeff_constr_b(waypoints, ts, n).shape)

    gd = ScipyMinimize(method='CG', fun=nn_reg_GD, jit=True)
    solution = gd.run(coeff)

    return solution.params.reshape((p, num_seg, order+1))
