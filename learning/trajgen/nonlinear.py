import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg as spl

from trajutils import _diff_coeff, _facln, _cost_matrix


class MinJerkReg(nn.Module):
    ''' A class that takes in a trajectory and computes its cost
    '''
    def __init__(self, ts, order, regularizer, coeff):
        super().__init__()
        # Compute cost matrices (note its shared across dimensions)
        num_seg = len(ts-1)
        durations = ts[1:] - ts[:-1]
        cost_mat = spl.block_diag(*[_cost_matrix(order, 3, d) for d in durations])
        self.ts = ts
        self.cost_mat = torch.tensor(cost_mat)
        self.regularizer = regularizer
        self.coeff = nn.Parameter(coeff)

    def forward(self, x0, p, rho, num_steps):
        # Compute jerk
        cost = 0
        for pp in range(p):
            cost += torch.dot(self.coeff[pp].reshape(-1),
                              self.cost_mat @ self.coeff[pp].reshape(-1))
        # Compute regularizer
        if self.regularizer is not None:
            ref = coeff2traj(self.coeff, self.ts, num_steps)[1].T.flatten()
            cost += rho *  self.regularizer.pred(x0, ref)[0]
        return cost


#def _coeff_constr_A(ts, coeffs):
def _coeff_constr_A(ts, n, num_coeffs):
    ''' Construct the matrix for the linear constraints on the coeffs.
    Assumes the coeffs are stacked as [c1, c2, ..., c_{#seg}].T
    Note: This is now applicable to min jerk only.
    '''
    #n = coeffs.shape[2]         # n := order of polynomial + 1
    num_seg = len(ts) - 1
    num_constraints = num_seg * 4 + 2
    #num_coeffs = np.prod(coeffs.shape[1:])
    #num_coeffs = coeffs.shape[1]
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


#def _coeff_constr_b(wps, ts, coeffs):
def _coeff_constr_b(wps, ts, n):
    ''' b of the linear constraints
    Input:
        - wps:      np.array(p, #segments+1)
        - coeffs:   np.array(p, #segments, order of polynomial)
    '''
    #n = coeffs.shape[2]         # n := order of polynomial + 1
    num_seg = len(ts) - 1
    num_constraints = num_seg * 4 + 2
    b = np.zeros((wps.shape[0], num_constraints))
    # Continuity constraints
    for i in range(num_seg):
        b[:, i*4] = wps[:, i]
        b[:, i*4+1] = wps[:, i+1]
    return b


def projectcoeff(wps, ts, coeffs):
    ''' Takes a list of coefficients and project them to the subspace of
    acceptable trajectories.
    '''
    A = torch.tensor(_coeff_constr_A(ts, coeffs)).double()
    b = torch.tensor(_coeff_constr_b(wps, ts, coeffs)).double()
    flat_coeff = coeffs.flatten(1).double()
    proj = torch.linalg.pinv(A) @ (b.T - A @ flat_coeff.T) + flat_coeff.T
    return proj.T.reshape(coeffs.shape)


def coeff2traj(coeffs, ts, numsteps):
    ''' Constructs a trajectory from polynomial coefficients
    Input:
        - coeffs:       np.array(p, #segments, polynomial order)
        - ts:           np.array(#segments+1), timestamps
        - numsteps:     Integer, number of time steps in the resulting traj
    Return:
        - traj:         np.array(numsteps), trajectory from coeffs
    '''
    p = coeffs.shape[0]
    ref = torch.zeros(p, numsteps)
    times = torch.linspace(ts[0], ts[-1], numsteps)
    k = 0
    for i, tt in enumerate(times):
        if tt > ts[k+1]: k += 1
        ref[:, i] = torch.tensor(
                _diff_coeff(coeffs.shape[2]-1, tt-ts[k], 0)) @ coeffs[:,k,:].T
    return times, ref


def generate(waypoints, ts, order, num_steps, p, rho, value_func, coeff0,
        num_iter=30, lr=1e-3):
    ''' Generate trajectory that minimizes the jerk with the dynamic
        regularizer using projected gradient descent.
    Input:
        - coeff0:       np.array(p, #segments, polynomial order), coeffcient
                        for warm-starting
        - num_iter:     Integer, number of GD steps
    '''
    costfn = MinJerkReg(ts, order, value_func, coeff0)
    optimizer = optim.SGD(costfn.parameters(), lr=lr)
    x0 = waypoints[:, 0]
    for _ in range(num_iter):
        optimizer.zero_grad()
        cost = costfn.forward(x0, p, rho, num_steps)
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            # TODO: figure out a better way to do this so that performance is
            # not compromised
            costfn.coeff.data = projectcoeff(waypoints, ts, costfn.coeff)
    return costfn.coeff.detach()
