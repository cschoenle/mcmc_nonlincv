import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import scipy
from .base import EnergyModelSeparated

def get_submat(N):
    wav_add = np.stack([np.roll(np.array([1,1]+(N-2)*[0,]), 2*i) for i in range(N//2)], axis = 0)
    wav_diff = np.stack([np.roll(np.array([1,-1]+(N-2)*[0,]), 2*i) for i in range(N//2)], axis = 0)
    wav = 1./np.sqrt(2) * np.concatenate([wav_add,wav_diff], axis = 0)
    return wav
def get_haar_wavelet_matrix(N):
    assert (N & (N-1) == 0) and N > 0, "N must be a power of 2"
    mat = get_submat(N)
    # print(mat)
    i = N//2
    while i >=2:
        mat = scipy.linalg.block_diag(get_submat(i),np.diag(np.ones(N-i))) @ mat
        # print(scipy.linalg.block_diag(get_submat(i),np.diag(np.ones(N-i))))
        i = i // 2
    return mat

class Phi41D(EnergyModelSeparated):
    """1D phi^4 model with Haar wavelet basis.
    The energy is given by:
    .. math::
        U(x) = \\frac{\\beta}{N} \\sum_{i=1}^{N} \\left(\\frac{1}{4a}(1 - x_i^2)^2 + h x_i\\right) + \\frac{\\beta a N}{2} \\sum_{i=1}^{N+1} (x_i - x_{i-1})^2
    Note that the CV variable z (from the wavelet transform) is related to the magnetisation M by z = sqrt(N) * M.
    Args:
        N: int, dimension of the grid (must be a power of 2)
        a: float, model energy parameter
        h: float, external magnetic field
        beta: float, inverse temperature
    """
    N: int = eqx.field(static=True)
    a: float = eqx.field(static=True)
    h: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    haar_forw: jax.Array
    haar_inv: jax.Array

    def __init__(self, N, a, h, beta):
        self.N = N
        self.a = a
        self.h = h
        self.beta = beta
        haar_forw = get_haar_wavelet_matrix(N)
        haar_inv = np.linalg.inv(haar_forw)

        self.haar_forw = jnp.array(haar_forw)
        self.haar_inv = jnp.array(haar_inv)

    def get_wavelet(self, x):
        if x.ndim == 1:
            y = self.haar_forw @ x  # shape (N,)
            return y[0], y[1:]
        elif x.ndim == 2:
            y = x @ self.haar_forw.T  # shape (T, N)
            return y[:, 0], y[:, 1:]
        else:
            raise ValueError("Input must be 1D or 2D (N,) or (T, N)")

    def get_field(self,z,x):
        y = jnp.insert(x,0,z)
        return self.haar_inv @ y

    def energy_full(self,x, state=None):
        V = ((1 - x ** 2) ** 2 / (4*self.a) + self.h * x).sum() / self.N
        x_ = jnp.pad(x,pad_width=1,mode='constant')
        ekin = ((x_[1:] - x_[:-1]) ** 2 / 2).sum() * self.a * self.N
        return self.beta * (ekin + V)

    def force(self,x, state=None):
        grad_V = (2*(1 - x ** 2)*(-2*x) / (4*self.a) + self.h) / (self.N)
        x_ = jnp.pad(x,pad_width=1,mode='constant')
        grad_ekin = self.a * self.N * (2*x - x_[:-2] - x_[2:])
        return self.beta * (grad_ekin + grad_V)
    
    @jax.jit
    def energy(self,z,wav, state=None):
        field = self.get_field(z,wav)
        return self.energy_full(field)
    @jax.jit
    def force_partialx(self,z,wav, state=None):
        field = self.get_field(z,wav)
        partial_field = self.force(field)
        partial_wav = self.haar_inv.T[1:] @ partial_field # Jacobian without first column, tranposed
        return partial_wav

def mode_change_cost_singlechain(mode_change, n_inter_traj):
# reduceat cannot deal with mode switch at last axis, therefore drop last value
    mode_switches = np.asarray(mode_change == True).nonzero()[0]
    if len(mode_switches) <= 2:
        return np.nan
    cost = np.add.reduceat(n_inter_traj, np.concatenate(([0,], mode_switches[:-1]+1)))[:-1]
    return np.mean(cost)

def get_mode_change_cost_phi4(z_traj_list, n_inter_traj_list):
    mode_traj_list = (z_traj_list > 0.)
    mode_change_list = mode_traj_list[:,1:] != mode_traj_list[:,:-1]
    cost = []
    for mode_change, n_inter_traj in zip(mode_change_list,n_inter_traj_list):
        cost.append(mode_change_cost_singlechain(mode_change, n_inter_traj))
    return np.mean(cost), np.std(cost)   