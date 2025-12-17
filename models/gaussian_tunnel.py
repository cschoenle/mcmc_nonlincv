import jax
import jax.numpy as jnp
import equinox as eqx
from .base import EnergyModelSeparated, EnergyModelCV
import numpy as np

from samplers.nonlinCV import GeneralLagrangeSolver, UnderdampedNonlinCV

class GaussianTunnel(EnergyModelCV, EnergyModelSeparated):
    """Gaussian tunnel model.
    Args:
        dimx: int, dimension of x
        d: float, distance between the two Gaussian peaks
        w1: float, weight of the first Gaussian peak
    """
    dimx: int = eqx.field(static=True)
    d: float = eqx.field(static=True)
    w1: float = eqx.field(static=True)
    sigmax_vals: jnp.ndarray

    def __init__(self, dimx, d, w1):
        self.dimx = dimx
        self.d = d
        self.w1 = w1
        self.sigmax_vals = jnp.linspace(0.5, 5, dimx)


    def marginalpz(self, z):
        return 1/jnp.sqrt(2*jnp.pi) * (
            self.w1 * jnp.exp(-0.5 * z**2).sum()
            + (1 - self.w1) * jnp.exp(-0.5 * (z-self.d)**2).sum())

    def mux(self,z):
        return jnp.cos(z/self.d*jnp.pi) * self.d/2

    def sigmax(self,z):
            return self.sigmax_vals

    def probfull_cond(self, z, x):
        """p(x|z)"""
        return 1/((2 * jnp.pi)**(self.dimx/2.) * jnp.prod(self.sigmax(z))) * jnp.exp(-0.5*jnp.sum(((x-self.mux(z))/self.sigmax(z))**2))

    def probfull(self, z, x):
        return self.marginalpz(z) * self.probfull_cond(z, x)

    @jax.jit
    def energy(self, z, x, state=None):
        return -jnp.log(self.probfull(z, x))

    
    @jax.jit
    def force_partialx(self, z, x, state=None):
        ### ∂_x -log(p) = -(∂_x p) * 1/p 
        return 1/self.sigmax(z)**2 * (x - self.mux(z))
    @jax.jit
    def energy_full(self, x, state=None):
        z = x[0:1]
        x = x[1:]
        return self.energy(z, x)
    @jax.jit
    def force(self, x, state=None):
        return jax.grad(self.energy_full)(x)
    
    def xi(self, x):
        return x[0:1]

    def grad_xi(self, x):
        """Output should be of shape (dimension, dim_CV)!!!"""      
        gradxi_full = jnp.zeros(self.dimx+1)
        gradxi_full = gradxi_full.at[0].set(1.)
        return gradxi_full.reshape(self.dimx+1, 1)
    

class GaussianTunnelLagrangeSolver(GeneralLagrangeSolver):
    """Lagrange solver for the dimer model."""
    EnergyModel: "GaussianTunnel"
    def get_gradxilagrange_pos(self,x, xnewtilde, z_new, dt, mass, grad_xi):
        """
        Solve x_new = x_new_tilde + dt * mass^{-1} * grad_xi @ lambda such that xi(x_new)=z_new
        """
        mass_diagonal = mass * jnp.ones(grad_xi.shape[0])
        lambda_ = mass_diagonal[0] / dt * (z_new - self.EnergyModel.xi(xnewtilde))
        return grad_xi @ lambda_

# Auxiliary class of sampling to avoid computation of Fixman term.    
class GaussianUnderdampedCV(UnderdampedNonlinCV):
    def energy_with_fixman(self, x, mass, state=None):
        energy_model = self.EnergyModel.energy_full(x, state)
        return energy_model
    
    def force_with_fixman(self, x, mass, state=None):
        force_model = self.EnergyModel.force(x, state)
        return force_model    
   
def mode_change_cost_singlechain(mode_change, n_inter_traj):
    mode_switches = np.asarray(mode_change == True).nonzero()[0]
    if len(mode_switches) <= 2:
        return np.nan
    cost = np.add.reduceat(n_inter_traj, np.concatenate(([0,], mode_switches[:-1]+1)))[:-1]
    return np.mean(cost)

def get_mode_change_cost_gaussian_tunnel(z_traj_list, n_inter_traj_list, mode_boundary):
    mode_traj_list = (z_traj_list > mode_boundary)
    mode_change_list = mode_traj_list[:,1:] != mode_traj_list[:,:-1]
    cost = []
    for mode_change, n_inter_traj in zip(mode_change_list,n_inter_traj_list):
        cost.append(mode_change_cost_singlechain(mode_change, n_inter_traj))
    return np.mean(cost), np.std(cost)