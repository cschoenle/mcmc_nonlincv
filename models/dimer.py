import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from .base import EnergyModelCV, EnergyModelCVSubset
from samplers.nonlinCV import GeneralLagrangeSolver, UnderdampedNonlinCV
from samplers.nonlinCV_subset import GeneralLagrangeSolverSubset, UnderdampedNonlinCVSubset

def apply_prb(x, L):
    """Restrict between [-L/2, L/2)"""
    return ((x + L/2.) % L) - L/2.

def energywell(r, h, sigmaLJ, w):
    r0 = sigmaLJ * 2**(1./6.)
    return h * (1-(r-r0-w)**2 / w**2)**2

def energyWCA(r, epsilon, sigmaLJ):
    r0 = sigmaLJ * 2**(1./6.)
    return jnp.where(r <= r0, 4*epsilon*((sigmaLJ/r)**12 - (sigmaLJ/r)**6)+epsilon, 0.)

def energyconfine(r,L):
    return r**2 / (2*L**2)  

class DimerModel(EnergyModelCV):
    """Gaussian tunnel model.
    Args:
        h: float, parameter of dimer interaction potential
        w: float, parameter of dimer interaction potential
        epsilon: float, parameter of LJ potential
        sigmaLJ: float, parameter of LJ potential and dimer interaction potential
        unitcellsize: float, size of the unit cell
        n: int, there are n^2 particles in the system
    """
    h: float = eqx.field(static=True)
    w: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    sigmaLJ: float = eqx.field(static=True)
    unitcellsize: float = eqx.field(static=True)
    n_particles: int = eqx.field(static=True)
    L: int = eqx.field(static=True)
    r0: float = eqx.field(static=True)
    zmin: float = eqx.field(static=True)
    zmax: float = eqx.field(static=True)

    def __init__(self, h, w, epsilon, sigmaLJ, unitcellsize, n):
        self.h = h 
        self.w = w
        self.epsilon = epsilon
        self.sigmaLJ = sigmaLJ
        self.unitcellsize = unitcellsize
        self.n_particles = n*n
        self.L = n * unitcellsize
        self.r0 = 2**(1./6) * sigmaLJ 
        self.zmin = -self.r0 / (2*w)
        self.zmax = float((self.L/np.sqrt(2) - self.r0) / (2*w))

    def energy_full_particles(self, x):
        """Energy of the system."""
        distance_dimer = jnp.linalg.norm(self.apply_boundaries(x[0]-x[1]), axis=-1)
        distances_solvent = x[2:][None,:,:] - x[2:][:,None,:]
        distances_solvent = distances_solvent[jnp.triu_indices(distances_solvent.shape[0], k=1)]
        distances_solvent = self.apply_boundaries(distances_solvent)
        distances_solvent = jnp.linalg.norm(distances_solvent, axis=-1)
        distance_dimer_solvent = jnp.linalg.norm(self.apply_boundaries(x[2:][None,:,:] - x[:2][:,None,:]), axis = -1)
        energy = (jnp.sum(energywell(distance_dimer, self.h, self.sigmaLJ, self.w))
                + jnp.sum(energyWCA(distances_solvent, self.epsilon, self.sigmaLJ))
                + jnp.sum(energyWCA(distance_dimer_solvent, self.epsilon, self.sigmaLJ))
                )
        return energy

    def energy_full(self,x, state=None):
        return self.energy_full_particles(x)

    def force(self, x, state=None):
        """Force on the particles."""
        return jax.grad(self.energy_full)(x)
    
    def xi(self, x):
        r0 = self.sigmaLJ * 2**(1./6.)
        return jnp.atleast_1d((jnp.linalg.norm(self.apply_boundaries(x[...,0,:]-x[...,1,:]), axis=-1) - r0) / (2*self.w))

    def fix_q1q2(self,x):
        """Change q1 such that distance (q1-q2) it NOT defined up to translation """
        distance_vec = (x[0] - x[1])
        distance_vec = self.apply_boundaries(distance_vec)
        return x.at[0].set(x[1] + distance_vec)  

    def grad_xi(self, x):
        """Output should be of shape (dimension, dim_CV)!!!"""        
        x_shifted = self.fix_q1q2(x)
        e12 = (x_shifted[0] - x_shifted[1]) / jnp.linalg.norm(x_shifted[0] - x_shifted[1], axis=-1, keepdims=True)
        gradxi = 1/(2*self.w) * jnp.stack([e12, -e12])
        gradxi_full = jnp.zeros_like(x)
        gradxi_full = gradxi_full.at[:2].set(gradxi)
        gradxi_full = gradxi_full.reshape(-1,1)
        return gradxi_full

    def apply_boundaries(self,x):
        return apply_prb(x, self.L)
    

class DimerModelSubset(DimerModel, EnergyModelCVSubset):
    """Dimer model with gradient only for dimer particles."""
    subset_indices: jnp.ndarray = eqx.field(static=True)

    def __init__(self, h, w, epsilon, sigmaLJ, unitcellsize, n):
        super().__init__(h, w, epsilon, sigmaLJ, unitcellsize, n)
        self.subset_indices = jnp.array([0,1], dtype=int)

    def xi_subset(self, x_subset):
        r0 = self.sigmaLJ * 2**(1./6.)
        return jnp.atleast_1d((jnp.linalg.norm(self.apply_boundaries(x_subset[0]-x_subset[1]), axis=-1) - r0) / (2*self.w))

    def grad_xi_subset(self, x_subset):
        """Output should be of shape (dimension_subset, dim_CV)!!!"""        
        x_shifted = self.fix_q1q2(x_subset)
        e12 = (x_shifted[0] - x_shifted[1]) / jnp.linalg.norm(x_shifted[0] - x_shifted[1], axis=-1, keepdims=True)
        gradxi = 1/(2*self.w) * jnp.stack([e12, -e12])
        return gradxi.reshape(-1,1)
    

class DimerLagrangeSolver(GeneralLagrangeSolver):
    """Lagrange solver for the dimer model."""
    EnergyModel: "DimerModel"
    def get_gradxilagrange_pos(self,x, xnewtilde, z_new, dt, mass, grad_xi):
        """
        Solve x_new = x_new_tilde + dt * mass^{-1} * grad_xi @ lambda such that xi(x_new)=z_new
        """
        L = self.EnergyModel.L
        r0 = self.EnergyModel.r0
        w = self.EnergyModel.w
        zmin = self.EnergyModel.zmin
        x = self.EnergyModel.fix_q1q2(x)
        xnewtilde = self.EnergyModel.fix_q1q2(xnewtilde)
        e12 = (x[0] - x[1]) / jnp.linalg.norm(x[0] - x[1], axis=-1, keepdims=True)
        b = jnp.sum(e12 * (xnewtilde[0] - xnewtilde[1]))
        delta = b**2 - (jnp.sum((xnewtilde[0] - xnewtilde[1])**2) - (2*w*z_new + r0)**2)
        lambda_n12 = w*mass/dt * jnp.where(b > 0, -b + jnp.sqrt(delta), -b - jnp.sqrt(delta))
        gradxi_old = jnp.zeros_like(x)
        gradxi_old = gradxi_old.at[:2].set(1/(2*w) * jnp.stack([e12, -e12]))
        output = gradxi_old * lambda_n12
        return jnp.where(z_new >= zmin, output, jnp.nan)   
    

class DimerLagrangeSolverSubset(GeneralLagrangeSolverSubset):
    """Lagrange solver for the dimer model."""
    EnergyModel: "DimerModel"
    def get_gradxilagrange_pos_subset(self,x, x_newtilde, z_new, dt, mass_subset, grad_xi_subset):
        """
        Solve x_new = x_new_tilde + dt * mass^{-1} * grad_xi @ lambda such that xi(x_new)=z_new
        Assume that both particles have the same mass
        """
        L = self.EnergyModel.L
        r0 = self.EnergyModel.r0
        w = self.EnergyModel.w
        zmin = self.EnergyModel.zmin
        x = self.EnergyModel.fix_q1q2(x)
        xnewtilde = self.EnergyModel.fix_q1q2(x_newtilde)
        e12 = (x[0] - x[1]) / jnp.linalg.norm(x[0] - x[1], axis=-1, keepdims=True)
        b = jnp.sum(e12 * (xnewtilde[0] - xnewtilde[1]))
        delta = b**2 - (jnp.sum((xnewtilde[0] - xnewtilde[1])**2) - (2*w*z_new + r0)**2)
        mass = mass_subset[0]
        lambda_n12 = w*mass/dt * jnp.where(b > 0, -b + jnp.sqrt(delta), -b - jnp.sqrt(delta))
        gradxi_old = 1/(2*w) * jnp.stack([e12, -e12])
        output = gradxi_old * lambda_n12
        return jnp.where(z_new >= zmin, output, jnp.nan)   


class DimerUnderdampedCV(UnderdampedNonlinCV):
    def energy_with_fixman(self, x, mass, state=None):
        energy_model = self.EnergyModel.energy_full(x)
        return energy_model

    def force_with_fixman(self, x, mass, state=None):
        force_model = self.EnergyModel.force(x)
        return force_model
    
class DimerUnderdampedCVSubset(UnderdampedNonlinCVSubset):
    def energy_with_fixman(self, x, mass, state=None):
        energy_model = self.EnergyModel.energy_full(x)
        return energy_model

    def force_with_fixman(self, x, mass, state=None):
        force_model = self.EnergyModel.force(x)
        return force_model

def get_tau_change_indices(x):
    tau = np.zeros_like(x, dtype=int)
    tau[x > .9] = 1   # Assign tau = 1 when x > 0.9
    tau[x < 0.1] = -1  # Assign tau = -1 when x < 0.1

    # Track state changes while ignoring 0s
    mask_tau = (tau != 0)
    tau_filtered = tau[mask_tau]  # Remove zeros for change detection
    indices_filtered = np.where(mask_tau)[0]  # Get corresponding indices

    # Find where tau switches between 1 and -1
    change_points = np.where(np.diff(tau_filtered) != 0)[0]  # Indexes in filtered array
    tau_change_indices = indices_filtered[change_points + 1]  # Convert to original indices    
    return tau_change_indices


def mode_change_cost_singlechain(mode_switch_indices, n_inter_traj):
# reduceat cannot deal with mode switch at last axis, therefore drop last value
    if len(mode_switch_indices) <= 2:
        return np.nan
    cost = np.add.reduceat(n_inter_traj, np.concatenate(([0,], mode_switch_indices[:-1]+1)))[:-1]
    return np.mean(cost)

def get_mode_change_cost_dimer(z_traj_list, n_inter_traj_list):
    cost = []
    for z_traj, n_inter_traj in zip(z_traj_list,n_inter_traj_list):
        tau_change_indices = get_tau_change_indices(z_traj)
        tau_change_indices = tau_change_indices[tau_change_indices < len(n_inter_traj)]
        cost.append(mode_change_cost_singlechain(tau_change_indices, n_inter_traj))
    return np.mean(cost), np.std(cost)    

