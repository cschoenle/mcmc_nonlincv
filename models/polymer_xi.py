#### Implement different child classes of Polymer for different choices of CV xi.
# import sys, os
# sys.path.append(os.path.abspath(".."))
from .polymer import PolymerBase
import jax.numpy as jnp
import jax
import numpy as np
import equinox as eqx
from models.base import EnergyModelCVSubset, EnergyModelLinCVSubset, EnergyModelSeparated
from samplers.nonlinCV_subset import GeneralLagrangeSolverSubset, UnderdampedNonlinCVSubset


def xi_end_to_end(polymer_positions):
    end_to_end= polymer_positions[-1] - polymer_positions[0]
    # diffs=jnp.concatenate((diffs,end_to_end[None,:]),axis=0)
    return jnp.linalg.norm(end_to_end[None,:], axis = -1)

class PolymerGeneralXi(PolymerBase, EnergyModelCVSubset):
    """Takes CV based on Polymer beads"""
    subset_indices: jax.Array
    xi_fn_polymer: callable = eqx.static_field()
    def __init__(self, config: dict, xi_fn_polymer):
        super().__init__(config)
        self.subset_indices = self.polymer_indices
        self.xi_fn_polymer = xi_fn_polymer

    def xi_subset(self,x_subset):
        return self.xi_fn_polymer(x_subset)

    def grad_xi_subset(self,x_subset):
        jacobian_matrix = jax.jacobian(self.xi_subset)(x_subset)
        grad_xi = jacobian_matrix.reshape(-1, self.dim * self.n_polymer).T
        return grad_xi


class PolymerLinSubset(PolymerBase, EnergyModelLinCVSubset):   
    def __init__(self, config: dict):
        PolymerBase.__init__(self, config)
        self.subset_indices=self.polymer_indices

    def polymer_from_xi(self, xi, n_polymer, dim, eps=1e-8):
        return xi.reshape(n_polymer, dim)

    def xi_from_polymer(self, polymer_positions):
        return self.xi_subset(polymer_positions)
    
class PolymerSeparated(PolymerBase,EnergyModelSeparated):
    @jax.jit 
    def energy(self,z,x,nbrs):
        position=jnp.concatenate((x,z.reshape(-1,self.dim)))
        return self.energy_full(position,nbrs)
    
    @jax.jit 
    def energy_lj(self,z,x,nbrs):
        position=jnp.concatenate((x,z.reshape(-1,self.dim)))
        return self.lj_energy_fn(position,nbrs)
    
    @jax.jit 
    def force_partialx(self,z,x,nbrs):
        position=jnp.concatenate((x,z.reshape(-1,self.dim)))
        return self.force(position,nbrs)[:self.n_solvent]    

    def update_state_subset(self,z,x,nbrs):
        position=jnp.concatenate((x,z.reshape(-1,self.dim)))
        return self.update_state(position,nbrs)        
 
    
class PolymerEndToEnd(PolymerBase, EnergyModelCVSubset):
    """Polymer model with CV the end to end distance."""
    subset_indices: jnp.ndarray = eqx.field(static=True)
    def __init__(self, config: dict):
        super().__init__(config)
        self.subset_indices = self.polymer_indices

    def xi_subset(self, x_subset):
        return jnp.atleast_1d(jnp.linalg.norm(x_subset[0]-x_subset[-1], axis=-1))

    def grad_xi_subset(self, x_subset):
        """Output should be of shape (dimension_subset, dim_CV)!!!"""        
        e19 = (x_subset[0] - x_subset[-1]) / jnp.linalg.norm(x_subset[0] - x_subset[-1], axis=-1, keepdims=True)
        zeros_rest = jnp.zeros((self.n_polymer -2, self.dim)).flatten()
        gradxi = jnp.concatenate((e19, zeros_rest, -e19), axis=0)
        return gradxi.reshape(-1,1)
       

class PolymerEndToEndLagrangeSolverSubset(GeneralLagrangeSolverSubset):
    """Lagrange solver for the dimer model."""
    EnergyModel: "PolymerEndToEnd"
    def get_gradxilagrange_pos_subset(self,x, x_newtilde, z_new, dt, mass_subset, grad_xi_subset):
        """
        Solve x_new = x_new_tilde + dt * mass^{-1} * grad_xi @ lambda such that xi(x_new)=z_new
        Assume that both particles have the same mass
        """
        x_newtilde_subset = x_newtilde[self.EnergyModel.subset_indices]
        e19 = grad_xi_subset[:3].flatten()
        b = jnp.sum(e19 * (x_newtilde_subset[0] - x_newtilde_subset[-1]))
        delta = b**2 - (jnp.sum((x_newtilde_subset[0] - x_newtilde_subset[-1])**2) - z_new**2)
        mass = mass_subset[0]
        lambda_n19 = 0.5*mass/dt * jnp.where(b > 0, -b + jnp.sqrt(delta), -b - jnp.sqrt(delta))
        return (grad_xi_subset * lambda_n19).reshape(x_newtilde_subset.shape) 
    

class PolymerEndToEndCVSubsetSampler(UnderdampedNonlinCVSubset):
    def energy_with_fixman(self, x, mass, state=None):
        energy_model = self.EnergyModel.energy_full(x, state)
        return energy_model

    def force_with_fixman(self, x, mass, state=None):
        force_model = self.EnergyModel.force(x, state)
        return force_model
    

def get_tau_change_indices(x, transition):
    tau = np.ones_like(x, dtype=int)
    tau[x < transition] = -1  # Assign tau = -1 when x < 0.1

    # Track state changes while ignoring 0s
    mask_tau = (tau != 0)
    tau_filtered = tau[mask_tau]  # Remove zeros for change detection
    indices_filtered = np.where(mask_tau)[0]  # Get corresponding indices

    # Find where tau switches between 1 and -1
    change_points = np.where(np.diff(tau_filtered) != 0)[0]  # Indexes in filtered array
    tau_change_indices = indices_filtered[change_points + 1]  # Convert to original indices    
    return tau_change_indices
  

def mode_change_cost_singlechain(mode_change, n_inter_traj):
# reduceat cannot deal with mode switch at last axis, therefore drop last value
    mode_switches = np.asarray(mode_change == True).nonzero()[0] + 1 #index refers to step after mode switch
    if len(mode_switches) == 0:
        return np.nan
    else:
        n_inter_traj = n_inter_traj[:mode_switches[-1]] # only keep cost up to last mode switch
    cost = np.add.reduceat(n_inter_traj, np.concatenate(([0,], mode_switches[:-1])))
    return np.mean(cost)

def get_mode_change_cost_polymer(x_polymer_traj_list, n_inter_traj_list, transition = 4.6):
    n_chains = x_polymer_traj_list.shape[0]
    n_time_steps = x_polymer_traj_list.shape[1]
    z_traj_list = jnp.linalg.norm(x_polymer_traj_list[...,0,:] - x_polymer_traj_list[...,-1,:], axis=-1).reshape(n_chains, n_time_steps)
    mode_traj_list = (z_traj_list > transition)
    mode_change_list = mode_traj_list[:,1:] != mode_traj_list[:,:-1]
    cost = []
    for mode_change, n_inter_traj in zip(mode_change_list,n_inter_traj_list):
        cost.append(mode_change_cost_singlechain(mode_change, n_inter_traj))
    return np.mean(cost), np.std(cost)   
