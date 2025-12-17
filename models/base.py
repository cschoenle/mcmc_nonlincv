import equinox as eqx
import jax.numpy as jnp

# DISCLAIMER: Note that the subroutine 'force' of a model is supposed to give the gradient of the energy
# (and therefore actually the negative force in the physics meaning)

class BaseEnergyModel(eqx.Module):
    def energy_full(self,x, state=None):
        return NotImplementedError
    def force(self, x, state=None):
        raise NotImplementedError
    def apply_boundaries(self,x, state=None):
        return x
    def update_state(self,x,state=None):
        return state
    
class EnergyModelSeparated(BaseEnergyModel):
    """Energy model with functions that take bipartition of variables (z,x), eg. polymer and solvent."""
    def energy(self, z, x, state=None):
        raise NotImplementedError
    def force_partialx(self, z, x, state=None):
        raise NotImplementedError
    def apply_boundaries_subset(self, x):
        return x
    def update_state_subset(self,z,x,state):
        return state
    
class EnergyModelCV(BaseEnergyModel):
    """Energy model with functions that take collective variables xi(x), x is then *full* state."""
    def xi(self, x):
        raise NotImplementedError
    def grad_xi(self, x):
        raise NotImplementedError
    def update_state(self, x, state):
        return state

class EnergyModelCVSubset(EnergyModelCV):
    """Energy model where CV is a function of a subset x_subset of the full state x."""
    subset_indices: jnp.ndarray  
    def xi_subset(self, x_subset):
        raise NotImplementedError
    def grad_xi_subset(self, x_subset):
        raise NotImplementedError
    def xi(self, x):
        return self.xi_subset(x[self.subset_indices])
    
class EnergyModelLinCVSubset(EnergyModelCVSubset):   
    """Energy model where CV is just a subset of the variables"""
    def xi_subset(self, x_subset):
        return x_subset.flatten()
    
    def grad_xi(self, x):
        x_subset = x[self.subset_indices]
        ndims = x.flatten().shape[0]
        nsub = x_subset.flatten().shape[0]
        grad_xi_full = jnp.zeros(x.shape + (nsub,))
        grad_xi_sub = self.grad_xi_subset(x_subset)
        grad_xi_full = grad_xi_full.at[self.subset_indices].set(grad_xi_sub.reshape(x_subset.shape + (nsub,))).reshape(ndims,nsub)  
        return grad_xi_full
    
    def grad_xi_subset(self, x_subset):
        nsub = x_subset.flatten().shape[0]
        return jnp.diag(jnp.ones(nsub))
    