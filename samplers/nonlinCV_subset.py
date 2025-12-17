import jax
import jax.numpy as jnp
from .base import BaseSteeringSampler
from .utils import accept_step
import equinox as eqx
from jaxopt import GaussNewton

# import sys
# import os
# # Add parent directory to sys.path
# sys.path.append(os.path.abspath(".."))
# from models.base import EnergyModelCVSubset

tolerance_solver = 1e-8
tolerance_check = 1e-6

class UnderdampedNonlinCVSubset(BaseSteeringSampler):
    """
    Sampling algorithm for a non-linear CV where CV just depends on a small subset of the full coordinates.
    Assumes that EnergyModel has 
        - an attribute self.subset_indices which gives the indices of the subset of coordinates necessary for the CV.
        - a method self.xi_subset(x_subset) which computes the CV from the subset coordinates.
        - a method self.grad_xi_subset(x_subset) which computes the gradient of the CV wrt the subset coordinates.
    To avoid the temperature entering into this algorithm, we assume that we are sampling exp(-p^2/(2*mass) - beta*V(x))
    The parameters mass, dt, and sigma use in the algorithm are then related to the physical parameters by:
    mass_phys = mass*beta, dt_phys = dt*beta, sigma_phys = sigma/sqrt(beta).
    """     
    LagrangeSolver : "BaseLagrangeSolverSubset"
    EnergyModel : "EnergyModelCVSubset"
    
    @eqx.filter_jit
    def onestep_schedule(self,z0, z1, x0, velocity, dt, key, log_acc_zsampler, parameters, state=None):
        mass = parameters['mass']  
        n_inter = self.z_scheduler.get_n_inter(z0, z1, velocity, dt)
        def body_fn_global(i, vals):
            work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, success_onestep, state = vals
            (work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, z_new, success, state) = \
                self.body_fn_core(work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, z0, z1, dt, i, n_inter, parameters, state)
            success_onestep = jnp.logical_and(success_onestep, success)
            return (work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, success_onestep, state)   
        work_p = jnp.zeros([])
        key, subkey = jax.random.split(key)
        p0tilde = jnp.sqrt(mass) * jax.random.normal(subkey, shape=x0.shape)
        v0 = self.z_scheduler.get_vz_step(0, z0, z1, n_inter, dt)
        x0_subset = x0[self.EnergyModel.subset_indices]
        grad_xi_subset = self.EnergyModel.grad_xi_subset(x0_subset)
        mass_diagonal = mass * jnp.ones(x0.flatten().shape[0]) # in case mass is a scalar
        mass_subset = mass_diagonal.reshape(x0.shape)[self.EnergyModel.subset_indices].flatten()
        gram_matrix_inv_subset = self.get_gram_matrix_inverse_subset(grad_xi_subset, mass_subset)
        grad_xi_gram_matrix_inv_subset = grad_xi_subset @ gram_matrix_inv_subset        
        gradxi_lambda0_subset = self.LagrangeSolver.get_gradxilagrange_mom_subset(x0, p0tilde, v0, grad_xi_subset, grad_xi_gram_matrix_inv_subset, mass_subset)
        p0 = p0tilde.at[self.EnergyModel.subset_indices].add(gradxi_lambda0_subset) 
        force_x0 = self.force_with_fixman(x0, mass, state)
        success_onestep = True
        init_vals = (work_p, key, x0, p0, force_x0, grad_xi_subset, grad_xi_gram_matrix_inv_subset, success_onestep, state)
        (work_p, key, xnew, pnew, force_xnew, grad_xi_new_subset, grad_xi_gram_matrix_inv_new_subset, success_onestep, statenew) = jax.lax.fori_loop(0, n_inter, body_fn_global, init_vals)
        work_total = work_p + self.energy_with_fixman(xnew, mass, statenew) - self.energy_with_fixman(x0, mass, state)
        log_acc_langevin = - work_total
        log_acc_tot = log_acc_langevin  + log_acc_zsampler
        log_acc_tot = jnp.where(jnp.logical_not(success_onestep), -jnp.inf, log_acc_tot) # force rejection if constraint not satisfied
        xnew, acceptance, key = accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        statenew = jax.lax.cond(
            acceptance,
            lambda _: statenew,
            lambda _: state,
            operand=None,
        )
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter, success_onestep, statenew
    
    def onestep_schedule_trajectory(self,z0,z1,x0,dt,key,n_inter,log_acc_zsampler, parameters, state=None):
        mass = parameters['mass']
        def body_fn_onestep(carry, i):  
            work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, success_onestep, state = carry
            (work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, z_new, success, state) = \
                self.body_fn_core(work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, z0, z1, dt, i, n_inter, parameters, state)
            success_onestep = jnp.logical_and(success_onestep, success)
            return (work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, success_onestep, state), (x, z_new, p) 
        work_p = jnp.zeros([])
        key, subkey = jax.random.split(key)
        p0tilde = jnp.sqrt(mass) * jax.random.normal(subkey, shape=x0.shape)
        v0 = self.z_scheduler.get_vz_step(0, z0, z1, n_inter, dt)
        x0_subset = x0[self.EnergyModel.subset_indices]
        grad_xi_subset = self.EnergyModel.grad_xi_subset(x0_subset)
        mass_diagonal = mass * jnp.ones(x0.flatten().shape[0]) # in case mass is a scalar
        mass_subset = mass_diagonal.reshape(x0.shape)[self.EnergyModel.subset_indices].flatten()
        gram_matrix_inv_subset = self.get_gram_matrix_inverse_subset(grad_xi_subset, mass_subset)
        grad_xi_gram_matrix_inv_subset = grad_xi_subset @ gram_matrix_inv_subset        
        gradxi_lambda0_subset = self.LagrangeSolver.get_gradxilagrange_mom_subset(x0, p0tilde, v0, grad_xi_subset, grad_xi_gram_matrix_inv_subset, mass_subset)
        p0 = p0tilde.at[self.EnergyModel.subset_indices].add(gradxi_lambda0_subset) 
        force_x0 = self.force_with_fixman(x0, mass, state)
        success_onestep=True
        init_vals = (work_p, key, x0, p0, force_x0, grad_xi_subset, grad_xi_gram_matrix_inv_subset, success_onestep, state)
        indices = jnp.arange(n_inter)
        (work_p, key, xnew, pnew, force_xnew, grad_xi_new_subset, grad_xi_gram_matrix_inv_new_subset, success_onestep, statenew), (xtraj, ztraj, ptraj) = jax.lax.scan(body_fn_onestep,init_vals,indices)
        log_acc_ediff = -(self.energy_with_fixman(xnew, mass, statenew) - self.energy_with_fixman(x0, mass, state))
        log_acc_langevin = - work_p
        log_acc_tot = log_acc_zsampler + log_acc_ediff + log_acc_langevin
        # success = 0.5 * jnp.sum((self.EnergyModel.xi(xnew) - z1)**2) < tolerance_check
        log_acc_tot = jnp.where(jnp.logical_not(success_onestep), -jnp.inf, log_acc_tot) # force rejection if constraint not satisfied
        return xtraj, ztraj, key, log_acc_tot, log_acc_zsampler, log_acc_ediff, log_acc_langevin, ptraj, success_onestep
  
    
    def body_fn_core(self, work_p, key, x, p, force_x, grad_xi_subset, grad_xi_gram_matrix_inv_subset, z0, z1, dt, i, n_inter, parameters, state):
        z_new = self.z_scheduler.get_step(i+1,z0,z1,n_inter)
        vz0 = self.z_scheduler.get_vz_step(i, z0, z1, n_inter, dt)
        vz1 = self.z_scheduler.get_vz_step(i+1, z0, z1, n_inter, dt)
        x_new, p_new, work_addition_momentum, force_x_new, grad_xi_new_subset, grad_xi_gram_matrix_inv_new_subset, key, state_new = \
            self.get_rattle_step(x, p, force_x, dt, z_new, vz0, vz1, parameters, key, grad_xi_subset, grad_xi_gram_matrix_inv_subset, state)
        work_p += work_addition_momentum
        x_new_subset = x_new[self.EnergyModel.subset_indices]
        success = 0.5 * jnp.sum((self.EnergyModel.xi_subset(x_new_subset) - z_new)**2) < tolerance_check
        return work_p, key, x_new, p_new, force_x_new, grad_xi_new_subset, grad_xi_gram_matrix_inv_new_subset, z_new, success, state_new
    
    def get_rattle_step(self, x, p, force_x, dt, z_new, vz0, vz1, parameters, key, grad_xi_subset, grad_xi_gram_matrix_inv_subset, state):
        mass = parameters['mass']
        p14, key = self.get_momentum_constrained(p, x, vz0, dt, parameters, key, grad_xi_subset, grad_xi_gram_matrix_inv_subset)
        xnewtilde = x + dt/mass * p14 - dt**2/(2*mass) * force_x
        mass_diagonal = mass * jnp.ones(x.flatten().shape[0]) # in case mass is a scalar
        mass_subset = mass_diagonal.reshape(x.shape)[self.EnergyModel.subset_indices].flatten()        
        lambda_n12_gradxi_old_subset = self.LagrangeSolver.get_gradxilagrange_pos_subset(x, xnewtilde, z_new, dt, mass_subset, grad_xi_subset)
        p12 = p14 - dt/2 * force_x
        p12 = p12.at[self.EnergyModel.subset_indices].add(lambda_n12_gradxi_old_subset)
        x_new = xnewtilde 
        x_new = x_new.at[self.EnergyModel.subset_indices].add(dt/mass * lambda_n12_gradxi_old_subset)
        state_new = self.EnergyModel.update_state(x_new, state)
        force_x_new = self.force_with_fixman(x_new, mass, state_new)
        p34tilde = p12 - dt/2 * force_x_new
        x_new = self.EnergyModel.apply_boundaries(x_new)
        # Compute new grad_xi and grad_xi@gram_matrix_inverse
        grad_xi_new_subset = self.EnergyModel.grad_xi_subset(x_new[self.EnergyModel.subset_indices])
        gram_matrix_inv_new_subset = self.get_gram_matrix_inverse_subset(grad_xi_new_subset, mass_subset)
        grad_xi_gram_matrix_inv_new_subset = grad_xi_new_subset @ gram_matrix_inv_new_subset        
        gradxi_lambda_n34_new_subset = self.LagrangeSolver.get_gradxilagrange_mom_subset(x_new, p34tilde, vz1, grad_xi_new_subset, grad_xi_gram_matrix_inv_new_subset, mass_subset)
        p34 = p34tilde.at[self.EnergyModel.subset_indices].add(gradxi_lambda_n34_new_subset)
        p_new, key = self.get_momentum_constrained(p34, x_new, vz1, dt, parameters, key, grad_xi_new_subset, grad_xi_gram_matrix_inv_new_subset) 
        work_addition_momentum = jnp.sum((p34**2 - p14**2)/(2*mass))
        return x_new, p_new, work_addition_momentum, force_x_new, grad_xi_new_subset, grad_xi_gram_matrix_inv_new_subset, key, state_new
    
    def get_momentum_constrained(self,p0, x, vz, dt, parameters,key, grad_xi_subset, grad_xi_gram_matrix_inv_subset):
        mass = parameters['mass']
        mass_diagonal = mass * jnp.ones(x.flatten().shape[0]) # in case mass is a scalar
        gamma = parameters['gamma']
        sigma_lagrange = jnp.sqrt(2*gamma)
        coefficient = dt/4*gamma/mass_diagonal
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=x.shape)
        shape_subset_nonflat = p0[self.EnergyModel.subset_indices].shape
        addition_subset = (grad_xi_gram_matrix_inv_subset @ vz).reshape(shape_subset_nonflat)
        addition_full = (jnp.zeros_like(p0).at[self.EnergyModel.subset_indices].set(addition_subset)).flatten()
        p14_tilde = ((1-coefficient) * p0.flatten()
            + dt/2 * gamma  / mass_diagonal * addition_full
            + jnp.sqrt(dt/2) * sigma_lagrange * noise.flatten()
            )/(1+coefficient)
        mass_subset = mass_diagonal.reshape(x.shape)[self.EnergyModel.subset_indices].flatten()
        coefficient_subset = dt/4*gamma/mass_subset
        skewed_gram_inv = self.get_skewed_gram_inv(grad_xi_subset,mass_subset, coefficient_subset)
        p14_tilde_subset = p14_tilde.reshape(p0.shape)[self.EnergyModel.subset_indices].flatten()
        correction = (grad_xi_subset / (1+coefficient_subset)[:, None]) @ skewed_gram_inv @ (vz -(grad_xi_subset / mass_subset[:, None]).T @ p14_tilde_subset)
        correction_unflat = correction.reshape(shape_subset_nonflat)
        p14_tilde_unflat = p14_tilde.reshape(p0.shape)
        p14 = p14_tilde_unflat.at[self.EnergyModel.subset_indices].add(correction_unflat)
        return p14, key 
    
    def get_gram_matrix_subset(self, grad_xi_subset, mass_subset):
        """ Expects grad_xi to be of shape (dim, dim_CV) and mass to be a scalar or an array of shape (dim,). """
        mass_diagonal = mass_subset * jnp.ones(grad_xi_subset.shape[0]) # in case mass is a scalar
        # Multiply each row of grad_xi by 1/mass_diagonal (broadcasted)
        gram_matrix = grad_xi_subset.T @ (grad_xi_subset / mass_diagonal[:, None])
        return gram_matrix        

    def get_gram_matrix_inverse_subset(self, grad_xi_subset, mass_subset):
        """ Expects grad_xi to be of shape (dim, dim_CV) and mass to be a scalar or an array of shape (dim,). """
        gram_matrix = self.get_gram_matrix_subset(grad_xi_subset, mass_subset)
        gram_matrix_inv = jnp.linalg.inv(gram_matrix)
        return gram_matrix_inv
    
    def get_skewed_gram_inv(self,grad_xi_subset,mass_subset, coefficient_subset):
        skewed_gram = grad_xi_subset.T @ (grad_xi_subset / (mass_subset*(1+coefficient_subset))[:, None])
        skewed_gram_inv = jnp.linalg.inv(skewed_gram)
        return skewed_gram_inv

    def fixman_term_subset(self, x_subset, mass_subset):
        grad_xi_subset = self.EnergyModel.grad_xi_subset(x_subset)
        gram_matrix = self.get_gram_matrix_subset(grad_xi_subset, mass_subset)
        return 0.5  * jnp.log(jnp.linalg.det(gram_matrix))

    def energy_with_fixman(self, x, mass, state):
        energy_model = self.EnergyModel.energy_full(x, state)
        x_subset = x[self.EnergyModel.subset_indices]
        mass_diagonal = mass * jnp.ones(x.flatten().shape[0]) # in case mass is a scalar
        mass_subset = mass_diagonal.reshape(x.shape)[self.EnergyModel.subset_indices].flatten()        
        fixmax_energy = self.fixman_term_subset(x_subset, mass_subset)
        return energy_model + fixmax_energy
    
    def force_with_fixman(self, x, mass, state):
        x_subset = x[self.EnergyModel.subset_indices]
        mass_diagonal = mass * jnp.ones(x.flatten().shape[0]) # in case mass is a scalar
        mass_subset = mass_diagonal.reshape(x.shape)[self.EnergyModel.subset_indices].flatten()              
        force_fixman = jax.grad(lambda x_subset: self.fixman_term_subset(x_subset, mass_subset))(x_subset)
        force_model = self.EnergyModel.force(x, state)
        force_total = force_model.at[self.EnergyModel.subset_indices].add(force_fixman)
        return force_total


class BaseLagrangeSolverSubset(eqx.Module):
    EnergyModel : "EnergyModelCVSubset"
    """Base class for Lagrange solvers."""   
    def get_gradxilagrange_pos_subset(self, x, x_newtilde, z_new, dt, mass_subset, grad_xi_subset):
        raise NotImplementedError   
    def get_gradxilagrange_mom_subset(self, x, ptilde, vz, grad_xi_subset, grad_xi_gram_matrix_inv_subset, mass_subset):
        raise NotImplementedError       

class GeneralLagrangeSolverSubset(BaseLagrangeSolverSubset):
    EnergyModel : "EnergyModelCVSubset"
    """General class for Lagrange solvers, numerical solution for position constraint."""   
    @eqx.filter_jit
    def get_gradxilagrange_pos_subset(self, x, x_newtilde, z_new, dt, mass_subset, grad_xi_subset):
        """Solve x_new = x_new_tilde + dt * mass^{-1} * grad_xi @ lambda such that xi(x_new)=z_new"""
        mass_prefactor = mass_subset[0]
        mass_diagonal_dimensionless = mass_subset / mass_prefactor
        x_newtilde_subset = x_newtilde[self.EnergyModel.subset_indices]
        

        def residual_fn(mu_):
            x_candidate = x_newtilde_subset + ((grad_xi_subset / mass_diagonal_dimensionless[:, None]) @ mu_).reshape(x_newtilde_subset.shape)
            return self.EnergyModel.xi_subset(x_candidate) - z_new
        solver = GaussNewton(residual_fn, maxiter=100, tol=tolerance_solver)
        mu_init = jnp.zeros(grad_xi_subset.shape[1])
        solution = solver.run(mu_init)
        mu_solution = solution.params
        lambda_solution = mu_solution * (mass_prefactor / dt)
        return (grad_xi_subset @ lambda_solution).reshape(x_newtilde_subset.shape)
    
    @eqx.filter_jit
    def get_gradxilagrange_mom_subset(self, x, ptilde, vz, grad_xi_subset, grad_xi_gram_matrix_inv_subset, mass_subset):
        """
        For pnew = ptilde + gradxi(x)*lambda with constraint v_xi(xnew)=z_dot,
        compute gradxi(x)*lambda. 
        Expects grad_xi to be of shape (dim, dim_CV) and mass to be a scalar or an array of shape (dim,).
        """
        ptilde_subset = ptilde[self.EnergyModel.subset_indices]
        grad_xi_lambda = grad_xi_gram_matrix_inv_subset @ (-(grad_xi_subset / mass_subset[:, None]).T @ ptilde_subset.flatten() + vz)
        grad_xi_lambda = jnp.reshape(grad_xi_lambda, ptilde_subset.shape)
        return grad_xi_lambda


class UnderdampedCVSamplerLinCVSubset(UnderdampedNonlinCVSubset):
    EnergyModel: "EnergyModelLinCVSubset"
    def energy_with_fixman(self, x, mass, state):
        return self.EnergyModel.energy_full(x, state)
    
    def force_with_fixman(self, x, mass, state):
        return self.EnergyModel.force(x, state)
    
    def get_skewed_gram_inv(self,grad_xi_subset,mass_subset, coefficient_subset):
        return jnp.diag(mass_subset*(1+coefficient_subset))
    
    def get_gram_matrix_subset(self, grad_xi_subset, mass_subset):
        return jnp.diag(1/mass_subset)  

    def get_gram_matrix_inverse_subset(self, grad_xi_subset, mass_subset):
        return jnp.diag(mass_subset)  
    
    
class LagrangeSolverLinCVSubset(GeneralLagrangeSolverSubset):
    """Lagrange solver for the Polymer when CV is just a subset."""
    EnergyModel: "EnergyModelLinCVSubset"
    def get_gradxilagrange_pos_subset(self, x, xnewtilde, z_new, dt, mass_subset, grad_xi_subset):
        """
        Solve x_new = x_new_tilde + dt * mass^{-1} * grad_xi @ lambda such that xi(x_new)=z_new
        """
        x_subset_shape = xnewtilde[self.EnergyModel.subset_indices].shape
        lambda_ = mass_subset / dt * (z_new - xnewtilde[self.EnergyModel.subset_indices].flatten())
        return (grad_xi_subset @ lambda_).reshape(x_subset_shape)    
