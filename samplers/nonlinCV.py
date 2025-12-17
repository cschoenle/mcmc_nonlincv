import jax
import jax.numpy as jnp
from .base import BaseSteeringSampler
from .utils import accept_step
import equinox as eqx
from jaxopt import GaussNewton,LBFGS

tolerance_solver = 1e-8
tolerance_check = 1e-6

class UnderdampedNonlinCV(BaseSteeringSampler):
    """To avoid the temperature entering into this algorithm, we assume that we are sampling exp(-p^2/(2*mass) - beta*V(x))
    The parameters mass, dt, and sigma use in the algorithm are then related to the physical parameters by:
    mass_phys = mass*beta, dt_phys = dt*beta, sigma_phys = sigma/sqrt(beta).
    """     
    LagrangeSolver : "BaseLagrangeSolver"
    EnergyModel : "EnergyModelCV"    
    
    @eqx.filter_jit
    def onestep_schedule(self,z0, z1, x0, velocity, dt, key, log_acc_zsampler, parameters, state=None):
        mass = parameters['mass']  
        n_inter = self.z_scheduler.get_n_inter(z0, z1, velocity, dt)
        def body_fn_global(i, vals):
            work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, success_onestep, state = vals
            (work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, z_new, success, state) = self.body_fn_core(work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, z0, z1, dt, i, n_inter, parameters, state)
            success_onestep = jnp.logical_and(success_onestep, success)
            return (work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, success_onestep, state)   
        work_p = jnp.zeros([])
        key, subkey = jax.random.split(key)
        p0tilde = jnp.sqrt(mass) * jax.random.normal(subkey, shape=x0.shape)
        v0 = self.z_scheduler.get_vz_step(0, z0, z1, n_inter, dt)
        grad_xi = self.EnergyModel.grad_xi(x0)
        gram_matrix = self.get_gram_matrix(grad_xi, mass)
        gram_matrix_inv = jnp.linalg.inv(gram_matrix)
        grad_xi_gram_matrix_inv = grad_xi @ gram_matrix_inv        
        gradxi_lambda0 = self.LagrangeSolver.get_gradxilagrange_mom(x0, p0tilde, v0, grad_xi, grad_xi_gram_matrix_inv, mass)   
        p0 = p0tilde + gradxi_lambda0
        force_x0 = self.force_with_fixman(x0, mass, state)
        success_onestep = True
        init_vals = (work_p, key, x0, p0, force_x0, grad_xi, grad_xi_gram_matrix_inv, success_onestep, state)
        (work_p, key, xnew, pnew, force_xnew, grad_xi_new, grad_xi_gram_matrix_inv_new, success_onestep, statenew) = jax.lax.fori_loop(0, n_inter, body_fn_global, init_vals)
        work_total = work_p + self.energy_with_fixman(xnew, mass, statenew) - self.energy_with_fixman(x0, mass, state)
        log_acc_langevin = - work_total
        log_acc_tot = log_acc_langevin  + log_acc_zsampler
        # success = 0.5 * jnp.sum((self.EnergyModel.xi(xnew) - z1)**2) < tolerance_check
        # first_error = jnp.logical_and(keep_going, jnp.logical_not(success))
        # Host-side logging
        # def report(vals):
        #     first_error, z0_val, z1_val, x0_val, p0_val, key_val = vals
        #     print(f"first error{first_error}, z0={z0_val}, z1={z1_val}, x0={x0_val}, p0={p0_val}, key={key_val}")
        # # Runtime-only callback (fires only when first_error=True)
        # jax.lax.cond(
        #     first_error,
        #     lambda _: jax.debug.callback(report, (first_error, z0, z1, x0, p0, key)),
        #     lambda _: None,
        #     operand=None,
        # )
        # keep_going = jnp.logical_and(keep_going, success)
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
            work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, success_onestep, state = carry
            (work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, z_new, success, state) = self.body_fn_core(work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, z0, z1, dt, i, n_inter, parameters, state)
            success_onestep = jnp.logical_and(success_onestep, success)
            return (work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, success_onestep, state), (x, z_new, p) 
        work_p = jnp.zeros([])
        key, subkey = jax.random.split(key)
        p0tilde = jnp.sqrt(mass) * jax.random.normal(subkey, shape=x0.shape)
        v0 = self.z_scheduler.get_vz_step(0, z0, z1, n_inter, dt)
        grad_xi = self.EnergyModel.grad_xi(x0)
        gram_matrix = self.get_gram_matrix(grad_xi, mass)
        gram_matrix_inv = jnp.linalg.inv(gram_matrix)
        grad_xi_gram_matrix_inv = grad_xi @ gram_matrix_inv        
        gradxi_lambda0 = self.LagrangeSolver.get_gradxilagrange_mom(x0, p0tilde, v0, grad_xi, grad_xi_gram_matrix_inv, mass)   
        p0 = p0tilde + gradxi_lambda0
        force_x0 = self.force_with_fixman(x0, mass, state)
        success_onestep=True
        init_vals = (work_p, key, x0, p0, force_x0, grad_xi, grad_xi_gram_matrix_inv, success_onestep, state)
        indices = jnp.arange(n_inter)
        (work_p, key, xnew, pnew, force_xnew, grad_xi_new, grad_xi_gram_matrix_inv_new, success_onestep, statenew), (xtraj, ztraj, ptraj) = jax.lax.scan(body_fn_onestep,init_vals,indices)
        log_acc_ediff = -(self.energy_with_fixman(xnew, mass, statenew) - self.energy_with_fixman(x0, mass, state))
        log_acc_langevin = - work_p
        log_acc_tot = log_acc_zsampler + log_acc_ediff + log_acc_langevin
        # success = 0.5 * jnp.sum((self.EnergyModel.xi(xnew) - z1)**2) < tolerance_check
        log_acc_tot = jnp.where(jnp.logical_not(success_onestep), -jnp.inf, log_acc_tot) # force rejection if constraint not satisfied
        return xtraj, ztraj, key, log_acc_tot, log_acc_zsampler, log_acc_ediff, log_acc_langevin, ptraj, success_onestep
  
    
    def body_fn_core(self, work_p, key, x, p, force_x, grad_xi, grad_xi_gram_matrix_inv, z0, z1, dt, i, n_inter, parameters, state):
        z_new = self.z_scheduler.get_step(i+1,z0,z1,n_inter)
        vz0 = self.z_scheduler.get_vz_step(i, z0, z1, n_inter, dt)
        vz1 = self.z_scheduler.get_vz_step(i+1, z0, z1, n_inter, dt)
        x_new, p_new, work_addition_momentum, force_x_new, grad_xi_new, grad_xi_gram_matrix_inv_new, key, state_new = self.get_rattle_step(x, p, force_x, dt, z_new, vz0, vz1, parameters, key, grad_xi, grad_xi_gram_matrix_inv, state)
        work_p += work_addition_momentum
        success = 0.5 * jnp.sum((self.EnergyModel.xi(x_new) - z_new)**2) < tolerance_check
        return work_p, key, x_new, p_new, force_x_new, grad_xi_new, grad_xi_gram_matrix_inv_new, z_new, success, state_new
    
    def get_rattle_step(self, x, p, force_x, dt, z_new, vz0, vz1, parameters, key, grad_xi, grad_xi_gram_matrix_inv, state):
        mass = parameters['mass']
        p14, key = self.get_momentum_constrained(p, x, vz0, dt, parameters, key, grad_xi, grad_xi_gram_matrix_inv)
        xnewtilde = x + dt/mass * p14 - dt**2/(2*mass) * force_x
        lambda_n12_gradxi_old = self.LagrangeSolver.get_gradxilagrange_pos(x, xnewtilde, z_new, dt, mass, grad_xi)
        p12 = p14 - dt/2 * force_x + lambda_n12_gradxi_old
        x_new = xnewtilde + dt/mass * lambda_n12_gradxi_old
        state_new = self.EnergyModel.update_state(x_new, state)
        force_x_new = self.force_with_fixman(x_new, mass, state_new)
        p34tilde = p12 - dt/2 * force_x_new
        x_new = self.EnergyModel.apply_boundaries(x_new)
        # Compute new grad_xi and grad_xi@gram_matrix_inverse
        grad_xi_new = self.EnergyModel.grad_xi(x_new)
        gram_matrix_new = self.get_gram_matrix(grad_xi_new, mass)
        gram_matrix_inv_new = jnp.linalg.inv(gram_matrix_new)
        grad_xi_gram_matrix_inv_new = grad_xi_new @ gram_matrix_inv_new        
        gradxi_lambda_n34_new = self.LagrangeSolver.get_gradxilagrange_mom(x_new, p34tilde, vz1, grad_xi_new, grad_xi_gram_matrix_inv_new, mass)
        p34 = p34tilde + gradxi_lambda_n34_new
        p_new, key = self.get_momentum_constrained(p34, x_new, vz1, dt, parameters, key, grad_xi_new, grad_xi_gram_matrix_inv_new) 
        work_addition_momentum = jnp.sum((p34**2 - p14**2)/(2*mass))
        return x_new, p_new, work_addition_momentum, force_x_new, grad_xi_new, grad_xi_gram_matrix_inv_new, key, state_new
    
    def get_momentum_constrained(self,p0, x, vz, dt, parameters,key, grad_xi, grad_xi_gram_matrix_inv):
        mass = parameters['mass']
        mass_diagonal = mass * jnp.ones(grad_xi.shape[0]) # in case mass is a scalar
        gamma = parameters['gamma']
        sigma_lagrange = jnp.sqrt(2*gamma)
        coefficient = dt/4*gamma/mass_diagonal
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=x.shape)
        p14_tilde = ((1-coefficient) * p0.flatten()
            + dt/2 * gamma  / mass_diagonal * (grad_xi_gram_matrix_inv @ vz)
            + jnp.sqrt(dt/2) * sigma_lagrange * noise.flatten()
            )/(1+coefficient)
        skewed_gram = grad_xi.T @ (grad_xi / (mass_diagonal*(1+coefficient))[:, None])
        skewed_gram_inv = jnp.linalg.inv(skewed_gram)
        correction = (grad_xi / (1+coefficient)[:, None]) @ skewed_gram_inv @ (vz -(grad_xi / mass_diagonal[:, None]).T @ p14_tilde)
        # grad_xi_lambda = self.LagrangeSolver.get_gradxilagrange_mom(x, p14_tilde, vz, grad_xi, grad_xi_gram_matrix_inv, mass)
        return jnp.reshape(p14_tilde + correction, p0.shape), key 
    
    def get_gram_matrix(self, grad_xi, mass):
        """ Expects grad_xi to be of shape (dim, dim_CV) and mass to be a scalar or an array of shape (dim,). """
        mass_diagonal = mass * jnp.ones(grad_xi.shape[0]) # in case mass is a scalar
        # Multiply each row of grad_xi by 1/mass_diagonal (broadcasted)
        return grad_xi.T @ (grad_xi / mass_diagonal[:, None])

    def fixman_term(self, x, mass):
        grad_xi = self.EnergyModel.grad_xi(x)
        gram_matrix = self.get_gram_matrix(grad_xi, mass)
        return 0.5  * jnp.log(jnp.linalg.det(gram_matrix))

    def energy_with_fixman(self, x, mass, state):
        energy_model = self.EnergyModel.energy_full(x, state)
        fixmax_energy = self.fixman_term(x, mass)
        return energy_model + fixmax_energy
    
    def force_with_fixman(self, x, mass, state):
        force_fixman = jax.grad(lambda x: self.fixman_term(x, mass))(x)
        force_model = self.EnergyModel.force(x, state)
        return force_model + force_fixman


class BaseLagrangeSolver(eqx.Module):
    EnergyModel : "EnergyModelCV"
    """Base class for Lagrange solvers."""   
    def get_gradxilagrange_pos(self, x, x_newtilde, z_new, dt, mass, grad_xi):
        raise NotImplementedError   
    def get_gradxilagrange_mom(self, x, ptilde, vz, grad_xi, grad_xi_gram_matrix_inv, mass):
        raise NotImplementedError       

class GeneralLagrangeSolver(BaseLagrangeSolver):
    EnergyModel : "EnergyModelCV"
    """General class for Lagrange solvers, numerical solution for position constraint."""   
    @eqx.filter_jit
    def get_gradxilagrange_pos(self, x, x_newtilde, z_new, dt, mass, grad_xi):
        """Solve x_new = x_new_tilde + dt * mass^{-1} * grad_xi @ lambda such that xi(x_new)=z_new"""
        mass_diagonal = mass * jnp.ones(grad_xi.shape[0])
        mass_prefactor = mass_diagonal[0]
        mass_diagonal_dimensionless = mass_diagonal / mass_prefactor

        def residual_fn(mu_):
            x_candidate = x_newtilde + ((grad_xi / mass_diagonal_dimensionless[:, None]) @ mu_).reshape(x.shape)
            return self.EnergyModel.xi(x_candidate) - z_new
        solver = GaussNewton(residual_fn, maxiter=100, tol=tolerance_solver)

        # def loss_fn(lambda_):
        #     x_candidate = x_newtilde + (dt * (grad_xi / mass_diagonal[:, None]) @ lambda_).reshape(x.shape)
        #     return jnp.sum((self.EnergyModel.xi(x_candidate) - z_new)**2)
        # solver = LBFGS(loss_fn, maxiter=50, tol=1e-8)        

        mu_init = jnp.zeros(grad_xi.shape[1])
        solution = solver.run(mu_init)
        mu_solution = solution.params
        lambda_solution = mu_solution * (mass_prefactor / dt)

        # jax.debug.print("Number of steps was {x} with difference {y}", x=solution.state.iter_num, y=residual_fn(mu_solution))

        # mse = 0.5*jnp.sum(residual_fn(mu_solution))**2 
        # all_good = (mse < tolerance_solver)

        # def dummy_fn(_):
        #     pass
        # def error_fn(_):
        #     jax.debug.print("At target z={z}, difference is {x}", z=z_new, x=residual_fn(mu_solution))
        # jax.lax.cond(all_good, dummy_fn, error_fn, mse)

        return (grad_xi @ lambda_solution).reshape(x.shape)
    @eqx.filter_jit
    def get_gradxilagrange_mom(self, x, ptilde, vz, grad_xi, grad_xi_gram_matrix_inv, mass):
        """
        For pnew = ptilde + gradxi(x)*lambda with constraint v_xi(xnew)=z_dot,
        compute gradxi(x)*lambda. 
        Expects grad_xi to be of shape (dim, dim_CV) and mass to be a scalar or an array of shape (dim,).
        """
        mass_diagonal = mass * jnp.ones(grad_xi.shape[0])
        grad_xi_lambda = grad_xi_gram_matrix_inv @ (-(grad_xi / mass_diagonal[:, None]).T @ ptilde.flatten() + vz)
        grad_xi_lambda = jnp.reshape(grad_xi_lambda, ptilde.shape)

        # p_final = ptilde + grad_xi_lambda
        # v_final = (grad_xi / mass_diagonal[:, None]).T @ p_final.flatten()
        # mse = 0.5*jnp.sum((v_final - vz)/vz)**2 
        # all_good = (mse < tolerance_check)

        # def dummy_fn(_):
        #     pass
        # def error_fn(_):
        #     jax.debug.print("Velocity difference is {x}", x=v_final - vz)
        # jax.lax.cond(all_good, dummy_fn, error_fn, mse)        

        return grad_xi_lambda

