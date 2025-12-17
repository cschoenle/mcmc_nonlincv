import jax
import jax.numpy as jnp
from .base import BaseSteeringSampler
from .utils import accept_step

class UnderdampedSubsetCV(BaseSteeringSampler): 
    """To avoid the temperature entering into this algorithm, we assume that we are sampling exp(-p^2/(2*mass) - beta*V(x))
    The parameters mass, dt, and sigma use in the algorithm are then related to the physical parameters by:
    mass_phys = mass*beta, dt_phys = dt*beta, sigma_phys = sigma/sqrt(beta).
    """     
    EnergyModel : "EnergyModelSeparated"
    @jax.jit
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key,log_acc_zsampler, parameters, state=None):
        gamma = parameters['gamma']
        mass = parameters['mass']  
        n_inter = self.z_scheduler.get_n_inter(z0, z1, velocity, dt)
        

        def body_fn_global(i, vals):
            work_p, key, x, p,state = vals
            (work_p, key, x, p,z_new,state) = self.body_fn_core(work_p, key, x, p, z0, z1, dt, i, n_inter, gamma, mass,state)
            return (work_p, key, x, p,state)   
        
        work_p = 0. 
        key, subkey = jax.random.split(key)
        p0 = jnp.sqrt(mass) * jax.random.normal(subkey, shape=x0.shape)              
        init_vals = (work_p, key, x0, p0,state)
        (work_p, key, xnew, pnew,state_new) = jax.lax.fori_loop(0, n_inter, body_fn_global, init_vals)
        
        old_energy=self.EnergyModel.energy(z0,x0,state)
        state_new=self.EnergyModel.update_state_subset(z1,xnew,state)
        work_total = work_p + self.EnergyModel.energy(z1,xnew,state_new) - old_energy
        log_acc_langevin = - work_total
        log_acc_tot = log_acc_langevin  + log_acc_zsampler
        xnew, acceptance, key = accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        keep_going = True
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter, keep_going, state

    def onestep_schedule_trajectory(self,z0,z1,x0,dt,key,n_inter,log_acc_zsampler, parameters, state=None):
        gamma = parameters['gamma']
        mass = parameters['mass']         
        def body_fn_onestep(carry, i):  
            work_p, key, x, p,state = carry
            (work_p, key, x, p, z_new,state_new) = self.body_fn_core(work_p, key, x, p, z0, z1, dt, i, n_inter, gamma, mass,state)
            return (work_p, key, x, p,state_new), (x, z_new,p)  
        work_p = 0. 
        key, subkey = jax.random.split(key)
        p0 = jnp.sqrt(mass) * jax.random.normal(subkey, shape=x0.shape) 
        
        init_vals = (work_p, key, x0, p0,state)
        indices = jnp.arange(n_inter)
        (work_p, key, xnew, pnew,state_new), (xtraj, ztraj,ptraj) = jax.lax.scan(body_fn_onestep,init_vals,indices)
        old_energy=self.EnergyModel.energy(z0,x0,state)
        state_new=self.EnergyModel.update_state_subset(z1,xnew,state)
        
        work_q = work_p + self.EnergyModel.energy(z1,xnew,state_new) - old_energy
        work_total = work_p + work_q
        log_acc_tot = -work_total  + log_acc_zsampler
        success = True
        return xtraj, ztraj, key, log_acc_tot, log_acc_zsampler, -work_q, -work_p,ptraj, success
  
    
    def body_fn_core(self, work_p, key, x, p, z0, z1, dt, i, n_inter, gamma, mass,state):
        z_old = self.z_scheduler.get_step(i,z0,z1,n_inter)
        z_new = self.z_scheduler.get_step(i+1,z0,z1,n_inter)
        sigma = jnp.sqrt(2*gamma)
        key, subkey1, subkey2 = jax.random.split(key, 3)
        noise1 = jax.random.normal(subkey1, shape=x.shape)
        noise2 = jax.random.normal(subkey2, shape=x.shape)
        p14 = ((1 - dt*gamma/(4*mass)) * p + jnp.sqrt(dt/2)*sigma*noise1) / (1+dt*gamma/(4*mass))
        p12 = p14 - dt/2 * self.EnergyModel.force_partialx(z_old,x,state)
        x_new = x + dt/mass*p12
        x_new = self.EnergyModel.apply_boundaries_subset(x_new)
        state_new=self.EnergyModel.update_state_subset(z_new,x_new,state)
        p34 = p12 - dt/2 * self.EnergyModel.force_partialx(z_new,x_new,state_new)
        p_new = ((1 - dt*gamma/(4*mass)) * p34 + jnp.sqrt(dt/2)*sigma*noise2) / (1+dt*gamma/(4*mass))
        work_p += jnp.sum(p34**2 - p14**2)/(2*mass)
        return work_p, key, x_new, p_new,z_new, state_new






