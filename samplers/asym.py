import jax
import jax.numpy as jnp
from .base import BaseSteeringSampler
from .utils import get_key_array, accept_step
import equinox as eqx
class AsymSampler(BaseSteeringSampler):   
    EnergyModel : "EnergyModelSeparated"
    @eqx.filter_jit
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key,log_acc_zsampler, parameters, state=None):
        n_inter = self.z_scheduler.get_n_inter(z0, z1, velocity, dt)
        log_acc_langevin = 0.  # log(rev) - log(forw)
        def body_fn_global(i, vals):
            log_acc, key, x = vals
            (log_acc, key, x, z_new) = self.body_fn_core(log_acc, key, x, z0, z1, dt, i, n_inter)
            return (log_acc, key, x)   
        init_vals = (log_acc_langevin, key, x0)
        (log_acc_langevin, key, xnew) = jax.lax.fori_loop(0, n_inter, body_fn_global, init_vals)
        log_acc_ediff = -(self.EnergyModel.energy(z1, xnew) - self.EnergyModel.energy(z0, x0))
        log_acc_tot = log_acc_zsampler + log_acc_ediff + log_acc_langevin
        xnew, acceptance, key = accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        keep_going = True
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter, keep_going, state
    
    def onestep_schedule_trajectory(self,z0,z1,x0,dt,key,n_inter,log_acc_zsampler, parameters, state=None):
        log_acc_langevin = 0.
        def body_fn_onestep(carry, i):  
            log_acc, key, x = carry
            (log_acc, key, x, z_new) = self.body_fn_core(log_acc, key, x, z0, z1, dt, i, n_inter)
            return (log_acc, key, x), (x, z_new)  
        init_vals = (log_acc_langevin, key, x0)
        indices = jnp.arange(n_inter)
        (log_acc_langevin, key, xnew), (xtraj, ztraj) = jax.lax.scan(body_fn_onestep,init_vals,indices)
        log_acc_ediff = -(self.EnergyModel.energy(z1,xnew) - self.EnergyModel.energy(z0,x0)) 
        log_acc_tot = log_acc_zsampler + log_acc_ediff + log_acc_langevin
        success = True
        return xtraj, ztraj, key, log_acc_tot, log_acc_zsampler, log_acc_ediff, log_acc_langevin, success
  
    
    def body_fn_core(self, log_acc, key, x, z0, z1, dt, i, n_inter):
        z_old = self.z_scheduler.get_step(i,z0,z1,n_inter)
        z_new = self.z_scheduler.get_step(i+1,z0,z1,n_inter)
        neg_grad = -dt * self.EnergyModel.force_partialx(z_new,x)
        
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2*dt)*jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        x += dx
        log_acc += -0.25 / dt * jnp.sum((-dx + dt * self.EnergyModel.force_partialx(z_old, x)) ** 2 - noise ** 2,
                                        axis=-1)
        return log_acc, key, x, z_new