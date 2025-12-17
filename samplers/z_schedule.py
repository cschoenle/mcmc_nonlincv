import jax.numpy as jnp
import equinox as eqx

class BaseZScheduler(eqx.Module):
    def get_n_inter(self, z0, z1, velocity, dt):
        raise NotImplementedError
    
    def get_step(self, i, z0, z1, n_inter):
        raise NotImplementedError
    
class ZSchedulerLinear(BaseZScheduler):
    def get_n_inter(self, z0, z1, velocity, dt):
        dz = z1 - z0
        abs_dz = jnp.sqrt(jnp.sum(dz ** 2))
        n_inter = jnp.asarray(jnp.ceil(abs_dz / (velocity * dt)), dtype=jnp.int32)
        return n_inter
    def get_step(self, i, z0, z1, n_inter):
        return z0 + (z1 - z0) / n_inter * i
    def get_vz_step(self, i, z0, z1, n_inter, dt):
        return (z1 - z0) / n_inter / dt

class ZSchedulerCos(BaseZScheduler):
    def get_n_inter(self, z0, z1, velocity, dt):
        dz = z1 - z0
        abs_dz = jnp.sqrt(jnp.sum(dz ** 2))
        n_inter = jnp.asarray(jnp.ceil(abs_dz / (velocity * dt)), dtype=jnp.int32)
        return n_inter
    def shift_function(self, time_frac):
        return -jnp.cos(time_frac * jnp.pi) / 2. + 0.5
    def shift_function_derivative(self, time_frac):
        return jnp.sin(time_frac * jnp.pi) / 2. * jnp.pi
    def get_step(self, i, z0, z1, n_inter):
        time_frac = i/n_inter
        return z0 + (z1 - z0) * self.shift_function(time_frac)
    def get_vz_step(self, i, z0, z1, n_inter, dt):
        time_frac = i/n_inter
        return (z1 - z0) * self.shift_function_derivative(time_frac) / (n_inter * dt) # multiply by 1/T