import jax
import jax.numpy as jnp
from .utils import get_key_array, accept_step
import equinox as eqx

class MalaSamplerFull(eqx.Module):
    EnergyModel : "BaseEnergyModel"    
    def mala_sampler(self,x0, dt, n_steps, key, state0=None, write_every=1):
        def bodyfn(carry,i):
            x, key, state, energy, output_arr_x, output_arr_energy = carry
            neg_grad = -dt*self.EnergyModel.force(x, state)
            key, subkey = jax.random.split(key)
            noise = jnp.sqrt(2*dt)*jax.random.normal(subkey, shape=x.shape)
            dx = neg_grad + noise
            xnew = x + dx
            statenew = self.EnergyModel.update_state(xnew, state)
            energynew = self.EnergyModel.energy_full(xnew, statenew)
            log_acc_mala = (-(energynew - energy) 
                            - 0.25/dt * jnp.sum((-dx + dt*self.EnergyModel.force(xnew, statenew))**2-noise**2))
            xnew, acceptance, key = accept_step(x, xnew, log_acc_mala, key)
            statenew = jax.lax.cond(
                acceptance,
                lambda _: statenew,
                lambda _: state,
                operand=None,
            )
            energynew = jnp.where(acceptance, energynew, energy)
            xnew = self.EnergyModel.apply_boundaries(xnew)

            def write_position(output_arr_x):
                return output_arr_x.at[i // write_every].set(xnew)
            output_arr_x = jax.lax.cond(i % write_every == 0, write_position, lambda arr_x: arr_x, output_arr_x)

            def write_energy(output_arr_e):
                return output_arr_e.at[i // write_every].set(energynew)
            output_arr_energy = jax.lax.cond(i % write_every == 0, write_energy, lambda arr_e: arr_e, output_arr_energy)

            return (xnew, key, statenew, energynew, output_arr_x, output_arr_energy), (acceptance)

        n_log_steps = n_steps // write_every
        output_arr_x = jnp.zeros((n_log_steps,) + x0.shape)
        output_arr_energy = jnp.zeros((n_log_steps,) + ())
        energy0 = self.EnergyModel.energy_full(x0, state0)
        init_vals = (x0, key, state0, energy0, output_arr_x, output_arr_energy)
        (x, key, state, energy, output_arr_x, output_arr_energy), acc_array = jax.lax.scan(bodyfn, init_vals, jnp.arange(n_steps))

        return acc_array, output_arr_x, output_arr_energy
    
    def get_samples_parallel(self, x0, dt, n_steps, n_chains, key, vectordimension=1, state0=None, write_every=1):
        def broadcast_neighborlist(state0, n_chains):
            return jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)) if isinstance(x, jax.Array) else x,
                state0
            )    
        if x0.ndim == vectordimension:
            x0 = jnp.broadcast_to(x0, (n_chains, *(x0.shape)))
        if state0 is None:
            key, subkeys = get_key_array(key, n_chains)
            (acc_array, output_arr_x, output_arr_energy) = eqx.filter_jit(jax.vmap(self.mala_sampler, in_axes=(0, None, None, 0, None, None)))(x0, dt, n_steps, subkeys, None, write_every)
        else:
            if state0.idx.shape[0] != n_chains:
                state0 = broadcast_neighborlist(state0, n_chains)           
            key, subkeys = get_key_array(key, n_chains)
            assert subkeys.shape[0] == n_chains
            (acc_array, output_arr_x, output_arr_energy) = jax.vmap(self.mala_sampler, in_axes=(0, None, None, 0, 0, None))(x0, dt, n_steps, subkeys, state0, write_every)
        return acc_array, output_arr_x, output_arr_energy