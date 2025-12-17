import jax
from jax import tree_util
import jax.numpy as jnp
from .utils import get_key_array, accept_step
import equinox as eqx


import optax

class BaseSteeringSampler(eqx.Module):
    """General sampler class.

    Contains all methods that all different implementations inherit from.
    Specific sampler only needs to specify 'onestep_schedule' and 'body_fn'
    to define procedure to generate move from z0 to z1.
    """
    EnergyModel : "BaseEnergyModel"
    CVSampler : "BaseCVSamplerModel"
    z_scheduler : "BaseZScheduler"

    @eqx.filter_jit
    def get_samples(self, x0, z0, velocity, dt, n_steps, key,parameters=None, state0=None):
        def scan_body(carry, i):
            key, z, x, state = carry
  
            zprop, key = self.CVSampler.sample(z, key)
            log_acc_zsampler = (self.CVSampler.log_prob(z, zprop) - self.CVSampler.log_prob(zprop, z))
            z, x, acceptance, key, log_acc_tot, log_acc_langevin, n_inter, success_onestep, state = self.onestep_schedule(z, zprop, x, velocity, dt, key, log_acc_zsampler, parameters, state)
            
            new_carry = (key, z, x, state)
            scan_output = (z, x, acceptance, n_inter, success_onestep)
            return new_carry, scan_output
        
        carry = (key, z0, x0, state0)
        indices = jnp.arange(n_steps)
        (final_key, _, _, _), outputs = jax.lax.scan(scan_body, carry, indices)
        z_trajectory, x_trajectory, accept_trajectory, n_inter_trajectory, success_traj = outputs
        z_traj = jnp.insert(z_trajectory, 0, z0, axis=0)
        x_traj = jnp.insert(x_trajectory, 0, x0, axis=0)
        acceptance_rate = accept_trajectory.mean()
        # first_false_index = jnp.where(~keep_going_traj)[0]
        # jax.debug.print("First index where test is False: {i}", i=first_false_index)
        return z_traj, x_traj, acceptance_rate, n_inter_trajectory, final_key, success_traj

    @eqx.filter_jit
    def get_samples_stack(self, x0, z0, velocity, dt, n_steps, key, parameters=None, state0=None):               
        z_proposals, logp_proposals, key = self.CVSampler.sample_and_log_prob_batch(key, n_steps)
        def scan_body(carry, i):
            key, z, logp_z, x, state = carry
            zprop = z_proposals[i]
            logp_z_prop = logp_proposals[i]
            log_acc_zsampler = (logp_z - logp_z_prop)
            z, x, acceptance, key, log_acc_tot, log_acc_langevin, n_inter, success_onestep, state = self.onestep_schedule(z, zprop, x, velocity, dt, key, log_acc_zsampler, parameters, state)
            logp_z = jnp.where(acceptance, logp_z_prop, logp_z)
            new_carry = (key, z, logp_z, x, state)
            scan_output = (z, x, acceptance, n_inter, success_onestep)
            return new_carry, scan_output
        
        logp_z0 = self.CVSampler.log_prob(z0, None)
        carry = (key, z0, logp_z0, x0, state0)
        indices = jnp.arange(n_steps)
        (final_key, _, _, _, _), outputs = jax.lax.scan(scan_body, carry, indices)
        z_trajectory, x_trajectory, accept_trajectory, n_inter_trajectory, success_traj = outputs
        z_traj = jnp.insert(z_trajectory, 0, z0, axis=0)
        x_traj = jnp.insert(x_trajectory, 0, x0, axis=0)
        acceptance_rate = accept_trajectory.mean()
        return z_traj, x_traj, acceptance_rate, n_inter_trajectory, final_key, success_traj    
       
    def get_samples_parallel(self, x0, z0, velocity, dt, n_steps, n_chains, key, vectordimension=1, parameters=None, state0=None, batched_proposal = False):
        if batched_proposal:
            get_samples_fn = self.get_samples_stack
        else:
            get_samples_fn = self.get_samples

        def broadcast_neighborlist(state0, n_chains):
            return tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)) if isinstance(x, jax.Array) else x,
                state0
            )    
        if x0.ndim == vectordimension:
            x0 = jnp.broadcast_to(x0, (n_chains, *(x0.shape)))
        if z0.ndim == 1:
            z0 = jnp.broadcast_to(z0, (n_chains, z0.shape[0]))

        if state0 is None:
            key, subkeys = get_key_array(key, n_chains)
            result = jax.vmap(get_samples_fn, in_axes=(0, 0, None, None, None, 0, None))(x0, z0, velocity, dt, n_steps, subkeys, parameters)
        else:
            if state0.idx.shape[0] != n_chains:
                state0 = broadcast_neighborlist(state0, n_chains)           
            key, subkeys = get_key_array(key, n_chains)
            assert subkeys.shape[0] == n_chains
            result = jax.vmap(get_samples_fn, in_axes=(0, 0, None, None, None, 0, None, 0))(x0, z0, velocity, dt, n_steps, subkeys, parameters, state0)
        return key, result

    def onestep_parallel_endpoints(self, z0, z1, x0, velocity, dt, n_chains, key, vectordimension=1, parameters=None, state0=None):
        log_acc_zsampler = self.CVSampler.log_prob(z0, z1) - self.CVSampler.log_prob(z1, z0)
        key, subkeys = get_key_array(key, n_chains)

        def onestep_schedule_wrapped(x0_i, state0_i, subkey):
            return self.onestep_schedule(z0, z1, x0_i, velocity, dt, subkey, log_acc_zsampler, parameters, state=state0_i)


        if x0.ndim == vectordimension + 1:
            onestep_parallelkeys = jax.vmap(onestep_schedule_wrapped, in_axes=(0, 0, 0))
        else:
            onestep_parallelkeys = jax.vmap(onestep_schedule_wrapped, in_axes=(None, None, 0))

        z1_results, xnew_results, acceptance_vals, keys, log_acc_tot, log_acc_langevin, n_inter, success_onestep, states = onestep_parallelkeys(x0, state0, subkeys)
        return xnew_results, z1_results, keys[0], acceptance_vals, log_acc_tot, log_acc_langevin

    def onestep_parallel_traj(self, z0, z1, x0, dt, n_inter, n_chains, key, parameters=None, vectordimension=1, state0=None):
        log_acc_zsampler = self.CVSampler.log_prob(z0, z1) - self.CVSampler.log_prob(z1, z0)
        key, subkeys = get_key_array(key, n_chains)

        def onestep_schedule_trajectory_wrapped(x0_i, state0_i, subkey):
            return self.onestep_schedule_trajectory(z0, z1, x0_i, dt, subkey, n_inter, log_acc_zsampler, parameters, state0_i)


        if x0.ndim == vectordimension + 1:
            onestep_parallelkeys = jax.vmap(onestep_schedule_trajectory_wrapped, in_axes=(0, 0, 0))
        else:
            onestep_parallelkeys = jax.vmap(onestep_schedule_trajectory_wrapped, in_axes=(None, None, 0))

        xtraj, ztraj, keys, log_acc_tot, log_acc_zsampler, log_acc_ediff, log_acc_langevin, success = onestep_parallelkeys(x0, subkeys, state0)
        return xtraj, ztraj, keys[0], log_acc_tot, log_acc_zsampler, log_acc_ediff, log_acc_langevin
    
    def onestep_parallel_traj_multi_z(self, z0, z1, x0, dt, n_inter, key, parameters=None, vectordimension=1, state0=None):
        n_chains = z0.shape[0]
        assert z1.shape[0] == n_chains, 'z1 and z0 must have same batch size'
        assert x0.shape[0] == n_chains, 'x0 and z0 must have same batch size'
        assert x0.ndim == vectordimension + 1, f'vectordimenion specified as {vectordimension}, but not compatible with shape of x0 {x0.shape}'
    
        key, subkeys = get_key_array(key, n_chains)
        if state0 is None:
            @eqx.filter_jit
            def onestep_schedule_trajectory_fixed(z0_i, z1_i, x0_i, key_i):
                return self.onestep_schedule_trajectory(z0_i, z1_i, x0_i, dt, key_i, n_inter, 0., parameters)
            assert subkeys.shape[0] == n_chains
            result = jax.vmap(onestep_schedule_trajectory_fixed, in_axes=(0, 0, 0, 0))(z0, z1, x0, subkeys)

        else:
            assert state0.idx.shape[0] == n_chains, 'state0 and z0 must have same batch size'
            @eqx.filter_jit
            def onestep_schedule_trajectory_fixed(z0_i, z1_i, x0_i, key_i, state0_i):
                return self.onestep_schedule_trajectory(z0_i, z1_i, x0_i, dt, key_i, n_inter, 0., parameters, state0_i)
            assert subkeys.shape[0] == n_chains
            result = jax.vmap(onestep_schedule_trajectory_fixed, in_axes=(0, 0, 0, 0, 0))(z0, z1, x0, subkeys, state0)
        xtraj, ztraj, _, log_acc_tot, _, log_acc_ediff, log_acc_langevin, ptraj, success_onestep = result
        return key, xtraj, ztraj, log_acc_tot, log_acc_ediff, log_acc_langevin, ptraj, success_onestep


    def onestep_endpoints_multi_z(self, z0, z1, x0, velocity, dt, key, parameters=None, vectordimension=1, state0=None):
        n_chains = z0.shape[0]
        assert z1.shape[0] == n_chains, 'z1 and z0 must have same batch size'
        assert x0.shape[0] == n_chains, 'x0 and z0 must have same batch size'
        assert x0.ndim == vectordimension + 1, f'vectordimension specified as {vectordimension}, but got {x0.shape}'

        key, subkeys = get_key_array(key, n_chains)

        if state0 is None:  
            @eqx.filter_jit
            def onestep_schedule_wrapped(z0_i, z1_i, x0_i, vel_i, key_i):
                return self.onestep_schedule(z0_i, z1_i, x0_i, vel_i, dt, key_i,0, parameters)
            
            result = jax.vmap(onestep_schedule_wrapped, in_axes=(0, 0, 0, None, 0))(z0, z1, x0, velocity, subkeys)

        else:
            assert state0.idx.shape[0] == n_chains, 'state0 and z0 must have same batch size'
            @eqx.filter_jit
            def onestep_schedule_wrapped(z0_i, z1_i, x0_i, vel_i, key_i, state_i):
                return self.onestep_schedule(z0_i, z1_i, x0_i, vel_i, dt, key_i, 0, parameters, state_i)
            
            result = jax.vmap(onestep_schedule_wrapped, in_axes=(0, 0, 0, None, 0, 0))(z0, z1, x0, velocity, subkeys, state0)

        znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter, keep_going, state = result

        return key[0], xnew, znew, log_acc_tot, log_acc_langevin, acceptance, n_inter
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        raise NotImplementedError("Please implement this method")

    def onestep_schedule_trajectory(self,z0,z1,x0,dt,key,n_inter,log_acc_zsampler, parameters, state=None):
        raise NotImplementedError("Please implement this method")
 