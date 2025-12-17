import equinox as eqx
import jax.random as jr
from jax import vmap
class BaseCVSamplerModel(eqx.Module):
    def sample(self, zold, key):
        raise NotImplementedError    
    def log_prob(self, z, zold):
        raise NotImplementedError
    def sample_and_log_prob(self, key):
        # helper method for iid sampler
        z, key = self.sample(None, key)
        logp = self.log_prob(z, None)
        return z, logp, key
    def sample_and_log_prob_batch(self, key, batch_size):
        keys = jr.split(key, batch_size+1)
        key = keys[0]
        subkeys = keys[1:]
        z, logp, keys = vmap(self.sample_and_log_prob, in_axes=(0,))(subkeys)
        return z, logp, key
