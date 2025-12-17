import jax
import jax.numpy as jnp
import equinox as eqx
from .base import BaseCVSamplerModel
import equinox as eqx
from flowjax.flows import masked_autoregressive_flow
from typing import Optional

from flowjax.bijections import RationalQuadraticSpline

"""A bimodal Gaussian sampler in 1D."""
class BiModalGaussian1D(BaseCVSamplerModel):
    """Args:
    - z1: float, the mean of the first Gaussian mode.
    - z2: float, the mean of the second Gaussian mode.
    - w1_proposal: float, the weight of the first Gaussian mode in the proposal distribution.
    """
    z1: float = eqx.field(static=True)
    z2: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    w1_proposal: float = eqx.field(static=True)

    def __init__(self, z1, z2, sigma=1.0, w1_proposal=0.5):
        self.z1 = z1
        self.z2 = z2
        self.sigma = sigma
        self.w1_proposal = w1_proposal
    @jax.jit
    def sample(self, z, key):
        subkeys = jax.random.split(key,4)
        key = subkeys[0]
        subkeys = subkeys[1:]
        coin = jax.random.uniform(subkeys[0]) <= self.w1_proposal
        mode1 = self.sigma * jax.random.normal(subkeys[1], shape=[1,]) + self.z1
        mode2 = self.sigma * jax.random.normal(subkeys[2], shape=[1,]) + self.z2
        result = jnp.where(coin, mode1, mode2)
        return result, key
    @jax.jit
    def log_prob(self, z, z_old):
        prob = 1/jnp.sqrt(2*jnp.pi * self.sigma**2) * (self.w1_proposal * jnp.exp(-0.5 * jnp.sum((z-self.z1)**2) / self.sigma**2) 
                                    + (1-self.w1_proposal) * jnp.exp(-0.5 * jnp.sum((z-self.z2)**2) / self.sigma**2))
        return jnp.log(prob)

class BiModalGaussian1DLowerBounded(BaseCVSamplerModel):
    z1: float = eqx.field(static=True)
    z2: float = eqx.field(static=True)
    z_min: jnp.ndarray  # 1d array, not static
    sigma: float = eqx.field(static=True)
    w1_proposal: float = eqx.field(static=True)

    def __init__(self, z1, z2, z_min, sigma=1.0, w1_proposal=0.5):
        assert z1 > z_min and z2 > z_min, "z1 and z2 must be greater than z_min"
        self.z1 = z1
        self.z2 = z2
        self.z_min = z_min        
        self.sigma = sigma
        self.w1_proposal = w1_proposal 

    @jax.jit
    def sample(self,z,key):
        """Sample from the bimodal Gaussian distribution with a lower bound."""
        def check_lowerbound(vals):
            z, key = vals
            return jnp.all(z < self.z_min)
        
        def z_sampler_single_body(vals):
            z, key = vals
            subkeys = jax.random.split(key,4)
            key = subkeys[0]
            subkeys = subkeys[1:]
            coin = jax.random.uniform(subkeys[0]) <= self.w1_proposal
            mode1 = self.sigma * jax.random.normal(subkeys[1], shape=[1,]) + self.z1
            mode2 = self.sigma * jax.random.normal(subkeys[2], shape=[1,]) + self.z2
            znew = jnp.where(coin, mode1, mode2)
            return (znew, key)        
        
        z_init = self.z_min - 1.
        init_vals = (z_init, key)
        z_new, key = jax.lax.while_loop(check_lowerbound, z_sampler_single_body, init_vals)
        return z_new, key

    @jax.jit
    def log_prob(self, z, z_old):
        """Ignore changed normalisation constant, does not matter for acceptance in MCMC."""
        prob = 1/jnp.sqrt(2*jnp.pi * self.sigma**2) * (self.w1_proposal * jnp.exp(-0.5 * jnp.sum((z-self.z1)**2) / self.sigma**2) 
                                    + (1-self.w1_proposal) * jnp.exp(-0.5 * jnp.sum((z-self.z2)**2) / self.sigma**2))
        return jnp.log(prob)

class BiModalGaussian1DLowerUpperBounded(BaseCVSamplerModel):
    z1: float = eqx.field(static=True)
    z2: float = eqx.field(static=True)
    z_min: jnp.ndarray  # 1d array, not static
    z_max: jnp.ndarray  # 1d array, not static
    sigma: float = eqx.field(static=True)
    w1_proposal: float = eqx.field(static=True)

    def __init__(self, z1, z2, z_min, z_max, sigma=1.0, w1_proposal=0.5):
        assert z1 > z_min and z2 > z_min, "z1 and z2 must be greater than z_min"
        self.z1 = z1
        self.z2 = z2
        self.z_min = z_min
        self.z_max = z_max
        self.sigma = sigma
        self.w1_proposal = w1_proposal
        assert z1 < z_max and z2 < z_max, "z1 and z2 must be smaller than z_max"
        assert z_max > z_min, "z_max must be greater than z_min"
        assert z1 > z_min and z2 > z_min, "z1 and z2 must be greater than z_min"

    @jax.jit
    def sample(self,z,key):
        """Sample from the bimodal Gaussian distribution with a lower bound."""
        def check_lowerupperbound(vals):
            z, key = vals
            return jnp.all(jnp.logical_or(z < self.z_min, z > self.z_max))
        
        def z_sampler_single_body(vals):
            z, key = vals
            subkeys = jax.random.split(key,4)
            key = subkeys[0]
            subkeys = subkeys[1:]
            coin = jax.random.uniform(subkeys[0]) <= self.w1_proposal
            mode1 = self.sigma * jax.random.normal(subkeys[1], shape=[1,]) + self.z1
            mode2 = self.sigma * jax.random.normal(subkeys[2], shape=[1,]) + self.z2
            znew = jnp.where(coin, mode1, mode2)
            return (znew, key)        
        
        z_init = self.z_min - 1. # start with invalid value below lower bound
        init_vals = (z_init, key)
        z_new, key = jax.lax.while_loop(check_lowerupperbound, z_sampler_single_body, init_vals)
        return z_new, key

    @jax.jit
    def log_prob(self, z, z_old):
        """Ignore changed normalisation constant, does not matter for acceptance in MCMC."""
        prob = 1/jnp.sqrt(2*jnp.pi * self.sigma**2) * (self.w1_proposal * jnp.exp(-0.5 * jnp.sum((z-self.z1)**2) / self.sigma**2) 
                                    + (1-self.w1_proposal) * jnp.exp(-0.5 * jnp.sum((z-self.z2)**2) / self.sigma**2))
        return jnp.log(prob)

    