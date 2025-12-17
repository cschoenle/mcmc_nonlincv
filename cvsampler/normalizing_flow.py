import jax
import jax.numpy as jnp
import equinox as eqx
from .base import BaseCVSamplerModel
import equinox as eqx

from flowjax.flows import coupling_flow
from flowjax.distributions import Transformed
from flowjax.bijections import Chain, RationalQuadraticSpline, AbstractBijection
import paramax

class FlowSampler(BaseCVSamplerModel):
    """Wrapper around a FlowJAX Transformed flow to fit the BaseCVSamplerModel API."""
    flow: Transformed = eqx.field()

    def sample(self, zold, key):
        """Draw a single sample (non-batched) from the flow."""
        # flow.sample expects a batch shape, so just wrap/unpack
        key, subkey = jax.random.split(key)
        sample = self.flow.sample(subkey, (1,))
        return jnp.squeeze(sample, axis=0), key

    def log_prob(self, z, zold):
        """Compute log-probability for a single point."""
        # flow.log_prob expects batches
        logp = self.flow.log_prob(jnp.expand_dims(z, 0))
        return jnp.squeeze(logp, axis=0)
    
    def sample_and_log_prob(self, key):
        z, logp = self.flow.sample_and_log_prob(key, (1,))
        return jnp.squeeze(z, axis=0), jnp.squeeze(logp, axis=0)

    def sample_and_log_prob_batch(self, key, batch_size):
        key, subkey = jax.random.split(key)
        z, logp = self.flow.sample_and_log_prob(subkey, (batch_size,))
        return z, logp, key

class ActNorm_Cov(AbstractBijection):
    """Per-dimension affine transform with data-dependent init."""

    x_mean: jnp.ndarray
    log_det: jnp.ndarray
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    L: jnp.ndarray
    whitening_matrix: jnp.ndarray

    def __init__(self, x_init, eps: float = 1e-6):
        # x_init should be [batch, dim])
        self.x_mean = jnp.mean(x_init, axis=0)
        self.shape = (x_init.shape[1],)
        self.cond_shape = None
        # Covariance matrix of your data
        def compute_covariance(X):
            # Center the data
            X_centered = X - jnp.mean(X, axis=0, keepdims=True)
            
            # Covariance matrix (dim x dim)
            cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
            return cov
        cov = compute_covariance(x_init)  # X: (batch, dim)
        # Eigendecomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov)  # symmetric matrix
        self.whitening_matrix = jnp.diag(1.0 / jnp.sqrt(eigenvalues)) @ eigenvectors.T
        # Scale and rotate
        # sqrt(eigenvalues) * Z in eigenvector basis
        self.L = eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues))
        self.log_det=0.5 * jnp.sum(jnp.log(eigenvalues))
        # X has shape (batch, dim)
        

    def transform_and_log_det(self, x, condition=None):
        y = x @ self.L.T + self.x_mean 
        return y, self.log_det

    def inverse_and_log_det(self, y, condition=None):
        x = y - self.x_mean
        # Whitened samples
        x = (x) @ self.whitening_matrix.T  # shape (batch, dim)
        return x, -self.log_det

def autoregressive_flow_with_actnorm(
    key,
    base_dist,
    x_init,
    flow_layers=8,
    knots=16,
    interval=4.0,
):
    # First get the autoregressive flow from flowjax
    maf = coupling_flow(
        key=key,
        base_dist=base_dist,
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
        flow_layers=flow_layers,
        nn_depth=2,
        nn_width=30
    )

    # Add ActNorm around it
    #inverse_batch=vmap(to_constrained.inverse)
    actnorm = paramax.non_trainable(ActNorm_Cov(x_init))

    # Chain them: ActNorm first, then the flow
    bijection = Chain([ maf.bijection,actnorm])
    #bijection = Chain([actnorm,])
    return Transformed(maf.base_dist, bijection )