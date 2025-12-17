import jax
import jax.numpy as jnp

def get_key_array(key, n_keys):
    keys = jax.random.split(key, num=n_keys + 1)
    return keys[0], keys[1:]

def accept_step(x0, xnew, log_acc_tot, key):
    """Accept or reject xnew based on log_acc_tot."""
    check_isnan = jnp.any(jnp.isnan(xnew))
    log_acc_tot = jnp.where(check_isnan, -jnp.inf, log_acc_tot)  # always reject
    key, subkey = jax.random.split(key)
    u1 = jax.random.uniform(subkey)
    acceptance = jnp.log(u1) <= log_acc_tot
    xout = jax.lax.cond(
        acceptance,
        lambda _: xnew,
        lambda _: x0,
        operand=None
    )
    # xnew = jnp.where(acceptance, xnew, x0)
    return xout, acceptance, key

