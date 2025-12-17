import equinox as eqx
import jax
import jax.numpy as jnp
from .base import BaseCVSamplerModel

class NormalizingFlowPolymer(BaseCVSamplerModel):
    flow: eqx.Module = eqx.field(static=True)
    signs: jnp.ndarray
    box_size: jnp.ndarray = eqx.field(static=True)  
    idx_to_keep :jnp.ndarray = eqx.field(static=True)

    def __init__(self, flow: eqx.Module, signs: jnp.ndarray,box_size: jnp.ndarray):
        self.flow = flow
        
        self.signs = signs
        self.box_size=box_size
        self.idx_to_keep=jnp.array(list(set(range(0,27))-set([0,1,2,4,5,8])))

    @eqx.filter_jit
    def sample(self, z_init, key):
        """Sample a single valid z, resampling until no NaN."""
        def cond(vals):
            z, key,idx = vals
            z = self.trilaterate_chain_lastcv(z, self.signs[idx])
            return jnp.isnan(z).any()

        def body(vals):
            _, key, idx = vals
            new_key, subkey = jax.random.split(key)
            z = self.flow.sample(key=subkey)
            return z, new_key, idx

        init_key, subkey, rot_key = jax.random.split(key, 3)
        z0 = self.flow.sample(key=subkey)
        idx = jax.random.randint(subkey, shape=(), minval=0, maxval=self.signs.shape[0])
        
        z, _, _ = jax.lax.while_loop(cond, body, (z0, init_key,idx))
        
        z=self.trilaterate_chain_lastcv(z, self.signs[idx])
        
        center_idx = z.shape[0] // 2
        shift = z[center_idx] 
        z = z - shift

        #rot_key, key = jax.random.split(key)
        z = self.rotate_polymer(z, rot_key)

        
        shift = z[center_idx] - jnp.asarray(self.box_size/2)
        z = z - shift
        return z.flatten(),key

    def sample_many(self, key, n_samples=100):
        """Sample n_samples independently (resampling only NaN samples)."""
        keys = jax.random.split(key, n_samples)
        zs,key,_ = jax.vmap(self.sample)(None, keys)
        return zs
    def log_prob(self, z, zold):
        cvs=self.cvs_from_polymer(z.reshape(9,3))
        value=self.flow.log_prob(cvs)+jnp.linalg.slogdet(jax.jacobian(lambda x : self.trilaterate_chain_lastcv(x,self.signs[0],None).flatten()[self.idx_to_keep])\
               (cvs))[1]
        return value



    def random_rotation_matrix(self, key):
        """
        Generate a uniform random rotation matrix in SO(3) using axis-angle method.
        key: jax.random.PRNGKey
        returns: 3x3 rotation matrix
        """
        k1, k2 = jax.random.split(key, 2)
        
        # Random unit vector (rotation axis)
        u = jax.random.normal(k1, (3,))
        n = u / jnp.linalg.norm(u)
        
        # Random angle
        theta = 2 * jnp.pi * jax.random.uniform(k2)
        
        nx, ny, nz = n
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        
        # Rodrigues' rotation formula
        R = jnp.array([
            [cos_theta + nx*nx*(1-cos_theta),     nx*ny*(1-cos_theta) - nz*sin_theta, nx*nz*(1-cos_theta) + ny*sin_theta],
            [ny*nx*(1-cos_theta) + nz*sin_theta,  cos_theta + ny*ny*(1-cos_theta),    ny*nz*(1-cos_theta) - nx*sin_theta],
            [nz*nx*(1-cos_theta) - ny*sin_theta,  nz*ny*(1-cos_theta) + nx*sin_theta, cos_theta + nz*nz*(1-cos_theta)]
        ])
        
        return R

    def rotate_polymer(self, z, key):
        """
        z_flat: shape (27,) representing 9 beads in 3D
        key: PRNGKey
        """
        R = self.random_rotation_matrix(key)
        
        z_rot = z @ R.T       # apply rotation
        return z_rot
    

    def cvs_from_polymer(self,positions):
        d1 = jnp.linalg.norm(positions[1:] - positions[:-1], axis=1)
        d2 = jnp.linalg.norm(positions[2:] - positions[:-2], axis=1)

        # replace d3 with six long-range distances: 0–3, 0–4, 0–5, 0–6, 0–7, 0–8
        d_long = jnp.array([
            jnp.linalg.norm(positions[0] - positions[3]),
            jnp.linalg.norm(positions[0] - positions[4]),
            jnp.linalg.norm(positions[0] - positions[5]),
            jnp.linalg.norm(positions[0] - positions[6]),
            jnp.linalg.norm(positions[0] - positions[7]),
            jnp.linalg.norm(positions[0] - positions[8]),
        ])

        # concatenate all CVs
        cvs_pred = jnp.concatenate([d1, d2, d_long])
        return cvs_pred
        
        
    def trilaterate_chain_lastcv(self,distances_flat, signs, center_point=0.0):
    # infer n from total length
        L = distances_flat.shape[0]
        n = int((L + 7) // 3)

        # split: d1 (n−1), d2 (n−2), d_long (6)
        idx1 = n - 1
        idx2 = idx1 + (n - 2)
        distances_nn = distances_flat[:idx1]        # 0–1, 1–2, 2–3, ...
        distances_2nn = distances_flat[idx1:idx2]   # 0–2, 1–3, 2–4, ...
        distances_long = distances_flat[idx2:]      # 0–3, 0–4, 0–5, 0–6, 0–7, 0–8

        # initialize positions
        pos = jnp.zeros((n, 3))

        # --- first 3 atoms
        d01 = distances_nn[0]
        pos = pos.at[0].set(jnp.array([0.0, 0.0, 0.0]))
        pos = pos.at[1].set(jnp.array([d01, 0.0, 0.0]))
        d02 = distances_2nn[0]
        d12 = distances_nn[1]
        x2 = (d02**2 - d12**2 + d01**2) / (2.0 * d01)
        y2 = jnp.sqrt(jnp.maximum(d02**2 - x2**2, 0.0))
        pos = pos.at[2].set(jnp.array([x2, y2, 0.0]))

        # --- trilaterate beads 3..8
        def body_fn(carry, inputs):
            pos, k = carry
            sgn = inputs

            # reference points
            P1, P2, P0 = pos[k-1], pos[k-2], pos[0]

            # distances
            r1 = distances_nn[k-1]              # (k-1, k)
            r2 = distances_2nn[k-2]             # (k-2, k)
            r0 = distances_long[k-3]            # (0, k)

            # local frame
            v12 = P2 - P1
            d = jnp.linalg.norm(v12)
            ex = v12 / d

            v10 = P0 - P1
            i = jnp.dot(ex, v10)
            temp = v10 - i * ex
            j_val = jnp.linalg.norm(temp)
            ey = temp / j_val
            ez = jnp.cross(ex, ey)

            # trilateration equations
            x = (r1**2 - r2**2 + d**2) / (2 * d)
            y = (r1**2 - r0**2 + i**2 + j_val**2) / (2 * j_val) - (i / j_val) * x
            z_sq = r1**2 - x**2 - y**2
            z = jnp.sqrt(z_sq) * sgn

            # set new bead
            Pk = P1 + x * ex + y * ey + z * ez
            pos = pos.at[k].set(Pk)

            return (pos, k + 1), None

        (pos_final, _), _ = jax.lax.scan(body_fn, (pos, 3), signs)
        pos = pos_final

        if center_point is not None:
            center_idx = n // 2
            shift = pos[center_idx] - jnp.asarray(center_point)
            pos = pos - shift

        return pos
    def dihedral_signs_from_positions(positions):
        """
        Compute the ±1 signs of the dihedral angles from Cartesian coordinates.
        positions: (n,3) array
        returns: (n-3,) array of ±1 signs
        """
        n = positions.shape[0]
        b1 = positions[1:-2] - positions[:-3]
        b2 = positions[2:-1] - positions[1:-2]
        b3 = positions[3:]   - positions[2:-1]

        n1 = jnp.cross(b1, b2)
        n2 = jnp.cross(b2, b3)

        # Compute the signed torsion term (b1 ⋅ n2)
        torsion_sign = jnp.sum(b1 * n2, axis=-1)

        # Turn into ±1 signs (avoid zero division)
        signs = jnp.sign(torsion_sign)
        # Replace zeros with +1 to have a deterministic choice
        signs = jnp.where(signs == 0, 1.0, signs)

        return signs