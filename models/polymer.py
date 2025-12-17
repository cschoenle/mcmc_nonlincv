import jax
import jax.numpy as jnp
import equinox as eqx
from .base import BaseEnergyModel, EnergyModelCV, EnergyModelCVSubset
from jax_md import space, quantity,simulate,energy
from jax_md.partition import NeighborFn, NeighborList
from typing import Callable

from jax import Array,jit

from jax_md import energy

from jax import vmap,lax


def build_zigzag_polymer_from_bond_angle(
        box_size, n_beads, bond_length, angle_CCC_deg, key,
        R_min=2.0, R_max=None, transverse_scale=0.3
):
    """
    Builds a polymer chain by first sampling an end-to-end distance,
    then placing beads along that vector with transverse fluctuations.

    Args:
        n_beads: number of monomers (≥ 2)
        bond_length: bond length (approximate, not enforced exactly)
        angle_CCC_deg: kept for API compatibility (not used explicitly)
        key: JAX PRNGKey
        R_min: minimum end-to-end distance
        R_max: maximum end-to-end distance (default = box_size/2)
        transverse_scale: amplitude of transverse noise

    Returns:
        R_polymer: array (n_beads, 3)
    """
    if R_max is None:
        R_max = box_size - 1

    # --- sample end-to-end vector ---
    key, subkey1, subkey2 = random.split(key, 3)
    R = random.uniform(subkey1, (), minval=R_min, maxval=R_max)

    # random direction on sphere
    vec = random.normal(subkey2, (3,))
    vec = vec / jnp.linalg.norm(vec)
    R_vec = R * vec

    # --- place beads along line ---
    t = jnp.linspace(0, 1, n_beads)[:, None]  # [n_beads, 1]
    base_positions = t * R_vec  # linear interpolation

    # --- add transverse fluctuations ---
    # generate orthonormal basis perpendicular to R_vec
    e1 = jnp.array([1.0, 0.0, 0.0])
    if jnp.abs(jnp.dot(e1, vec)) > 0.9:
        e1 = jnp.array([0.0, 1.0, 0.0])
    e1 = e1 - jnp.dot(e1, vec) * vec
    e1 = e1 / jnp.linalg.norm(e1)
    e2 = jnp.cross(vec, e1)

    key, subkey = random.split(key)
    coeffs = transverse_scale * random.normal(subkey, (n_beads, 2))
    transverse = coeffs[:, 0:1] * e1 + coeffs[:, 1:2] * e2

    positions = base_positions + transverse

    # --- recenter in box ---
    com = jnp.mean(positions, axis=0)
    box_center = jnp.array([box_size / 2] * 3)
    R_polymer = (positions - com + box_center) % box_size

    return R_polymer


def initialize_solvent_away_from_polymer(
        n_solvent,
        box_size,
        polymer_pos,
        exclusion_radius,
        key,
        dim=3,
        noise_scale=0.05,
):
    """
    Place solvent particles on a nearly-uniform grid inside a cubic box,
    excluding a spherical zone around each polymer bead to avoid overlaps.

    Args:
        n_solvent: number of solvent particles
        box_size: cubic box length
        polymer_pos: array (n_polymer, dim)
        exclusion_radius: minimum distance from any polymer bead
        key: JAX PRNGKey
        dim: dimensionality (default 3)
        noise_scale: random jitter to break perfect lattice symmetry

    Returns:
        solvent_pos: array (n_solvent, dim)
        key: updated PRNGKey
    """
    # Grid size large enough to host at least n_solvent points
    L = int(jnp.ceil(n_solvent ** (1 / dim)))
    grid = jnp.stack(
        jnp.meshgrid(*[jnp.linspace(0, box_size, L, endpoint=False)] * dim, indexing="ij"),
        axis=-1,
    ).reshape(-1, dim)

    # Shuffle grid so we don’t always pick the same subset
    key, subkey = random.split(key)
    perm = random.permutation(subkey, grid.shape[0])
    grid = grid[perm]

    # Add small random noise to avoid perfect lattice artifacts
    key, subkey = random.split(key)
    noise = noise_scale * random.normal(subkey, grid.shape)
    grid = (grid + noise) % box_size

    # Exclude points too close to polymer
    diff = grid[:, None, :] - polymer_pos[None, :, :]
    diff = (diff + box_size / 2) % box_size - box_size / 2
    dist = jnp.linalg.norm(diff, axis=-1)
    min_dist = jnp.min(dist, axis=1)
    valid = grid[min_dist > exclusion_radius]

    if valid.shape[0] < n_solvent:
        raise RuntimeError(
            f"Only {valid.shape[0]} valid solvent sites found, but {n_solvent} required. "
            "Try reducing exclusion_radius or n_solvent."
        )

    solvent_pos = valid[:n_solvent]
    return solvent_pos, key

class PolymerBase(BaseEnergyModel):
    box_size: jnp.ndarray
    dim: int = eqx.field(static=True)
    n_solvent: int = eqx.field(static=True)
    n_polymer: int = eqx.field(static=True)
    # density: float = eqx.field(static=True)
    # angle_deg: float = eqx.field(static=True)
    bond_length: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    bond_length_1: float = eqx.field(static=True)
    bond_length_2: float = eqx.field(static=True)
    end_to_end_length: float = eqx.field(static=True)
    k_bond_1: float = eqx.field(static=True)
    k_bond_2: float = eqx.field(static=True)
    k_end_to_end: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True) 
    epsilon: float = eqx.field(static=True)
    r_onset: float = eqx.field(static=True)
    r_cutoff: float = eqx.field(static=True)
    first_idx: jnp.ndarray = eqx.field(static=True)
    last_idx: jnp.ndarray = eqx.field(static=True)
    n_pairs: int = eqx.field(static=True)
    # dt: float = eqx.field(static=True)
    kT: float = eqx.field(static=True)
    # gamma: float = eqx.field(static=True)


    neighbor_fn: NeighborFn
    lj_energy_fn: Callable[[Array, NeighborList], Array] = eqx.field(static=True)
    polymer_indices: jnp.ndarray 
    displacement_fn: Callable[[Array, Array], Array] =eqx.field(static=True)

    def __init__(self, config: dict):

        self.dim = config["dim"]
        self.n_solvent = config["n_solvent"]
        self.n_polymer = config["n_polymer"]
        self.alpha=config["alpha"]

        self.bond_length = config["bond_length"]
        self.bond_length_1 = config["bond_length_1"]
        self.bond_length_2 = config["bond_length_2"]
        self.end_to_end_length = config["end_to_end_length"]
        self.k_bond_1 = config["k_bond_1"]
        self.k_bond_2 = config["k_bond_2"]
        self.k_end_to_end = config["k_end_to_end"]
        self.sigma = config["sigma"]
        self.epsilon = config["epsilon"]  
         # Precompute polymer pairs for the CV
        self.n_pairs = self.n_polymer - 3
        self.first_idx = jnp.arange(self.n_pairs)         # first bead of each pair
        self.last_idx  = self.first_idx + 3          # last bead of each pair
        # Precompute constant Jacobian for grad_xi


        if "r_onset" in config:
            self.r_onset = config["r_onset"]
        else:
            self.r_onset = 2.
        if "r_cutoff" in config:
            self.r_cutoff = config["r_cutoff"]
        else:
            self.r_cutoff = 2.5
        self.kT = config["kT"]
        # self.density = config["density"]
        # self.angle_deg = config["angle_deg"]        
        # self.dt = config["dt"]
        # self.gamma = config["gamma"]
        # self.key = config["key"]

        if "box_size" in config:
            self.box_size = config["box_size"]
            if "density" in config:
                print("Warning: box_size and density are both specified. box_size is used.")
        else:
            self.box_size = quantity.box_size_at_number_density(
            particle_count=self.n_solvent,
            number_density=config["density"],
            spatial_dimension=self.dim
            )

        # Species array
        species = jnp.concatenate([jnp.zeros(self.n_solvent, dtype=int), jnp.ones(self.n_polymer, dtype=int)])
        self.displacement_fn, shift_fn = space.periodic(self.box_size)

        # Define energy and neighbor functions
        sigma_matrix = self.sigma * jnp.array([[1.0, 1.0],
                                               [1.0, 0.0]]) 

        epsilon_matrix = self.epsilon * jnp.array([[1.0, 1.0],
                                                   [1.0, 0.0]])          
        self.neighbor_fn, self.lj_energy_fn = energy.lennard_jones_neighbor_list(
            self.displacement_fn,
            self.box_size,
            species=species,
            sigma=sigma_matrix,
            epsilon=epsilon_matrix,
            r_onset=self.r_onset,
            r_cutoff=self.r_cutoff,
            capacity_multiplier=2.0)

        # Store polymer parameters as attributes
        self.polymer_indices = jnp.where(species == 1)[0]
       
    def total_energy_fn(self,positions, neighbor):
        lj_energy = self.lj_energy_fn(positions, neighbor=neighbor)
        bond_en = self.bond_energy(positions)
        return lj_energy + bond_en
    
    

    def biased_energy_fn(self, positions):
        """Biased polymer energy without the end-to-end term."""

        # --- unwrap polymer chain, center on 5th bead ---
        poly_unwrapped = positions[self.polymer_indices]

        # --- bonded terms ---

        # spring i--i+1
        delta1 = poly_unwrapped[1:] - poly_unwrapped[:-1]
        dist1 = jnp.linalg.norm(delta1, axis=-1)
        energy1 = 0.5 * self.k_bond_1 * jnp.sum((dist1 - self.bond_length_1) ** 2)

        # spring i--i+2
        delta2 = poly_unwrapped[2:] - poly_unwrapped[:-2]
        dist2 = jnp.linalg.norm(delta2, axis=-1)
        energy2 = 0.5 * self.k_bond_2 * jnp.sum((dist2 - self.bond_length_2) ** 2)

        # --- LJ interactions for i -> i+3 and beyond (vectorized) ---
        lj_cutoff_fn = energy.multiplicative_isotropic_cutoff(
            energy.lennard_jones, self.r_onset * self.sigma, self.r_cutoff * self.sigma
        )

        # create all i,j pairs with j >= i+3
        i_idx, j_idx = jnp.triu_indices(self.n_polymer, k=3)
        delta = poly_unwrapped[i_idx]- poly_unwrapped[j_idx]
        dist = jnp.linalg.norm(delta, axis=-1)
        lj_energy = jnp.sum(lj_cutoff_fn(dist))

            # --- first-to-last distance  ---
        delta_end = poly_unwrapped[-1]- poly_unwrapped[0]
        end_to_end_dist = jnp.linalg.norm(delta_end)

        return energy1 + energy2 + lj_energy
    
    def total_energy_fn_biased(self,positions, neighbor):
        lj_energy = self.lj_energy_fn(positions, neighbor=neighbor)
        bond_en= self.biased_energy_fn(positions)
        return lj_energy + bond_en
    


    def compute_log_weight(self, positions, nbrs=None):
        """
        Compute log-weight for biased MD, accounting for PBC.

        Args:
            positions: (N, dim) full configuration
            nbrs: ignored, for vmap signature

        Returns:
            log_weight: float
        """
        # Extract polymer beads
        poly_pos = positions[self.polymer_indices]

    
        delta_end =poly_pos[-1]- poly_pos[0]
        dist_end = jnp.linalg.norm(delta_end)
        d_rel = dist_end - self.end_to_end_length

        # Bias potential
        V_end = 0.25 * self.k_end_to_end * (d_rel**4) / (self.alpha**4) \
                - self.k_end_to_end * (d_rel**2) / (self.alpha**2)

        # Log weight = -beta * V_end
        log_weight = - (1.0 / self.kT) * V_end
        return log_weight  
    


    
    def bond_energy(self, positions):
        """
        Bonded polymer energy using unwrapped polymer for chain terms,
        while keeping LJ (i,i+3 and beyond) computed with PBC.
        Also returns the PBC-aware end-to-end distance.
        """
        """Biased polymer energy without the end-to-end term."""

      
        poly_unwrapped = positions[self.polymer_indices]

        # --- bonded terms ---

        # spring i--i+1
        delta1 = poly_unwrapped[1:] - poly_unwrapped[:-1]
        dist1 = jnp.linalg.norm(delta1, axis=-1)
        energy1 = 0.5 * self.k_bond_1 * jnp.sum((dist1 - self.bond_length_1) ** 2)

        # spring i--i+2
        delta2 = poly_unwrapped[2:] - poly_unwrapped[:-2]
        dist2 = jnp.linalg.norm(delta2, axis=-1)
        energy2 = 0.5 * self.k_bond_2 * jnp.sum((dist2 - self.bond_length_2) ** 2)

        # --- LJ interactions for i -> i+3 and beyond (vectorized) ---
        lj_cutoff_fn = energy.multiplicative_isotropic_cutoff(
            energy.lennard_jones, self.r_onset * self.sigma, self.r_cutoff * self.sigma
        )

        # create all i,j pairs with j >= i+3
        i_idx, j_idx = jnp.triu_indices(self.n_polymer, k=3)
        delta = poly_unwrapped[i_idx]- poly_unwrapped[j_idx]
        dist = jnp.linalg.norm(delta, axis=-1)
        lj_energy = jnp.sum(lj_cutoff_fn(dist))

        
        # --- End-to-end vector  ---
        delta_end = poly_unwrapped[-1]- poly_unwrapped[0]
        dist_end = jnp.linalg.norm(delta_end)
        d_rel = dist_end - self.end_to_end_length
        energy_end = 0.25 * self.k_end_to_end * (d_rel**4) / (self.alpha**4) \
                    - self.k_end_to_end * (d_rel**2) / (self.alpha**2)

    

        # --- Total energy ---
        total_energy = energy1 + energy2 + energy_end + lj_energy

        return total_energy
    @jax.jit
    def energy_full(self, x,nbrs):
        return self.total_energy_fn(x,nbrs) / self.kT
    
    @jax.jit
    def force(self, x, nbrs):
        energy_fn = lambda x_: self.energy_full(x_, nbrs)
        return -quantity.canonicalize_force(energy_fn)(x) / self.kT
    
    def update_state(self,x,nbrs):
        nbrs = nbrs.update(x)
        return nbrs
    
    def apply_boundaries(self, x):
        chain_positions = x[self.polymer_indices]
        # central bead index 
        center_idx = self.n_polymer // 2

        # coordinates of the central bead
        shift = chain_positions[center_idx]

        # shift *all* atoms (solvent + polymer)
        new_positions = x - shift+self.box_size/2 # center around L//2
        return new_positions % self.box_size

    def apply_boundaries_subset(self,x):
        return x % self.box_size

    def center(self,state):
            positions = state.position
            new_positions = self.apply_boundaries(positions)
            state = state.set(position=new_positions)
            return state      

    def run_md_ensemble(self, R_init_array, key_array, steps, write_every, dt_init, gamma=1.0, biased=False, save_solvent=True):
        """
        Fully vectorized MD over an ensemble using lax.scan for better GPU throughput.
        Maintains (batch, N, dim) structure.
        """
        n_states, _, dim = R_init_array.shape

        # Select energy function
        energy_fn = self.total_energy_fn_biased if biased else self.total_energy_fn

        elastic_energy_fn = self.biased_energy_fn if biased else self.bond_energy

        # Integrator
        displacement_fn, shift_fn = space.periodic(self.box_size)
        init_fn, apply_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt_init, self.kT, gamma=gamma)

        # Initialize neighbor lists
        nbrs_0 = self.neighbor_fn.allocate(R_init_array[0])
        all_nbrs = vmap(self.neighbor_fn.update, (0, None))(R_init_array, nbrs_0)

        # --- Batch wrappers ---
        batch_init_fn = vmap(lambda R, k, nbr: init_fn(k, R, neighbor=nbr))
        batch_apply_fn = vmap(lambda s, n: apply_fn(s, kT=self.kT, neighbor=n))
        batch_update_nbrs = vmap(lambda s, n: n.update(s.position))
        batch_bond_energy = vmap(lambda s, n: self.energy_full(s.position,n))
        batch_log_weights = vmap(lambda s, n: self.compute_log_weight(s.position, n))
        batch_center = vmap(lambda s: self.center(s))
        batch_positions = vmap(lambda s: s.position)

      

        # batch_xi = vmap(lambda s: self.xi_from_polymer(s.position[self.polymer_indices]))

        # Initialize states
        states = batch_init_fn(R_init_array, key_array, all_nbrs)

        # Preallocate logs (polymer-only)
        n_log = steps // write_every
        if save_solvent:
            log = {
                'H': jnp.zeros((n_states, n_log)),
                'position': jnp.zeros((n_states, n_log, self.n_polymer+self.n_solvent, dim)),
                # 'xi': jnp.zeros((n_states,n_log,) + shape_xi),
            }
        else:
            log = {
                'H': jnp.zeros((n_states, n_log)),
                'position': jnp.zeros((n_states, n_log, self.n_polymer, dim)),
                # 'xi': jnp.zeros((n_states,n_log,) + shape_xi),
            }
        if biased:
            log['log_weights'] = jnp.zeros((n_states, n_log))

        # --- Scan body ---
        def step_fn(carry, i):
            state, nbrs, log = carry
            
            # 1. Update neighbors
            nbrs = batch_update_nbrs(state, nbrs)

            # 2. Integrate one step
            state = batch_apply_fn(state, nbrs)

            # 3. Center wrt central bead of chain

            state =batch_center(state)
            
        
            # 3. Logging
            def write_logs(log):
                step_idx = i // write_every
                
                # Bond energy
                log['H'] = log['H'].at[:, step_idx].set(batch_bond_energy(state, nbrs))
                
                # Polymer positions
                if save_solvent:
                    log['position'] = log['position'].at[:, step_idx].set(state.position)
                else:
                    log['position'] = log['position'].at[:, step_idx].set(state.position[:, self.polymer_indices])
                
                # end to end distance
                

                # Xi values
                # log['xi'] = log['xi'].at[:, step_idx].set(batch_xi(state))
                
                # Log weights if biased
                if biased:
                    log['log_weights'] = log['log_weights'].at[:, step_idx].set(batch_log_weights(state, nbrs))
                
                return log

            log = lax.cond(i % write_every == 0, write_logs, lambda l: l, log)

            return (state, nbrs, log), None

        # Run integration
        (states, all_nbrs, log), _ = lax.scan(step_fn, (states, all_nbrs, log), jnp.arange(steps))

        # Extract final positions and keys
        final_positions = batch_positions(states)
        final_keys = vmap(lambda s: s.rng)(states)

        return final_positions, all_nbrs, log, final_keys


    def initialize_polymer_state(self,key,
                                angle_deg, exclusion_radius, max_attempts):
        """
        Build a single polymer + solvent configuration.
        """
        key, subkey = jax.random.split(key)
        

        R_polymer = build_zigzag_polymer_from_bond_angle(
                self.box_size,
                self.n_polymer,
                self.bond_length,
                angle_deg,
                subkey
            )
        
        R_solvent, key = initialize_solvent_away_from_polymer(
                self.n_solvent,
                self.box_size,
                R_polymer,
                exclusion_radius=exclusion_radius,
                key=subkey,
                dim=self.dim
                
            )
     
       
        R_init = jnp.concatenate([R_solvent, R_polymer], axis=0)
        return R_init, key

    def initialize_polymer_batch(self, key, ensemble_size, angle_deg, exclusion_radius, max_attempts):
        """
        Build a batch of polymer + solvent configurations sequentially.
        """
        R_list = []
        key_list = []

        for _ in range(ensemble_size):
            R_init, key = self.initialize_polymer_state(key, angle_deg, exclusion_radius, max_attempts)
            R_list.append(R_init)
            key_list.append(key)

        # Stack results into arrays of shape (ensemble_size, n_particles, dim)
        R_init_array = jnp.stack(R_list)
        key_array = jnp.stack(key_list)  # shape (ensemble_size, 2) for PRNGKeys

        return R_init_array, key_array      
    
    

class Polymer(PolymerBase, EnergyModelCVSubset):   
    grad_xi_const: jnp.ndarray 
    grad_xi_const_subset: jnp.ndarray   
    def __init__(self, config: dict):
        PolymerBase.__init__(self, config)
        
        self.grad_xi_const, self.grad_xi_const_subset = self.build_grad_xi_and_subset()
        self.subset_indices=self.polymer_indices

    def build_grad_xi_and_subset(self):
        n_particles = self.n_solvent + self.n_polymer
        n_pairs = self.first_idx.size
        dim = self.dim  # usually 3

        # Row indices for Jacobian (one row per component of each pair vector)
        row_idx = jnp.arange(n_pairs * dim).reshape(n_pairs, dim)

        # Column indices for first and last beads
        col_first = self.first_idx[:, None] * dim + jnp.arange(dim)
        col_last  = self.last_idx[:, None] * dim + jnp.arange(dim)

        # Values (-1 for first bead, +1 for last bead)
        vals_first = -jnp.ones_like(col_first, dtype=jnp.float32)
        vals_last  =  jnp.ones_like(col_last,  dtype=jnp.float32)

        # Flatten for scatter
        rows   = jnp.concatenate([row_idx.ravel(), row_idx.ravel()])
        cols   = jnp.concatenate([col_first.ravel(), col_last.ravel()])
        values = jnp.concatenate([vals_first.ravel(), vals_last.ravel()])

        # Build full Jacobian in shape (n_particles*dim, n_pairs*dim)
        gradxi = jnp.zeros((n_particles * dim, n_pairs * dim), dtype=jnp.float32)
        gradxi = gradxi.at[rows, cols].set(values)
        grad_xi_subset = gradxi.reshape(n_particles, dim, n_pairs*dim)[self.polymer_indices].reshape(self.n_polymer*dim, n_pairs*dim)

        return gradxi, grad_xi_subset

    def polymer_from_xi(self, xi, n_polymer, dim, eps=1e-8):
        """
        Reconstruct polymer positions from xi, assuming the middle bead is pinned
        at the center of a cubic box with side self.box_size.

        Args:
            xi : array, shape ((n_pairs + 2) * dim,)
                Flattened deltas in the same order produced by xi_from_polymer.
            n_polymer : int
            dim : int
            eps : float
                Small regularizer for numerical stability.

        Returns:
            polymer_positions : array, shape (n_polymer, dim)
        """
        # deltas = xi.reshape(n_polymer-1, dim)
        # mid = n_polymer // 2
        # # left side (before mid)
        # left = jnp.cumsum(deltas[:mid][::-1], axis=0)[::-1] * -1.0  # reverse cumsum for left side
        # # right side (after mid)
        # right = jnp.cumsum(deltas[mid:], axis=0)
        # # combine with pinned middle bead
        # pinned = jnp.full((1, dim), 0)
        # polymer_positions = jnp.vstack([left, pinned, right])
        # return polymer_positions
        xi_mat = xi.reshape(-1, dim)  # (n_eqs, dim)
        n_eqs = xi_mat.shape[0]

        # Pinned bead = middle one
        pin_idx = n_polymer // 2

        # -------------------------------
        # Build A matrix (n_eqs, n_polymer)
        # -------------------------------
        A = jnp.zeros((n_eqs, n_polymer))

        # Main first/last pairs
        row_idx = jnp.arange(len(self.first_idx))
        A = A.at[row_idx, jnp.array(self.first_idx)].set(-1.0)
        A = A.at[row_idx, jnp.array(self.last_idx)].set(+1.0)

        # Extra constraints: v1 = p4 - p3, v2 = p4 - p5
        
        A = A.at[6, 7].set(-1.0).at[6, 0].set(+1.0)
        A = A.at[7, 8].set(-1.0).at[7, 1].set(+1.0)

        # -------------------------------
        # Solve least squares with pinned bead
        # -------------------------------
        A_reduced = jnp.concatenate([A[:, :pin_idx], A[:, pin_idx+1:]], axis=1)

        AtA = A_reduced.T @ A_reduced
        reg = eps * jnp.eye(AtA.shape[0])
        pinv = jnp.linalg.solve(AtA + reg, A_reduced.T)

        P_reduced = pinv @ xi_mat   # (n_polymer-1, dim)

        # Insert pinned bead at box center
        pinned = jnp.full((1, dim), 0)
        left  = P_reduced[:pin_idx, :]
        right = P_reduced[pin_idx:, :]
        polymer_positions = jnp.vstack([left, pinned, right])+self.box_size/2

        return polymer_positions
    

    def xi_from_polymer(self, polymer_positions):
        """
        Compute xi given only the polymer positions.

        Args:
            polymer_positions : array of shape (n_polymer, dim)
                Positions of polymer beads only.

        Returns:
            xi : array of shape (n_pairs * dim,)
                Flattened bond-vector differences between beads i and i+3.
        """

   
        # # consecutive differences: p[i+1] - p[i]
        # deltas = polymer_positions[1:] - polymer_positions[:-1]  # (n_polymer-1, dim)
        
        # # flatten to 1D
        # return deltas.reshape(-1)  # ((n_polymer-1)*dim,)
        v1=polymer_positions[0]-polymer_positions[7]
        v2=polymer_positions[1]-polymer_positions[8]



        firsts = jnp.take(polymer_positions, self.first_idx, axis=0)  # (n_pairs, dim)
        lasts  = jnp.take(polymer_positions, self.last_idx, axis=0)   # (n_pairs, dim)


        deltas = lasts - firsts                                       # (n_pairs, dim)

        deltas=jnp.append(deltas,v1[None,:],axis=0)
        deltas=jnp.append(deltas,v2[None,:],axis=0)
        return deltas.reshape(-1)                                     # (n_pairs*dim,)

    def xi_subset(self, x_subset):
        return self.xi_from_polymer(x_subset)

    def grad_xi(self, x):
        """Return precomputed grad_xi for a single configuration."""
        return self.grad_xi_const  # shape (n_particles*dim, n_pairs*dim)    
    
    def grad_xi_subset(self, x_subset):
        return self.grad_xi_const_subset
    
