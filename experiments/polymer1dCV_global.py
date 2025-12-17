import sys
import os
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(".."))

import jax
import jax.numpy as jnp
import numpy as np
from models.polymer_xi import PolymerEndToEnd, PolymerEndToEndCVSubsetSampler, PolymerEndToEndLagrangeSolverSubset, get_mode_change_cost_polymer

from cvsampler.bimodal_gaussian import BiModalGaussian1DLowerBounded
from samplers import ZSchedulerLinear
from models.dimer import get_mode_change_cost_dimer
import argparse

def run_simulation_global(working_dir,alpha1,alpha2,K_equiv,n_chains,n_steps,savechains=False, savex=False, checkfile=False):
    print("Running simulation with parameters:")
    print(f"alpha1={alpha1}, alpha2={alpha2}, K_equiv={K_equiv}, n_chains={n_chains}, n_steps={n_steps}")
    # Initialize model, sampler, and scheduler
    with open(os.path.join(working_dir, 'polymer_params.json')) as json_file:
        config_polymer = json.load(json_file)    
    with open(os.path.join(working_dir,'CV1d','propsampler_params.json')) as json_file:
        cv_prop_params = json.load(json_file) 
    sigma_prop = cv_prop_params["sigma_prop"]
    w1_proposal = cv_prop_params["w1_proposal"]
    z1_prop = cv_prop_params["z1"]
    z2_prop = cv_prop_params["z2"]        

    polymer = PolymerEndToEnd(config_polymer)
    polymer_lagrange_solver = PolymerEndToEndLagrangeSolverSubset(polymer)
    cv_sampler = BiModalGaussian1DLowerBounded(z1 = z1_prop, z2 = z2_prop, z_min = jnp.array([0.,]), sigma = sigma_prop, w1_proposal = w1_proposal)
    z_scheduler = ZSchedulerLinear()

    sampler = PolymerEndToEndCVSubsetSampler(polymer, cv_sampler, z_scheduler, polymer_lagrange_solver)



    output_dir = os.path.join(working_dir,'CV1d','global')
    os.makedirs(output_dir, exist_ok=True)

    x0 = jnp.load(os.path.join(working_dir, 'x0.npy')).reshape(polymer.n_solvent + polymer.n_polymer, polymer.dim)
    z0 = polymer.xi(x0)  
    nbr0 = polymer.neighbor_fn.allocate(x0)

    mass = 1.
    dt = np.sqrt(alpha2*mass)
    gamma = 4*alpha1*mass/dt    
    velocity = float(jnp.abs(z2_prop - z1_prop)) / (dt*K_equiv)  
    parameters = {'mass' : mass,
                'gamma' : gamma}
    key = jax.random.PRNGKey(42)
    output_string = f"n{n_steps}nc{n_chains}_a1{alpha1:.4E}a2{alpha2:.4E}K{K_equiv}"
    if checkfile and os.path.isfile(os.path.join(output_dir, 'pacc' + output_string+'.npy')):
        print("File exists, aborting.")
        return None, None, None, None    
    key, (z_traj_list, x_traj_list, pacc_list, n_inter_traj_list, key_arr, success_array) = sampler.get_samples_parallel(x0, z0, 
                                                                                                                            velocity, dt, 
                                                                                                                            n_steps, n_chains, 
                                                                                                                            key,vectordimension=2, 
                                                                                                                            parameters=parameters, 
                                                                                                                            state0=nbr0)    
    print('pacc=',pacc_list.mean())
    error_rate = 1-success_array.mean(axis=-1)
    if jnp.any(jnp.logical_not(success_array)):
        print("Error: Some trajectories did not complete all steps.")
        print("The Error rate is",  error_rate.mean())
    else:
        print("All trajectories completed all steps.")
    np.save(os.path.join(output_dir, 'pacc' + output_string+'.npy'), pacc_list)
    np.save(os.path.join(output_dir, 'errorrate' + output_string+'.npy'), error_rate)
    if savechains:
        np.save(os.path.join(output_dir, 'z_traj_' + output_string+'.npy'), z_traj_list)
        np.save(os.path.join(output_dir, 'n_inter_traj_' + output_string+'.npy'), n_inter_traj_list)
    if savex:
        np.save(os.path.join(output_dir, 'x_traj_' + output_string+'.npy'), x_traj_list)

    transition_point = (z1_prop + z2_prop) / 2.
    transition_point = jnp.asarray(transition_point)
    if transition_point.shape == (1,):
        transition_point = transition_point.squeeze()    
    cost_mean, cost_std = get_mode_change_cost_polymer(x_traj_list[:,:,polymer.polymer_indices], n_inter_traj_list, transition = transition_point)
    np.save(os.path.join(output_dir, 'modeswitchcost_' + output_string+'.npy'), 
            np.array([cost_mean, cost_std]))
    
    return z_traj_list, success_array, pacc_list, cost_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--working_dir", help="working directory")
    parser.add_argument("-a1", "--alpha1", help="alpha1 parameter", type=float)
    parser.add_argument("-a2", "--alpha2", help="alpha2 parameter", type=float)
    parser.add_argument("-K", "--K_equiv", help="no of steps for a unit length in CV space", type=int)
    parser.add_argument("-nc", "--n_chains", help="no of chains", type=int)
    parser.add_argument("-n", "--n_steps", help="no of chains", type=int)
    parser.add_argument("--savechains", help = "save chains", action = "store_true", default = False)
    parser.add_argument("--savex", help = "save x", action = "store_true", default = False)
    parser.add_argument("--checkfile", help = "check if output file already exists", action = "store_true", default = False)

    args = parser.parse_args()
    working_dir = args.working_dir
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    K_equiv = args.K_equiv
    n_chains = int(args.n_chains)
    n_steps = int(args.n_steps)
    savechains = args.savechains
    savex = args.savex
    checkfile = args.checkfile
    z, success_arr,pacc_list, cost_mean = run_simulation_global(working_dir, 
                                                                alpha1, alpha2, 
                                                                K_equiv, n_chains=n_chains, n_steps=n_steps, 
                                                                savechains=savechains, savex=savex, checkfile=checkfile)