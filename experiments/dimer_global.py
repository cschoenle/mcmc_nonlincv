import sys
import os
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(".."))

import jax
import jax.numpy as jnp
import numpy as np
from models import DimerModel, DimerLagrangeSolver, DimerUnderdampedCV
from cvsampler import BiModalGaussian1DLowerUpperBounded
from samplers import ZSchedulerLinear
from models.dimer import get_mode_change_cost_dimer
import argparse

def run_simulation_global(working_dir,alpha1,alpha2,K_equiv,n_chains,n_steps,savechains=False, savex=False, checkfile=False):
    print("Running simulation with parameters:")
    z0 = 0.
    print(f"z0={z0}, alpha1={alpha1}, alpha2={alpha2}, K_equiv={K_equiv}, n_chains={n_chains}, n_steps={n_steps}")
    # Initialize model, sampler, and scheduler
    with open(os.path.join(working_dir, 'dimer_params.json')) as json_file:
        dimer_params = json.load(json_file)    
    with open(os.path.join(working_dir, 'propsampler_params.json')) as json_file:
        cv_prop_params = json.load(json_file) 
    sigma_prop = cv_prop_params["sigma_prop"]
    w1_proposal = cv_prop_params["w1_proposal"]
    z1_prop = cv_prop_params["z1"]
    z2_prop = cv_prop_params["z2"]        

    energymodel = DimerModel(**dimer_params)
    dimer_lagrange_solver = DimerLagrangeSolver(energymodel) # analytical solution for the dimer model
    # general_lagrange_solver = GeneralLagrangeSolver(energymodel) # general numerical solver for Lagrange multipliers
    cv_sampler = BiModalGaussian1DLowerUpperBounded(z1_prop, z2_prop, jnp.array([energymodel.zmin,]), jnp.array([energymodel.zmax,]), sigma_prop, w1_proposal)
    z_scheduler = ZSchedulerLinear()

    sampler = DimerUnderdampedCV(energymodel, cv_sampler, z_scheduler, dimer_lagrange_solver)

    L = energymodel.L
    n = dimer_params["n"]
    r0 = energymodel.r0
    w = energymodel.w

    x0 = np.load(os.path.join(working_dir, 'x0.npy'))
    x0 = jnp.array(x0)
    x0 = energymodel.apply_boundaries(x0)


    output_string = f"z0{z0:.1f}n{n_steps}nc{n_chains}_a1{alpha1:.4E}a2{alpha2:.4E}K{K_equiv}"

    output_dir = os.path.join(working_dir,'global')
    os.makedirs(output_dir, exist_ok=True)

    if checkfile and os.path.isfile(os.path.join(output_dir, 'pacc' + output_string+'.npy')):
        print("File exists, aborting.")
        return None, None, None, None

    z0 = energymodel.xi(x0)    
    # gamma = 2 * alpha1
    # dt = alpha2 / 2.
    # mass = dt / 2.

    mass = 1.
    dt = np.sqrt(alpha2*mass)
    gamma = 4*alpha1*mass/dt    
    velocity = 1./(dt*K_equiv)  
    parameters = {'mass' : mass,
                'gamma' : gamma}       
    key = jax.random.PRNGKey(42)
    key, (z_traj_list, x_traj_list, pacc_list, n_inter_traj_list, key_arr, success_array) = sampler.get_samples_parallel(x0, z0, 
                                                                                                                            velocity, dt, 
                                                                                                                            n_steps, n_chains, 
                                                                                                                            key,vectordimension=2, 
                                                                                                                            parameters=parameters, 
                                                                                                                            state0=None)    
    print('pacc=',pacc_list.mean())
    error_rate = 1-success_array.mean(axis=-1)
    if jnp.any(jnp.logical_not(success_array)):
        print("Error: Some trajectories did not complete all steps (constraint was not satisfied).")
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

    cost_mean, cost_std = get_mode_change_cost_dimer(z_traj_list, n_inter_traj_list)
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