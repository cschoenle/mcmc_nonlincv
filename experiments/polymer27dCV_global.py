import sys
import os
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(".."))

import jax
import jax.numpy as jnp
import numpy as np
from models.polymer_xi import get_mode_change_cost_polymer, PolymerSeparated

from samplers import ZSchedulerLinear
from samplers.underdamped import UnderdampedSubsetCV
from cvsampler.normalizing_flow import FlowSampler, autoregressive_flow_with_actnorm
import argparse
import equinox as eqx
from flowjax.distributions import StandardNormal
from cvsampler import NormalizingFlowPolymer
import itertools

def run_simulation_global(working_dir,alpha1,alpha2,K_equiv,n_chains,n_steps,savechains=False, savex=False, checkfile=False,dt_mala=None,n_local_steps=0):
    print("Running simulation with parameters:")
    print(f"alpha1={alpha1}, alpha2={alpha2}, K_equiv={K_equiv}, n_chains={n_chains}, n_steps={n_steps}")
    # Initialize model, sampler, and scheduler
    with open(os.path.join(working_dir, 'polymer_params.json')) as json_file:
        config_polymer = json.load(json_file)    
    # specified here to determine the velocity and later the mode switches of the trajectory
    z1_prop = 3.529472
    z2_prop = 5.651341      
    
    polymer = PolymerSeparated(config_polymer)
    key, subkey = jax.random.split(jax.random.key(0))

    base_dist=StandardNormal((21,))
    flow = autoregressive_flow_with_actnorm(
        subkey,
        base_dist=base_dist,
        knots=32,
        x_init=jnp.ones((100,21)),
        interval=4,
        flow_layers=20
    )

    key, subkey = jax.random.split(key)
    foldername_flow = os.path.join(working_dir, 'CV27d', "train_flow")

    flow = eqx.tree_deserialise_leaves(os.path.join(foldername_flow, 'Trained_model.eqx'), flow)

    values = [+1, -1]
    signs_all = jnp.array(list(itertools.product(values, repeat=6)))
    cv_sampler=NormalizingFlowPolymer(flow, signs_all,polymer.box_size)

    z_scheduler_lin=ZSchedulerLinear()

    sampler = UnderdampedSubsetCV(EnergyModel=polymer,CVSampler=cv_sampler,z_scheduler=z_scheduler_lin)



    output_dir = os.path.join(working_dir,'CV27d','global')
    os.makedirs(output_dir, exist_ok=True)

    x0 = jnp.load(os.path.join(working_dir, 'x0.npy')).reshape(polymer.n_solvent + polymer.n_polymer, polymer.dim)
    z0 = x0[polymer.polymer_indices].flatten()
    x0_solvent=x0[:polymer.n_solvent]
    nbr0 = polymer.neighbor_fn.allocate(x0)

    mass = 1.
    dt = np.sqrt(alpha2*mass)
    if dt_mala is None:
        dt_mala = dt    
    gamma = 4*alpha1*mass/dt    
    velocity = float(jnp.abs(z2_prop - z1_prop)) / (dt*K_equiv)  
    parameters = {'mass' : mass,
                'gamma' : gamma,
                'n_local_steps' : n_local_steps,
                'dt_mala' : dt_mala}       
    key = jax.random.key(42)
    if n_local_steps > 0:
        output_string = f"n{n_steps}nc{n_chains}_a1{alpha1:.4E}a2{alpha2:.4E}K{K_equiv}_nlocal{n_local_steps}_dtmala{dt_mala:.3E}"
    else:
        output_string = f"n{n_steps}nc{n_chains}_a1{alpha1:.4E}a2{alpha2:.4E}K{K_equiv}"
    if checkfile and os.path.isfile(os.path.join(output_dir, 'pacc' + output_string+'.npy')):
        print("File exists, aborting.")
        return None, None, None, None    
    key, (z_traj_list, x_traj_list, pacc_list, n_inter_traj_list, key_arr, success_array) = sampler.get_samples_parallel(x0_solvent, z0, 
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
    cost_mean, cost_std = get_mode_change_cost_polymer(z_traj_list.reshape(z_traj_list.shape[:-1]+(9,3)), n_inter_traj_list, transition = transition_point)
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
    parser.add_argument("--dt_mala", help="time step for MALA local moves", type=float, default=None)
    parser.add_argument("--n_local_steps", help="number of local MALA steps per global step", type=int, default=0)    

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
    dt_mala = args.dt_mala
    n_local_steps = args.n_local_steps    
    z, success_arr,pacc_list, cost_mean = run_simulation_global(working_dir, 
                                                                alpha1, alpha2, 
                                                                K_equiv, n_chains=n_chains, n_steps=n_steps, 
                                                                savechains=savechains, savex=savex, checkfile=checkfile,
                                                                dt_mala=dt_mala, n_local_steps=n_local_steps)
