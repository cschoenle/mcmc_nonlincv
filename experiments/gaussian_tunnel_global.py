import sys
import os
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(".."))

import jax
import jax.numpy as jnp
import numpy as np
from models import GaussianTunnel
from cvsampler import BiModalGaussian1D
from samplers import ZSchedulerLinear, UnderdampedSubsetCV
import argparse
from models.gaussian_tunnel import get_mode_change_cost_gaussian_tunnel

def run_simulation_global(working_dir,alpha1,alpha2,K_equiv,n_chains,n_steps, savechains=False, savex=False, checkfile=False):
    print("Running simulation with parameters:")
    print(f"alpha1={alpha1}, alpha2={alpha2}, K_equiv={K_equiv}, n_chains={n_chains}, n_steps={n_steps}")
    # Initialize model, sampler, and scheduler
    with open(os.path.join(working_dir, 'gaussian_params.json')) as json_file:
        gaussian_params = json.load(json_file)
    dimx = gaussian_params['dimx']
    d = gaussian_params['d']
    with open(os.path.join(working_dir, 'propsampler_params.json')) as json_file:
        cv_prop_params = json.load(json_file)
    energymodel = GaussianTunnel(**gaussian_params)
    cv_sampler = BiModalGaussian1D(**cv_prop_params)
    z_scheduler = ZSchedulerLinear()

    z0 = jnp.array([0., ])
    x0val = energymodel.mux(z0)[0]
    x0 = jnp.array(dimx * [x0val, ])

    sampler = UnderdampedSubsetCV(energymodel, cv_sampler, z_scheduler)

    output_string = f"n{n_steps}nc{n_chains}_a1{alpha1:.4E}a2{alpha2:.4E}K{K_equiv}"

    output_dir = os.path.join(working_dir,'global')
    os.makedirs(output_dir, exist_ok=True)

    if checkfile and os.path.isfile(os.path.join(output_dir, 'pacc' + output_string+'.npy')):
        try:
            pacc = np.load(os.path.join(output_dir, 'pacc' + output_string+'.npy'))
            print("File exists, aborting.")
            return None, None, None, None
        except:
            pass

    mass = 1.
    dt = np.sqrt(alpha2*mass)
    gamma = 4*alpha1*mass/dt    
    velocity = jnp.array(d/(dt*K_equiv))
    parameters = {'mass' : mass,
                'gamma' : gamma}       
    key = jax.random.PRNGKey(42)
    key, (z_traj_list, x_traj_list, pacc_list, n_inter_traj_list, key_arr, _) = sampler.get_samples_parallel(x0, z0,
                                                                                                             velocity, dt,
                                                                                                             n_steps, n_chains,
                                                                                                             key,vectordimension=1,
                                                                                                             parameters=parameters)
    print('pacc=',pacc_list.mean())
    np.save(os.path.join(output_dir, 'pacc' + output_string+'.npy'), pacc_list)
    if savechains:
        np.save(os.path.join(output_dir, 'z_traj_' + output_string+'.npy'), z_traj_list)
        np.save(os.path.join(output_dir, 'n_inter_traj_' + output_string+'.npy'), n_inter_traj_list)
    if savex:
        np.save(os.path.join(output_dir, 'x_traj_' + output_string+'.npy'), x_traj_list)

    cost_mean, cost_std = get_mode_change_cost_gaussian_tunnel(z_traj_list, n_inter_traj_list, d/2)
    print("mean cost", cost_mean)
    np.save(os.path.join(output_dir, 'modeswitchcost_' + output_string+'.npy'), 
            np.array([cost_mean, cost_std]))
    
    return z_traj_list, _, pacc_list, cost_mean


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
    z, _, pacc_list, cost_mean = run_simulation_global(working_dir, 
                                                                alpha1, alpha2, 
                                                                K_equiv, n_chains=n_chains, n_steps=n_steps, 
                                                                savechains=savechains, savex=savex, checkfile=checkfile)