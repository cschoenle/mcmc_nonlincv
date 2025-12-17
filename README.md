# Monte-Carlo sampling with non-local CV updates
 [**Link to the Paper**](https://arxiv.org/???)
 
This is the repository with code connected to the following paper:

C.Schönle, D. Carbone, M. Gabrié, T. Lelièvre, G. Stoltz. 
"Efficient Monte-Carlo sampling of metastable systems using non-local collective variable updates", 
arXiv:2512.??? (2025)

## Please Note
(December 2025) At present, the latest version of `jax_md` can only be installed and run with Python 3.12 due to compatibility issues with `jaxlib`. 
Within our package, however, this only concerns the polymer example. 

## Code structure
The code is structured around the different ingredients of the algorithm.
### Sampling algorithms
Different versions of the sampling algorithm are implemented in `samplers`. The overarching structure is implemented in `base.by`

For the simplest case where the CV is a represented by a subset of the degrees of freedom:
- `asym.py`: The asymmetric sampler with overdamped dynamics considered in [**arXiv:2405.18160**](https://arxiv.org/abs/2405.18160)
- `underdamped.py`: The more general sampler with underdamped Langevin dynamics

For the general case of a non-linear CV, the samplers as well as a general solver for the Lagrange multipliers can be found in:
- `nonlinCV.py`
- `nonlineCV_subset.py` (when the CV is a non-linear function of a subset of degrees of freedom)

For completeness, implementation of a simple MALA sampler in `mala.py`

Several choices of schedule for the constrained dynamics are implemented in `z_schedule.py`

### CV Samplers
One important ingredient of the algorithm is the proposal sampler in CV space, several choices can be found in `cvsampler`.

### Model systems
In the paper, we consider four different model system, and they are implemented in `models`. 
In `dimer.py`, we also have implemented the analytical solution of the Lagrange multiplier for the position constraint.

### Experiments and Notebooks
We provide scripts in `experiments` that demonstrate how to use the code to generate MCMC trajectories. Example prompts are:
```aiignore
python3 -m experiments.gaussian_tunnel_global -dir experiments/data/gaussian_tunnel -a1 0. -a2 0.67 -K 100 -nc 7 -n 100
python3 -m experiments.phi4_global -dir experiments/data/phi4 -a1 0. -a2 1.4e-3 -K 1300 -nc 7 -n 100
python3 -m experiments.dimer_global -dir experiments/data/dimer -a1 0. -a2 4.5e-4 -K 400 -nc 7 -n 17
python3 -m experiments.polymer1dCV_global -dir experiments/data/polymer -a1 0. -a2 2e-4 -K 400 -nc 3 -n 5
python3 -m experiments.polymer27dCV_global -dir experiments/data/polymer -a1 0. -a2 5e-4 -K 400 -nc 3 -n 5
```
specifying in each case the parameters $\alpha_1$ and $\alpha_2$, 
the number of steps $K$ for a certain reference distance in CV space, as well as the number of chains $n_c$ 
and the number of overall iterations $n$. 

Within `Notebooks`: 
- In `01_Gaussian_Tunnel.ipynb`, we apply two different version of the algorithm on the Gaussian tunnel.
- In `02_initialize_dimer.ipynb`, we provide a script to initialize the dimer.
