from functools import partial
import os
import pickle as pkl
from datetime import datetime
from itertools import product
import argparse
import time
import traceback

os.environ["DDE_BACKEND"] = "jax"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import tqdm

from jax import config
config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

import numpy as np
import jax
import jax.numpy as jnp

try:
    print(f'Jax: CPUs={jax.local_device_count("cpu")} - GPUs={jax.local_device_count("gpu")}')
except:
    pass


from pied.test_cases import prep_damped_oscillator, prep_1d_wave_equation, prep_2d_eikonal_equation, prep_burgers_equation, \
    prep_groundwater, prep_2d_pipe_flow_scalar, prep_cell_population, prep_cooling, prep_chromatography
from pied.ed import RandomMethod, SimulatorEnsembleMethod, GPMutualInformationMethod, \
    PINNEnsembleInverseMethod, PINNFewStepInverseSolverTraining, PINNEnsembleWithMINEMethod, PINNEnsembleWithVBOEDMethod, \
    PINNFewStepInverseSolverTraining, PINNModelTrainingEstimation, PINNTolerableInverseParams
from pied.utils import sample_from_uniform
from pied.models.pinn_ensemble import PINNEnsemble

from test_problems import OSC_PARAMS, WAVE_PARAMS, NS_PARAMS, EIK_PARAMS, GROUNDWATER_PARAMS, CELL_PARAMS, COOLING_PARAMS, CHROMATOGRAPHY_PARAMS, BURGERS_PARAMS


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default='osc')
parser.add_argument('--crits', type=str, nargs='+', default=None)
parser.add_argument('--crit-inc', type=str, default=None)
parser.add_argument('--crit-exc', type=str, default=None)
parser.add_argument('--out-folder', type=str, default=f'{DIR_PATH}/../results-ed_loop')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
print(args)

rand_str = f'round-{args.seed}'
# rand_str = f'{n_exp}_{n_obs}_{args.seed}'
# rand_str = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + str(np.random.randint(100))
FILE_OUT_PREFIX = f'{args.out_folder}/{args.problem}/{rand_str}'
os.makedirs(FILE_OUT_PREFIX, exist_ok=True)

rng = jax.random.PRNGKey(args.seed)
CRITS = args.crits

crit_include = args.crit_inc
crit_exclude = args.crit_exc

"""
SET UP PROBLEM
"""


CRITS_ALL_LIST = [
    'random',
    'fixed-obs',
    
    'mote-None__share-init', 
    'mote-0__share-init', 
    'mote-1000__share-init', 
    
    'mote-None', 
    'mote-0', 
    'mote-1000', 
    
    'fist-1.0_50__share-init',
    'fist-0.5_50__share-init',
    'fist-0.1_50__share-init',
    'fist-0.01_50__share-init',
    'fist-1.0_100__share-init',
    'fist-0.5_100__share-init',
    'fist-0.1_100__share-init',
    'fist-0.01_100__share-init',
    'fist-1.0_200__share-init',
    'fist-0.5_200__share-init',
    'fist-0.1_200__share-init',
    'fist-0.01_200__share-init',
    
    'fist-1.0_50',
    'fist-0.5_50',
    'fist-0.1_50',
    'fist-0.01_50',
    'fist-1.0_100',
    'fist-0.5_100',
    'fist-0.1_100',
    'fist-0.01_100',
    'fist-1.0_200',
    'fist-0.5_200',
    'fist-0.1_200',
    'fist-0.01_200',
    
    'mi',
    'mine',
    'vboed',
]

if args.problem == 'osc':
    
    if CRITS is None:
        CRITS = CRITS_ALL_LIST + ['random-numsim', 'vboed-numsim']
    
    problem_fn = prep_damped_oscillator
    CONFIG_DICT = OSC_PARAMS
    

elif args.problem == 'osc_ood':
    
    if CRITS is None:
        CRITS = [
            'random',
            'fixed-obs',
            'fist-1.0_50__share-init',
            'fist-0.5_50__share-init',
            'fist-0.1_50__share-init',
            'mote-None__share-init', 
            'mote-1000__share-init', 
            'mi',
            'mine',
            'vboed',
        ]
        
    problem_fn = prep_damped_oscillator
    CONFIG_DICT = OSC_PARAMS
    

elif args.problem == 'wave':
    
    if CRITS is None:
        CRITS = CRITS_ALL_LIST
    
    problem_fn = prep_1d_wave_equation
    CONFIG_DICT = WAVE_PARAMS 
    
elif args.problem == 'burgers':
    
    if CRITS is None:
        CRITS = CRITS_ALL_LIST
    
    problem_fn = prep_burgers_equation
    CONFIG_DICT = BURGERS_PARAMS 
    
elif args.problem == 'gw':
    
    if CRITS is None:
        CRITS = CRITS_ALL_LIST
    
    problem_fn = prep_groundwater
    CONFIG_DICT = GROUNDWATER_PARAMS 

elif args.problem == 'cell':
    
    if CRITS is None:
        CRITS = CRITS_ALL_LIST
    
    problem_fn = prep_cell_population
    CONFIG_DICT = CELL_PARAMS     
    
elif args.problem == 'cool':
    
    if CRITS is None:
        CRITS = CRITS_ALL_LIST
    
    problem_fn = prep_cooling
    CONFIG_DICT = COOLING_PARAMS
    
elif args.problem == 'chroma':
    
    if CRITS is None:
        CRITS = CRITS_ALL_LIST + [
            'fist-1.0_20__share-init',
            'fist-0.5_20__share-init',
            'fist-0.1_20__share-init',
        ]
    
    problem_fn = prep_chromatography
    CONFIG_DICT = CHROMATOGRAPHY_PARAMS    
        
elif args.problem == 'eik':
    
    if CRITS is None:
        CRITS = CRITS_ALL_LIST + ['random-numsim', 'vboed-numsim', 'mi-numsim']
    
    problem_fn = prep_2d_eikonal_equation
    CONFIG_DICT = EIK_PARAMS
    
elif args.problem == 'ns':
    
    if CRITS is None:
        CRITS = [c for c in CRITS_ALL_LIST if ('200' not in c) and ('1000' not in c) and ('mine' not in c)]
    
    problem_fn = prep_2d_pipe_flow_scalar
    CONFIG_DICT = NS_PARAMS
            
else:
    raise ValueError(f'Invalid problem - {args.problem}')

problem = problem_fn(args.seed)
if 'ood' in args.problem:
    problem['inv_param_in_domain'] = problem['inv_param_in_domain'] / 2.
    print(f"{problem['inv_param_in_domain']=}")


if crit_include is not None:
    CRITS = [c for c in CRITS if crit_include in c]
    
if crit_exclude is not None:
    CRITS = [c for c in CRITS if crit_exclude not in c]

import random
random.shuffle(CRITS)

print(f'CRITERIA TO RUN =\n{CRITS}')


ed_rounds = CONFIG_DICT['ed_rounds']
fixed_exp_param = CONFIG_DICT['fixed_exp_param']

pinn_ensemble_size = CONFIG_DICT['pinn_ensemble_size']
exp_setup_rounds = CONFIG_DICT['exp_setup_rounds']
obs_setup_rounds = CONFIG_DICT['obs_setup_rounds']
obs_search_time_limit = CONFIG_DICT['obs_search_time_limit']
n_inv = CONFIG_DICT['n_inv']

pinn_init_meta_rounds = CONFIG_DICT['pinn_init_meta_rounds']
pinn_init_meta_steps = CONFIG_DICT['pinn_init_meta_steps']
pinn_meta_eps = CONFIG_DICT['pinn_meta_eps']

noise_std = CONFIG_DICT['noise_std']
ed_criterion = CONFIG_DICT['ed_criterion']
llh_function = CONFIG_DICT['llh_function']
llh_args = CONFIG_DICT['llh_args']

obs_optim_gd_params = CONFIG_DICT['obs_optim_gd_params']
obs_optim_grad_clip = CONFIG_DICT['obs_optim_grad_clip']
obs_optim_grad_jitter = CONFIG_DICT['obs_optim_grad_jitter']
obs_optim_grad_zero_rate = CONFIG_DICT['obs_optim_grad_zero_rate']
obs_optim_use_lbfgs = CONFIG_DICT['obs_optim_use_lbfgs']
lbfgs_gd_params = CONFIG_DICT['lbfgs_gd_params']
if obs_optim_use_lbfgs:
    obs_optim_gd_params = lbfgs_gd_params

mi_pool_size = CONFIG_DICT['mi_pool_size']

mine_nn_args = CONFIG_DICT['mine_nn_args']
mine_train_steps = CONFIG_DICT['mine_train_steps']
mine_train_set_size = CONFIG_DICT['mine_train_set_size']

vbed_clusters = CONFIG_DICT['vbed_clusters']
vbed_optim_args = CONFIG_DICT['vbed_optim_args']
vbed_steps = CONFIG_DICT['vbed_steps']

pde_colloc_sample_num = CONFIG_DICT['pde_colloc_sample_num']
icbc_colloc_sample_num = CONFIG_DICT['icbc_colloc_sample_num']
max_inv_embedding_dim = CONFIG_DICT['max_inv_embedding_dim']
chunk_size = CONFIG_DICT['chunk_size']
pmt_use_single_batch_step_num = CONFIG_DICT['pmt_use_single_batch_step_num']
reg = CONFIG_DICT['reg']

MIN_OBS_ROUND_BO = CONFIG_DICT['min_obs_bo']
MIN_OBS_ROUND_GD = CONFIG_DICT['min_obs_gd']
# gd_reps = CONFIG_DICT['gd_reps']



"""
SET UP PROBLEM
"""
rng, k1_, k2_ = jax.random.split(rng, num=3)
    
rand_seeds = jax.random.randint(k1_, (n_inv,), 0, 1000000)
TRUE_INVS_EMBS = []
ORACLES = []
for i in rand_seeds:
    i = int(i)
    p = problem_fn(seed=i)
    TRUE_INVS_EMBS.append(p['true_inv_embedding'])
    ORACLES.append(p['oracle'])
TRUE_INVS_EMBS = jnp.array(TRUE_INVS_EMBS)
if 'ood' in args.problem:
    print(f'{TRUE_INVS_EMBS=}')
    
# TRUE_INV_PARAM = problem['aux'].get('true_inv_param', None)
# TRUE_INV_EMBEDDING = problem['true_inv_embedding']
# oracle = problem['oracle']


# EXP_PARAMS_LIST = sample_from_uniform(
#     n=ed_rounds, 
#     bounds=problem['exp_in_domain'], 
#     sample_dim=problem['exp_input_dim'], 
#     rng=k2_
# )
EXP_PARAMS_LIST = jnp.array([fixed_exp_param for _ in range(ed_rounds)])

# print()
# if TRUE_INV_EMBEDDING.shape[0] <= 5:
#     print(f'True Inv Param Embedding = {TRUE_INV_EMBEDDING}\n')
if EXP_PARAMS_LIST[0].shape[0] <= 5:
    print('Exp params to use:')
    for x in EXP_PARAMS_LIST: 
        print(x)

PROBLEM_SPECS = {
    # 'true_inv_param': TRUE_INV_PARAM,
    'true_inv_embeddings': np.array(TRUE_INVS_EMBS),
    'exp_params_list': np.array(EXP_PARAMS_LIST),
}


"""
RUN ED LOOP
"""

for crit_method_full in CRITS:
    
    share_init = '__share-init' in crit_method_full
    crit_method = crit_method_full.removesuffix('__share-init')
    
    if os.path.exists(f'{FILE_OUT_PREFIX}/{crit_method_full}.pkl'):
        continue

    print(f'==================== {crit_method_full} ====================')
        
    try:
        
        """
        CONSTRUCT CRITERION FOR EXP
        """
        rng, k_ = jax.random.split(rng)
        seed = int(jax.random.randint(k_, tuple(), 0, 1000000))

        # baseline method - MCMC but enough times that it is accurate
        if ('random' in crit_method) or ('fixed-obs' in crit_method):
            print(f'random method')
            ed = RandomMethod(
                simulator_xs=problem['simulator_xs'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],
                noise_std=noise_std, 
                ensemble_size=pinn_ensemble_size, 
                pinn_share_init=share_init,
                pinn_init_meta_rounds=pinn_init_meta_rounds,
                pinn_init_meta_steps=pinn_init_meta_steps,
                pinn_meta_eps=pinn_meta_eps,
                pinn_ensemble_args=problem['ed_args']['pinn_ensemble_args'],
                ensemble_steps=problem['ed_args']['pinn_ensemble_training_steps'], 
                seed=seed,
            )
            
        elif crit_method == 'mi':
            print(f'mi method')
            ed = GPMutualInformationMethod(
                simulator_xs=problem['simulator_xs'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],
                ensemble_size=pinn_ensemble_size, 
                pinn_ensemble_args=problem['ed_args']['pinn_ensemble_args'],
                ensemble_steps=problem['ed_args']['pinn_ensemble_training_steps'], 
                pinn_share_init=share_init,
                pinn_init_meta_rounds=pinn_init_meta_rounds,
                pinn_init_meta_steps=pinn_init_meta_steps,
                pinn_meta_eps=pinn_meta_eps,
                pool_size=mi_pool_size,
                noise_std=noise_std, 
                acq_fn='ucb',
                exp_setup_rounds=exp_setup_rounds,
                obs_setup_rounds=obs_setup_rounds,
                obs_search_time_limit=obs_search_time_limit,
                min_obs_rounds=MIN_OBS_ROUND_BO,
                seed=seed,
            )
            
        elif crit_method == 'mi-numsim':
            print(f'mi-numsim method')
            ed = GPMutualInformationMethod(
                simulator_xs=problem['simulator_xs'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],
                use_pinns=False,
                ensemble_size=pinn_ensemble_size, 
                pinn_ensemble_args=problem['ed_args']['pinn_ensemble_args'],
                ensemble_steps=problem['ed_args']['pinn_ensemble_training_steps'], 
                pool_size=mi_pool_size,
                noise_std=noise_std, 
                acq_fn='ucb',
                exp_setup_rounds=exp_setup_rounds,
                obs_setup_rounds=obs_setup_rounds,
                obs_search_time_limit=obs_search_time_limit,
                min_obs_rounds=MIN_OBS_ROUND_BO,
                seed=seed,
            )
            
        elif crit_method.startswith('nmc-'):
            
            N, M = crit_method.strip('nmc-').split('_')
            N = int(N)
            M = int(M)
            print(f'nmc method : N={N}, M={M}')
            
            ed = SimulatorEnsembleMethod(
                simulator_xs=problem['simulator_xs'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],     
                ensemble_size=pinn_ensemble_size,
                pinn_ensemble_args=problem['ed_args']['pinn_ensemble_args'],
                ensemble_steps=problem['ed_args']['pinn_ensemble_training_steps'], 
                pinn_share_init=share_init,
                pinn_init_meta_rounds=pinn_init_meta_rounds,
                pinn_init_meta_steps=pinn_init_meta_steps,
                pinn_meta_eps=pinn_meta_eps,
                use_pinns=True,
                ed_criterion=ed_criterion,
                llh_function=llh_function,  
                llh_args=llh_args,
                N=N,
                M=M,
                noise_std=noise_std,
                acq_fn='ucb',
                exp_setup_rounds=exp_setup_rounds,
                obs_setup_rounds=obs_setup_rounds,
                obs_optim_gd_params=obs_optim_gd_params,
                obs_optim_grad_clip=obs_optim_grad_clip,
                obs_search_time_limit=obs_search_time_limit, 
                seed=seed,
            )
                
        elif crit_method == 'mine':
            print(f'mine method : debiased loss')
            ed = PINNEnsembleWithMINEMethod(
                simulator_xs=problem['simulator_xs'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],
                ensemble_size=pinn_ensemble_size, 
                pinn_ensemble_args=problem['ed_args']['pinn_ensemble_args'],
                ensemble_steps=problem['ed_args']['pinn_ensemble_training_steps'], 
                pinn_share_init=share_init,
                pinn_init_meta_rounds=pinn_init_meta_rounds,
                pinn_init_meta_steps=pinn_init_meta_steps,
                pinn_meta_eps=pinn_meta_eps,
                ed_criterion=ed_criterion,
                llh_function=llh_function, 
                llh_args=llh_args,               
                mine_use_pinns=True,
                mine_use_sampled_invs_as_prior=False,
                mine_train_set_size=mine_train_set_size,
                mine_nn_args=mine_nn_args,
                mine_optim_args=dict(learning_rate=1e-3),
                mine_debias_loss_alpha=0.01,
                mine_train_steps=mine_train_steps,
                acq_fn='ucb',
                exp_setup_rounds=exp_setup_rounds,
                obs_setup_rounds=obs_setup_rounds,
                obs_search_time_limit=obs_search_time_limit,
                min_obs_rounds=MIN_OBS_ROUND_BO,
                seed=seed,
            )
            
        elif crit_method == 'vboed':
            print(f'vboed method')
            ed = PINNEnsembleWithVBOEDMethod(
                simulator_xs=problem['simulator_xs'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],
                ensemble_size=pinn_ensemble_size, 
                pinn_ensemble_args=problem['ed_args']['pinn_ensemble_args'],
                ensemble_steps=problem['ed_args']['pinn_ensemble_training_steps'], 
                pinn_share_init=share_init,
                pinn_init_meta_rounds=pinn_init_meta_rounds,
                pinn_init_meta_steps=pinn_init_meta_steps,
                pinn_meta_eps=pinn_meta_eps,
                ed_criterion=ed_criterion,
                llh_function=llh_function,
                llh_args=llh_args,
                vbed_use_pinns=True,
                vbed_method='marg',
                vbed_clusters=vbed_clusters,
                vbed_optim_args=vbed_optim_args,
                vbed_steps=vbed_steps,
                acq_fn='ucb',
                exp_setup_rounds=exp_setup_rounds,
                obs_setup_rounds=obs_setup_rounds,
                obs_search_time_limit=obs_search_time_limit,
                min_obs_rounds=MIN_OBS_ROUND_BO,
                seed=seed,
            )
            
        elif crit_method == 'vboed-numsim':
            print(f'vboed-numsim method')
            ed = PINNEnsembleWithVBOEDMethod(
                simulator_xs=problem['numerical_solver'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],
                ensemble_size=pinn_ensemble_size, 
                ed_criterion=ed_criterion,
                llh_function=llh_function,
                llh_args=llh_args,
                vbed_use_pinns=False,
                vbed_vectorise_simulator=False,
                vbed_method='marg',
                vbed_clusters=vbed_clusters,
                vbed_optim_args=vbed_optim_args,
                vbed_steps=vbed_steps,
                acq_fn='ucb',
                exp_setup_rounds=exp_setup_rounds,
                obs_setup_rounds=obs_setup_rounds,
                obs_search_time_limit=obs_search_time_limit,
                min_obs_rounds=MIN_OBS_ROUND_BO,
                seed=seed,
            )
            
        elif crit_method.startswith('mote-'):
            
            a = crit_method[5:]
            pretraining_steps = None if (a == 'None') else int(a)
            use_forward_params = (pretraining_steps is None)
                
            print(f'mote method : pretraining_steps={pretraining_steps}')
            
            ed = PINNModelTrainingEstimation(
                simulator_xs=problem['simulator_xs'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],
                ensemble_size=pinn_ensemble_size, 
                pinn_ensemble_args=problem['ed_args']['pinn_ensemble_args'],
                ensemble_steps=problem['ed_args']['pinn_ensemble_training_steps'], 
                inverse_ensemble_pretraining_steps=pretraining_steps,
                inverse_ens_use_forward_params=use_forward_params,
                inverse_ens_do_perturb=False,
                pinn_share_init=share_init,
                pinn_init_meta_rounds=pinn_init_meta_rounds,
                pinn_init_meta_steps=pinn_init_meta_steps,
                pinn_meta_eps=pinn_meta_eps,
                pde_colloc_sample_num=pde_colloc_sample_num,
                icbc_colloc_sample_num=icbc_colloc_sample_num,
                reg=reg,
                exp_setup_rounds=exp_setup_rounds,
                obs_setup_rounds=obs_setup_rounds,
                min_obs_rounds=MIN_OBS_ROUND_GD,  # v3 - changed from 3
                obs_optim_use_lbfgs=obs_optim_use_lbfgs,
                obs_optim_gd_params=obs_optim_gd_params,
                obs_optim_grad_clip=obs_optim_grad_clip,
                obs_optim_grad_jitter=obs_optim_grad_jitter,
                obs_optim_grad_zero_rate=obs_optim_grad_zero_rate,
                obs_search_time_limit=obs_search_time_limit,
                seed=seed,
            )
            
        elif crit_method.startswith('fist-'):
            
            a, b = crit_method.removeprefix('fist-').split('_')
            perturb_val = float(a)
            inverse_ensemble_training_steps = int(b)
            
            if pmt_use_single_batch_step_num <= inverse_ensemble_training_steps:
                chunk_size_round = 1
            else:
                chunk_size_round = chunk_size
                
            print(f'fist method : perturb_val={perturb_val}, inverse_ensemble_training_steps={inverse_ensemble_training_steps}, chunk_size_round={chunk_size_round}')
            
            ed = PINNFewStepInverseSolverTraining(
                simulator_xs=problem['simulator_xs'],
                pde=problem['pde'], 
                pde_domain=problem['pde_domain'], 
                exp_design_fn=problem['exp_design_fn'], 
                obs_design_fn=problem['obs_design_fn'],
                inv_embedding=problem['inv_embedding'],
                inv_param_in_domain=problem['inv_param_in_domain'], 
                exp_in_domain=problem['exp_in_domain'], 
                obs_in_domain=problem['obs_in_domain'],
                inv_input_dim=problem['inv_input_dim'], 
                exp_input_dim=problem['exp_input_dim'], 
                obs_input_dim=problem['obs_input_dim'], 
                obs_reading_count=problem['obs_reading_count'],
                x_input_dim=problem['x_input_dim'],
                y_output_dim=problem['y_output_dim'],
                ensemble_size=pinn_ensemble_size, 
                pinn_ensemble_args=problem['ed_args']['pinn_ensemble_args'],
                ensemble_steps=problem['ed_args']['pinn_ensemble_training_steps'], 
                do_fresh_pretraining=False,
                inv_perturb=True,
                net_perturb=True,
                inv_perturb_val=perturb_val,
                inverse_ensemble_pretraining_steps=0,
                inverse_ensemble_training_steps=inverse_ensemble_training_steps,
                pinn_share_init=share_init,
                pinn_init_meta_rounds=pinn_init_meta_rounds,
                pinn_init_meta_steps=pinn_init_meta_steps,
                pinn_meta_eps=pinn_meta_eps,
                pde_colloc_sample_num=pde_colloc_sample_num,
                icbc_colloc_sample_num=icbc_colloc_sample_num,
                obs_optim_with_gd=True,
                do_jit=True,
                exp_setup_rounds=exp_setup_rounds,
                obs_setup_rounds=obs_setup_rounds,
                min_obs_rounds=MIN_OBS_ROUND_GD,  # v3 - changed from 3
                obs_optim_use_lbfgs=obs_optim_use_lbfgs,
                obs_optim_gd_params=obs_optim_gd_params,
                obs_optim_grad_clip=obs_optim_grad_clip,
                obs_optim_grad_jitter=obs_optim_grad_jitter,
                obs_optim_grad_zero_rate=obs_optim_grad_zero_rate,
                obs_search_time_limit=obs_search_time_limit,
                chunk_size=chunk_size_round,
                seed=seed,
            )
        
            
        else:
            print(f'Invalid crit_method - {crit_method_full}')
            exit()


        rounds_data = []

        for ed_round, exp_param in enumerate(EXP_PARAMS_LIST):
            
            t = time.time()
            if crit_method == 'fixed-obs':
                fo = problem['ed_args']['grid_obs_param']
                print(f'Using fixed obs_design = {fo}')
                best_exp, best_obs, ed_aux = ed.run_experiment_round(
                    given_exp_design=exp_param, 
                    given_obs_design=fo,
                )
            else:
                best_exp, best_obs, ed_aux = ed.run_experiment_round(given_exp_design=exp_param)
            ed_running_time = time.time() - t
            
            YS_OBS = []
            for oracle in ORACLES:
                rng, k = jax.random.split(rng)
                ys_obs_single, _ = oracle(best_exp, best_obs, k)
                YS_OBS.append(ys_obs_single)
            YS_OBS = jnp.array(YS_OBS)
            
            t = time.time()
            
            if 'numsim' in crit_method:
                
                import torch
                from gpytorch.mlls import ExactMarginalLogLikelihood
                from botorch.fit import fit_gpytorch_mll
                from botorch.models import SingleTaskGP
                from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, qSimpleRegret
                from botorch.optim.initializers import gen_batch_initial_conditions, initialize_q_batch_nonneg
                from botorch.generation import gen_candidates_torch, get_best_candidates
                from botorch.sampling.stochastic_samplers import StochasticSampler
                
                inv_samples = []
                
                inv_in_domain = problem['inv_param_in_domain']
                
                for inv_case in tqdm.trange(len(YS_OBS), desc='Inverse problems'):
                                        
                    YS_CHECK = YS_OBS[inv_case]
                    
                    inv_transform = lambda inv: (inv - inv_in_domain[:,0]) / (inv_in_domain[:,1] - inv_in_domain[:,0])
                    inv_rev_transform = lambda x: inv_in_domain[:,0] + x * (inv_in_domain[:,1] - inv_in_domain[:,0])
                    hypercube = jnp.array([[0., 1.] for _ in range(inv_in_domain.shape[0])])
                
                    ran_inv_params_untransformed = []
                    ran_inv_params = []
                    ran_inv_scores = []
                    
                    t2 = time.time()
                        
                    for rr in range(200):
                        
                        if rr < 2:
                            
                            rng, k_ = jax.random.split(rng)
                            inv_design_candidate_untransformed = sample_from_uniform(
                                n=1, 
                                bounds=hypercube, 
                                sample_dim=hypercube.shape[0], 
                                rng=k_
                            )[0]
                        
                        else:
                        
                            train_X = torch.tensor(np.array(ran_inv_params_untransformed))
                            train_Y = torch.tensor(np.array(ran_inv_scores).reshape(-1, 1))
                            train_Y = (train_Y - torch.mean(train_Y)) / torch.std(train_Y)
                            
                            model = SingleTaskGP(train_X, train_Y)
                            mll = ExactMarginalLogLikelihood(model.likelihood, model)
                            fit_gpytorch_mll(mll)
                            
                            sampler = StochasticSampler(sample_shape=torch.Size([128]))
                            q_fn = qUpperConfidenceBound(model, beta=1., sampler=sampler)
                            
                            inv_domain_np = np.array(hypercube)
                            Xinit = gen_batch_initial_conditions(q_fn, torch.tensor(inv_domain_np.T), q=1, num_restarts=25, raw_samples=500)
                            batch_candidates, batch_acq_values = gen_candidates_torch(
                                initial_conditions=Xinit,
                                acquisition_function=q_fn,
                                lower_bounds=torch.tensor(inv_domain_np[:,0]),
                                upper_bounds=torch.tensor(inv_domain_np[:,1]),
                            )
                            inv_design_candidate_untransformed = jnp.array(get_best_candidates(batch_candidates, batch_acq_values)[0].cpu().detach().numpy())
                        
                        inv_design_candidate = inv_rev_transform(inv_design_candidate_untransformed)
                        ys_pred = problem['numerical_solver'](best_exp, inv_design_candidate)(best_obs)
                        assert ys_pred.shape == YS_CHECK.shape
                        score = - jnp.log(jnp.linalg.norm(ys_pred - YS_CHECK))
                        
                        ran_inv_params_untransformed.append(inv_design_candidate_untransformed)
                        ran_inv_params.append(inv_design_candidate)
                        ran_inv_scores.append(score)
                        
                        t3 = time.time() - t2
                                    
                    best_obs_i = np.argmax(ran_inv_scores)
                    inv_score_round = ran_inv_scores[best_obs_i]
                    inv_guess_round = ran_inv_params[best_obs_i]
                    
                    inv_samples.append(inv_guess_round)
                    print(f'Inv prob {inv_case} -- perfoemd {rr} rounds | pred = {problem["inv_embedding"](inv_guess_round)}, true = {TRUE_INVS_EMBS[inv_case]} | score = {jnp.exp(-inv_score_round):.5f}, time = {t3:.2f}')
                    
                inv_samples = jnp.array(inv_samples)
            
            else:
                
                # inv_samples, infer_aux = ed.process_observation(ys_obs, n_inv=n_inv)
                
                ensinv = PINNEnsemble(
                    pde=problem['pde'],
                    pde_domain=problem['pde_domain'],
                    exp_design_fn=problem['exp_design_fn'],
                    obs_design_fn=problem['obs_design_fn'],
                    inv_problem=True,
                    **problem['ed_args']['pinn_ensemble_args'],
                )
                obs_design_rep = jnp.repeat(best_obs[None,:], repeats=n_inv, axis=0).reshape(n_inv, -1)
                
                inv_params_guesses = sample_from_uniform(
                    n=n_inv,
                    bounds=problem['inv_param_in_domain'],
                    sample_dim=problem['inv_input_dim'],
                )
                
                if share_init:
                    param = ed._generate_shared_params(n=n_inv, inv=inv_params_guesses)
                    ensinv.prep_simulator(exp_params=best_exp, inv_params_guesses=inv_params_guesses, new_nn_params=param)
                else:
                    ensinv.prep_simulator(exp_params=best_exp, inv_params_guesses=inv_params_guesses)

                for _ in tqdm.trange(problem['ed_args']['pinn_ensemble_training_steps'], mininterval=2):
                    ensinv.step_opt(obs_design=obs_design_rep, observation=YS_OBS)
                    
                inv_samples = ensinv.params['inv']
            
            obs_processing_time = time.time() - t
            
            inv_emb_diff = jnp.linalg.norm(jax.vmap(problem['inv_embedding'])(inv_samples) - TRUE_INVS_EMBS, axis=1)
            mean_log_score = jnp.mean(jnp.log(inv_emb_diff))
            mean_score = jnp.mean(inv_emb_diff)
            median_score = jnp.median(inv_emb_diff)
            
            ed_aux_keys_to_store = {
                'exp_param',
                'best_score',
                'best_obs_param',
                'inv_prior_samples',
                'forward_ensemble_nn_params',
                'obs_param_candidates',
                'obs_param_scores',
                'gd_paths',
                'ran_obs_params',
                'ran_obs_params_untransformed',
                'ran_obs_scores',
                'exp_round_time_elapsed',
            }
            ed_aux_filtered = {k: ed_aux[k] for k in ed_aux.keys() if k in ed_aux_keys_to_store}
            
            rounds_data.append({
                'ed_round': ed_round,
                'best_exp': np.array(best_exp),
                'best_obs': np.array(best_obs),
                'obs': np.array(YS_OBS),
                'inferred_inv_samples': np.array(inv_samples),
                'true_inv_samples': np.array(TRUE_INVS_EMBS),
                'inv_emb_diff': np.array(inv_emb_diff),
                'mean_score': float(mean_score),
                'mean_log_score': float(mean_log_score),
                'median_score': float(median_score),
                # 'ed_aux': ed_aux_filtered,
                # 'pinn_shared_init_evol': ed._pinn_shared_init_params_records,
                # 'nn_params': ensinv.params,
                'timing': {
                    'ed_running': ed_running_time,
                    'obs_processing': obs_processing_time,
                },
            })
            
            if best_obs.shape[0] <= 5:
                print(f'Chosen Obs = {best_obs}')
            print(f'Round {ed_round + 1} mean score = {mean_score}')
            print(f'Round {ed_round + 1} median score = {median_score}')
            print(f'Round {ed_round + 1} score: \n{np.sort(inv_emb_diff)}')
            
        records = {
            'problem_specs': PROBLEM_SPECS,
            'rounds_data': rounds_data,
        }
        
        with open(f'{FILE_OUT_PREFIX}/{crit_method_full}.pkl', 'wb+') as f:
            pkl.dump(records, f)
        
    except Exception:
        print(traceback.format_exc())
