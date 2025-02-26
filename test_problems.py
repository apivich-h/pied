OSC_PARAMS = dict(
    
    n_exp = 10,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [1., 0.],
    
    pinn_ensemble_size = 20,
    exp_setup_rounds = 1,
    obs_setup_rounds = 100,
    obs_search_time_limit = 0,
    n_inv = 50,
    
    pinn_init_meta_rounds = 20,
    pinn_init_meta_steps = 2000,
    pinn_meta_eps = 0.1,
    
    noise_std = 1e-3,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = False,
    lbfgs_gd_params = dict(maxiter=500),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=500, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = None,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 100,
    icbc_colloc_sample_num = 1,
    max_inv_embedding_dim = 2,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 10000,
    reg = 1e-5,
    
    min_obs_bo = 1000,
    min_obs_gd = 5,
    gd_reps = 1,
   
)


WAVE_PARAMS = dict(
    
    n_exp = 10,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [0.],
    
    pinn_ensemble_size = 5,
    exp_setup_rounds = 1,
    obs_setup_rounds = 500,
    obs_search_time_limit = 0,
    n_inv = 10,
    
    pinn_init_meta_rounds = 20,
    pinn_init_meta_steps = 5000,
    pinn_meta_eps = 0.1,
    
    noise_std = 1e-2,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = False,
    lbfgs_gd_params = dict(maxiter=500),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=500, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = None,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 1000,
    icbc_colloc_sample_num = 300,
    max_inv_embedding_dim = 100,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 10000,
    reg = 1e-3,
    
    min_obs_bo = 1000,
    min_obs_gd = 3,
    gd_reps = 1,
   
)


GROUNDWATER_PARAMS = dict(
    
    n_exp = 10,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [0.],
    
    pinn_ensemble_size = 10,
    exp_setup_rounds = 1,
    obs_setup_rounds = 200,
    obs_search_time_limit = 0,
    n_inv = 10,
    
    pinn_init_meta_rounds = 20,
    pinn_init_meta_steps = 2000,
    pinn_meta_eps = 0.1,
    
    noise_std = 1e-2,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = False,
    lbfgs_gd_params = dict(maxiter=500),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=500, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = 0.5,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 200,
    icbc_colloc_sample_num = 1,
    max_inv_embedding_dim = 1,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 10000,
    reg = 1e-5,
    
    min_obs_bo = 1000,
    min_obs_gd = 5,
    gd_reps = 1,
   
)


COOLING_PARAMS = dict(
    
    n_exp = 10,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [0.],
    
    pinn_ensemble_size = 10,
    exp_setup_rounds = 1,
    obs_setup_rounds = 200,
    obs_search_time_limit = 0,
    n_inv = 10,
    
    pinn_init_meta_rounds = 20,
    pinn_init_meta_steps = 2000,
    pinn_meta_eps = 0.1,
    
    noise_std = 0.5,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = False,
    lbfgs_gd_params = dict(maxiter=500),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=500, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = 0.2,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 200,
    icbc_colloc_sample_num = 1,
    max_inv_embedding_dim = 1,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 10000,
    reg = 1e-3,
    
    min_obs_bo = 1000,
    min_obs_gd = 5,
    gd_reps = 1,
   
)


CHROMATOGRAPHY_PARAMS = dict(
    
    n_exp = 10,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [0.],
    
    pinn_ensemble_size = 10,
    exp_setup_rounds = 1,
    obs_setup_rounds = 100,
    obs_search_time_limit = 0,
    n_inv = 12,
    
    pinn_init_meta_rounds = 20,
    pinn_init_meta_steps = 1000,
    pinn_meta_eps = 0.1,
    
    noise_std = 1e-3,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = True,
    lbfgs_gd_params = dict(maxiter=500),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=500, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = 0.,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 500,
    icbc_colloc_sample_num = 100,
    max_inv_embedding_dim = 2,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 1000,
    reg = 1e-3,
    
    min_obs_bo = 1000,
    min_obs_gd = 3,
    gd_reps = 1,
   
)


CELL_PARAMS = dict(
    
    n_exp = 1,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [0.],
    
    pinn_ensemble_size = 10,
    exp_setup_rounds = 1,
    obs_setup_rounds = 200,  # 100,
    obs_search_time_limit = 0,  # 3 * 60,
    n_inv = 8,
    
    pinn_init_meta_rounds = 20,
    pinn_init_meta_steps = 2000,
    pinn_meta_eps = 0.1,
    
    noise_std = 1e-4,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = False,
    lbfgs_gd_params = dict(maxiter=500),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=500, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = 0.5,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 200,
    icbc_colloc_sample_num = 100,
    max_inv_embedding_dim = 2,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 10000,
    reg = 1e-5,
    
    min_obs_bo = 200,
    min_obs_gd = 5,
    gd_reps = 5,
   
)


BURGERS_PARAMS = dict(
    
    n_exp = 10,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [0.],
    
    pinn_ensemble_size = 10,
    exp_setup_rounds = 1,
    obs_setup_rounds = 200,
    obs_search_time_limit = 0,
    n_inv = 10,
    
    pinn_init_meta_rounds = 20,
    pinn_init_meta_steps = 2000,
    pinn_meta_eps = 0.1,
    
    noise_std = 1e-2,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = False,
    lbfgs_gd_params = dict(maxiter=500),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=500, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = 0.2,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 500,
    icbc_colloc_sample_num = 200,
    max_inv_embedding_dim = 4,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 1000,
    reg = 1e-3,
    
    min_obs_bo = 200,
    min_obs_gd = 3,
    gd_reps = 1,
   
)


EIK_PARAMS = dict(
    
    n_exp = 10,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [1.5, 1.5],
    
    pinn_ensemble_size = 10,
    exp_setup_rounds = 1,
    obs_setup_rounds = 1000,
    obs_search_time_limit = 0,
    n_inv = 50,
    
    pinn_init_meta_rounds = 20,
    pinn_init_meta_steps = 2000,
    pinn_meta_eps = 0.1,
    
    noise_std = 1e-2,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = False,
    lbfgs_gd_params = dict(maxiter=1000),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=1000, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = 0.2,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 1000,
    icbc_colloc_sample_num = 1,
    max_inv_embedding_dim = 100,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 10000,
    reg = 1e-3,
    
    min_obs_bo = 1000,
    min_obs_gd = 3,
    gd_reps = 1,
   
)


NS_PARAMS = dict(
    
    n_exp = 10,
    n_obs = 100,
    
    ed_rounds = 1,
    fixed_exp_param = [0.],
    
    pinn_ensemble_size = 5,
    exp_setup_rounds = 1,
    obs_setup_rounds = 500,
    obs_search_time_limit = 60 * 60,
    n_inv = 10,
    
    pinn_init_meta_rounds = 10,
    pinn_init_meta_steps = 1000,
    pinn_meta_eps = 0.1,
    
    noise_std = 1e-2,
    ed_criterion = 'eig',
    llh_function = 'nllh',
    llh_args = dict(noise_std=0.1),
    
    obs_optim_use_lbfgs = False,
    lbfgs_gd_params = dict(maxiter=1000),
    obs_optim_gd_params = dict(stepsize=0.01, maxiter=1000, acceleration=True),
    obs_optim_grad_clip = 1.,
    obs_optim_grad_jitter = None,
    obs_optim_grad_zero_rate = 0.2,
    
    mi_pool_size = 100,
    
    mine_nn_args = dict(hidden_layers=2, hidden_dim=256, activation='elu'),
    mine_train_steps = 2000,
    mine_train_set_size = 128,
    
    vbed_clusters = 20,
    vbed_optim_args = dict(learning_rate=0.01),
    vbed_steps = 500,
    
    pde_colloc_sample_num = 100,
    icbc_colloc_sample_num = 50,
    max_inv_embedding_dim = 100,
    chunk_size = 10,
    pmt_use_single_batch_step_num = 10000,
    reg = 1e-3,
    
    min_obs_bo = 100,
    min_obs_gd = 1,
    gd_reps = 3,
   
)