from functools import partial
import os
import sys
import pickle as pkl
from collections.abc import MutableMapping
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import tqdm
from scipy.integrate import odeint

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.interpolate import RegularGridInterpolator
# from jax.experimental.ode import odeint

import flax
from flax import linen as nn
import optax
# import gpjax as gpx
import diffrax as dfx
import pykonal

import torch
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement, qNoisyExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

import pied.deepxde as dde
from .utils import sample_from_uniform
from .utils.jax_utils import flatten
from .utils.interp2d import GridInterpolator
from .icbc_patch import generate_periodic_bc, generate_velocity_bc, generate_fixed_function_bc, generate_fixed_points_bc, generate_arbitrary_bc
from .models.model_loader import construct_net


CURR_DIR = os.path.dirname(os.path.realpath(__file__))


"""
DAMPED OSCILLATOR EXAMPLE
"""
def prep_damped_oscillator(seed=0):
    
    t_max = 20.
    M = 1.
    N_obs = 3  #5

    pde_domain = dde.geometry.TimeDomain(0, t_max)
    
    
    def pde(x, y, const, exp_design):
        mu, k = const
        dy_t = dde.grad.jacobian(y, x, j=0)[0]
        dy_tt = dde.grad.hessian(y, x, i=0, j=0)[0]
        return (M * dy_tt + mu * dy_t + k * y[0],)
    
    xs_ic = jnp.array([[0.]])
    
    def inital_pos(params, net_apply, exp, inv, xs):
        ys_pred = net_apply(params, xs)
        return (ys_pred - exp[0]).reshape(-1)
    
    def inital_vel(params, net_apply, exp, inv, xs):
        ys_pred = jax.vmap(jax.jacobian(lambda x_: net_apply(params, x_)))(xs)
        return (ys_pred - exp[1]).reshape(-1)
    
    exp_design_fn = [
        (inital_pos, xs_ic),
        (inital_vel, xs_ic),
    ]

    def reading(obs_param):
        return obs_param.reshape(N_obs, 1)
        # return jnp.linspace(obs_param[0], obs_param[1], num=N_obs, endpoint=True).reshape(-1, 1)

    def obs_design_fn(f, obs_param):
        return f(reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 1.], [-1., 1.]])  # x0, v0
    # obs_in_domain = jnp.array([[0., t_max], [0., t_max]])
    obs_in_domain = jnp.array([[0., t_max] for _ in range(N_obs)])

    inv_param_in_domain = jnp.array([[0., 4.], [0., 4.]])  # mu, k
    inv_param_prior_log_pdf = lambda b: jnp.array(0.)
    
    # true_inv_param = jnp.array([1.4, 3.])
    true_inv_param = sample_from_uniform(
        n=1, 
        bounds=inv_param_in_domain, 
        sample_dim=2, 
        rng=jax.random.PRNGKey(seed)
    )[0]
    
    inv_embedding = lambda inv: inv
    # compare_true_inv_fn = lambda inv: jnp.linalg.norm(inv - true_inv_param)
    
    # =================================================
    
    noise_std = 1e-3
    
    def _crit_case(t, omega0, gamma, x0, v0):
        w = - gamma
        A = x0
        B = v0 + gamma * A
        return (A + B * t) * jnp.exp(w * t)
    
    def _noncrit_case(t, omega0, gamma, x0, v0):
        omega0 = omega0 + 0j
        gamma = gamma + 0j
        w1 = - gamma + 1j * ((omega0 ** 2 - gamma ** 2) ** 0.5)
        w2 = - gamma - 1j * ((omega0 ** 2 - gamma ** 2) ** 0.5)
        A = (v0 - x0 * w2) / (w1 - w2)
        B = x0 - A
        return (A * jnp.exp(w1 * t) + B * jnp.exp(w2 * t)).real
    
    def oscillation_solution(t, omega0, gamma, x0, v0):
        return jax.lax.cond(omega0 == gamma, _crit_case, _noncrit_case, t, omega0, gamma, x0, v0)
    
    def closed_form_soln(exp_design, xs, inv):
        x0, v0 = exp_design
        mu, k = inv
        omega0 = jnp.sqrt(k / M)
        gamma = mu / (2 * M)
        return oscillation_solution(xs, omega0, gamma, x0, v0)
    
    def generate_closed_form_fn(exp_design, inv, rng=jax.random.PRNGKey(42)):
        
        def _fn(xs):
            ys = closed_form_soln(exp_design, xs, inv)
            return ys
        
        return _fn
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        # xs = obs_design_fn(obs_design)
        f = generate_closed_form_fn(exp_design, true_inv_param, rng=rng)
        ys = obs_design_fn(f, obs_design)
        noise = noise_std * jax.random.normal(rng, shape=ys.shape)
        return ys + noise, dict()
    
    def numerical_solver(exp_design, inv, rng=jax.random.PRNGKey(42)):
        
        from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
        
        A = jnp.array([[0., 1.], [-inv[1], -inv[0]]])

        def eqn(t, y, _):
            return A @ y

        sol = diffeqsolve(
            terms=ODETerm(eqn), 
            solver=Dopri5(), 
            t0=0, t1=10, dt0=0.01, y0=exp_design, 
            saveat=SaveAt(dense=True),
        )

        def pred_fn(x): 
            y = jax.vmap(sol.interpolation.evaluate)(x.reshape(-1))[:,0]
            return y.reshape(x.shape)
        
        return pred_fn
    
    # =================================================
        
    def plot_function(func, **kwargs):
        ts = np.linspace(0., t_max, 501).reshape(-1, 1)
        plt.plot(ts, func(ts), **kwargs)
        
    def plot_solution(exp_param, obs_param=None, beta=None, rng=jax.random.PRNGKey(0), **kwargs):
        fn = generate_closed_form_fn(exp_design=exp_param, inv=(true_inv_param if beta is None else beta), rng=rng)
        plot_function(func=fn, **kwargs)
        if obs_param is not None:
            xs = reading(obs_param)
            ys = obs_design_fn(fn, obs_param)
            plt.plot(xs.reshape(-1), ys.reshape(-1), 'ok')
            
    def plot_criterion_landscape(crit_func):
        xi = np.linspace(0., t_max, 51)
        yi = np.linspace(0., t_max, 51)
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        pts = jnp.array([xs, ys]).T
        
        xs, ys, zs = [], [], []
        for p in tqdm.tqdm(pts):
            x, y = p
            if x <= y:
                z = crit_func(p)
                xs.append(x)
                ys.append(y)
                zs.append(z)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        
        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = plt.contourf(xi, yi, zi, levels=50, cmap="RdBu_r", alpha=0.7, antialiased=True)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        plt.colorbar()
            
    # =================================================
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': generate_closed_form_fn,
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': 2, 
        'exp_input_dim': 2, 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_obs,
        'x_input_dim': 1, 
        'y_output_dim': 1,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': true_inv_param,
        'xs_reading': reading,
        'numerical_solver': numerical_solver,
        'helper_fns': {
            'closed_form_soln': closed_form_soln, 
            'plot_criterion_landscape': plot_criterion_landscape,
            'plot_some_mc_samples': generate_closed_form_fn,
            'plot_function': plot_function,
            'plot_solution': plot_solution,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'hidden_layers': 6, 
                    'hidden_dim': 8,
                },
                'n_pde_collocation_pts': 300,
                'n_icbc_collocation_pts': 1,
                'optim_method': 'adam',
                'optim_args': {
                    'learning_rate': 0.01,
                }
            },
            'pinn_ensemble_training_steps': 30000,
            'grid_obs_param': jnp.array([t_max / 3., 2 * t_max / 3, t_max]),
        },
        'aux': {
            't_max': t_max,
            'true_inv_param': true_inv_param,
        },
    }
    

"""
EIKONAL EQUATION EXAMPLE
- 2D field example
- Inverse parameter is to deduce a function - for solving purpose, only need value from discrete grid
- Only two exp design params (for each coordinates) and 8 obs design params (for 4 sensors)
"""
def prep_2d_eikonal_equation(seed=0):
        
    # sensor_granularity = 10
    L = 5.
    # Lgrid_count = 11
    pde_domain = dde.geometry.Rectangle(xmin=[0., 0.], xmax=[L, L])
        
    invnet_out_transform = lambda x, y: jnp.abs(y) + 0.2
    inv_net = construct_net(input_dim=2, hidden_layers=1, hidden_dim=16, output_dim=1, activation='sin', output_transform=invnet_out_transform)[0]
    arr = jnp.ones(shape=(1, 2))
    inv_net_mock_params = inv_net.init(jax.random.PRNGKey(0), arr)
    inv_net_mock_params = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=arr.dtype), inv_net_mock_params)
    pflat, unflatten_fn = flatten(inv_net_mock_params)
    
    def pde(x, y, const, exp_design):
        dT_dx = dde.grad.jacobian(y, x, i=0, j=0)[0]
        dT_dy = dde.grad.jacobian(y, x, i=0, j=1)[0]
        T_mag = dT_dx**2 + dT_dy**2
        v_mag = inv_net.apply(unflatten_fn(const), x) ** 2
        assert T_mag.shape == v_mag.shape, (T_mag.shape, v_mag.shape)
        return (T_mag * v_mag - 1.,)
    
    exp_design_fn = []
    
    def exp_design_out_transform(exp_design):
    
        @jax.jit
        def out_transform(x, y):
            dx1 = x[..., 0:1] - exp_design[0]
            dx2 = x[..., 1:2] - exp_design[1]
            dist = jnp.maximum(dx1**2 + dx2**2, 1e-12)
            return (y[..., 0:1] ** 2) * jnp.sqrt(dist)
        
        return out_transform

    # def obs_design_fn(obs_param):
    #     x1_space = jnp.linspace(obs_param[0], obs_param[1], num=sensor_granularity)
    #     x2_space = jnp.linspace(obs_param[2], obs_param[3], num=sensor_granularity)
    #     arrs = jnp.array(jnp.meshgrid(x1_space, x2_space)).reshape(2, sensor_granularity**2).T
    #     mid = arrs[12]
    #     dx = arrs - mid[None, :]
    #     theta = obs_param[4] * jnp.pi
    #     rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta),  jnp.cos(theta)]])
    #     return jnp.clip((rot @ dx.T).T + mid[None,:], 0, L)


    # exp_in_domain = jnp.array([[0., L], [0., L]])
    # obs_in_domain = jnp.array([[0., L] for _ in range(4)] + [[0., 1.]])
    
    N_readings = 30
    
    def xs_reading(obs_param):
        return obs_param.reshape(N_readings, 2)
        # x1_space = jnp.linspace(obs_param[0], obs_param[1], num=sensor_granularity)
        # x2_space = jnp.linspace(obs_param[2], obs_param[3], num=sensor_granularity)
        # return jnp.array(jnp.meshgrid(x1_space, x2_space)).reshape(2, sensor_granularity**2).T
    
    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., L], [0., L]])
    obs_in_domain = jnp.array([[0., L] for _ in range(2 * N_readings)])

    inv_range = 0.5
    inv_param_in_domain = jnp.array([[-inv_range, inv_range] for _ in range(pflat.shape[0])])
    
    # =================================================
    
    npts_emb = 201    
    axis_coord = jnp.linspace(0., L, num=npts_emb, endpoint=True)
    pt_sample_grid_emb = jnp.concatenate(jnp.meshgrid(axis_coord, axis_coord)).reshape(2, -1).T[:,::-1]
    
    inverse_fn = lambda const: partial(inv_net.apply, unflatten_fn(const))
    
    @jax.jit
    def inv_embedding(inv):
        return inverse_fn(inv)(pt_sample_grid_emb).reshape(-1)
    
    # # true_inv_function = lambda xs: 0.5 + 0.3 * xs[:,1] + 0.1 * (xs[:,0] - 3.)**2 - 0.01 * (xs[:,0] - 2.) * (xs[:,1] - 1.)**2
    
    # coeff1 = jax.random.normal(key=jax.random.PRNGKey(seed), shape=(12,))
    # coeff2 = jax.random.uniform(key=jax.random.PRNGKey(seed+1), shape=(6,))
    # A1 = 0.1 * coeff1[0:4].reshape(2, 2)
    # A2 = 0.3 * coeff1[4:8].reshape(2, 2)
    # A3 = 0.7 * coeff1[8:12].reshape(2, 2)
    # b1 = 5 * coeff2[0:2]
    # b2 = 5 * coeff2[2:4]
    # b3 = 5 * coeff2[4:6]
    # # A1 = 0.1 * coeff1[0:4].reshape(2, 2)
    # # A2 = 0.3 * coeff1[4:8].reshape(2, 2)
    # # A3 = 0.5 * coeff1[8:12].reshape(2, 2)
    # # b1 = 5 * coeff2[0:2]
    # # b2 = 5 * coeff2[2:4]
    # # b3 = 5 * coeff2[4:6]
    
    # def true_inv_function(xs):
    #     n1 = jnp.linalg.norm((xs - b1[None,:]) @ (A1 @ A1.T), axis=1) ** 2
    #     n2 = jnp.linalg.norm((xs - b2[None,:]) @ (A2 @ A2.T), axis=1) ** 2
    #     n3 = jnp.linalg.norm((xs - b3[None,:]) @ (A3 @ A3.T), axis=1) ** 2
    #     # return jnp.exp(-n1) + jnp.exp(-n2) + jnp.exp(-n3)
    #     return jnp.exp(-n1) + jnp.exp(-n2) + jnp.exp(-n3) + 0.1
    
    true_inv_val = jax.random.uniform(
        key=jax.random.PRNGKey(seed),
        shape=pflat.shape,
        minval=-inv_range,
        maxval=inv_range,
    )
    true_inv_function = lambda x: inv_net.apply(unflatten_fn(true_inv_val), x)
    
    true_inv_embedding = true_inv_function(pt_sample_grid_emb).reshape(-1)
    assert true_inv_embedding.shape == inv_embedding(inv_param_in_domain[:,0]).shape
    
    # @jax.jit
    # def compare_inv_fn(inv1, inv2):
    #     f1 = inverse_fn(inv1)(pt_sample_grid_emb)
    #     f2 = inverse_fn(inv2)(pt_sample_grid_emb)
    #     return jnp.linalg.norm(f1 - f2) / pt_sample_grid_emb.shape[0]
    
    # def compare_true_inv_fn(inv):
    #     f1 = inverse_fn(inv)(pt_sample_grid_emb)
    #     ftrue = true_inv_function(pt_sample_grid_emb)
    #     return jnp.linalg.norm(f1 - ftrue) / pt_sample_grid_emb.shape[0]
    
    # =================================================
    
    npts_eiksim = 501
    node_intervals_eiksim = L / (npts_eiksim - 1)
    axis_coord = jnp.linspace(0., L, num=npts_eiksim, endpoint=True)
    pt_sample_grid = jnp.concatenate(jnp.meshgrid(axis_coord, axis_coord)).reshape(2, -1).T[:,::-1]
    solution_interp = GridInterpolator(lower_bound=0., upper_bound=L, grid_size=npts_eiksim, dims=2)
    
    noise_std = 1e-3
    
    def eikonal_solver(vel_field_fn, source_x, source_y):
        
        # Initialize the solver.
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = 0, 0, 0
        solver.velocity.node_intervals = 1., node_intervals_eiksim, node_intervals_eiksim
        solver.velocity.npts = 1, npts_eiksim, npts_eiksim
        
        solver.velocity.values = np.array(vel_field_fn(pt_sample_grid).reshape(solver.velocity.npts))

        # Initialize the source.
        src_idx = 0, int((source_x + 0.5 * node_intervals_eiksim) // node_intervals_eiksim), int((source_y + 0.5 * node_intervals_eiksim) // node_intervals_eiksim)
        solver.traveltime.values[src_idx] = 0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)

        # Solve the system.
        solver.solve()
        
        traveltime = jnp.array(solver.traveltime.values[0].T).flatten()
        return lambda x: solution_interp(traveltime, x)
    
    def noisy_simulator_xs(exp_design, inv, rng=jax.random.PRNGKey(42)):
        
        inv_field = lambda x: inv_net.apply(unflatten_fn(inv), x)
        soln_fn = eikonal_solver(inv_field, float(exp_design[0]), float(exp_design[1]))
        
        def _fn(xs):
            return jax.vmap(soln_fn)(xs)
        
        return _fn
    
    def oracle_function(exp_design, rng=jax.random.PRNGKey(42)):
        soln_fn = eikonal_solver(true_inv_function, float(exp_design[0]), float(exp_design[1]))
        return jax.vmap(soln_fn)
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        soln_fn = eikonal_solver(true_inv_function, float(exp_design[0]), float(exp_design[1]))
        f = jax.vmap(soln_fn)
        ys = obs_design_fn(f, obs_design)
        return jnp.round(ys, decimals=3), dict(soln_fn=soln_fn)

    def numerical_solver(exp_design, inv, rng=jax.random.PRNGKey(42)):
                
        inv_field = lambda x: inv_net.apply(unflatten_fn(inv), x)
        soln_fn = eikonal_solver(inv_field, float(exp_design[0]), float(exp_design[1]))
        
        def _fn(obs):
            xs = xs_reading(obs)
            return jax.vmap(soln_fn)(xs)
                
        return _fn
        
    # =================================================
        
    def plot_function(func, ax=None, exp_param=None, obs_param=None, res=51, levels=50, cmap="RdBu_r", cbar=False, **contour_kwargs):
        if ax is None:
            ax = plt.gca()
        
        xi = np.linspace(0., L, res)
        yi = np.linspace(0., L, res)
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        pts = jnp.array([xs, ys]).T
        zs = np.array(func(pts)).reshape(-1)

        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, alpha=0.7, antialiased=True, **contour_kwargs)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        ax.axis('scaled')
        # if cbar:
        #     plt.colorbar(cnt, ax=ax)
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            ax.plot(xs[:,0], xs[:,1], 'xk')
            
        if exp_param is not None:
            ax.plot([exp_param[0]], [exp_param[1]], 'or')
            
        return cnt
            
    def plot_grid(zs, exp_param=None, obs_param=None, ax=None, res=51, levels=50, cmap="RdBu_r", cbar=False, **contour_kwargs):
        # if ax is None:
        #     ax = plt.gca()
        
        xi = np.linspace(0., L, res)
        yi = np.linspace(0., L, res)
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        pts = jnp.array([xs, ys]).T
        zs = zs.reshape(-1)

        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, alpha=0.7, antialiased=True, **contour_kwargs)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        ax.axis('scaled')
        # if cbar:
        #     plt.colorbar(cnt, ax=ax)
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            ax.plot(xs[:,0], xs[:,1], 'xk')
            
        if exp_param is not None:
            ax.plot([exp_param[0]], [exp_param[1]], 'or')
            
        return cnt
            
    a, b = 5, 6
    assert a * b == N_readings
    fixed_grid_obs = jnp.concatenate(jnp.meshgrid(
        jnp.linspace(0., L, num=a, endpoint=True), 
        jnp.linspace(0., L, num=b, endpoint=True)
    )).reshape(2, -1).T.reshape(-1)
                
    # =================================================
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': noisy_simulator_xs,
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': pflat.shape[0],
        'exp_input_dim': 2, 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_readings,  # sensor_granularity**2,
        'x_input_dim': 2, 
        'y_output_dim': 1,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': true_inv_embedding,
        'xs_reading': xs_reading,
        'numerical_solver': numerical_solver,
        'helper_fns': {
            'eikonal_solver': eikonal_solver,
            'plot_function': plot_function, 
            'plot_grid': plot_grid,
            'inverse_fn': inverse_fn,
            'oracle_function': oracle_function,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'hidden_layers': 6, 
                    'hidden_dim': 8,
                },
                'output_transform_generator_fn': exp_design_out_transform,
                'n_pde_collocation_pts': 10000,
                'n_icbc_collocation_pts': 1,
                'optim_method': 'adam',
                'optim_args': {
                    'learning_rate': 0.001,
                }
            },
            'pinn_ensemble_training_steps': 50000,
            # 'grid_obs_param': jnp.array(pde_domain.random_points(N_readings, 'Hammersley').reshape(-1)),
            'grid_obs_param': fixed_grid_obs,
        },
        'aux': {
            'true_inv_function': true_inv_function,
            'pt_sample_grid': pt_sample_grid,
        },
    }
    
   
def prep_1d_wave_equation(seed=0):
    
    minval = 0.5
    maxval = 2.
    TRUE_CS = jax.random.uniform(key=jax.random.PRNGKey(seed), shape=(2,), minval=minval, maxval=maxval)
    C_1 = float(TRUE_CS[0])
    C_2 = float(TRUE_CS[1])
    
    AMP = 1.
    INIT_X = 1.
    WIDTH = 0.1
    
    #Spatial mesh - i indices
    L_x = 6. #Range of the domain according to x [m]
    dx = 0.01 #Infinitesimal distance
    N_x = int(L_x/dx) #Points number of the spatial mesh
    X = np.linspace(0,L_x,N_x+1) #Spatial array



    #Temporal mesh with CFL < 1 - j indices
    L_t = 6 #Duration of simulation [s]
    dt = 0.01*dx  #Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
    N_t = int(L_t/dt) #Points number of the temporal mesh
    T = np.linspace(0,L_t,N_t+1) #Temporal array

    #Def of the initial condition   
    def I(x):
        """
        two space variables depending function 
        that represent the wave form at t = 0
        """
        return AMP * np.exp(-(x-INIT_X)**2/WIDTH)
    
    def get_U(c1, c2):

        #Def of velocity (spatial scalar field)
        def celer(x):
            return c1 + ((c2 - c1) * (x > 4.))
        
        loop_exec = 1  # Processing loop execution flag
        left_bound_cond = 1  #Boundary cond 1 : Dirichlet, 2 : Neumann, 3 Mur
        right_bound_cond = 1  #Boundary cond 1 : Dirichlet, 2 : Neumann, 3 Mur
        
        #Velocity array for calculation (finite elements)
        c = np.zeros(N_x+1, float)
        for i in range(0,N_x+1):
            c[i] = celer(X[i])




        ############## CALCULATION CONSTANTS ###############
        c_1 = c[0]
        c_2 = c[N_x]

        C2 = (dt/dx)**2

        CFL_1 = c_1*(dt/dx)
        CFL_2 = c_2*(dt/dx)




        ############## PROCESSING LOOP ###############

        if loop_exec:
            # $\forall i \in {0,...,N_x}$
            u_jm1 = np.zeros(N_x+1,float)   #Vector array u_i^{j-1}
            u_j = np.zeros(N_x+1,float)     #Vector array u_i^j
            u_jp1 = np.zeros(N_x+1,float)   #Vector array u_i^{j+1}
            
            q = np.zeros(N_x+1,float)
            q[0:N_x+1] = c[0:N_x+1]**2
            
            U = np.zeros((N_x+1,N_t+1),float) #Global solution
            
            #init cond - at t = 0
            u_j[0:N_x+1] = I(X[0:N_x+1])
            U[:,0] = u_j.copy()
            
            
            #init cond - at t = 1
            #without boundary cond
            u_jp1[1:N_x] =  u_j[1:N_x] + 0.5*C2*( 0.5*(q[1:N_x] + q[2:N_x+1])*(u_j[2:N_x+1] - u_j[1:N_x]) - 0.5*(q[0:N_x-1] + q[1:N_x])*(u_j[1:N_x] - u_j[0:N_x-1]))
            
            
            #left boundary conditions
            if left_bound_cond == 1:
                #Dirichlet bound cond
                u_jp1[0] = 0
                
            elif left_bound_cond == 2:
                #Nuemann bound cond
                #i = 0
                u_jp1[0] = u_j[0] + 0.5*C2*( 0.5*(q[0] + q[0+1])*(u_j[0+1] - u_j[0]) - 0.5*(q[0] + q[0+1])*(u_j[0] - u_j[0+1]))

            elif left_bound_cond == 3:
                #Mur bound cond
                #i = 0
                u_jp1[0] = u_j[1] + (CFL_1 -1)/(CFL_1 + 1)*( u_jp1[1] - u_j[0])

            
            
            #right boundary conditions
            if right_bound_cond == 1:
                #Dirichlet bound cond
                u_jp1[N_x] = 0
                
                
            elif right_bound_cond == 2:
                #Nuemann bound cond
                #i = N_x
                u_jp1[N_x] =  u_j[N_x] + 0.5*C2*( 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x-1] - u_j[N_x]) - 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x] - u_j[i-1]))
                
                
            elif right_bound_cond == 3:
                #Mur bound cond
                #i = N_x
                u_jp1[N_x] = u_j[N_x-1] + (CFL_2 -1)/(CFL_2 + 1)*(u_jp1[N_x-1] - u_j[N_x])
            
            u_jm1 = u_j.copy()  #go to the next step
            u_j = u_jp1.copy()  #go to the next step
            U[:,1] = u_j.copy()
            
            
            #Process loop (on time mesh)
            for j in range(1, N_t):
                #calculation at step j+1
                #without boundary cond
                u_jp1[1:N_x] = -u_jm1[1:N_x] + 2*u_j[1:N_x] + C2*( 0.5*(q[1:N_x] + q[2:N_x+1])*(u_j[2:N_x+1] - u_j[1:N_x]) - 0.5*(q[0:N_x-1] + q[1:N_x])*(u_j[1:N_x] - u_j[0:N_x-1]))
                
                
                #left bound conditions
                if left_bound_cond == 1:
                    #Dirichlet bound cond
                    u_jp1[0] = 0

                elif left_bound_cond == 2:
                    #Nuemann bound cond
                    #i = 0
                    u_jp1[0] = -u_jm1[0] + 2*u_j[0] + C2*( 0.5*(q[0] + q[0+1])*(u_j[0+1] - u_j[0]) - 0.5*(q[0] + q[0+1])*(u_j[0] - u_j[0+1]))       
                    
                elif left_bound_cond == 3:
                    #Mur bound cond
                    #i = 0
                    u_jp1[0] = u_j[1] + (CFL_1 -1)/(CFL_1 + 1)*( u_jp1[1] - u_j[0])



                #right bound conditions
                if right_bound_cond == 1:
                    #Dirichlet bound cond
                    u_jp1[N_x] = 0
                    
                elif right_bound_cond == 2:
                    #Nuemann bound cond
                    #i = N_x
                    u_jp1[N_x] = -u_jm1[N_x] + 2*u_j[N_x] + C2*( 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x-1] - u_j[N_x]) - 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x] - u_j[N_x-1]))
                    
                elif right_bound_cond == 3:
                    #Mur bound cond
                    #i = N_x
                    u_jp1[N_x] = u_j[N_x-1] + (CFL_2 -1)/(CFL_2 + 1)*(u_jp1[N_x-1] - u_j[N_x])

            
                
                u_jm1[:] = u_j.copy()   #go to the next step
                u_j[:] = u_jp1.copy()   #go to the next step
                U[:,j] = u_j.copy()
            
        return U
    
    U = get_U(C_1, C_2)

    # softclip from https://github.com/yonesuke/softclip/blob/main/src/softclip/softclip.py
    hinge_softness = 1.
    # softclip_forward = lambda x: minval + (maxval - minval) * nn.sigmoid(x / hinge_softness)
    # softclip_inverse = lambda y: hinge_softness * jnp.log(y - minval) - hinge_softness * jnp.log(maxval - y)
    softclip_forward = lambda x: minval + jnp.abs(x - minval)
    
    #Def of the initial condition   
    def ic_fn_jax(x, exp, inv):
        """
        two space variables depending function 
        that represent the wave form at t = 0
        """
        return AMP * jnp.exp(-((x[...,0:1] - INIT_X)**2/WIDTH))
    
    def ic_no_vel(x, exp, inv):
        return x

    def c_fn_jax(x, const):
        c1, c2 = const
        return c1 + ((c2 - c1) * (x[...,0:1] > 4.))
    
    def pde(x, y, const, exp_design):
        # const = jnp.clip(const, a_min=minval, a_max=maxval)
        const = softclip_forward(const)
        c2U_fn = lambda x_: c_fn_jax(x_, const)**2 * y[1](x_)
        c2U = c2U_fn(x)
        dc2U_dxx = dde.grad.hessian((c2U, c2U_fn), x, i=0, j=0)[0]
        dU_dtt = dde.grad.hessian(y, x, i=1, j=1)[0]
        return (dc2U_dxx - dU_dtt,)

    geom = dde.geometry.Interval(0, L_x)
    timedomain = dde.geometry.TimeDomain(0, L_t)
    pde_domain = dde.geometry.GeometryXTime(geom, timedomain)
    
    # def exp_design_fn(exp_param):
    #     return [
    #         dde.icbc.boundary_conditions.PeriodicBC(pde_domain, 0, boundary_r),
    #         dde.icbc.IC(
    #             pde_domain, 
    #             lambda x_: ic_fn(x_[:,0], exp_param).reshape(-1, 1), lambda _, 
    #             on_initial: on_initial
    #         ),
    #     ]
    
    xs_ic = jnp.array(pde_domain.random_initial_points(10000))
    xs_bc = jnp.array(pde_domain.random_boundary_points(10000))
    
    exp_design_fn = [
        (generate_fixed_function_bc(boundary_func=ic_fn_jax), xs_ic),
        (generate_velocity_bc(boundary_func=lambda x, exp, inv: 0., i=0, j=1), xs_ic),
        (generate_fixed_function_bc(boundary_func=lambda x, exp, inv: 0.), xs_bc),
    ]

    N_readings = 3
    TIMESTEPS = T[::2000]
    timesteps_arr = jnp.array(TIMESTEPS)
    N_timesteps = TIMESTEPS.shape[0]
    
    @jax.jit
    def xs_reading(obs_param):
        locs = obs_param.reshape(N_readings, 1)
        return jnp.concatenate([jnp.repeat(locs, N_timesteps, axis=0), jnp.tile(timesteps_arr, reps=N_readings)[:,None]], axis=1)

    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 0.]])
    obs_in_domain = jnp.tile(jnp.array([[0., L_x]]), reps=(N_readings, 1))

    inv_param_in_domain = jnp.array([[minval, maxval] for _ in range(2)])
    
    @jax.jit
    def inv_embedding(inv):
        return softclip_forward(inv)
    
    true_inv_param = jnp.array([C_1, C_2])
    
    # =================================================
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        xs = xs_reading(obs_design)
        idxs = (xs // jnp.array([dx, dt])[None,:]).astype(int)
        r = jnp.array([U[i1, i2] for (i1, i2) in idxs])
        return jnp.round(r.reshape(-1), decimals=6), dict()
        
    # =================================================
        
    def plot_function(func, exp_param=None, obs_param=None):
        xi = np.linspace(0., L_x, 101)
        yi = np.linspace(0., L_t, 101)
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        pts = jnp.array([xs, ys]).T
        zs = np.array(func(pts)).reshape(-1)

        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = plt.contourf(xi, yi, zi, levels=50, cmap="RdBu_r", alpha=0.7, antialiased=True)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        plt.colorbar()
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            plt.plot(xs[:,0], xs[:,1], 'xk')
        
    def plot_solution(obs_param=None, cvals=None, rng=jax.random.PRNGKey(0)):
        
        c1, c2 = (true_inv_param if cvals is None else cvals)
        
        xi = X
        yi = T[::400]
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        zs = np.array(get_U(c1, c2)[:,::400].T).reshape(-1)

        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = plt.contourf(xi, yi, zi, levels=50, cmap="RdBu_r", alpha=0.7, antialiased=True)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        plt.colorbar()
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            plt.plot(xs[:,0], xs[:,1], 'xk')
            
    # =================================================
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': None,
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': inv_param_in_domain.shape[0], 
        'exp_input_dim': exp_in_domain.shape[0], 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_readings * TIMESTEPS.shape[0],
        'x_input_dim': 2, 
        'y_output_dim': 1,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': true_inv_param,
        'xs_reading': xs_reading,
        'helper_fns': {
            'plot_function': plot_function,
            'plot_solution': plot_solution,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'activation': 'sin',
                    'hidden_layers': 3, 
                    'hidden_dim': 16,
                },
                'n_pde_collocation_pts': 15000,  #20k
                'n_icbc_collocation_pts': 2000,  #2k
                # 'optim_method': 'adam',
                # 'optim_args': {
                #     'learning_rate': 0.001,
                # },
                'optim_method': 'lbfgs',
                'maxiter': 200000,
                'optim_args': None,
            },
            'pinn_ensemble_training_steps': 200000,
            'grid_obs_param': jnp.array([0., L_x / 2., L_x]),
        },
        'aux': {
            'X': X,
            'T': T,
            'U': U,
            'c_transform': softclip_forward,
        }
    }
    
    
def prep_population(seed=0):
    
    def pde(x, y, const, exp_design):
        # H = prey
        # L = predator
        a, b, c, d, k, r = const
        dH_dt = dde.grad.jacobian(y, x, i=0, j=0)[0]
        dL_dt = dde.grad.jacobian(y, x, i=1, j=0)[0]
        H_val = y[0][..., 0:1]
        L_val = y[0][..., 1:2]
        
        H_change = r * H_val * (1. - (H_val / k)) - ((a * H_val * L_val) / (c + H_val))
        L_change = ((a * b * H_val * L_val) / (c + H_val)) - d * L_val
        return (jnp.concatenate([dH_dt - H_change, dL_dt - L_change]),)
        
    data = np.loadtxt(f'{CURR_DIR}/dataset/population.txt')    
    ts = data[:,0]
    prey_data = data[:,1]
    predator_data = data[:,2]
    
    ts = ts - ts[0]

    pde_domain = dde.geometry.TimeDomain(ts[0], ts[-1])
    
    xs_ic = jnp.array([[0.]])
    ys_ic = jnp.array([[prey_data[0], predator_data[0]]])
    
    def inital_pos(params, net_apply, exp, inv, xs):
        ys_pred = net_apply(params, xs)
        return (ys_pred - ys_ic).reshape(-1)
    
    exp_design_fn = [
        (inital_pos, xs_ic),
    ]
    
    def exp_design_in_transform(exp_design):
    
        @jax.jit
        def in_transform(x):
            return 0.1 * x
        
        return in_transform
    
    def exp_design_out_transform(exp_design):
    
        @jax.jit
        def out_transform(x, y):
            return jnp.exp(y)
        
        return out_transform

    N_readings = 10
    
    @jax.jit
    def xs_reading(obs_param):
        return obs_param.reshape(-1, 1)

    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 0.]])
    obs_in_domain = jnp.tile(jnp.array([[ts[0], ts[-1]]]), reps=(N_readings, 1))

    inv_param_in_domain = jnp.array([[0., 1.] for _ in range(6)])
    
    # @jax.jit
    # def inv_embedding(inv):
    #     return softclip_forward(inv)
    
    # true_inv_param = jnp.array([[0., 1.] for _ in range(6)])
    
    # =================================================
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        prey_obs = jnp.interp(obs_design, ts, prey_data)
        predator_obs = jnp.interp(obs_design, ts, predator_data)
        d = jnp.array([prey_obs, predator_obs]).T
        return d.reshape(-1), dict()
        
    # =================================================
        
    def plot_function(func, obs_param=None, **kwargs):
        ts_interp = jnp.linspace(ts[0], ts[-1], 200)
        plt.plot(ts_interp, func(ts_interp.reshape(-1, 1)), **kwargs)
        
    def plot_solution(obs_param=None, rng=jax.random.PRNGKey(0)):
        plt.plot(ts, prey_data)
        plt.plot(ts, predator_data)
        
    def plot_solution_phase(obs_param=None, rng=jax.random.PRNGKey(0)):
        plt.plot(prey_data, predator_data)
            
    # =================================================
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': None,
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': inv_param_in_domain.shape[0], 
        'exp_input_dim': exp_in_domain.shape[0], 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_readings,
        'x_input_dim': 1, 
        'y_output_dim': 2,
        # 'inv_embedding': inv_embedding,
        # 'true_inv_embedding': true_inv_param,
        'xs_reading': xs_reading,
        'helper_fns': {
            'plot_function': plot_function,
            'plot_solution': plot_solution,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'activation': 'sin',
                    'hidden_layers': 6, 
                    'hidden_dim': 8,
                    'input_dim': 1,
                    'output_dim': 2,
                },
                'input_transform_generator_fn': exp_design_in_transform,
                'output_transform_generator_fn': exp_design_out_transform,
                'n_pde_collocation_pts': 500,  #20k
                'n_icbc_collocation_pts': 1,  #2k
                # 'optim_method': 'adam',
                # 'optim_args': {
                #     'learning_rate': 0.001,
                # },
                'optim_method': 'lbfgs',
                'maxiter': 50000,
                'optim_args': None,
            },
            'pinn_ensemble_training_steps': 50000,
            # 'grid_obs_param': jnp.array([0., L_x / 2., L_x]),
        },
        'aux': {
            'ts': ts,
            'prey_data': prey_data,
            'predator_data': predator_data,
        }
    }
    
    
def prep_groundwater(seed=0):
        
    from scipy.io import loadmat
    K_val = 0.0355  # according to the paper, rather than the theoretical value
    SPEEDS = [55, 85, 100, 125, 150, 210, 250, 315, 350, 410, 505]
    data = loadmat(f'{CURR_DIR}/dataset/groundwater/2mm/{SPEEDS[seed % len(SPEEDS)]}mL_per_minute_2mm_beads_9_June_analysed.mat')

    Q = data['Q'][0,0]
    W = data['W'][0,0]
    q = Q / W
    K_div_q_TRUE = K_val / q
    inv_scaling = 1.e3
       
    hs = data['hexp'][:,0]
    xs = data['xexp'][0]
    nanvals = np.isnan(hs)
    hs = jnp.array(hs[~nanvals])
    xs = jnp.array(xs[~nanvals])
    L_x = np.max(xs)
    xs = L_x - xs
    xs = xs[::-1]
    hs = hs[::-1]
    
    # scaling factor for PDE to match paper
    scaling = np.mean(hs) ** 2
    
    def pde(x, y, const, exp_design):
        K_div_q = inv_scaling * const[0]
        h = y[0]
        dh_x = dde.grad.jacobian(y, x, j=0)[0]
        dh_xx = dde.grad.hessian(y, x, i=0, j=0)[0]
        return (scaling * (1. + (K_div_q * h * dh_x)),)
        # return (scaling * (1. + (K_div_q * h * dh_x) + (1 / 3.) * (h * dh_xx + dh_x ** 2)),)

    pde_domain = dde.geometry.Interval(0., L_x)
    
    # def exp_design_fn(exp_param):
    #     return [
    #         dde.icbc.boundary_conditions.PeriodicBC(pde_domain, 0, boundary_r),
    #         dde.icbc.IC(
    #             pde_domain, 
    #             lambda x_: ic_fn(x_[:,0], exp_param).reshape(-1, 1), lambda _, 
    #             on_initial: on_initial
    #         ),
    #     ]
    
    xs_ic = jnp.array([[0.]])
    
    def inital_pos(params, net_apply, exp, inv, xs):
        ys_pred = net_apply(params, xs)
        return (ys_pred - inv[1]).reshape(-1)
    
    exp_design_fn = [
        (inital_pos, xs_ic),
    ]
    
    # def exp_design_in_transform(exp_design):
    
    #     @jax.jit
    #     def in_transform(x):
    #         return 0.1 * x
        
    #     return in_transform
    
    # def exp_design_out_transform(exp_design):
    
    #     @jax.jit
    #     def out_transform(x, y):
    #         return jnp.exp(y)
        
    #     return out_transform

    N_readings = 10
    
    @jax.jit
    def xs_reading(obs_param):
        return obs_param.reshape(-1, 1)

    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 0.]])
    obs_in_domain = jnp.tile(jnp.array([[0.1, L_x - 1e-6]]), reps=(N_readings, 1))

    inv_param_in_domain = jnp.array([[0.1, 1.5], [0.1, 0.5]])
    
    @jax.jit
    def inv_embedding(inv):
        return inv_scaling * inv[0:1]
    
    true_inv_param = jnp.array([K_div_q_TRUE])
    
    # =================================================
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        return jnp.interp(obs_design, xs, hs).reshape(-1), dict()
        
    # =================================================
        
    def plot_function(func, obs_param=None, **kwargs):
        ts_interp = jnp.linspace(0., L_x, 200)
        plt.plot(ts_interp, func(ts_interp.reshape(-1, 1)))
        
    def plot_solution(obs_param=None, fn=None, rng=jax.random.PRNGKey(0)):
        plt.plot(xs, hs, '.', alpha=0.1, color='gray', markersize=5)
        if fn is not None:
            ts_interp = jnp.linspace(0., L_x, 200)
            plt.plot(ts_interp, fn(ts_interp.reshape(-1, 1)), '--', color='black', alpha=0.8)
        if obs_param is not None:
            plt.plot(obs_param, oracle(None, obs_param)[0], 'o', color='blue', markersize=5)
        plt.xlabel('x')
        plt.ylabel('h(x)')
            
    # =================================================
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': None,
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': inv_param_in_domain.shape[0], 
        'exp_input_dim': exp_in_domain.shape[0], 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_readings,
        'x_input_dim': 1, 
        'y_output_dim': 1,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': true_inv_param,
        'xs_reading': xs_reading,
        'helper_fns': {
            'plot_function': plot_function,
            'plot_solution': plot_solution,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'activation': 'tanh',
                    'hidden_layers': 2, 
                    'hidden_dim': 8,
                    'input_dim': 1,
                    'output_dim': 1,
                },
                # 'input_transform_generator_fn': exp_design_in_transform,
                # 'output_transform_generator_fn': exp_design_out_transform,
                'n_pde_collocation_pts': 500,  #20k
                'n_icbc_collocation_pts': 1,  #2k
                # 'optim_method': 'adam',
                # 'optim_args': {
                #     'learning_rate': 0.001,
                # },
                'optim_method': 'lbfgs',
                'maxiter': 50000,
                'optim_args': None,
            },
            'pinn_ensemble_training_steps': 50000,
            'grid_obs_param': jnp.linspace(start=obs_in_domain[0,0], stop=obs_in_domain[0,1], num=N_readings),
        },
        'aux': {
            'x': xs,
            'h': hs,
            'K': K_val,
            'Q': Q,
            'W': W,
        }
    }
    
    
def prep_cooling(seed=0):
        
    import pickle
    with open(f'{CURR_DIR}/dataset/cooling/case{seed % 2}.pkl', 'rb') as f:
        data = pickle.load(f)

    ts = data['time']
    ys = data['temp']
    K = data['lambda']
    T0 = data['T0']
    Tinit = data['T_init']
    
    L_t = 300.
    
    # scaling factor for PDE to match paper
    inv_scaling = 1.e-4
    T_scaling = 10.
    
    def pde(x, y, const, exp_design):
        K, T0, Tinit = const
        K = inv_scaling * jnp.abs(K)
        T0 = T_scaling * T0
        Tinit = T_scaling * Tinit 
        T = y[0]
        dT_t = dde.grad.jacobian(y, x, j=0)[0]
        return (1.e3 * (dT_t + K * (T - T0)),)
        # return (scaling * (1. + (K_div_q * h * dh_x) + (1 / 3.) * (h * dh_xx + dh_x ** 2)),)

    pde_domain = dde.geometry.Interval(0., L_t)
    
    xs_ic = jnp.array([[0.]])
    
    def inital_pos(params, net_apply, exp, inv, xs):
        K, T0, Tinit = inv
        ys_pred = net_apply(params, xs)
        return (ys_pred - Tinit).reshape(-1)
    
    exp_design_fn = [
        (inital_pos, xs_ic),
    ]
    
    def exp_design_in_transform(exp_design):
    
        @jax.jit
        def in_transform(x):
            return 1.e-2 * x
        
        return in_transform
    
    def exp_design_out_transform(exp_design):
    
        @jax.jit
        def out_transform(x, y):
            return 1.e2 * y
        
        return out_transform

    N_readings = 4
    
    @jax.jit
    def xs_reading(obs_param):
        return obs_param.reshape(-1, 1)

    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 0.]])
    obs_in_domain = jnp.tile(jnp.array([[0., L_t]]), reps=(N_readings, 1))

    # K, T0, Tinit
    inv_param_in_domain = jnp.array([[1., 100.], [2., 4.], [5., 8.]])
    
    @jax.jit
    def inv_embedding(inv):
        return inv_scaling * jnp.abs(inv[0:1])
    
    true_inv_param = jnp.array([K])
    
    # =================================================
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        return jnp.interp(obs_design, ts, ys).reshape(-1), dict()
        
    # =================================================
        
    def plot_function(func, obs_param=None, **kwargs):
        ts_interp = jnp.linspace(0., L_t, 200)
        plt.plot(ts_interp, func(ts_interp.reshape(-1, 1)), **kwargs)
        
    def plot_solution(obs_param=None, rng=jax.random.PRNGKey(0), **kwargs):
        plt.plot(ts, ys, '.', **kwargs)
            
    # =================================================
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': None,
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': inv_param_in_domain.shape[0], 
        'exp_input_dim': exp_in_domain.shape[0], 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_readings,
        'x_input_dim': 1, 
        'y_output_dim': 1,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': true_inv_param,
        'xs_reading': xs_reading,
        'helper_fns': {
            'plot_function': plot_function,
            'plot_solution': plot_solution,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'activation': 'tanh',
                    'hidden_layers': 2, 
                    'hidden_dim': 8,
                    'input_dim': 1,
                    'output_dim': 1,
                },
                # 'input_transform_generator_fn': exp_design_in_transform,
                # 'output_transform_generator_fn': exp_design_out_transform,
                'n_pde_collocation_pts': 500,  #20k
                'n_icbc_collocation_pts': 1,  #2k
                # 'optim_method': 'adam',
                # 'optim_args': {
                #     'learning_rate': 0.001,
                # },
                'optim_method': 'lbfgs',
                'maxiter': 50000,
                'optim_args': None,
            },
            'pinn_ensemble_training_steps': 50000,
            'grid_obs_param': jnp.linspace(start=obs_in_domain[0,0], stop=obs_in_domain[0,1], num=N_readings),
        },
        'aux': {
            't': ts,
            'T': ys,
            'raw_inv': np.array([K / inv_scaling, T0, Tinit])
        }
    }
    

def prep_cell_population(seed=0):
    
    if seed % 4 == 0:
        a = 'c'
        c = [530.39, 0.066, 46.42]
    elif seed % 4 == 1:
        a = 'd'
        c = [484.74, 0.065, 43.15]
    elif seed % 4 == 2:
        a = 'e'
        c = [636.68, 0.070, 45.48]
    else:
        a = 'f'
        c = [982.26, 0.078, 47.65]
        
    c1, c2, c3 = c
    true_inv_param = jnp.array([1.e-2 * c1, 1.e2 * c2, 1.e-1 * c3])
        
    from scipy.io import loadmat
    data = loadmat(f'{CURR_DIR}/dataset/cells/data_{a}.mat')

    Y_data = jnp.array(data['C'], dtype=float)
    T_data = jnp.array(data['t'].flatten(), dtype=float)
    X_data = jnp.array(data['x'].flatten(), dtype=float)
    L_x = 1900
    
    def pde(x, y, const, exp_design):
        # scaling
        c1, c2, c3 = const
        c1 = 1.e2 * c1
        c2 = 1.e-2 * c2
        c3 = 1.e1 * c3
        # pde
        p = y[0]
        p_t = dde.grad.jacobian(y, x, j=1)[0]
        p_xx = dde.grad.hessian(y, x, i=0, j=0)[0]
        return (10. * (p_t - c1 * p_xx - c2 * p + c3 * p**2),)

    geom = dde.geometry.Interval(0, L_x)
    timedomain = dde.geometry.TimeDomain(0, T_data[-1])
    pde_domain = dde.geometry.GeometryXTime(geom, timedomain)
    
    xs_ic = jnp.array([X_data, jnp.zeros_like(X_data)]).T
    xs_bc = jnp.array(pde_domain.random_boundary_points(1000))
    
    exp_design_fn = [
        (generate_fixed_points_bc(0, xs_ic, Y_data[0]), xs_ic),
        (generate_velocity_bc((lambda xs, exp, inv: 0.), i=0, j=0, scale=10.), xs_bc),
    ]
    
    def exp_design_in_transform(exp_design):
    
        @jax.jit
        def in_transform(x):
            x = x.at[..., 0:1].multiply(1.e-3)
            x = x.at[..., 1:2].multiply(1.e-1)
            return x
        
        return in_transform
    
    def exp_design_out_transform(exp_design):
    
        @jax.jit
        def out_transform(x, y):
            return 1.e-3 * y**2
        
        return out_transform

    N_readings = 5
    N_timesteps = T_data.shape[0] - 1
    
    @jax.jit
    def xs_reading(obs_param):
        locs = obs_param.reshape(N_readings, 1)
        return jnp.concatenate([jnp.repeat(locs, N_timesteps, axis=0), jnp.tile(T_data[1:], reps=N_readings)[:,None]], axis=1)

    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 0.]])
    obs_in_domain = jnp.tile(jnp.array([[X_data[0], X_data[-1]]]), reps=(N_readings, 1))

    inv_param_in_domain = jnp.array([[4., 10.], [4., 10.], [4., 10.]])
    
    @jax.jit
    def inv_embedding(inv):
        return inv
    
    # =================================================
    
    import scipy
    interp = scipy.interpolate.RBFInterpolator(
        np.array(jnp.meshgrid(X_data, T_data)).reshape(2, -1).T, 
        np.array(Y_data.reshape(-1))
    )
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        xs = xs_reading(obs_design)
        return interp(xs).reshape(-1), dict()
            
    # =================================================
    
    def plot_function_2d(func, obs_param=None):
        xi = np.linspace(0., L_x, 101)
        yi = np.linspace(0., 48., 101)
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        pts = jnp.array([xs, ys]).T
        zs = np.array(func(pts)).reshape(-1)

        plt.ticklabel_format(style='sci', axis='both', scilimits=(-1, 1))
        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = plt.contourf(xi, yi, zi, levels=50, cmap="RdBu_r", alpha=0.5, antialiased=True)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        cbar = plt.colorbar()
        cbar.formatter.set_powerlimits((0, 0))
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            plt.plot(xs[:,0], xs[:,1], 'ob')
            
        plt.xlabel('x', fontsize=14)
        plt.ylabel('t', fontsize=14)
    
    def plot_function(func, t, **kwargs):
        xs = jnp.linspace(0., L_x, num=100)
        plt.plot(xs, func(jnp.array([xs, t * jnp.ones_like(xs)]).T), **kwargs)
        
    def plot_solution(t=None, obs_param=None, rng=jax.random.PRNGKey(0)):
        ts_all = [0, 12, 24, 36, 48]
        plot_all = (t is None)
        if not plot_all:
            assert t in ts_all
            ts = [t]
        else:
            ts = ts_all
        plt.ticklabel_format(style='sci', axis='both', scilimits=(-1, 1))
        for t in ts:
            i = t // 12
            plt.plot(X_data, Y_data[i], '.-', label=f't={t}hrs', markersize=4, alpha=(0.6 if i==0 else 0.2))
        if obs_param is not None:
            for x in obs_param:
                plt.axvline(x, linestyle='--', color='gray', alpha=0.5)
            ys = oracle(None, obs_param)[0].reshape(obs_param.shape[0], -1)
            plt.plot(obs_param, ys, 'ob', alpha=0.8, markersize=4)
        if plot_all:
            plt.legend(loc='upper right')
        plt.xlabel('x', fontsize=14)
        plt.ylabel('ρ(x,t)', fontsize=14)
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': None,
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': inv_param_in_domain.shape[0], 
        'exp_input_dim': exp_in_domain.shape[0], 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_readings * N_timesteps,
        'x_input_dim': 2, 
        'y_output_dim': 1,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': true_inv_param,
        'xs_reading': xs_reading,
        'helper_fns': {
            'plot_function_2d': plot_function_2d,
            'plot_function': plot_function,
            'plot_solution': plot_solution,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'activation': 'tanh',
                    'hidden_layers': 2, 
                    'hidden_dim': 8,
                    'input_dim': 2,
                    'output_dim': 1,
                },
                'input_transform_generator_fn': exp_design_in_transform,
                'output_transform_generator_fn': exp_design_out_transform,
                'n_pde_collocation_pts': 1000,
                'n_icbc_collocation_pts': 100,
                'optim_method': 'adam',
                'optim_args': {
                    'learning_rate': 0.001,
                },
                # 'optim_method': 'lbfgs',
                # 'maxiter': 50000,
                # 'optim_args': None,
            },
            'pinn_ensemble_training_steps': 50000,
            'grid_obs_param': jnp.linspace(obs_in_domain[0,0], obs_in_domain[0,1], num=N_readings),
        },
        'aux': {
            'x': X_data,
            't': T_data,
            'y': Y_data,
            'L_x': L_x,
        }
    }
    
    
def prep_chromatography(seed=0):
    
    # these constants are from the paper, unless stated otherwise
    v0 = 1.964875841  # mL / min
    t0 = 21.  # mins
    eps = 0.648  # dimensionless
    L_x = 10.  # cm
    Nttbb = 420.  # number of theoretical plates for more-retained enentiomer of Troger's base as reported in paper
    Dax = (v0 * L_x) / (2. * Nttbb)  # theoretical value of dispersion coefficient as computed
    
    scale_Dax = 1.e-2
    scale_K = 10.
    
    # scaling of pde to make quantities unitless as in the paper
    a1_true = v0 * t0 / L_x
    a2_true = Dax * t0 / L_x**2
    a3_true = (1. - eps) / eps

    geom = dde.geometry.Interval(0, 1.)
    timedomain = dde.geometry.TimeDomain(0, 1.)
    pde_domain = dde.geometry.GeometryXTime(geom, timedomain)
    
    # we only extract data for the pure more-retained enentiomer of Troger's base
    data = np.genfromtxt(f'{CURR_DIR}/dataset/chromatography/chroma{seed % 4}.csv', delimiter=',')
    ts = jnp.array(data[:,0] / t0)
    y_start = jnp.array(data[:,1] / data[0,1])
    y_end = jnp.array(data[:,2] / data[0,1])
    
    def pde(x, y, const, exp_design):
        # constants
        D_ax_guess, K_guess = jnp.abs(const)
        a2 = (scale_Dax * D_ax_guess) * t0 / L_x**2
        a3 = scale_K * K_guess * (1. - eps) / eps
        # pde stuff
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)[0]
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)[0]
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)[0]
        # assume linear adsorpotion isotherm
        return (5. * ((1. + a3) * dy_t + a1_true * dy_x - a2 * dy_xx),)
        
    
    def bc_fn_jax(x, exp_param, inv):
        t_during_inject = (x[:,1:2] < 0.1)
        return 1. * t_during_inject
    
    def ic_fn_jax(x, exp_param, inv):
        return 0.
    
    
    xs_ic = jnp.array(pde_domain.random_initial_points(1000)).at[:,1].set(0.)
    xs_bc = jnp.array(pde_domain.random_boundary_points(1000)).at[:,0].set(0.)
    # xs_bc_edge = jnp.array(pde_domain.random_boundary_points(1000)).at[:,0].set(1.)
    
    exp_design_fn = [
        (generate_fixed_function_bc(boundary_func=bc_fn_jax, scale=5.), xs_bc),
        # (generate_velocity_bc(boundary_func=lambda x, exp, inv: 0., i=0, j=0), xs_bc_edge),
        (generate_fixed_function_bc(boundary_func=ic_fn_jax, scale=5.), xs_ic),
    ]

    N_obs = 30
    
    def xs_reading(obs_param):
        # return jnp.vstack([jnp.ones_like(obs_param), obs_param]).T
        x_ = jnp.linspace(obs_param[0], obs_param[1], N_obs)
        return jnp.vstack([jnp.ones_like(x_), x_]).T


    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 0.]])
    obs_in_domain = jnp.array([[0., 1.], [0., 1.]])
    
    
    # N_obs = 25
    
    # def xs_reading(obs_param):
    #     return jnp.vstack([jnp.ones_like(obs_param), obs_param]).T

    # def obs_design_fn(f, obs_param):
    #     return f(xs_reading(obs_param)).reshape(-1)


    # exp_in_domain = jnp.array([[0., 0.]])
    # obs_in_domain = jnp.array([[0., 1.] for _ in range(N_obs)])
    

    # Dax, adsorption isotherm constant
    inv_param_in_domain = jnp.array([[1., 6.], [0.3, 0.4]])
    
    true_inv_param = jnp.array([Dax])
    inv_embedding = lambda inv: scale_Dax * jnp.abs(inv[0:1])
    
    def exp_design_out_transform(exp_design):
    
        @jax.jit
        def out_transform(x, y):
            return 0.2 * jax.nn.softplus(5. * y)
        
        return out_transform
    
    # =================================================
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        x_ = jnp.linspace(obs_design[0], obs_design[1], N_obs).reshape(-1)
        # x_ = obs_design
        return jnp.interp(x_, ts, y_end).reshape(-1), dict()
        
    # =================================================
        
    def plot_function(func, obs_param=None):
        xi = np.linspace(0., 1., 101)
        yi = np.linspace(0., 1., 101)
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        pts = jnp.array([xs, ys]).T
        zs = np.array(func(pts)).reshape(-1)

        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = plt.contourf(xi, yi, zi, levels=50, cmap="RdBu_r", alpha=0.7, antialiased=True)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        plt.colorbar()
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            plt.plot(xs[:,0], xs[:,1], 'xk')
        
    def plot_elution_pred(func, **kwargs):
        ts_interp = jnp.linspace(0., 1., 200)
        plt.plot(ts_interp, func(xs_reading(ts_interp)), **kwargs)
        
    def plot_elution_true(obs_param=None, rng=jax.random.PRNGKey(0), **kwargs):
        plt.plot(ts, y_end, '.', **kwargs)
        if obs_param is not None:
            plt.plot(obs_param, oracle(None, obs_param)[0], 'o', **kwargs)
            
    # =================================================
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': None,
        # 'sample_inv_param_prior': lambda n, rng: sample_from_uniform(n, inv_param_in_domain, 2, rng=rng),
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': inv_param_in_domain.shape[0], 
        'exp_input_dim': exp_in_domain.shape[0], 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_obs,
        'x_input_dim': 2, 
        'y_output_dim': 1,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': true_inv_param,
        'xs_reading': xs_reading,
        'helper_fns': {
            'plot_function': plot_function,
            'plot_elution_pred': plot_elution_pred,
            'plot_elution_true': plot_elution_true,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'hidden_layers': 4, 
                    'hidden_dim': 8,
                },
                'n_pde_collocation_pts': 1500,
                'n_icbc_collocation_pts': 300,
                # 'optim_method': 'adam',
                # 'optim_args': {
                #     'learning_rate': 0.001,
                # }
                'optim_method': 'lbfgs',
                'maxiter': 150000,
                'optim_args': None,
            },
            'output_transform_generator_fn': exp_design_out_transform,
            'pinn_ensemble_training_steps': 100000,
            # 'grid_obs_param': jnp.linspace(0., 1., N_obs+1)[1:],
            'grid_obs_param': jnp.array([0., 1.]),
        },
        'aux': {
            'true_inv_param': true_inv_param,
            'ts': ts,
            'y_start': y_start,
            'y_end': y_end,
            'constants': {
                'v0': v0,
                't0': t0,
                'eps': eps,
                'L_x': L_x,
                'Nttbb': Nttbb,
                'Dax': Dax,
            }
        },
    }
    

def prep_burgers_equation(seed=0):
    
    # adapted from https://github.com/sachabinder/Burgers_equation_simulation/blob/main/Burgers_solver_SP.py
    
    #Spatial mesh
    L_x = 12 #Range of the domain according to x [m]
    dx = 0.01 #Infinitesimal distance
    N_x = int(L_x/dx) #Points number of the spatial mesh
    X = jnp.linspace(0,L_x,N_x) #Spatial array

    #Temporal mesh
    L_t = 5. #Duration of simulation [s]
    dt = 0.01  #Infinitesimal time
    N_t = int(L_t/dt) #Points number of the temporal mesh
    T = jnp.linspace(0,L_t,N_t) #Temporal array
    
    MID = 6.
    WIDTH = 0.8

    def _boundary_r(x, on_boundary, xL, xR):
        return (on_boundary and jnp.isclose(x[0], xL)) or (on_boundary and jnp.isclose(x[0], xR))

    boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, 0, L_x)


    geom = dde.geometry.Interval(0, L_x)
    timedomain = dde.geometry.TimeDomain(0, L_t)
    pde_domain = dde.geometry.GeometryXTime(geom, timedomain)
    
    
    def pde(x, y, const, exp_design):
        mu = jnp.exp(const[0])
        nu = jnp.exp(const[1])
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)[0]
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)[0]
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)[0]
        return (dy_t + mu * y[0] * dy_x - nu * dy_xx,)
        
    
    def ic_fn_jax(x, exp_param, inv):
        mid = MID  #inv[2]
        width = WIDTH
        x_ = jnp.minimum((x[:,0:1] - mid) % L_x, (mid - x[:,0:1]) % L_x)
        return jnp.exp(- x_ ** 2 / width)
    
    
    xs_ic = jnp.array(pde_domain.random_initial_points(1000))
    xs_bc = jnp.array(pde_domain.random_boundary_points(1000))
    
    exp_design_fn = [
        (generate_periodic_bc(comp=0, L_bound=0., R_bound=L_x), xs_bc),
        (generate_fixed_function_bc(boundary_func=ic_fn_jax), xs_ic),
    ]

    N_obs = 11
    N_x_obs = 3
    
    def xs_reading(obs_param):
        xloc = obs_param % L_x
        return jnp.concatenate(jnp.meshgrid(xloc, jnp.linspace(0., L_t, N_obs)), axis=0).reshape(2, -1).T


    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 0.]])
    obs_in_domain = jnp.array([[0., L_x] for _ in range(N_x_obs)])

    # mu, nu, ic_centre, ic_height
    inv_param_in_domain = jnp.array([[-4., -1.], [-3., -1.]])
    
    # we make the parameter the log of these ones
    # # mu = 1
    # # nu = 0.01
    # true_inv_param = jnp.array([0., -2.])
    true_inv_param = sample_from_uniform(
        n=1, 
        bounds=inv_param_in_domain, 
        sample_dim=inv_param_in_domain.shape[0], 
        rng=jax.random.PRNGKey(seed)
    )[0]
    
    inv_embedding = lambda inv: inv[0:2]
    # compare_inv_fn = lambda inv1, inv2: jnp.linalg.norm(inv1 - inv2)
    # compare_true_inv_fn = lambda inv: compare_inv_fn(inv, true_inv_param)
    
    # =================================================
    
    # noise_prior_oracle = gpx.Prior(
    #     mean_function=gpx.mean_functions.Zero(), 
    #     kernel=gpx.kernels.RBF(variance=1e-6, lengthscale=1e-3)
    # )
    
    noise_std = 1e-3
    
    def ic_fn(x, exp_param, inv):
        mid = MID  #inv[2]
        width = WIDTH
        x_ = jnp.minimum((x[:,0] - mid) % L_x, (mid - x[:,0]) % L_x)
        return jnp.exp(- x_ ** 2 / width)
    
    def _solve_grid_burgers(exp_design, inv):
        
        u0 = jnp.array(ic_fn(X.reshape(-1, 1), exp_design, inv)).reshape(-1)
        mu = jnp.exp(inv[0])
        nu = jnp.exp(inv[1])
        k = 2*jnp.pi*jnp.fft.fftfreq(N_x, d = dx)  # Wave number discretization
        
        # Definition of ODE system (PDE ---(FFT)---> ODE system)
        @jax.jit
        def burg_system(u, t):
            # Spatial derivative in the Fourier domain
            u_hat = jnp.fft.fft(u)
            u_hat_x = 1j*k*u_hat
            u_hat_xx = -k**2*u_hat
            
            # Switching in the spatial domain
            u_x = jnp.fft.ifft(u_hat_x)
            u_xx = jnp.fft.ifft(u_hat_xx)
            
            # ODE resolution
            u_t = -mu*u*u_x + nu*u_xx
            return u_t.real
        
        # PDE resolution (ODE system resolution)
        U = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, u, args: burg_system(u, t)), 
            y0=u0, 
            t0=0.,
            t1=L_t,
            dt0=dt,
            saveat=dfx.SaveAt(ts=T),
            solver=dfx.Tsit5(),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            max_steps=100000,
            throw=False,  # ignore all errors, might cause some issues!
        ).ys.T
        
        # u0 = np.array(ic_fn(X, exp_design))
        # mu = float(10. ** inv[0])
        # nu = float(10. ** inv[1])
        # k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)  # Wave number discretization
        
        # def burg_system(u, t):
        #     #Spatial derivative in the Fourier domain
        #     u_hat = np.fft.fft(u)
        #     u_hat_x = 1j*k*u_hat
        #     u_hat_xx = -k**2*u_hat
            
        #     #Switching in the spatial domain
        #     u_x = np.fft.ifft(u_hat_x)
        #     u_xx = np.fft.ifft(u_hat_xx)
            
        #     #ODE resolution
        #     u_t = -mu*u*u_x + nu*u_xx
        #     return u_t.real

        # U = odeint(burg_system, u0, T, atol=1e-4, rtol=1e-4, mxstep=1000).T
        
        return jnp.array(U)
    
    def burgers_solver(exp_design, inv):
        U = _solve_grid_burgers(exp_design, inv)
        interp = RegularGridInterpolator([X, T], U, method='nearest')
        return lambda xs: interp(xs).reshape(-1, 1)
    
    def noisy_closed_form_soln(exp_design, inv, rng=jax.random.PRNGKey(42)):
        return burgers_solver(exp_design, inv)
        
        # def _fn(xs):
        #     # prior_dist = noise_prior_oracle.predict(xs)
        #     ys = burgers_solver(exp_design, inv)(xs)
        #     return ys
        #     # noise = prior_dist.sample(seed=rng, sample_shape=(1,)).reshape(ys.shape)
        #     # assert ys.shape == noise.shape, (ys.shape, noise.shape)
        #     # noise = noise_std * jax.random.normal(rng, shape=ys.shape)
        #     # return ys + noise
        
        # return _fn
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        f = noisy_closed_form_soln(exp_design, true_inv_param, rng=rng)
        xs = xs_reading(obs_design)
        ys = obs_design_fn(f, obs_design)
        noise = noise_std * jax.random.normal(rng, shape=ys.shape)
        return (ys + noise).reshape(-1), dict()
        
    # =================================================
        
    def plot_function(func, obs_param=None):
        xi = np.linspace(0., L_x, 101)
        yi = np.linspace(0., L_t, 101)
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        pts = jnp.array([xs, ys]).T
        zs = np.array(func(pts)).reshape(-1)

        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = plt.contourf(xi, yi, zi, levels=50, cmap="RdBu_r", alpha=0.7, antialiased=True)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        plt.colorbar()
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            plt.plot(xs[:,0], xs[:,1], 'xk')
        
    def plot_solution(obs_param=None, inv=None, rng=jax.random.PRNGKey(0)):
        
        if inv is None:
            inv = true_inv_param[0]
            
        plot_function(noisy_closed_form_soln(exp_design=jnp.array([0.]), inv=inv, rng=rng))
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            plt.plot(xs[:,0], xs[:,1], '.k')
            
    # =================================================
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': noisy_closed_form_soln,
        # 'sample_inv_param_prior': lambda n, rng: sample_from_uniform(n, inv_param_in_domain, 2, rng=rng),
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': inv_param_in_domain.shape[0], 
        'exp_input_dim': exp_in_domain.shape[0], 
        'obs_input_dim': obs_in_domain.shape[0],
        'obs_reading_count': N_obs * obs_in_domain.shape[0],
        'x_input_dim': 2, 
        'y_output_dim': 1,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': inv_embedding(true_inv_param),
        'xs_reading': xs_reading,
        'helper_fns': {
            'solve_grid_burgers': _solve_grid_burgers,
            'burgers_solver': burgers_solver, 
            # 'noisy_closed_form_soln': noisy_closed_form_soln,
            # 'plot_criterion_landscape': plot_criterion_landscape,
            'plot_function': plot_function,
            'plot_solution': plot_solution,
            'ic_fn': ic_fn,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    'hidden_layers': 6, 
                    'hidden_dim': 8,
                },
                'n_pde_collocation_pts': 1000,
                'n_icbc_collocation_pts': 500,
                'optim_method': 'adam',
                'optim_args': {
                    'learning_rate': 0.001,
                }
            },
            'pinn_ensemble_training_steps': 100000,
            'grid_obs_param': jnp.array([2., 6., 10.]),
        },
        'aux': {
            'true_inv_param': true_inv_param,
            'X_grid': X,
            'T_grid': T,
        },
    }

def prep_2d_pipe_flow_scalar(seed=0):
            
    rng = jax.random.PRNGKey(seed)
    
    
    """
    
    Geometry:
     
                      4     4
        -----+-----+-----+-----
                                2
                 (8,2)---+-----
                   |            2
        -----+-----+
          4     4
          
    """
    
    
    RHO_BASE = 1e-3 * 998.2  # kg m^-3
    MU_BASE = 0.001003  # kg m^-1 s^-1
    v0 = 2.25 # m s^-1

    length_scaling = 1.
    xcorner = 5.  # m
    ycorner = 1. # m
    xmin = 0.  # m
    xmax = 8.  # m
    ymin = 0.  # m
    ymax = 3. # m
    inverse_section_xmin = 2.  # m
    inverse_section_xmax = xcorner
    
    pde_domain = dde.geometry.CSGDifference(
        dde.geometry.Rectangle([xmin, ymin], [xmax, ymax]), 
        dde.geometry.Rectangle([xcorner, ymin], [xmax, ycorner]), 
    )

    # Define Navier Stokes Equations (Time-dependent PDEs)
    def pde(x, y, const, exp_design):
        
        rho = jnp.abs(const[0])
        # mu = 1.e-3 * jnp.abs(const[1])
        mu = MU_BASE
        
        # rho = 1.e3 * (10. ** const[0])
        # mu = 1.e-3 * (10. ** const[1])
        
        u = y[0][:, 0:1]  # m s^-1
        v = y[0][:, 1:2]  # m s^-1
        p = y[0][:, 2:3]  # kg m^-1 s^-2
        
        # scalings to make SI dimensions correct
        du_x = dde.grad.jacobian(y, x, i=0, j=0)[0]  # s^-1
        du_y = dde.grad.jacobian(y, x, i=0, j=1)[0]  # s^-1
        dv_x = dde.grad.jacobian(y, x, i=1, j=0)[0]  # s^-1
        dv_y = dde.grad.jacobian(y, x, i=1, j=1)[0]  # s^-1
        dp_x = dde.grad.jacobian(y, x, i=2, j=0)[0]  # kg m^-2 s^-2
        dp_y = dde.grad.jacobian(y, x, i=2, j=1)[0]  # kg m^-2 s^-2
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)[0]  # m^-1 s^-1
        du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)[0]  # m^-1 s^-1
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)[0]  # m^-1 s^-1
        dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)[0]  # m^-1 s^-1
        
        continuity = du_x + dv_y
        x_momentum = rho * (u * du_x + v * du_y) - mu * (du_xx + du_yy) + dp_x
        y_momentum = rho * (u * dv_x + v * dv_y) - mu * (dv_xx + dv_yy) + dp_y
        
        # scale loss function to make magnitude about the same
        return (jnp.concatenate([continuity, x_momentum, y_momentum]).reshape(-1, 1), )
    
    N_boundary = 500
    rng, k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(rng, 10)
    
    x_top = jnp.hstack([
        jax.random.uniform(key=k1, shape=(N_boundary, 1), minval=xmin, maxval=xmax),
        ymax * jnp.ones(shape=(N_boundary, 1)), 
    ])
    x_bottom_lowerstep = jnp.hstack([
        jax.random.uniform(key=k2, shape=(N_boundary, 1), minval=xmin, maxval=inverse_section_xmin),
        ymin * jnp.ones(shape=(N_boundary, 1)), 
    ])
    x_bottom_upperstep = jnp.hstack([
        jax.random.uniform(key=k3, shape=(N_boundary, 1), minval=xcorner, maxval=xmax),
        ycorner * jnp.ones(shape=(N_boundary, 1)), 
    ])
    x_upstep= jnp.hstack([
        xcorner * jnp.ones(shape=(N_boundary, 1)), 
        jax.random.uniform(key=k4, shape=(N_boundary, 1), minval=ymin, maxval=ycorner),
    ])
    
    x_left = jnp.hstack([
        xmin * jnp.ones(shape=(N_boundary, 1)), 
        jax.random.uniform(key=k1, shape=(N_boundary, 1), minval=ymin, maxval=ymax),
    ])
    x_right = jnp.hstack([
        xmax * jnp.ones(shape=(N_boundary, 1)), 
        jax.random.uniform(key=k5, shape=(N_boundary, 1), minval=ycorner, maxval=ymax),
    ])
    x_inv = jnp.hstack([
        jax.random.uniform(key=k6, shape=(N_boundary, 1), minval=inverse_section_xmin, maxval=inverse_section_xmax),
        ymin * jnp.ones(shape=(N_boundary, 1)), 
    ])
    
    
    def noslip_boundaries(x, y, exp_param, inv):
        return (y[...,:2] - 0.).reshape(-1)
    
    def inflow(x, y, exp_param, inv):
        c = (ymax + ymin) / 2.
        r = (ymax - ymin) / 2.
        u_res = y[:,0].reshape(-1) - v0 * (1. - ((x[:,1] - c) / r)**2).reshape(-1)
        v_res = y[:,1].reshape(-1)
        p_res = y[:,2].reshape(-1)
        return jnp.concatenate([u_res, v_res, p_res]).reshape(-1)
    
    # def outflow(x, y, exp_param, inv):
    #     du_x = dde.grad.jacobian(y, x, i=0, j=0)[0].reshape(-1)
    #     dv_y = dde.grad.jacobian(y, x, i=1, j=1)[0].reshape(-1)
    #     u_res = y[:,2].reshape(-1) - ((1. / Re) * du_x).reshape(-1)
    #     v_res = dv_y
    #     return jnp.concatenate([u_res, v_res]).reshape(-1)
    
    def inverse_flow(x, y, exp_param, inv):
        u_res = y[:,0].reshape(-1)
        v_res = y[:,1].reshape(-1) + 1.e-1 * (x[:,0:1] - inverse_section_xmin).reshape(-1)
        return jnp.concatenate([u_res, v_res]).reshape(-1)
    
    def exp_design_in_transform(exp_design):
    
        @jax.jit
        def in_transform(x):
            return x
        
        return in_transform
    
    def exp_design_out_transform(exp_design):
    
        @jax.jit
        def out_transform(x, y):
            # dy = 1.e3 * (x[..., 1:2] - ymax)
            # y = y.at[..., 0:2].multiply(dy)
            # y = y.at[..., 2:3].multiply(1.e3)  # apply value scaling to make things easier to learn
            return y
        
        return out_transform
    
    exp_design_fn = [
        (generate_arbitrary_bc(noslip_boundaries), x_top),
        (generate_arbitrary_bc(noslip_boundaries), x_bottom_lowerstep),
        (generate_arbitrary_bc(noslip_boundaries), x_bottom_upperstep),
        (generate_arbitrary_bc(noslip_boundaries), x_upstep),
        (generate_arbitrary_bc(inflow), x_left),
        # (generate_arbitrary_bc(outflow), x_right),
        (generate_arbitrary_bc(inverse_flow), x_inv),
    ]
    
    
    N_readings = 15  # 30
    
    def xs_reading(obs_param):
        return obs_param.reshape(N_readings, 2)
    
    def obs_design_fn(f, obs_param):
        return f(xs_reading(obs_param)).reshape(-1)


    exp_in_domain = jnp.array([[0., 0.]])
    # obs_in_domain = jnp.hstack([jnp.array([[xmin, ycorner], [xmax, ymax]]) for _ in range(N_readings)]).T
    obs_in_domain = jnp.hstack([jnp.array([[xmin + 1.e-6, ycorner + 1.e-6], 
                                           [xmax - 1.e-6, ymax - 1.e-6]]) for _ in range(N_readings)]).T
    
    # N_readings_mesh_dim = 5  # 30
    
    # def xs_reading(obs_param):
    #     z1 = jnp.linspace(obs_param[0], obs_param[1], N_readings_mesh_dim)
    #     z2 = jnp.linspace(obs_param[2], obs_param[3], N_readings_mesh_dim)
    #     return jnp.array(jnp.meshgrid(z1, z2)).reshape(2, -1).T
    
    # def obs_design_fn(f, obs_param):
    #     return f(xs_reading(obs_param)).reshape(-1)


    # exp_in_domain = jnp.array([[0., 0.]])
    # # obs_in_domain = jnp.hstack([jnp.array([[xmin, ycorner], [xmax, ymax]]) for _ in range(N_readings)]).T
    # obs_in_domain = jnp.array([
    #     [xmin + 1.e-6, xmax - 1.e-6], 
    #     [xmin + 1.e-6, xmax - 1.e-6], 
    #     [ycorner + 1.e-6, ymax - 1.e-6],
    #     [ycorner + 1.e-6, ymax - 1.e-6],
    # ])

    case_name = 'export.csv'
    # example_used = seed % 3
    # if example_used == 0:
    #     case_name = 'export.csv'
    # elif example_used == 1:
    #     (2.25-((y*1000[m^-1])-0.5)^2)*1[m s^-1]
    #     case_name = 'export_2.csv'
    # elif example_used == 2:
    #     case_name = 'export_3.csv'
    # else:
    #     raise ValueError('Hmmm?')
        
    import pandas as pd
    fname = f'{CURR_DIR}/dataset/ns-data/{case_name}'
    df = pd.read_csv(fname, on_bad_lines='skip', header=None, names=['x', 'y', 'z', 'p', 'u', 'v', '_x', '_y'])
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
    
    xs = 1e3 * df['x'].values + xcorner
    ys = 1e3 * df['y'].values + ycorner
    us = df['u'].values
    vs = df['v'].values
    ps = 1e-3 * df['p'].values
    
    on_b_x = (inverse_section_xmin < xs) & (xs < inverse_section_xmax)
    on_b_y = np.isclose(ys, ymin, atol=1e-6, rtol=1e-6)
    on_b = on_b_x & on_b_y
    xs_border = xs[on_b]
    vs_border = vs[on_b]
    xs_emb, idx = np.unique(xs_border, return_index=True)
    vs_emb = vs_border[idx]
    
    # inv_param_in_domain = jnp.array([[-1., 1.], [-1., 1.]])
    # inv_param_in_domain = jnp.array([[-5., 0.]])
    inv_param_in_domain = jnp.array([[0.1, 2.]])
    
    @jax.jit
    def inv_embedding(inv):
        # return 10. ** inv
        # return inv
        return jnp.array([jnp.abs(inv[0])])
    
    # true_inv_embedding = jnp.array([jnp.log10(MU_BASE)])
    true_inv_embedding = jnp.array([RHO_BASE])
    assert true_inv_embedding.shape == inv_embedding(inv_param_in_domain[:,0]).shape
    
    # =================================================
    
    # # @jax.jit
    # # def inv_embedding(inv):
    # #     return inv
    
    # # dx = ob_xyt_reshaped[1,0,0,0] - ob_xyt_reshaped[0,0,0,0]
    # # dy = ob_xyt_reshaped[0,1,0,1] - ob_xyt_reshaped[0,0,0,1]
    # # dt = 0.1
    # # x_origin = ob_xyt_reshaped[0,0,0]
    # # d_array = jnp.array([dx, dy, dt])
    
    from scipy.interpolate import LinearNDInterpolator
    
    # interp_dict = dict()

    # for t in TIMESTEPS:
    #     d = data[t]
    #     x = jnp.hstack([d['xs'].reshape(-1, 1), d['ys'].reshape(-1, 1)])
    #     y = jnp.hstack([d['u'].reshape(-1, 1), d['v'].reshape(-1, 1)])
    #     interp_dict[t] = [LinearNDInterpolator(points=x, values=y[:,0]), LinearNDInterpolator(points=x, values=y[:,1])]
    
    all_pts = jnp.vstack([xs, ys]).T
    interp_u = LinearNDInterpolator(points=all_pts, values=us)
    interp_v = LinearNDInterpolator(points=all_pts, values=vs)
    interp_p = LinearNDInterpolator(points=all_pts, values=ps)
    
    def oracle(exp_design, obs_design, rng=jax.random.PRNGKey(42)):
        xs = xs_reading(obs_design)
        # r = jnp.array([[interp_u(x)[0], interp_v(x)[0]] for x in xs])
        r = jnp.array([[interp_u(x)[0], interp_v(x)[0], interp_p(x)[0]] for x in xs])
        return r.reshape(-1), dict()
        
    # # =================================================
        
    def plot_function(func, comp=0, obs_param=None, cmap="RdBu_r", **contour_kwargs):
        
        xi = np.linspace(xmin, xmax, 201)
        yi = np.linspace(ymin, ymax, 201)
        Xi, Yi = np.meshgrid(xi, yi)
        xs = Xi.flatten()
        ys = Yi.flatten()
        
        pts = jnp.concatenate([pde_domain.random_points(1000)] + 
                              [xs[:200] for xs in [x_top, x_bottom_lowerstep, x_bottom_upperstep, x_upstep, x_left, x_right, x_inv]])
        # pts = jnp.array([xs, ys]).T
        zs = np.array(func(pts))[:,comp].reshape(-1)

        triang = tri.Triangulation(pts[:,0], pts[:,1])
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = plt.contourf(xi, yi, zi, levels=50, cmap=cmap, alpha=0.7, antialiased=True, **contour_kwargs)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        plt.colorbar()
        plt.gca().add_patch(plt.Rectangle((xcorner, ymin), xmax-xcorner, ycorner-ymin, color='black'))
        plt.axis('scaled')
        
        if obs_param is not None:
            xs = xs_reading(obs_param)
            plt.plot(xs[:,0], xs[:,1], 'xk')
            
    def plot_solution(comp=0, obs_param=None, cmap="RdBu_r", **contour_kwargs):
        
        xi = np.linspace(xmin, xmax, 201)
        yi = np.linspace(ymin, ymax, 201)
        Xi, Yi = np.meshgrid(xi, yi)
        
        pts = jnp.array([xs, ys]).T
        zs = [us, vs, ps][comp]

        triang = tri.Triangulation(pts[:,0], pts[:,1])
        interpolator = tri.LinearTriInterpolator(triang, zs)
        zi = interpolator(Xi, Yi)
        cnt = plt.contourf(xi, yi, zi, levels=50, cmap=cmap, alpha=0.7, antialiased=True, **contour_kwargs)
        for c in cnt.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.)
        plt.colorbar()
        plt.gca().add_patch(plt.Rectangle((xcorner, ymin), xmax-xcorner, ycorner-ymin, color='black'))
        plt.axis('scaled')
        
        if obs_param is not None:
            x_ = xs_reading(obs_param)
            plt.plot(x_[:,0], x_[:,1], 'xk')
            
    def plot_colloc(n_pde=1000, n_ibc=100):
        xs_pde = pde_domain.random_points(n_pde)
        plt.plot(xs_pde[:,0], xs_pde[:,1], '.')
        for xs in [x_top, x_bottom_lowerstep, x_bottom_upperstep, x_upstep, x_left, x_right, x_inv]:
            plt.plot(xs[:n_ibc,0], xs[:n_ibc,1], '.')
        plt.gca().set_aspect('equal', adjustable='box')
            
    def plot_inverse(inv=None):
        if inv is not None:
            raise ValueError
        else:
            plt.plot(xs_emb, vs_emb.reshape(-1))
            
    # =================================================
    
    import scipy
    p = scipy.stats.qmc.Sobol(d=2).random_base2(m=6)[:N_readings]
    obs = jnp.array(p.reshape(-1))
    grid_params = obs_in_domain[:,0] + obs * (obs_in_domain[:,1] - obs_in_domain[:,0])
    
    # grid_params = jnp.array([obs_in_domain[0,0], obs_in_domain[1,1], obs_in_domain[2,0], obs_in_domain[3,1]])
    
    return {
        'pde': pde,
        'pde_domain': pde_domain,
        'exp_design_fn': exp_design_fn,
        'obs_design_fn': obs_design_fn,
        'simulator_xs': None,
        'oracle': oracle,
        'exp_in_domain': exp_in_domain,
        'obs_in_domain': obs_in_domain,
        'inv_param_in_domain': inv_param_in_domain,
        'inv_input_dim': 2,
        'exp_input_dim': 1, 
        'obs_input_dim': obs_in_domain.shape[0],
        # 'obs_reading_count': (N_readings_mesh_dim**2) * 3,  # N_readings * 3,
        'obs_reading_count': N_readings * 3,
        'x_input_dim': 2, 
        'y_output_dim': 3,
        'inv_embedding': inv_embedding,
        'true_inv_embedding': true_inv_embedding,
        'xs_reading': xs_reading,
        'helper_fns': {
            'plot_function': plot_function, 
            'plot_solution': plot_solution,
            'plot_colloc': plot_colloc,
            'plot_inverse': plot_inverse,
        },
        'ed_args': {
            'pinn_ensemble_args': {
                'nn_construct_params': {
                    # 'arch': 'laaf',
                    # 'activation': 'tanh',
                    'activation': 'sin',
                    'input_dim': 2,
                    'output_dim': 3,
                    'hidden_layers': 6, 
                    'hidden_dim': 16,
                },
                'input_transform_generator_fn': exp_design_in_transform,
                'output_transform_generator_fn': exp_design_out_transform,
                'n_pde_collocation_pts': 2000,
                'n_icbc_collocation_pts': 300,
                'optim_method': 'adam',
                'optim_args': {
                    'learning_rate': 0.001,
                },
                # 'optim_method': 'lbfgs',
                # 'maxiter': 200000,
                # 'optim_args': None,
                'pde_colloc_rand_method': 'pseudo',
            },
            'pinn_ensemble_training_steps': 100000,
            'grid_obs_param': grid_params,
        },
        'aux': {
            'data': {
                'xs': xs,
                'ys': ys,
                'us': us,
                'vs': vs,
                'ps': ps,
            },
            'x_top': x_top,
            'x_bottom_lowerstep': x_bottom_lowerstep,
            'x_bottom_upperstep': x_bottom_upperstep,
            'x_upstep': x_upstep,
            'x_left': x_left,
            'x_right': x_right,
            'x_inv': x_inv,
        },
    }
