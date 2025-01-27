# aim:
'''
Generate data for other problems for which the quad. manifold method can be compared to the RL NLDR approach.
Code using torch.tensor so the ops can be done on GPU for the original big example (2^12 x 2000).
'''

import math
from math import sqrt
import numpy as np
import opinf
import matplotlib.pyplot as plt
import scipy
import torch
import os
from compute_results.generate_snapshots import compute_S
from compute_results.reconstruction_errors import compute_linear_quadratic_reconstruction_error
from plot_makers.plot_figures import plot_initial_values, plot_sing_vals, plot_snapshot_energy_spectrum
from plot_makers.plot_figures import plot_comparison_FOM_ROM

from compute_results.solvers import full_order_solve, linear_rom_solve, quad_rom_solve
from compute_results.reconstruction_errors import relative_err_computation
n_spt = 2**6
n_time = 20
# n_spt = 2**12
# n_time = 2000
c_val = 10
m_val = 0.10

plot_directory = f'plots_nx{n_spt}_nt{n_time}_c{c_val}_mu{m_val}/'

threshold = 1. # fraction of cumulative energy needed in Linear ROM.
gamma = 1e9
regularizer  = math.sqrt(gamma/2)

x_vals = np.linspace(0, 1, n_spt)
t_vals = np.linspace(0,0.1,n_time)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

S_0 = (1/sqrt( 2e-4 * np.pi)) * np.exp( -( (x_vals - np.repeat([m_val], x_vals.shape))**2/ 2e-4 )  )
results = compute_S(S_0, n_spt, n_time, c_val)

S = results['S']
S_ref = results['S_ref']
S_centered = results['S_centered']
FOM_A = results['FOM_A']
U_svd = results['U_svd']
Sigma_svd = results['Sigma_svd']

# plot Figure 3.
plot_initial_values(S, plot_directory)
# S_prior = S
plot_sing_vals(results, plot_directory)
# print(np.allclose(S, S_prior)) # True

results_reconstr_errs = compute_linear_quadratic_reconstruction_error(S, threshold, regularizer)
U_trunc = results_reconstr_errs['U_trunc']
S_hat = results_reconstr_errs['S_hat']
S_kron = results_reconstr_errs['S_kron']
V_bar = results_reconstr_errs['V_bar']
linear_reconstr_err = results_reconstr_errs['linear_reconstr_err']
quad_reconstr_err = results_reconstr_errs['quad_reconstr_err']

print('linear reconstruction error:', linear_reconstr_err) 
print('quadratic reconstruction error:', quad_reconstr_err) 

# errors for rom predictions using new initial condition - Figure 4:
plot_snapshot_energy_spectrum(results, regularizer, plot_directory)

# Fig 5.
# Linear ROM
m_test = 0.12547
S_0_test = (1/sqrt( 2e-4 * np.pi)) * np.exp( -( (x_vals - np.repeat([m_test], x_vals.shape))**2/ 2e-4 )  )
t_test_vals = np.arange(0, 0.08+1e-6, 1e-6)

# opinf requires numpy arrays:
c_hat = U_trunc.T @ FOM_A @ S_ref[:,0] # (r, 1)
A_hat = U_trunc.T @ FOM_A @ U_trunc # (r, r)
H_hat = U_trunc.T @ FOM_A @ V_bar #(r, r^2)

s_0_test = U_trunc.T @ S_0_test

from utils.tools import ckron_numpy
s_0_test_kron = ckron_numpy(s_0_test)

with opinf.utils.TimedBlock("Full-order solve"):
    S_fom_test = full_order_solve(FOM_A, S_0_test, t_test_vals) # (x, t)

with opinf.utils.TimedBlock("Linear ROM solve"):
    S_rom_lin_test = linear_rom_solve(c_hat, A_hat, s_0_test, t_test_vals)

with opinf.utils.TimedBlock("Quadratic ROM solve"):
    S_rom_quad_test = quad_rom_solve(c_hat, A_hat, H_hat, s_0_test, t_test_vals)

S_rom_linear_test = U_trunc @ S_rom_lin_test
S_rom_quadratic_test = U_trunc @ S_rom_quad_test

model_results = {
    'FOM_reconstruction': S_fom_test,
    'Linear_ROM_reconstruction': S_rom_linear_test,
    'Quadratic_ROM_reconstruction': S_rom_quadratic_test
}

# Fig 5. time vals to compare FOM, ROM.
time_instances = [0.02, 0.04, 0.06, 0.08]

from plot_makers.plot_figures import plot_model_results
plot_model_results(model_results, x_vals, t_test_vals, time_instances, plot_directory)

#***************************
# Fig 6. Using r = 15
# print(U_trunc.shape) # r = 19 here.
# For Fig 7., need to change 'r' in U_trunc, before computing the model_results.
# plot_comparison_FOM_ROM(model_results, x_vals, t_test_vals, plot_directory)

# Fig 8.
# Later.

# relative error computation:
fom_linear_rel_err, fom_quadratic_rel_err = relative_err_computation(model_results)



# uptill now compare results with RL_NLDR.



# plt.figure(figsize=(8, 6))
# plt.pcolormesh(x_vals, t_test_vals, S_rom_linear_test.T, shading='auto', cmap='viridis')  # Transpose data for correct orientation
# plt.colorbar(label="Variable Value")  # Add a color bar

# # Add labels and title
# plt.xlabel("Space (x-values)")
# plt.ylabel("Time (t-values)")
# plt.title("Space-Time Evolution of the Variable")
# plt.savefig('test_fig.png')


# print(S_fom_test.shape, ":", S_rom_linear_test.shape, ":", S_rom_quadratic_test.shape)

# plots for Figure 5.
# print(t_test_vals)
# t_instances = [0.02, 0.04, 0.06, 0.08]
# index = np.argmin( abs(t_test_vals - t_instances[0]* np.ones(len(t_test_vals))))
# print(index, ":", t_test_vals[index])


# 1. FOM
# linear_FOM_model =opinf.models.ContinuousModel(
#         operators="A")
# dt_test = t_test_vals[1] - t_test_vals[0]

# FOM_prediciton = linear_FOM_model.predict(S_0_test, t_test_vals, method='BDF', max_step = dt_test)



# moving forward in time:
# 1. FOM



# 2. Linear ROM
# 3. Quadratic ROM







 


# S = compute_S(S_0, n_spt, n_time, c_val)


# NLDR reconstruction - just send snapshot array and get best sample array.

# moving ROM forward in time for different input parameters:

















