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

n_spt = 2**6
n_time = 20
# n_spt = 2**12
# n_time = 2000
c_val = 10
m_val = 0.10

plot_directory = f'plots_nx{n_spt}_nt{n_time}_c{c_val}_mu{m_val}/'

threshold = 0.95
gamma = 1e9
regularizer  = math.sqrt(gamma/2)


x_vals = np.linspace(  0, 1, n_spt)
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
c_hat = U_svd[:,: ]



# S = compute_S(S_0, n_spt, n_time, c_val)


# NLDR reconstruction - just send snapshot array and get best sample array.

# moving ROM forward in time for different input parameters:

















