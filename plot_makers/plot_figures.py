import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from utils.tools import ckron_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_initial_values(S, plot_directory):
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    plt.plot(S[:,0])
    plt.xlabel('x-coordinate')
    plt.ylabel('solution s(x,0)')
    plt.savefig(f'{plot_directory}initial_values')
    plt.close()
    return


def plot_sing_vals(results, plot_directory):
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    
    S = results['S']
    S_ref = results['S_ref']
    S_centered = results['S_centered']

    S = torch.tensor(S, device = device)
    S_ref = torch.tensor(S_ref, device = device)
    S_centered = torch.tensor(S_centered, device = device)

    _, Sigma, _ = torch.svd(S_centered)

    del S
    del S_ref
    del S_centered
    torch.cuda.empty_cache()  # Free up GPU memory
    # S = S.cpu()
    # S_ref = S_ref.cpu()
    # S_centered = S_centered.cpu()
    Sigma = Sigma.cpu()

    plt.plot(Sigma)
    plt.xlabel('singular value index')
    plt.ylabel('normalized singular values')
    plt.savefig(f'{plot_directory}normalized_sing_vals')
    plt.close()
    return

def plot_snapshot_energy_spectrum(results, regularizer, plot_directory):
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    S = results['S']
    S_ref = results['S_ref']
    S_centered = results['S_centered']

    S = torch.tensor(S, device = device)
    S_ref = torch.tensor(S_ref, device = device)
    S_centered = torch.tensor(S_centered, device = device)

    U, _, _ = torch.svd(S_centered)

    S_cent_norm = torch.linalg.norm(S_centered, 'fro').item()
    lin_constr_norms, quad_constr_norms = [], []
    
    for r_val in range(1,S_centered.shape[1]):
        U_trunc = U[: , : r_val]
        S_hat = U_trunc.T @ S_centered
        lin_constr = U_trunc @ S_hat
        lin_constr_norms.append(torch.linalg.norm(lin_constr, 'fro').item()) 

        S_kron = ckron_tensor(S_hat)
        D_mat = S_kron.T
        Z_mat = S_centered - lin_constr
    
        # svd of D_mat
        DU_matrix, DS_matrix, DYt_matrix = torch.linalg.svd(D_mat, full_matrices = False)
        # regularization
        DS_matrix_inv = DS_matrix / (DS_matrix **2 + regularizer **2)
        # computing V_bar
        V_bar = Z_mat @ DU_matrix @ torch.diag(DS_matrix_inv) @ DYt_matrix

        S_quad_reconstr = lin_constr + V_bar @ S_kron
        quad_constr_norms.append(torch.linalg.norm(S_quad_reconstr, 'fro').item()) 

    del S
    del S_ref
    del S_centered
    del U
    torch.cuda.empty_cache()  # Free up GPU memory

    # S = S.cpu()
    # S_ref = S_ref.cpu()
    # S_centered = S_centered.cpu()
    # U =  U.cpu()

    lin_constr_norms[:] = [x / S_cent_norm for x in lin_constr_norms]
    quad_constr_norms[:] = [x / S_cent_norm for x in quad_constr_norms]

    plt.scatter(np.arange(len(lin_constr_norms)), lin_constr_norms, label = 'Linear Manifold')
    plt.scatter(np.arange(len(quad_constr_norms)), quad_constr_norms, label = 'Quadratic Manifold')
    plt.legend()
    plt.xlabel('Reduced basis dimension r')
    plt.ylabel('Snapshot retained energy')
    plt.savefig(f'{plot_directory}snapshot_energy_spectrum')


