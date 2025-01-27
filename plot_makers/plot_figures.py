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
    Sigma /= Sigma.max()

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

    ticks = np.arange(0, len(Sigma)+1, 10)
    labels = ticks
    plt.xticks(ticks=ticks, labels=np.round(labels, 2)) 
    plt.xlim(0, len(Sigma) )

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

    S_cent_norm = torch.linalg.norm(S_centered, 'fro').item()**2
    lin_constr_norms, quad_constr_norms = [], []
    
    for r_val in range(1,S_centered.shape[1]):
        U_trunc = U[: , : r_val]
        S_hat = U_trunc.T @ S_centered
        lin_constr = U_trunc @ S_hat # VS_hat
        lin_constr_norms.append(torch.linalg.norm(lin_constr, 'fro').item()**2) 

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
        quad_constr_norms.append(torch.linalg.norm(S_quad_reconstr, 'fro').item()**2)  # VS_hat + V_bar[S_hat \ckron S_hat]

    del S
    del S_ref
    del S_centered
    del U
    torch.cuda.empty_cache() 

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
    plt.close()

def plot_model_results(model_results, xvals_global, time_vals_global, t_instances, plot_directory):
    fom_results = model_results['FOM_reconstruction']
    linear_rom_results = model_results['Linear_ROM_reconstruction']
    quadratic_rom_results = model_results['Quadratic_ROM_reconstruction']

    indices = []
    for i in range(len(t_instances)):
        index = np.argmin( abs(time_vals_global - t_instances[i]* np.ones(len(time_vals_global))))
        indices.append(index)

    for index in indices:
        time_val = time_vals_global[index] 
        plt.scatter(np.arange(len(fom_results[:,0])), fom_results[:,index], label = 'Exact', color = 'k')
        plt.plot( linear_rom_results[:,index], label = 'Linear ROM', color = 'b')
        plt.plot( quadratic_rom_results[:,index], label = 'Quadratic ROM', color = 'r')
        plt.legend()
        
        ticks = np.linspace(0, len(linear_rom_results[:, index]) - 1, 6)  # Positions
        labels = np.linspace(0, 1, 6)  # Labels (0, 0.2, 0.4, 0.6, 0.8, 1)
        plt.xticks(ticks=ticks, labels=np.round(labels, 2))  # Round labels for neatness
        plt.xlim(0, len(linear_rom_results[:,index]) - 1)

        plt.xlabel('x-coordinate')
        plt.ylabel('solution s(x, t)')
        plt.title(f'Values at time = {time_val}')
        plt.savefig(f'{plot_directory}FOM_ROM_test_{time_val}.png')
        plt.close()
        
    return None


def plot_comparison_FOM_ROM(model_results, x_vals, t_vals, plot_directory):
    S_fom = model_results['FOM_reconstruction']
    S_linear_rom = model_results['Linear_ROM_reconstruction']
    S_quadratic_rom = model_results['Quadratic_ROM_reconstruction']

    compare_folder = f'FOM_ROM_compare_{x_vals[0]}-{x_vals[-1]}_{t_vals[0]}-{t_vals[-1]}/'
    compare_directory = plot_directory + compare_folder

    if not os.path.exists(compare_directory):
        os.makedirs(compare_directory)

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (15,5))
    fig.subplots_adjust(wspace=0.5) 

    fig.suptitle('Comparison of time evolution of FOM and ROMs.')
    pc1 = ax1.pcolormesh(x_vals, t_vals, S_fom.T, shading='auto', cmap='viridis')  # Transpose data for correct orientation
    ax1.set_xlabel("FOM")
    ax1.set_ylabel("Time (t-values)")

    pc2 = ax2.pcolormesh(x_vals, t_vals, S_linear_rom.T, shading='auto', cmap='viridis')  # Transpose data for correct orientation
    ax2.set_xlabel("Linear ROM")

    pc3 = ax3.pcolormesh(x_vals, t_vals, S_quadratic_rom.T, shading='auto', cmap='viridis')  # Transpose data for correct orientation
    ax3.set_xlabel("Quadratic ROM")

    fig.colorbar(pc1, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.1)
    fig.colorbar(pc2, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.1)
    fig.colorbar(pc3, ax=ax3, orientation='horizontal', fraction=0.046, pad=0.1)


    plt.savefig(f'{compare_directory}compare_time_evolution.png')


