import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ckron_tensor(arr):
    return torch.cat([arr[i] * arr[: i + 1] for i in range(arr.shape[0])], axis = 0)

def ckron_numpy(arr):
    return np.concatenate([arr[i] * arr[: i + 1] for i in range(arr.shape[0])], axis = 0)

def compute_Utrunc_Vbar_given_truncdim(S, S_ref, regularizer, trunc_dim): 
    S = torch.tensor(S, device = device) #1.
    S_ref = torch.tensor(S_ref, device = device) #2.
    S_centered = S - S_ref # 3.

    U, Sigma, _ = torch.svd(S_centered) # 4., 5.

    U_trunc = U[:,: trunc_dim]
    S_hat = U_trunc.T @ (S_centered) # 9.

    S_kron = ckron_tensor(S_hat) # 12.
    D_mat = S_kron.T # 13.
    Z_mat = S_centered - U_trunc @ S_hat # 14.
    
    # svd of D_mat
    DU_matrix, DS_matrix, DYt_matrix = torch.linalg.svd(D_mat, full_matrices = False) # 15, 16, 17
    # regularization
    DS_matrix_inv = DS_matrix / (DS_matrix **2 + regularizer **2) # 18
    # computing V_bar
    V_bar = Z_mat @ DU_matrix @ torch.diag(DS_matrix_inv) @ DYt_matrix #19

    U_trunc_cpu = U_trunc.cpu().numpy()
    V_bar_cpu = V_bar.cpu().numpy()

    del S, S_ref, S_centered, U, Sigma, U_trunc, S_hat, S_kron, D_mat
    del Z_mat, DU_matrix, DS_matrix, DYt_matrix, DS_matrix_inv, V_bar

    torch.cuda.empty_cache()  # Free up GPU memory

    return U_trunc_cpu, V_bar_cpu
