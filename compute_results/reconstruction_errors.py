import torch
import numpy as np
from utils.tools import ckron


# convert all numpy arrays to torch tensors for faster computation on gpu.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_linear_quadratic_reconstruction_error(S: np.array, threshold : float, regularizer: float) -> float:
    '''
    Inputs: 
    S (snapshot matrix)
    threshold: value to decide number of SVD modes to retain (described below).
    regularizer: value to be used in the computation of V_bar in quadratic reconstruction (described below). 


    S - S_ref ~ V S_hat where,

    S_ref is the mean centered column (i.e. across time) of the snapshot matrix S.
    
    V S_hat are obtained from the SVD of S - S_ref, using a 'threshold' for the
    number of retained modes, as a ratio of cumulative energy:
    \sum_[j = 1 to r]{\sigma^2}/  \sum_[ all ]{sigma^2} > thershold,
    'r' modes are retained.
    _____________________________________________________________________________

    S  - S_ref ~ VS_hat
    S_lin =  V_Shat
    linear_reconstruction_error = ||S - S_ref - S_lin||_F/ ||S - S_ref||_F    
    _____________________________________________________________________________
    
    S - S_ref ~ VS_hat + V_bar[S_hat \ckron S_hat]
    S_quad = VS_hat + V_bar[S_hat \ckron S_hat]
    quadratic_reconstruction_error = ||S - S_quad||_F/ ||S||_F    

    \ckron preserves only the r(r+1)/2 unique values in the r^2
    values in the Kronecker product of an 'r' dimensional array.

    regularizer is used in the computation of 
    V_bar = argmin ||(S - S_ref - VS_hat - V_bar[S_hat \ckron S_hat]||_F^{2} + ||\lambda {V_bar}||_F^{2}  

    See: https://willcox-research-group.github.io/rom-operator-inference-Python3/source/api/_autosummaries/opinf.lstsq.L2Solver.html#opinf.lstsq.L2Solver
    and how they compute V_bar: https://willcox-research-group.github.io/rom-operator-inference-Python3/_modules/opinf/lstsq/_tikhonov.html#L2Solver
    
    The computation steps in the above (second link, (fit and solve method)) are replicated in the computation of V_bar below.
    _____________________________________________________________________________

    All of the SVD operations are performed on torch tensors so they can be performed on GPU.
    '''

    S_ref = np.mean(S , axis = 1).reshape(-1, 1).repeat(S.shape[1], axis = 1) # (x, t)
    S = torch.tensor(S, device = device)
    S_ref = torch.tensor(S_ref, device = device)
    S_centered = S - S_ref
    U, Sigma, _ = torch.svd(S_centered)

    energy = Sigma**2 / torch.sum(Sigma**2)
    cumulative_energy = torch.cumsum(energy, dim = 0)

    index_threshold = np.argmax(cumulative_energy.cpu() >= threshold) + 1  # +1 for 1-based index

    U_trunc = U[:, : index_threshold]
    S_hat = U_trunc.T @ (S_centered)
    S_linear_reconstr = U_trunc @ S_hat
    linear_reconstr_err = torch.linalg.norm(S_centered - S_linear_reconstr, 'fro')/torch.linalg.norm(S_centered, 'fro')
    
    # quadratic reconstruction error:
    S_kron = ckron(S_hat)
    D_mat = S_kron.T
    Z_mat = S_centered - U_trunc @ S_hat
    
    # svd of D_mat
    DU_matrix, DS_matrix, DYt_matrix = torch.linalg.svd(D_mat, full_matrices = False)
    # regularization
    DS_matrix_inv = DS_matrix / (DS_matrix **2 + regularizer **2)
    # computing V_bar
    V_bar = Z_mat @ DU_matrix @ torch.diag(DS_matrix_inv) @ DYt_matrix

    S_quad_reconstr = S_linear_reconstr + V_bar @ S_kron

    quad_reconstr_err = torch.linalg.norm(S_centered - S_quad_reconstr, 'fro')/torch.linalg.norm(S_centered, 'fro')

    return linear_reconstr_err.item(), quad_reconstr_err.item()

