import scipy
import numpy as np
import opinf
import torch
from compute_results.solvers import full_order_solve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def compute_S(S_init: np.array , n_spt: int, n_time: int, c_val: float) -> np.array:
    
    '''
    Compute the snapshot matrix S (n_spt, n_time)
    for ds/dt = -c ds/dx for a given S(x,0) = S_init
    '''

    x_vals = np.linspace(0, 1, n_spt)
    t_vals = np.linspace(0,0.1,n_time)

    dx = x_vals[1] - x_vals[0]
    diags = np.array([1, 0, -1]) *  ( -c_val/(2*dx) )
    A = scipy.sparse.diags(diags, [-1, 0, 1], (n_spt, n_spt))
    with opinf.utils.TimedBlock("Full-order solve"):
        S = full_order_solve(A, S_init, t_vals) # (x, t)

    S_ref = np.mean(S , axis = 1).reshape(-1, 1).repeat(S.shape[1], axis = 1) # (x, t)
    S_centered = S - S_ref

    S_centered = torch.tensor(S_centered, device = device)

    U, Sigma, _ = torch.svd(S_centered)
    
    S_centered_cpu = S_centered.cpu().numpy()

    del S_centered
    del U
    del Sigma
    torch.cuda.empty_cache()  # Free up GPU memory


    results = {
        'S': S,
        'S_ref' : S,
        'S_centered': S_centered_cpu,
        'FOM_A': A
    } 


    return results

