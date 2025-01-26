import scipy
from utils.tools import ckron_numpy

def full_order_solve(A, initial_condition, time_domain):
    '''
    Solve ivp to move the initial condition forward in time subject 
    to spatial and temporal domain.
    ds/dt = A@s
    '''
    return scipy.integrate.solve_ivp(
        fun=lambda t, s: A @ s,
        t_span=[time_domain[0], time_domain[-1]],
        y0=initial_condition,
        t_eval=time_domain,
        method="BDF",
    ).y


def linear_rom_solve(c_hat, A_hat, initial_condition, time_domain):
    '''
    Solve ivp to move the initial condition forward in time subject 
    to spatial and temporal domain.
    ds'/dt =  c' + A's'
    '''
    return scipy.integrate.solve_ivp(
        fun=lambda t, s: c_hat + A_hat @ s,
        t_span=[time_domain[0], time_domain[-1]],
        y0=initial_condition,
        t_eval=time_domain,
        method="BDF",
    ).y

def quad_rom_solve(c_hat, A_hat, H_hat, initial_condition, time_domain):
    '''
    Solve ivp to move the initial condition forward in time subject 
    to spatial and temporal domain.
    ds'/dt = c' + A's' + H'[s' \ckron s']
    '''
    return scipy.integrate.solve_ivp(
        fun=lambda t, s: c_hat + A_hat @ s + H_hat @ ckron_numpy(s),
        t_span=[time_domain[0], time_domain[-1]],
        y0=initial_condition,
        t_eval=time_domain,
        method="BDF",
    ).y

