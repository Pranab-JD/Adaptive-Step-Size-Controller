"""
Created on Fri Aug 07 11:47:39 2020

@author: Pranab JD

Description: -
        Consists of different adaptive step size methods
"""


import numpy as np

################################################################################################

def Traditional_Controller(Method, u, dt_inp, p, error, tol):
    """
    Parameters
    ----------
    Method              : Solution using desired integration scheme (u_sol, u_ref, u, num_mv)
    u                   : 1D vector u
    dt_inp              : dt input
    p                   : Order of Method
    error               : Initial error (> tol)
    tol                 : Maximum tolerance

    Returns
    ---------
    u_sol               : Solution using desired integrator
    u_ref               : Reference solution (same as input, if error < tol)
    dt_inp              : dt input
    dt_used             : dt used in this time step
    dt_new              : dt for next time step
    counts              : Number of matrix-vector products

    """

    ## Max. number of iters to achieve tolerance in a single time loop
    n_iters = 100
    counts = 0              # Counter for matrix-vector products

    dt = dt_inp

    for mm in range(n_iters):

        # print('Step size Rejected!!! Error =', error)

        ### Traditional step size controller ###
        new_dt = dt * (tol/error)**(1/(p + 1))
        dt = 0.8 * new_dt          # Safety factor

        ## Re-calculate u_ref, u_sol, and error
        u_sol, u_ref, u, its_method = Method(u, dt)
        error = np.mean(abs(u_ref - u_sol))

        ## Count of matrix-vector products
        counts = counts + its_method

        if error <= tol:
            # print('Error within limits. dt accepted!! Error = ', error)
            dt_used = dt
            # dt_new = dt
            
            ## For Cost_Controller_3N, no 2 step sizes can be the same!!
            new_dt = dt * (tol/error)**(1/(p + 1))
            dt_new = 0.8 * new_dt          # Safety factor
            
            break

        ## Error alert
        if mm == (n_iters - 1):
            print('Max iterations reached. Check parameters!!!')

    return u_sol, u_ref, dt_inp, dt_used, dt_new, counts

################################################################################################

def Cost_Controller_2N(cost_n, dt_n, cost_n_1, dt_n_1, Pen_Nonpen):
    """
    Parameters
    ----------
    cost_n        : Cost at time step 'n' using dt_n
    dt_n          : dt at time step 'n'
    cost_n_1      : Cost at time step 'n-1' usinf dt_n_1
    dt_n_1        : dt at time step 'n-1'

    Returns
    ---------
    dt            : dt (optimised for cost control)

    """

    def Non_penalized():

        alpha = 0.65241444
        beta  = 0.26862269
        lambd = 1.37412002
        delta = 0.64446017

        return alpha, beta, lambd, delta

    def Penalized():

        alpha = 1.19735982
        beta  = 0.44611854
        lambd = 1.38440318
        delta = 0.73715227

        return alpha, beta, lambd, delta

    if Pen_Nonpen == 0:
        alpha, beta, lambd, delta = Non_penalized()
    elif Pen_Nonpen == 1:
        alpha, beta, lambd, delta = Penalized()
    else:
        print('Error!! Check controller')

    Del = (np.log(cost_n) - np.log(cost_n_1))/(np.log(dt_n) - np.log(dt_n_1))

    s = np.exp(-alpha * np.tanh(beta * Del))

    if 1 <= s < lambd:
        dt = dt_n * lambd

    elif delta <= s < 1:
        dt = dt_n * delta

    else:
        dt = dt_n * s

    return dt

################################################################################################

