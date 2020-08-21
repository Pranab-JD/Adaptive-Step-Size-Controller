"""
Created on Fri Aug  7 11:47:39 2020

@author: Pranab JD

Description: -
        Consists of different adpative step size methods
"""

import numpy as np

################################################################################################

def Richardson_Extrapolation(method, p, u, dt, tol):
    """
    Parameters
    ----------
    method  : Time integration scheme (method should take in 'u', 'dt', and counts as input and return
              new 'u', 'dt', and counts as output)
    p       : Order of method
    u       : Vector u
    dt      : Given dt
    tol     : Maximum tolerance

    Returns
    ---------
    u_sol   : Solution (u)
    u_ref   : Reference (u)
    error   : Mean error between u_sol and u_ref (<= tol)
    dt      : New dt

    """

    ## Max. number of iters to achieve tolerance in a single time loop
    n_iters = 1000
    counts = 0              # Counter for matrix-vector products 

    for mm in range(n_iters):

        u_ref_1, dt_1, c1 = method(u, dt/2)
        u_ref, dt_2, c2 = method(u_ref_1, dt/2)

        u_sol, dt, c3 = method(u, dt)

        ### Error estimate ###
        error = np.mean(abs(u_ref - u_sol))
        
        ### Matrix-vector products
        counts = counts + c1 + c2 + c3

        ### Step size controller ###
        new_dt = dt * (tol/error)**(1/(p + 1))
        dt = 0.8 * new_dt          # Safety factor

        if error <= tol:
            # print('Error within limits. dt accepted!! Error = ', error)
            break

        ## Error alert
        if mm == (n_iters - 1):
            print('Max iterations reached. Check parameters!!!')

    return u_sol, u_ref, error, dt, counts


################################################################################################

def Higher_Order_Method(method_ref, method, p, u, dt, tol):
    """
    Parameters
    ----------
    method_ref  : Higher order scheme as reference (method should take in 'u', 'dt', and
                    counts as input and return new 'u', 'dt', and counts as output)
    method      : Time integration scheme (method should take in 'u', 'dt', and counts
                  as input and return new 'u', 'dt', and counts as output)
    p           : Order of method
    u           : Vector u
    dt          : Given dt
    tol         : Maximum tolerance

    Returns
    ---------
    u_sol       : Solution (u)
    u_ref       : Reference (u)
    error       : Mean error between u_sol and u_ref (<= tol)
    dt          : New dt

    """

    ## Max. number of iters to achieve tolerance in a single time loop
    n_iters = 1000
    counts = 0              # Counter for matrix-vector products 

    for mm in range(n_iters):

        u_ref, dt_1, c1 = method_ref(u, dt)

        u_sol, dt, c2 = method(u, dt)

        ### Error estimate ###
        error = np.mean(abs(u_ref - u_sol))
        
        ### Matrix-vector products
        counts = counts + c1 + c2

        ### Step size controller ###
        new_dt = dt * (tol/error)**(1/(p + 1))
        dt = 0.8 * new_dt          # Safety factor

        if error <= tol:
            # print('Error within limits. dt accepted!! Error = ', error)
            break

        ## Error alert
        if mm == (n_iters - 1):
            print('Max iterations reached. Check parameters!!!')

    return u_sol, u_ref, error, dt, counts


################################################################################################
