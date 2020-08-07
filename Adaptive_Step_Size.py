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
    method  : Time integration scheme (method should take in 'u' and 'dt' as input and return
              new 'u' and 'dt' as output)
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

    for mm in range(n_iters):

        u_ref_1 = method(u, dt/2)[0]
        u_ref = method(u_ref_1, dt/2)[0]

        u_sol, dt = method(u, dt)

        ### Error estimate ###
        error = np.mean(abs(u_ref - u_sol))

        ### Step size controller ###
        new_dt = dt * (tol/error)**(1/(p + 1))
        dt = 0.8 * new_dt          # Safety factor

        if error <= tol:
            # print('Error within limits. dt accepted!! Error = ', error)
            break

        ## Error alert
        if mm == (n_iters - 1):
            print('Max iterations reached. Check parameters!!!')

    return u_sol, u_ref, error, dt


################################################################################################

def Higher_Order_Method(method_ref, method, p, u, dt, tol):
    """
    Parameters
    ----------
    method_ref  : Higher order scheme as reference (method should take in 'u' and 'dt' as input
                  and return new 'u' and 'dt' as output)
    method      : Time integration scheme (method should take in 'u' and 'dt' as input and
                  return new 'u' and 'dt' as output)
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

    for mm in range(n_iters):

        u_ref = method_ref(u, dt)[0]

        u_sol, dt = method(u, dt)

        ### Error estimate ###
        error = np.mean(abs(u_ref - u_sol))

        ### Step size controller ###
        new_dt = dt * (tol/error)**(1/(p + 1))
        dt = 0.8 * new_dt          # Safety factor

        if error <= tol:
            print('Error within limits. dt accepted!! Error = ', error)
            break

        ## Error alert
        if mm == (n_iters - 1):
            print('Max iterations reached. Check parameters!!!')

    return u_sol, u_ref, error, dt


################################################################################################
