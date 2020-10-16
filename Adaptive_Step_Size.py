"""
Created on Fri Aug 07 11:47:39 2020

@author: Pranab JD

Description: -
        Consists of different adaptive step size methods
"""

import numpy as np

################################################################################################

def Higher_Order_Method_1(A_adv, m_adv, Method, p, Method_ref, u_sol, u, dt_inp, tol):
    """
    Parameters
    ----------
    p           		: Order of Method
    A_adv 		        : N x N advection matrix
    m_adv			    : Index of u for advection
    Method      		: Time integration scheme (function: Solution)
    Method_ref  		: Reference time integration scheme
    u_sol       		: Solution using desired integrator
    u            		: 1D vector u
    dt_inp      		: dt input
    tol         		: Maximum tolerance

    Returns
    ---------
    u_sol       		: Solution using desired integrator (same as input, if error < tol)
    u_ref       		: Reference solution (same as input, if error < tol)
    dt_inp          	: dt input
    dt_used				: dt used in this time step (= dt_inp, if error < tol)
    dt_new				: dt for next time step (= dt used, if error > tol)
    counts+its_ref_1    : Number of matrix-vector products (= its_ref_1, if error < tol)

    """

    ## Max. number of iters to achieve tolerance in a single time loop
    n_iters = 1000
    counts = 0              # Counter for matrix-vector products

    u_ref, its_ref_1 = Method_ref(A_adv, m_adv, u, dt_inp)

    ### Error estimate ###
    error = np.mean(abs(u_ref - u_sol))

    ### If error > tol, reduce dt till error < tol
    if error > tol:

        dt = dt_inp

        for mm in range(n_iters):

            ### Step size controller ###
            new_dt = dt * (tol/error)**(1/(p + 1))
            dt = 0.875 * new_dt          # Safety factor

            ## Re-calculate u_ref, u_sol, and error
            u_ref, its_ref_2 = Method_ref(A_adv, m_adv, u, dt)
            u_sol, u, its_method = Method(u, dt)
            error = np.mean(abs(u_ref - u_sol))

            ### Matrix-vector products
            counts = counts + its_method + its_ref_2

            if error <= tol:
                # print('Error within limits. dt accepted!! Error = ', error)
                dt_used = dt
                dt_new = dt
                break

            ## Error alert
            if mm == (n_iters - 1):
                print('Max iterations reached. Check parameters!!!')

    ## Increase/decrease dt for next time step
    else:

        dt_used = dt_inp

        ### Step size controller ###
        new_dt = dt_inp * (tol/error)**(1/(p + 1))
        dt_new = 0.875 * new_dt          # Safety factor

    return u_sol, u_ref, dt_inp, dt_used, dt_new, counts + its_ref_1

################################################################################################

def Higher_Order_Method_2(A_adv, m_adv, A_dif, m_dif, Method, p, Method_ref, u_sol, u, dt_inp, tol):
    """
    Parameters
    ----------
    p           		: Order of Method
    A_adv/A_dif 		: N x N advection and diffusion matrices
    m_adv/m_dif			: Index of u for advection/diffusion
    Method      		: Time integration scheme (function: Solution)
    Method_ref  		: Reference time integration scheme
    u_sol       		: Solution using desired integrator
    u            		: 1D vector u
    dt_inp      		: dt input
    tol         		: Maximum tolerance

    Returns
    ---------
    u_sol       		: Solution using desired integrator (same as input, if error < tol)
    u_ref       		: Reference solution (same as input, if error < tol)
    dt_inp          	: dt input
    dt_used				: dt used in this time step (= dt_inp, if error < tol)
    dt_new				: dt for next time step (= dt used, if error > tol)
    counts+its_ref_1    : Number of matrix-vector products (= its_ref_1, if error < tol)

    """

    ## Max. number of iters to achieve tolerance in a single time loop
    n_iters = 1000
    counts = 0              # Counter for matrix-vector products

    u_ref, its_ref_1 = Method_ref(A_adv, m_adv, A_dif, m_dif, u, dt_inp)

    ### Error estimate ###
    error = np.mean(abs(u_ref - u_sol))

    ### If error > tol, reduce dt till error < tol
    if error > tol:

        dt = dt_inp

        for mm in range(n_iters):

            ### Step size controller ###
            new_dt = dt * (tol/error)**(1/(p + 1))
            dt = 0.875 * new_dt          # Safety factor

            ## Re-calculate u_ref, u_sol, and error
            u_ref, its_ref_2 = Method_ref(A_adv, m_adv, A_dif, m_dif, u, dt)
            u_sol, u, its_method = Method(u, dt)
            error = np.mean(abs(u_ref - u_sol))

            ### Matrix-vector products
            counts = counts + its_method + its_ref_2

            if error <= tol:
                # print('Error within limits. dt accepted!! Error = ', error)
                dt_used = dt
                dt_new = dt
                break

            ## Error alert
            if mm == (n_iters - 1):
                print('Max iterations reached. Check parameters!!!')


    ## Increase/decrease dt for next time step
    else:

        dt_used = dt_inp

        ### Step size controller ###
        new_dt = dt_inp * (tol/error)**(1/(p + 1))
        dt_new = 0.875 * new_dt          # Safety factor

    return u_sol, u_ref, dt_inp, dt_used, dt_new, counts + its_ref_1

################################################################################################

def Richardson_Extrapolation_1(Method, p, u_sol, u, dt_inp, tol):
    """
    Parameters
    ----------
    Method      		: Time integration scheme (function: Solution)
    p           		: Order of Method
    u_sol       		: Solution using desired integrator
    u            		: 1D vector u (input)
    dt_inp      		: dt input
    tol         		: Maximum tolerance

    Returns
    ---------
    u_sol       		: Solution using desired integrator (same as input, if error < tol)
    u_ref       		: Reference solution (same as input, if error < tol)
    dt_inp          	: dt input
    dt_used				: dt used in this time step (= dt_inp, if error < tol)
    dt_new				: dt for next time step (= dt used, if error > tol)
    counts+its_ref_1    : Number of matrix-vector products (= its_ref_1, if error < tol)

    """

    ## Max. number of iters to achieve tolerance in a single time loop
    n_iters = 1000
    counts = 0              # Counter for matrix-vector products

    u_ref_1, u, its_ref_1 = Method(u, dt_inp/2)
    u_ref, u, its_ref_2 = Method(u_ref_1, dt_inp/2)

    ### Error estimate ###
    error = np.mean(abs(u_ref - u_sol))

    ### If error > tol, reduce dt till error < tol
    if error > tol:

        dt = dt_inp

        for mm in range(n_iters):

            ### Step size controller ###
            new_dt = dt * (tol/error)**(1/(p + 1))
            dt = 0.875 * new_dt          # Safety factor

            ## Re-calculate u_ref, u_sol, and error
            u_ref_1, u, its_ref_3 = Method(u, dt/2)
            u_ref, u, its_ref_4 = Method(u_ref_1, dt/2)
            u_sol, u, its_method = Method(u, dt)

            error = np.mean(abs(u_ref - u_sol))

            ### Matrix-vector products
            counts = counts + its_method + its_ref_3 + its_ref_4

            if error <= tol:
                # print('Error within limits. dt accepted!! Error = ', error)
                dt_used = dt
                dt_new = dt
                break

            ## Error alert
            if mm == (n_iters - 1):
                print('Max iterations reached. Check parameters!!!')

    ## Increase/decrease dt for next time step
    else:

        dt_used = dt_inp

        ### Step size controller ###
        new_dt = dt_inp * (tol/error)**(1/(p + 1))
        dt_new = 0.875 * new_dt          # Safety factor

    return u_sol, u_ref, dt_inp, dt_used, dt_new, counts + its_ref_1 + its_ref_2


################################################################################################

def Richardson_Extrapolation_2(Method, p, u_sol, u, dt_inp, tol):
    """
    Parameters
    ----------
    p           		: Order of Method
    A_adv/A_dif 		: N x N advection and diffusion matrices
    m_adv/m_dif			: Index of u for advection/diffusion
    Method      		: Time integration scheme (function: Solution)
    u_sol       		: Solution using desired integrator
    u            		: 1D vector u
    dt_inp      		: dt input
    tol         		: Maximum tolerance

    Returns
    ---------
    u_sol       		: Solution using desired integrator (same as input, if error < tol)
    u_ref       		: Reference solution (same as input, if error < tol)
    dt_inp          	: dt input
    dt_used				: dt used in this time step (= dt_inp, if error < tol)
    dt_new				: dt for next time step (= dt used, if error > tol)
    counts+its_ref_1    : Number of matrix-vector products (= its_ref_1, if error < tol)

    """

    ## Max. number of iters to achieve tolerance in a single time loop
    n_iters = 1000
    counts = 0              # Counter for matrix-vector products

    u_ref_1, u, its_ref_1 = Method(u, dt_inp/2)
    u_ref, u, its_ref_2 = Method(u_ref_1, dt_inp/2)

    ### Error estimate ###
    error = np.mean(abs(u_ref - u_sol))

    ### If error > tol, reduce dt till error < tol
    if error > tol:

        dt = dt_inp

        for mm in range(n_iters):

            ### Step size controller ###
            new_dt = dt * (tol/error)**(1/(p + 1))
            dt = 0.875 * new_dt          # Safety factor

            ## Re-calculate u_ref, u_sol, and error
            u_ref_1, u, its_ref_3 = Method(u, dt/2)
            u_ref, u, its_ref_4 = Method(u_ref_1, dt/2)
            u_sol, u, its_method = Method(u, dt)

            error = np.mean(abs(u_ref - u_sol))

            ### Matrix-vector products
            counts = counts + its_method + its_ref_3 + its_ref_4

            if error <= tol:
                # print('Error within limits. dt accepted!! Error = ', error)
                dt_used = dt
                dt_new = dt
                break

            ## Error alert
            if mm == (n_iters - 1):
                print('Max iterations reached. Check parameters!!!')

    ## Increase/decrease dt for next time step
    else:

        dt_used = dt_inp

        ### Step size controller ###
        new_dt = dt_inp * (tol/error)**(1/(p + 1))
        dt_new = 0.875 * new_dt          # Safety factor

    return u_sol, u_ref, dt_inp, dt_used, dt_new, counts + its_ref_1 + its_ref_2


################################################################################################

def Trad_Controller(Method, p, u_sol_3, u_sol_4, u, dt_inp, tol):

    error = np.mean(abs(u_sol_3 - u_sol_4))

    counts_3 = 0; counts_4 = 0; n_iters = 1000

    if error > tol:

        dt = dt_inp

        for mm in range(n_iters):

            ### Step size controller ###
            new_dt = dt * (tol/error)**(1/(p + 1))
            dt = 0.875 * new_dt          # Safety factor

            ## Re-calculate u_ref, u_sol, and error
            u_sol_3, u_sol_4, u, num_mv_sol_3_err, num_mv_sol_4_err = Method(u, dt)
            error = np.mean(abs(u_sol_3 - u_sol_4))

            ### Matrix-vector products
            counts_3 = counts_3 + num_mv_sol_3_err
            counts_4 = counts_4 + num_mv_sol_4_err

            if error <= tol:
                dt_used = dt
                dt_new = dt
                break

            ## Error alert
            if mm == (n_iters - 1):
                print('Max iterations reached. Check parameters!!!')

    ## Increase/decrease dt for next time step
    else:

        dt_used = dt_inp

        ### Step size controller ###
        new_dt = dt_inp * (tol/error)**(1/(p + 1))
        dt_new = 0.875 * new_dt          # Safety factor

    return u_sol_3, u_sol_4, dt_inp, dt_used, dt_new, counts_3, counts_4


################################################################################################

def Step_Size_Controller(count_mat_vec_n, dt_n, count_mat_vec_n_1, dt_n_1):

    cost_n = count_mat_vec_n/dt_n
    cost_n_1 = count_mat_vec_n_1/dt_n_1

    def Non_penalized():

        alpha = 0.65241444
        beta = 0.26862269
        lambd = 1.37412002
        delta = 0.64446017

        return alpha, beta, lambd, delta

    def Penalized():

        alpha = 1.19735982
        beta = 0.44611854
        lambd = 1.38440318
        delta = 0.73715227

        return alpha, beta, lambd, delta

    alpha, beta, lambd, delta = Non_penalized()
    # alpha, beta, lambd, delta = Penalized()

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
