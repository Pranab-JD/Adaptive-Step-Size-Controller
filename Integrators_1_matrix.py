"""
Created on Wed Sep 04 16:15:14 2020

@author: Pranab JD

Description: -
        Contains several integrators for single
        matrix equations (A.u^m)

"""

from Leja_Interpolation import *

################################################################################################

### Explicit Integrators ###

def RK2(A, m, u, dt):
    """
    Parameters
    ----------
    A       : Matrix (A)
    m       : Index of u (u^m)
    u       : 1D vector u (input)
    dt      : dt

    Returns
    -------
    u_rk2   : 1D vector u (output) after time dt (2nd order)
    2       : # of matrix-vector products

    """

    k1 = dt * (A.dot(u**m))
    k2 = dt * (A.dot((u + k1)**m))

    ## Solution
    u_rk2 = u + 1./2. * (k1 + k2)

    return u_rk2, 2

##############################################################################

def RK4(A, m, u, dt):
    """
    Parameters
    ----------
    A       : Matrix (A)
    m       : Index of u (u^m)
    u       : 1D vector u (input)
    dt      : dt

    Returns
    -------
    u_rk4   : 1D vector u (output) after time dt (4th order)
    2       : # of matrix-vector products

    """

    k1 = dt * (A.dot(u**m))
    k2 = dt * (A.dot((u + k1/2)**m))
    k3 = dt * (A.dot((u + k2/2)**m))
    k4 = dt * (A.dot((u + k3)**m))

    ## Solution
    u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

    return u_rk4, 4

##############################################################################

def RKF45(A, m, u, dt):
    """
    Parameters
    ----------
    A       : Matrix (A)
    m       : Index of u (u^m)
    u       : 1D vector u (input)
    dt      : dt

    Returns
    -------
    u_rkf4  : 1D vector u (output) after time dt (4th order)
    5       : # of matrix-vector products for u_rkf4
    u_rkf5  : 1D vector u (output) after time dt (5th order)
    6       : # of matrix-vector products for u_rkf5

    """

    k1 = dt * (A.dot(u**m))
    k2 = dt * (A.dot((u + k1/4)**m))
    k3 = dt * (A.dot((u + 3./32.*k1 + 9./32.*k2)**m))
    k4 = dt * (A.dot((u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**m))
    k5 = dt * (A.dot((u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**m))
    k6 = dt * (A.dot((u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**m))

    ### Solution
    u_rkf4 = u + (25./216.*k1 + 1408./2565.*k3 + 2197./4101.*k4 - 1./5.*k5)
    u_rkf5 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

    return u_rkf4, 5, u_rkf5, 6

################################################################################################

### Exponential Integrators ###

def ETD(A_adv, m_adv, u, dt, c, Gamma, Real_Imag_Leja):
    """
    Parameters
    ----------
    A               : Matrix (A)
    m               : Index of u (u^m)
    u               : 1D vector u (Input)
    dt              : dt
    c, Gamma        : Parameters for Leja extrapolation
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_etd           : 1D vector u (output) after time dt (2nd order)
    mat_vec_num     : # of matrix-vector products

    """

    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    ### RHS of PDE at u
    f_u = A_adv.dot(u**m_adv)
    
    u_sol, mat_vec_num = Leja_phi(u, f_u, dt, c, Gamma, phi_1, A_adv, m_adv)

    ### ETD Solution
    u_etd = u + (u_sol * dt)

    return u_etd, mat_vec_num

##############################################################################

def EXPRB42(A, m, u, dt, c, Gamma, Real_Imag_Leja):
    """
    Parameters
    ----------
    A               : Matrix (A)
    m               : Index of u (u^m)
    u               : 1D vector u (Input)
    dt              : dt
    c, Gamma        : Parameters for Leja extrapolation
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_exprb43       : 1D vector u (output) after time dt (4th order)
    mat_vec_num     : # of matrix-vector products

    """
    
    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    epsilon = 1e-7

    ### RHS of PDE at u
    f_u = A.dot(u**m)

    ############## --------------------- ##############

    ### Internal stage 1
    a_n_f, its_a = Leja_phi(u, f_u, 3*dt/4, c, Gamma, phi_1, A, m)
    a_n = u + (a_n_f * 3*dt/4)

    ############## --------------------- ##############
    
    ### J(u) * u
    Linear_u = (A.dot((u + (epsilon * u))**m) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u

    ### J(u) * a
    Linear_a = (A.dot((u + (epsilon * a_n))**m) - f_u)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A.dot(a_n**m) - Linear_a

    ############## --------------------- ##############

    u_1, its_1 = Leja_phi(u, f_u, dt, c, Gamma, phi_1, A, m)
    u_nl_3, its_3 = Leja_phi(u, (Nonlin_a - Nonlin_u), dt, c, Gamma, phi_3, A, m)

    ### 4th order solution
    u_exprb42 = u + (u_1 * dt) + (u_nl_3 * 32*dt/9)

    mat_vec_num = 4 + its_a + its_1 + its_3

    return u_exprb42, mat_vec_num

##############################################################################

def EXPRB32(A, m, u, dt, c, Gamma, Real_Imag_Leja):
    """
    Parameters
    ----------
    A               : Matrix (A)
    m               : Index of u (u^m)
    u               : 1D vector u (Input)
    dt              : dt
    c, Gamma        : Parameters for Leja extrapolation
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_exprb2        : 1D vector u (output) after time dt (2nd order)
    mat_vec_num_2   : # of matrix-vector products for 2nd order solution
    u_exprb3        : 1D vector u (output) after time dt (3rd order)
    mat_vec_num_3   : # of matrix-vector products for 3rd order solution

    """

    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    epsilon = 1e-7

    ### RHS of PDE at u
    f_u = A.dot(u**m)

    ############## --------------------- ##############

    ### Internal stage 1 (2nd order solution)
    a_n_f, its_a = Leja_phi(u, f_u, dt, c, Gamma, phi_1, A, m)
    a_n = u + (a_n_f * dt)

    u_exprb2 = a_n; mat_vec_num_2 = 1 + its_a

    ############## --------------------- ##############

    ### J(u) * u
    Linear_u = (A.dot((u + (epsilon * u))**m) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = A.dot(u**m) - Linear_u

    ### J(u) * a
    Linear_a = (A.dot((u + (epsilon * a_n))**m) - f_u)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A.dot(a_n**m) - Linear_a

    ############## --------------------- ##############

    ### 3rd order solution
    u_3, its_3 = Leja_phi(u, 2*(Nonlin_a - Nonlin_u), dt, c, Gamma, phi_3, A, m)
    u_exprb3 = u_exprb2 + (u_3 * dt)

    mat_vec_num_3 = 5 + its_a + its_3

    return u_exprb2, mat_vec_num_2, u_exprb3, mat_vec_num_3

##############################################################################

def EXPRB43(A, m, u, dt, c, Gamma, Real_Imag_Leja):
    """
    Parameters
    ----------
    A               : Matrix (A)
    m               : Index of u (u^m)
    u               : 1D vector u (Input)
    dt              : dt
    c, Gamma        : Parameters for Leja extrapolation
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_exprb3        : 1D vector u (output) after time dt (3rd order)
    mat_vec_num_3   : # of matrix-vector products for 3rd order solution
    u_exprb4        : 1D vector u (output) after time dt (4th order)
    mat_vec_num_4   : # of matrix-vector products for 4th order solution

    """

    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    epsilon = 1e-7

    ### RHS of PDE at u
    f_u = A.dot(u**m)

    ############## --------------------- ##############

    ### Internal stage 1
    a_n_f, its_a = Leja_phi(u, f_u, dt/2, c, Gamma, phi_1, A, m)
    a_n = u + a_n_f * dt/2

    ############## --------------------- ##############

    ### J(u) * u
    Linear_u = (A.dot((u + (epsilon * u))**m) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u

    ### J(u) * a
    Linear_a = (A.dot((u + (epsilon * a_n))**m) - f_u)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A.dot(a_n**m) - Linear_a

    ############# --------------------- ##############

    ### Internal stage 2
    b_n_f, its_b_1 = Leja_phi(u, f_u, dt, c, Gamma, phi_1, A, m)
    b_n_nl, its_b_2 = Leja_phi(u, (Nonlin_a - Nonlin_u), dt, c, Gamma, phi_1, A, m)

    b_n = u + (b_n_f * dt) + (b_n_nl * dt)

    ############# --------------------- ##############

    ### J(u) * b
    Linear_b = (A.dot((u + (epsilon * b_n))**m) - f_u)/epsilon

    ### F(b) = f(b) - (J(u) * b)
    Nonlin_b = A.dot(b_n**m) - Linear_b

    ############# --------------------- ##############

    ### 3rd and 4th order solutions
    u_1 = b_n_f
    u_nl_3, its_3 = Leja_phi(u, (-14*Nonlin_u + 16*Nonlin_a - 2*Nonlin_b), dt, c, Gamma, phi_3, A, m)
    u_nl_4, its_4 = Leja_phi(u, (36*Nonlin_u - 48*Nonlin_a + 12*Nonlin_b), dt, c, Gamma, phi_4, A, m)

    u_exprb3 = u + (u_1 * dt) + (u_nl_3 * dt)
    u_exprb4 = u_exprb3 + (u_nl_4 * dt)

    mat_vec_num_3 = 6 + its_a + its_b_1 + its_b_2 + its_3
    mat_vec_num_4 = 6 + its_a + its_b_1 + its_b_2 + its_3 + its_4

    return u_exprb3, mat_vec_num_3, u_exprb4, mat_vec_num_4

##############################################################################