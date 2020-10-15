"""
Created on Wed Aug 19 16:17:29 2020

@author: Pranab JD

Description: -
        Contains several integrators for viscous Burgers'
        equation (A_Adv.u^m_adv + A_Diff.u^m_dif)

"""

from Leja_Interpolation import *

################################################################################################

### Reference Integrators (Explicit methods)

def RK4(A_adv, m_adv, A_dif, m_dif, u, dt):
    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    m_adv   : Index of u (u^m_adv); advection
    A_dif   : Diffusion matrix (A)
    m_dif   : Index of u (u^m_dif); diffusion
    u       : 1D vector u (input)
    dt      : dt

    Returns
    -------
    u_rk4 : 1D vector u (output) after time dt
    dt    : dt (unchanged)
    8     : # of matrix-vector products

    """
    
    k1 = dt * (A_adv.dot(u**m_adv) + A_dif.dot(u**m_dif))
    k2 = dt * (A_adv.dot((u + k1/2)**m_adv) + A_dif.dot((u + k1/2)**m_dif))
    k3 = dt * (A_adv.dot((u + k2/2)**m_adv) + A_dif.dot((u + k1/2)**m_dif))
    k4 = dt * (A_adv.dot((u + k3)**m_adv) + A_dif.dot((u + k1/2)**m_dif))
    
    ## Solution
    u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

    return u_rk4, 8

############## --------------------- ##############

def RKF5(A_adv, m_adv, A_dif, m_dif, u, dt):
    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    m_adv   : Index of u (u^m_adv); advection
    A_dif   : Diffusion matrix (A)
    m_dif   : Index of u (u^m_dif); diffusion
    u       : 1D vector u (input)
    dt      : dt

    Returns
    -------
    u_rkf5 : 1D vector u (Output) after time dt
    dt     : dt (unchanged)
    12     : # of matrix-vector products

    """

    k1 = dt * (A_adv.dot(u**m_adv) + A_dif.dot(u**m_dif))

    k2 = dt * (A_adv.dot((u + k1/4)**m_adv) + A_dif.dot((u + k1/4)**m_dif))

    k3 = dt * (A_adv.dot((u + 3./32.*k1 + 9./32.*k2)**m_adv) + A_dif.dot((u + 3./32.*k1 + 9./32.*k2)**m_dif))

    k4 = dt * (A_adv.dot((u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**m_adv) \
            + A_dif.dot((u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**m_dif))

    k5 = dt * (A_adv.dot((u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**m_adv) \
            + A_dif.dot((u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**m_dif))

    k6 = dt * (A_adv.dot((u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**m_adv) \
            + A_dif.dot((u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**m_dif))

    ### Solution
    u_rkf5 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

    return u_rkf5, 12

##############################################################################

### ETD ###

def ETD(A_adv, m_adv, A_dif, m_dif, u, dt, Leja_X, c, Gamma):
    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    m_adv   : Index of u (u^m_adv); advection
    A_dif   : Diffusion matrix (A)
    m_dif   : Index of u (u^m_dif); diffusion
    u       : 1D vector u (Input)
    dt      : dt
    Leja_X  : Leja points
    c, Gamma: Parameters for Leja extrapolation

    Returns
    -------
    u_etd       : 1D vector u (output) after time dt
    mat_vec_num : # of matrix-vector products

    """
    
    epsilon = 1e-7
    
    ### Matrix-vector function
    f_u = A_adv.dot(u**2) + A_dif.dot(u)

    # ### J(u) * u
    # Linear_u = (A_adv.dot((u + (epsilon * u))**2) + A_dif.dot((u + (epsilon * u))) - f_u)/epsilon
    # 
    # ### F(u) = f(u) - (J(u) * u)
    # Nonlin_u = f_u - Linear_u
    
    ############## --------------------- ##############
    
    # def Real_Leja():
    # 
    #     ### Solution
    #     u_lin, its_lin = real_Leja_exp(A_adv, A_dif, u, 2, 1, u, dt, Leja_X, c, Gamma)
    #     u_nl, its_nl = real_Leja_phi(A_adv, A_dif, u, 2, 1, Nonlin_u, dt, Leja_X, c, Gamma, phi_1)
    #     u_nl = u_nl * dt
    #     
    #     return u_lin, u_nl, its_lin + its_nl + 2
    
    ############## --------------------- ##############
    
    def Imag_Leja():
        
        ### Solution
        # u_lin, its_lin = imag_Leja_exp(A_adv, A_dif, u, 2, 1, u, dt, Leja_X, c, Gamma)
        u_nl, its_nl = imag_Leja_phi(u, f_u, dt, Leja_X, c, Gamma, phi_1, A_adv, m_adv, A_dif, m_dif)
        u_nl = u_nl * dt
        
        return u_nl, its_nl + 2
    
    ############## --------------------- ##############
    
    # u_lin, u_nl, mat_vec_num = Real_Leja()
    u_nl, mat_vec_num = Imag_Leja()
    
    ### ETD1 Solution ###
    u_etd = u_nl
    
    return u_etd, mat_vec_num

##############################################################################

### ETDRK2 ###
    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    m_adv   : Index of u (u^m_adv); advection
    A_dif   : Diffusion matrix (A)
    m_dif   : Index of u (u^m_dif); diffusion
    u       : 1D vector u (Input)
    dt      : dt
    Leja_X  : Leja points
    c, Gamma: Parameters for Leja extrapolation

    Returns
    -------
    u_etdrk2    : 1D vector u (output) after time dt
    mat_vec_num : # of matrix-vector products

    """

def ETDRK2(A_adv, m_adv, A_dif, m_dif, u, dt, Leja_X, c, Gamma):

    epsilon = 1e-12
    
    ############## --------------------- ##############
    
    ### Matrix-vector function
    f_u = A_adv.dot(u**2) + A_dif.dot(u)

    ### J(u) * u
    Linear_u = (A_adv.dot((u + (epsilon * u))**2) + A_dif.dot((u + (epsilon * u))) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u

    ############## --------------------- ##############

    def Real_Leja():
        
        ### Solution
        u_lin, its_lin = real_Leja_exp(A_adv, A_dif, u, 2, 1, u, dt, Leja_X, c, Gamma)
        u_nl_1, its_nl_1 = real_Leja_phi(A_adv, A_dif, u, 2, 1, Nonlin_u, dt, Leja_X, c, Gamma, phi_1)
        u_nl_1 = u_nl_1 * dt
        
        a_n = u_lin + u_nl_1

        ############## --------------------- ##############

        ### RK2 ###
        ### J(u) * a
        Linear_a = (A_adv.dot((u + (epsilon * a_n))**2) + A_dif.dot((u + (epsilon * a_n))) - f_u)/epsilon
        
        ### F(a) = f(a) - (J(u) * a)
        Nonlin_a = A_adv.dot(a_n**2) + A_dif.dot(a_n) - Linear_a
        
        ## Nonlinear Term
        u_nl_2, its_nl_2 = real_Leja_phi(A_adv, A_dif, u, 2, 1, (Nonlin_a - Nonlin_u), dt, Leja_X, c, Gamma, phi_2)
        u_nl_2 = u_nl_2 * dt
        
        return a_n, u_nl_2, its_lin + its_nl_1 + its_nl_2 + 8

    ############## --------------------- ##############

    def Imag_Leja():
        
        ### Solution
        u_lin, its_lin = imag_Leja_exp(A_adv, A_dif, u, 2, 1, u, dt, Leja_X, c, Gamma)
        u_nl_1, its_nl_1 = imag_Leja_phi(A_adv, A_dif, u, 2, 1, Nonlin_u, dt, Leja_X, c, Gamma, phi_1)
        u_nl_1 = u_nl_1 * dt
        
        a_n = u_lin + u_nl_1
        
        ############## --------------------- ##############
    
        ### RK2 ###
        ### J(u) * a
        Linear_a = (A_adv.dot((u + (epsilon * a_n))**2) + A_dif.dot((u + (epsilon * a_n))) - f_u)/epsilon
        
        ### F(a) = f(a) - (J(u) * a)
        Nonlin_a = A_adv.dot(a_n**2) + A_dif.dot(a_n) - Linear_a
        
        ## Nonlinear Term
        u_nl_2, its_nl_2 = imag_Leja_phi(A_adv, A_dif, u, 2, 1, (Nonlin_a - Nonlin_u), dt, Leja_X, c, Gamma, phi_2)
        u_nl_2 = u_nl_2 * dt
        
        return a_n, u_nl_2, its_lin + its_nl_1 + its_nl_2 + 8
    
    ############## --------------------- ##############
    
    # a_n, u_nl_2, mat_vec_num = Real_Leja()
    a_n, u_nl_2, mat_vec_num = Imag_Leja()
    
    ### ETDRK2 Solution ###
    u_etdrk2 = a_n + u_nl_2
    
    return u_etdrk2, mat_vec_num

##############################################################################

### EXPRB42 ###

def EXPRB42(A_adv, m_adv, A_dif, m_dif, u, dt, c, Gamma):
    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    m_adv   : Index of u (u^m_adv); advection
    A_dif   : Diffusion matrix (A)
    m_dif   : Index of u (u^m_dif); diffusion
    u       : 1D vector u (Input)
    dt      : dt
    c, Gamma: Parameters for Leja extrapolation

    Returns
    -------
    u_exprb42   : 1D vector u (output) after time dt
    mat_vec_num : # of matrix-vector products

    """
    
    epsilon = 1e-7
    
    ############## --------------------- ##############

    ### Matrix-vector function
    f_u = A_adv.dot(u**m_adv) + A_dif.dot(u**m_dif)

    ### J(u) * u
    Linear_u = (A_adv.dot((u + (epsilon * u))**m_adv) + A_dif.dot((u + (epsilon * u))**m_dif) - f_u)/epsilon
    
    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u
    
    ############## --------------------- ##############
    
    a_n_f, its_a = imag_Leja_phi(u, f_u, 3*dt/4, c, Gamma, phi_1, A_adv, m_adv, A_dif, m_dif)
    
    a_n = u + a_n_f * 3*dt/4
    
    ############## --------------------- ##############
    
    ### J(u) * a
    Linear_a = (A_adv.dot((u + (epsilon * a_n))**m_adv) + A_dif.dot((u + (epsilon * a_n))**m_dif) - f_u)/epsilon
    
    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A_adv.dot(a_n**m_adv) + A_dif.dot(a_n**m_dif) - Linear_a
    
    ############## --------------------- ##############

    u_1, its_1 = imag_Leja_phi(u, f_u, dt, c, Gamma, phi_1, A_adv, m_adv, A_dif, m_dif)
    u_nl_3, its_3 = imag_Leja_phi(u, (Nonlin_a - Nonlin_u), dt, c, Gamma, phi_3, A_adv, m_adv, A_dif, m_dif)
    
    u_exprb42 = u + (u_1 * dt) + (u_nl_3 * 32*dt/9)
    
    return u_exprb42, 8 + its_a + its_1 + its_3

##############################################################################

### EXPRB43

def EXPRB43(A_adv, m_adv, A_dif, m_dif, u, dt, c, Gamma, tol):
    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    m_adv   : Index of u (u^m_adv); advection
    A_dif   : Diffusion matrix (A)
    m_dif   : Index of u (u^m_dif); diffusion
    u       : 1D vector u (Input)
    dt      : dt
    c, Gamma: Parameters for Leja extrapolation

    Returns
    -------
    u_exprb43   : 1D vector u (output) after time dt (3rd and/or 4th order)
    mat_vec_num : # of matrix-vector products

    """
    
    epsilon = 1e-7
        
    ### Matrix-vector function
    f_u = A_adv.dot(u**m_adv) + A_dif.dot(u**m_dif)

    ### J(u) * u
    Linear_u = (A_adv.dot((u + (epsilon * u))**m_adv) + A_dif.dot((u + (epsilon * u))**m_dif) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u
    
    ############## --------------------- ##############
    
    a_n_f, its_a = imag_Leja_phi(u, f_u, dt/2, c, Gamma, phi_1, tol, A_adv, m_adv, A_dif, m_dif)
    b_n_f, its_b = imag_Leja_phi(u, f_u, dt, c, Gamma, phi_1, tol, A_adv, m_adv, A_dif, m_dif)
    
    a_n = u + a_n_f * dt/2
    b_n = u + b_n_f * dt
    
    ############# --------------------- ##############
    
    ### J(u) * a
    Linear_a = (A_adv.dot((u + (epsilon * a_n))**m_adv) + A_dif.dot((u + (epsilon * a_n))**m_dif) - f_u)/epsilon
    
    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A_adv.dot(a_n**m_adv) + A_dif.dot(a_n**m_dif) - Linear_a
    
    ############## --------------------- ##############
    
    ### J(u) * b
    Linear_b = (A_adv.dot((u + (epsilon * b_n))**m_adv) + A_dif.dot((u + (epsilon * b_n))**m_dif) - f_u)/epsilon
    
    ### F(b) = f(b) - (J(u) * b)
    Nonlin_b = A_adv.dot(b_n**m_adv) + A_dif.dot(b_n**m_dif) - Linear_b
    
    ############# --------------------- ##############
    
    u_1 = b_n_f
    u_nl_3, its_3 = imag_Leja_phi(u, (-14*Nonlin_u + 16*Nonlin_a - 2*Nonlin_b), dt, c, Gamma, phi_3, tol, A_adv, m_adv, A_dif, m_dif)
    u_nl_4, its_4 = imag_Leja_phi(u, (36*Nonlin_u - 48*Nonlin_a + 12*Nonlin_b), dt, c, Gamma, phi_4, tol, A_adv, m_adv, A_dif, m_dif)
    
    u_exprb3 = u + (u_1 * dt) + (u_nl_3 * dt)
    u_exprb4 = u + (u_1 * dt) + (u_nl_3 * dt) + (u_nl_4 * dt)
     
    return u_exprb3, 12 + its_a + its_b + its_3, u_exprb4, 12 + its_a + its_b + its_3 + its_4


##############################################################################