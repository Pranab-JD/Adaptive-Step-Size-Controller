"""
Created on Wed Aug 19 16:17:29 2020

@author: Pranab JD

Description: -
        Contains several integrators for viscous Burgers'
        equation (A_Adv.u^2 + A_Diff.u)

"""

from Leja_Header import *

################################################################################################

### Reference Integrators (Explicit methods)

def RK4(A_adv, u, dt):
    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    A_dif   : Diffusion matrix (A)
    u       : 1D vector u (Input)
    dt      : dt

    Returns
    -------
    u_rk4 : 1D vector u (Output) after time dt
    dt    : dt (unchanged)
    8     : # of matrix-vector products

    """

    k1 = dt * A_adv.dot(u**2)
    k2 = dt * A_adv.dot((u + k1/2)**2)
    k3 = dt * A_adv.dot((u + k2/2)**2)
    k4 = dt * A_adv.dot((u + k3)**2)

    ## Solution
    u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

    return u_rk4, dt, 8

############## --------------------- ##############

def RKF5(A_adv, A_dif, u, dt):
    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    A_dif   : Diffusion matrix (A)
    u       : 1D vector u (Input)
    dt      : dt

    Returns
    -------
    u_rkf5 : 1D vector u (Output) after time dt
    dt     : dt (unchanged)
    12     : # of matrix-vector products

    """

    k1 = dt * (A_adv.dot(u**2) + A_dif.dot(u))

    k2 = dt * (A_adv.dot((u + k1/4)**2) + A_dif.dot(u + k1/4))

    k3 = dt * (A_adv.dot((u + 3./32.*k1 + 9./32.*k2)**2) + A_dif.dot(u + 3./32.*k1 + 9./32.*k2))

    k4 = dt * (A_adv.dot((u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**2) \
            + A_dif.dot(u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3))

    k5 = dt * (A_adv.dot((u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**2) \
            + A_dif.dot(u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4))

    k6 = dt * (A_adv.dot((u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**2) \
            + A_dif.dot(u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5))

    ### Solution
    u_rkf5 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

    return u_rkf5, dt, 12

##############################################################################

### ETD

def ETD(A, u, dt, Leja_X, c, Gamma):

    epsilon = 1e-7

    ############## --------------------- ##############

    def Real_Leja():

        ### Matrix-vector product
        A_dot_u_1 = A.dot(u**2)

        ### J(u) * u
        Linear_u = (A.dot((u + (epsilon * u))**2) - A_dot_u_1)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u = A_dot_u_1 - Linear_u

        ### Solution
        u_lin, its = real_Leja_exp(A, u, 2, dt, Leja_X, c, Gamma)
        u_nl = real_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt

        return u_lin, u_nl, its + 2

    ############## --------------------- ##############

    def Imag_Leja():

        ### Matrix-vector product
        A_dot_u_1 = A.dot(u**2)

        ### J(u) * u
        Linear_u = (A.dot((u + (epsilon * u))**2) - A_dot_u_1)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u = A_dot_u_1 - Linear_u

        ### Solution
        u_lin, its = imag_Leja_exp(A, u, 2, dt, Leja_X, c, Gamma)
        u_nl = imag_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt

        return u_lin, u_nl, its + 2

    ############## --------------------- ##############

    u_lin, u_nl, mat_vec_num = Real_Leja()
    # u_lin, u_nl, mat_vec_num = Imag_Leja()

    ### ETD1 Solution ###
    u_etd = u_lin + u_nl

    return u_etd, mat_vec_num

##############################################################################

### ETDRK2

def ETDRK2(A, u, dt, Leja_X, c, Gamma):

    epsilon = 1e-7

    ############## --------------------- ##############

    def Real_Leja():

        ### ETD1 ###
        A_dot_u_1 = A.dot(u**2)

        ### J(u) * u
        Linear_u = (A.dot((u + (epsilon * u))**2) - A_dot_u_1)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u = A_dot_u_1 - Linear_u

        ### ETD1 Solution
        u_lin_adv, mat_vec_num = real_Leja_exp(A, u, 2, dt, Leja_X, c, Gamma)
        u_nl_1 = real_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt
        a_n = u_lin_adv + u_nl_1

        ############## --------------------- ##############

        ### ETDRK2 ###
        A_dot_u_2 = A.dot(a_n**2)

        ### J(u) * u
        Linear_u2 = (A.dot((a_n + (epsilon * a_n))**2) - A_dot_u_2)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u2 = A_dot_u_2 - Linear_u2

        ## Nonlinear Term
        u_nl_2 = real_Leja_phi(phi_2, (Nonlin_u2 - Nonlin_u), dt, Leja_X, c, Gamma) * dt

        return a_n, u_nl_2, mat_vec_num + 4

    ############## --------------------- ##############

    def Imag_Leja():

        ### ETD1 ###
        A_dot_u_1 = A.dot(u**2)

        ### J(u) * u
        Linear_u = (A.dot((u + (epsilon * u))**2) - A_dot_u_1)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u = A_dot_u_1 - Linear_u

        ### ETD1 Solution
        u_lin_adv, mat_vec_num = imag_Leja_exp(A, u, 2, dt, Leja_X, c, Gamma)
        u_nl_1 = imag_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt
        a_n = u_lin_adv + u_nl_1

        ############## --------------------- ##############

        ### ETDRK2 ###
        A_dot_u_2 = A.dot(a_n**2)

        ### J(u) * u
        Linear_u2 = (A.dot((a_n + (epsilon * a_n))**2) - A_dot_u_2)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u2 = A_dot_u_2 - Linear_u2

        ## Nonlinear Term
        u_nl_2 = imag_Leja_phi(phi_2, (Nonlin_u2 - Nonlin_u), dt, Leja_X, c, Gamma) * dt

        return a_n, u_nl_2, mat_vec_num + 4

    ############## --------------------- ##############

    a_n, u_nl_2, mat_vec_num = Real_Leja()
    # a_n, u_nl_2, mat_vec_num = Imag_Leja()

    ### ETDRK2 Solution ###
    u_etdrk2 = a_n + u_nl_2

    return u_etdrk2, mat_vec_num

##############################################################################

### ETDRK4

def ETDRK4(A, u, dt, Leja_X, c, Gamma):

    epsilon = 1e-7          # Amplitude of perturbation

    ############## --------------------- ##############

    def Real_Leja():

        ############## --------------------- ##############

        ### Linear and Nonlinear u ###

        ### Matrix-vector product
        A_dot_u_1 = A.dot(u**2)

        ### J(u) * u
        Linear_u = (A.dot((u + (epsilon * u))**2) - A_dot_u_1)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u = A_dot_u_1 - Linear_u

        ############## --------------------- ##############

        ### a_n ###
        u_lin_a, its_a = real_Leja_exp(A, u, 2, dt/2, Leja_X, c, Gamma)
        u_nl_a = real_Leja_phi(phi_1, Nonlin_u, dt/2, Leja_X, c, Gamma) * dt/2

        a_n = u_lin_a + u_nl_a

        ############## --------------------- ##############

        ### Linear and Nonlinear a ###

        ### Matrix-vector product
        A_dot_a = A.dot(a_n**2)

        ### J(u) * u
        Linear_a = (A.dot((a_n + (epsilon * a_n))**2) - A_dot_a)/epsilon

        ### F(u_n) - (J(u_n) * u)
        Nonlin_a = A_dot_a - Linear_a

        ############## --------------------- ##############

        ### b_n ###
        u_lin_b = u_lin_a
        u_nl_b = real_Leja_phi(phi_1, Nonlin_a, dt/2, Leja_X, c, Gamma) * dt/2

        b_n = u_lin_b + u_nl_b

        ############## --------------------- ##############

        ### Linear and Nonlinear b ###

        ### Matrix-vector product
        A_dot_b = A.dot(b_n**2)

        ### J(u) * u
        Linear_b = (A.dot((b_n + (epsilon * b_n))**2) - A_dot_b)/epsilon

        ### F(u_n) - (J(u_n) * u)
        Nonlin_b = A_dot_b - Linear_b

        ############## --------------------- ##############

        ### c_n ###
        u_lin_c, its_c = real_Leja_exp(A, a_n, 2, dt/2, Leja_X, c, Gamma)
        u_nl_c = real_Leja_phi(phi_1, (2*Nonlin_b - Nonlin_u), dt/2, Leja_X, c, Gamma) * dt/2

        c_n = u_lin_c + u_nl_c

        ############## --------------------- ##############

        ### Linear and Nonlinear c ###

        ### Matrix-vector product
        A_dot_c = A.dot(c_n**2)

        ### J(u) * u
        Linear_c = (A.dot((c_n + (epsilon * c_n))**2) - A_dot_c)/epsilon

        ### F(u_n) - (J(u_n) * u)
        Nonlin_c = A_dot_c - Linear_c

        ############## --------------------- ##############

        ### Linear part of solution ###
        u_lin_adv, its_lin = real_Leja_exp(A, u, 2, dt, Leja_X, c, Gamma)

        ### Nonlinear part of solution ###
        u_nl_1 = real_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt
        u_nl_2 = real_Leja_phi(phi_2, (-3*Nonlin_u + 2*(Nonlin_a + Nonlin_b) - Nonlin_c), dt, Leja_X, c, Gamma) * dt
        u_nl_3 = real_Leja_phi(phi_3, (4 * (Nonlin_u - Nonlin_a - Nonlin_b + Nonlin_c)), dt, Leja_X, c, Gamma) * dt

        return u_lin_adv, u_nl_1, u_nl_2, u_nl_3, its_a + its_c + its_lin + 8

    ############## --------------------- ##############

    def Imag_Leja():

        ################### Advective Term ###################

        ### Linear and Nonlinear u ###

        ### Matrix-vector product
        A_dot_u_1 = A.dot(u**2)

        ### J(u) * u
        Linear_u = (A.dot((u + (epsilon * u))**2) - A_dot_u_1)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u = A_dot_u_1 - Linear_u

        ############## --------------------- ##############

        ### a_n ###
        u_lin_a, its_a = imag_Leja_exp(A, u, 2, dt/2, Leja_X, c, Gamma)
        u_nl_a = imag_Leja_phi(phi_1, Nonlin_u, dt/2, Leja_X, c, Gamma) * dt/2

        a_n = u_lin_a + u_nl_a

        ############## --------------------- ##############

        ### Linear and Nonlinear a ###

        ### Matrix-vector product
        A_dot_a = A.dot(a_n**2)

        ### J(u) * u
        Linear_a = (A.dot((a_n + (epsilon * a_n))**2) - A_dot_a)/epsilon

        ### F(u_n) - (J(u_n) * u)
        Nonlin_a = A_dot_a - Linear_a

        ############## --------------------- ##############

        ### b_n ###
        u_lin_b = u_lin_a
        u_nl_b = imag_Leja_phi(phi_1, Nonlin_a, dt/2, Leja_X, c, Gamma) * dt/2

        b_n = u_lin_b + u_nl_b

        ############## --------------------- ##############

        ### Linear and Nonlinear b ###

        ### Matrix-vector product
        A_dot_b = A.dot(b_n**2)

        ### J(u) * u
        Linear_b = (A.dot((b_n + (epsilon * b_n))**2) - A_dot_b)/epsilon

        ### F(u_n) - (J(u_n) * u)
        Nonlin_b = A_dot_b - Linear_b

        ############## --------------------- ##############

        ### c_n ###
        u_lin_c, its_c = imag_Leja_exp(A, a_n, 2, dt/2, Leja_X, c, Gamma)
        u_nl_c = imag_Leja_phi(phi_1, (2*Nonlin_b - Nonlin_u), dt/2, Leja_X, c, Gamma) * dt/2

        c_n = u_lin_c + u_nl_c

        ############## --------------------- ##############

        ### Linear and Nonlinear c ###

        ### Matrix-vector product
        A_dot_c = A.dot(c_n**2)

        ### J(u) * u
        Linear_c = (A.dot((c_n + (epsilon * c_n))**2) - A_dot_c)/epsilon

        ### F(u_n) - (J(u_n) * u)
        Nonlin_c = A_dot_c - Linear_c

        ############## --------------------- ##############

        ### Linear part of solution ###
        u_lin_adv, its_lin = imag_Leja_exp(A, u, 2, dt, Leja_X, c, Gamma)

        ### Nonlinear part of solution ###
        u_nl_1 = imag_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt
        u_nl_2 = imag_Leja_phi(phi_2, (-3*Nonlin_u + 2*(Nonlin_a + Nonlin_b) - Nonlin_c), dt, Leja_X, c, Gamma) * dt
        u_nl_3 = imag_Leja_phi(phi_3, (4 * (Nonlin_u - Nonlin_a - Nonlin_b + Nonlin_c)), dt, Leja_X, c, Gamma) * dt

        return u_lin_adv, u_nl_1, u_nl_2, u_nl_3, its_a + its_c + its_lin + 8

    ############## --------------------- ##############

    # u_lin_adv, u_nl_1, u_nl_2, u_nl_3, mat_vec_num = Real_Leja()
    u_lin_adv, u_nl_1, u_nl_2, u_nl_3, mat_vec_num = Imag_Leja()

    ### ETDRK4 Solution ###
    u_etdrk4 = u_lin_adv + u_nl_1 + u_nl_2 + u_nl_3

    return u_etdrk4, mat_vec_num

##############################################################################

### EXPRB43

def EXPRB43_A(A, u, dt, Leja_X, c, Gamma):
    
    f_u = A.dot(u**2)
    
    a_n = u + imag_Leja_phi(phi_1, f_u, dt/2, Leja_X, c, Gamma) * dt/2
    b_n = u + imag_Leja_phi(phi_1, f_u, dt, Leja_X, c, Gamma) * dt
    
    f_a = A.dot(a_n**2)
    f_b = A.dot(b_n**2)
    
    u_exprb4 = u + imag_Leja_phi(phi_1, f_u, dt, Leja_X, c, Gamma) * dt \
                 + imag_Leja_phi(phi_3, (-14*f_u + 16*f_a - 2*f_b), dt, Leja_X, c, Gamma) * dt \
                 + imag_Leja_phi(phi_4, (36*f_u - 48*f_a + 12*f_b), dt, Leja_X, c, Gamma) * dt

    u_exprb3 = u + imag_Leja_phi(phi_1, f_u, dt, Leja_X, c, Gamma) * dt \
                 + imag_Leja_phi(phi_3, (-14*f_u + 16*f_a - 2*f_b), dt, Leja_X, c, Gamma) * dt
            
    return u_exprb4, u_exprb3

##############################################################################

### EXPRB43

def EXPRB43_B(A, u, dt, Leja_X, c, Gamma):
    
    epsilon = 1e-7
    
    ############## --------------------- ##############
    
    ### Linear and Nonlinear u ###
    
    ### Matrix-vector product
    A_dot_u_1 = A.dot(u**2)

    ### J(u) * u
    Linear_u = (A.dot((u + (epsilon * u))**2) - A_dot_u_1)/epsilon

    ### F(u) - (J(u) * u)
    Nonlin_u = A_dot_u_1 - Linear_u
    
    ############## --------------------- ##############
    
    u_lin = real_Leja_exp(A, u, 2, dt, Leja_X, c, Gamma)[0]
    u_lin_a = real_Leja_exp(A, u, 2, dt/2, Leja_X, c, Gamma)[0]
    
    a_n = u_lin_a + real_Leja_phi(phi_1, Nonlin_u, dt/2, Leja_X, c, Gamma) * dt/2
    b_n = u_lin + real_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt
    
    ############## --------------------- ##############
    
    ### Linear and Nonlinear a ###

    ### Matrix-vector product
    A_dot_a = A.dot(a_n**2)

    ### J(u) * u
    Linear_a = (A.dot((a_n + (epsilon * a_n))**2) - A_dot_a)/epsilon

    ### F(u_n) - (J(u_n) * u)
    Nonlin_a = A_dot_a - Linear_a
        
    ############## --------------------- ##############

    ### Linear and Nonlinear b ###

    ### Matrix-vector product
    A_dot_b = A.dot(b_n**2)

    ### J(u) * u
    Linear_b = (A.dot((b_n + (epsilon * b_n))**2) - A_dot_b)/epsilon

    ### F(u_n) - (J(u_n) * u)
    Nonlin_b = A_dot_b - Linear_b
    
    ############## --------------------- ##############
    
    u_exprb4 = u_lin + real_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt \
                     + real_Leja_phi(phi_3, (-14*Nonlin_u + 16*Nonlin_a - 2*Nonlin_b), dt, Leja_X, c, Gamma) * dt \
                     + real_Leja_phi(phi_4, (36*Nonlin_u - 48*Nonlin_a + 12*Nonlin_b), dt, Leja_X, c, Gamma) * dt

    u_exprb3 = u_lin + imag_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c, Gamma) * dt \
                     + imag_Leja_phi(phi_3, (-14*Nonlin_u + 16*Nonlin_a - 2*Nonlin_b), dt, Leja_X, c, Gamma) * dt
            
    return u_exprb4, u_exprb3

##############################################################################