"""
Created on Fri Oct 16 11:58:08 2020

@author: Pranab JD

Description: Implementing the cost controller on nonlinear problems
            (Burgers' Equation) using exponential Rosenbrock methods
"""

import numpy as np
from Leja_Interpolation import *
from scipy.sparse import issparse, eye
from scipy.sparse.linalg import cg, gmres, bicgstab

#
# utility
#

class counter:
    def __init__(self):
        self.cnt=0
    def incr(self,x):
        self.cnt+=1

def GMRES(A,b,x0,tol):
    c=counter()
    return gmres(A,b,x0=x0,callback=c.incr,tol=tol)[0],c.cnt

##############################################################################
##############################################################################

### Numerical Methods

def Leja_params(A, m, u):
    """
    Parameters
    ----------
    A       : N x N matrix A (1 or matrices; Adv, Diff)
    m       : Index of u (u^m) (1 or 2 values)
    u       : Vector u

    Returns
    -------
    c, Gamma

    Description: Calculate eigen values of the advection (power iterations)
                    and diffusion (Gershgorin's disks) matrices;
                    calculate c and Gamma
    """

    ## Define matrices
    if len(A) == 1:
        A_adv = A; m_adv = m
    elif len(A) == 2:
        A_adv = A[0]; m_adv = m[0]
        A_dif = A[1]; m_dif = m[1]
    else:
        print("Error! Check number of matrices!!")

    ### Advection
    eigen_min_adv = 0
    eigen_max_adv, eigen_imag_adv, its_power = Power_iteration(A_adv, u, 2)   # Max real, imag eigen value
    eigen_max_adv = eigen_max_adv * 1.25                                           # Safety factor
    eigen_imag_adv = eigen_imag_adv * 1.25                                         # Safety factor

    c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
    Gamma_real_adv = 0.25 * (eigen_max_adv - eigen_min_adv)
    c_imag_adv = 0
    Gamma_imag_adv = 0.25 * (eigen_imag_adv - (- eigen_imag_adv))

    ### Diffusion
    eigen_min_dif = 0
    eigen_max_dif, eigen_imag_dif = Gershgorin(A_dif)      # Max real, imag eigen value
    c_real_dif = 0.5 * (eigen_max_dif + eigen_min_dif)
    Gamma_real_dif = 0.25 * (eigen_max_dif - eigen_min_dif)

    return c_imag_adv, Gamma_imag_adv, c_real_dif, Gamma_real_dif

##############################################################################

### 1 matrix

def EXPRB42(A_adv, m_adv, u, dt, c, Gamma):

    epsilon = 1e-7

    ############## --------------------- ##############

    ### Matrix-vector function
    f_u = A_adv.dot(u**m_adv)

    ### J(u) * u
    Linear_u = (A_adv.dot((u + (epsilon * u))**m_adv) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u

    ############## --------------------- ##############

    a_n_f, its_a = imag_Leja_phi(u, f_u, 3*dt/4, c, Gamma, phi_1, A_adv, m_adv)

    a_n = u + a_n_f * 3*dt/4

    ############## --------------------- ##############

    ### J(u) * a
    Linear_a = (A_adv.dot((u + (epsilon * a_n))**m_adv) - f_u)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A_adv.dot(a_n**m_adv) - Linear_a

    ############## --------------------- ##############

    u_1, its_1 = imag_Leja_phi(u, f_u, dt, c, Gamma, phi_1, A_adv, m_adv)
    u_nl_3, its_3 = imag_Leja_phi(u, (Nonlin_a - Nonlin_u), dt, c, Gamma, phi_3, A_adv, m_adv)

    u_exprb42 = u + (u_1 * dt) + (u_nl_3 * 32*dt/9)

    return u_exprb42, 4 + its_a + its_1 + its_3

##############################################################################

def EXPRB43(A_adv, m_adv, u, dt, c, Gamma):

    """
    Parameters
    ----------
    A_adv   : Advection matrix (A)
    m_adv   : Index of u (u^m_adv); advection
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
    f_u = A_adv.dot(u**m_adv)

    ### J(u) * u
    Linear_u = (A_adv.dot((u + (epsilon * u))**m_adv) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u

    ############## --------------------- ##############

    a_n_f, its_a = imag_Leja_phi(u, f_u, dt/2, c, Gamma, phi_1, A_adv, m_adv)
    b_n_f, its_b = imag_Leja_phi(u, f_u, dt, c, Gamma, phi_1, A_adv, m_adv)

    a_n = u + a_n_f * dt/2
    b_n = u + b_n_f * dt

    ############# --------------------- ##############

    ### J(u) * a
    Linear_a = (A_adv.dot((u + (epsilon * a_n))**m_adv) - f_u)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A_adv.dot(a_n**m_adv) - Linear_a

    ############## --------------------- ##############

    ### J(u) * b
    Linear_b = (A_adv.dot((u + (epsilon * b_n))**m_adv) - f_u)/epsilon

    ### F(b) = f(b) - (J(u) * b)s
    Nonlin_b = A_adv.dot(b_n**m_adv) - Linear_b

    ############# --------------------- ##############

    u_1 = b_n_f
    u_nl_3, its_3 = imag_Leja_phi(u, (-14*Nonlin_u + 16*Nonlin_a - 2*Nonlin_b), dt, c, Gamma, phi_3, A_adv, m_adv)
    u_nl_4, its_4 = imag_Leja_phi(u, (36*Nonlin_u - 48*Nonlin_a + 12*Nonlin_b), dt, c, Gamma, phi_4, A_adv, m_adv)

    u_exprb3 = u + (u_1 * dt) + (u_nl_3 * dt)
    u_exprb4 = u + (u_1 * dt) + (u_nl_3 * dt) + (u_nl_4 * dt)

    return u_exprb3, 6 + its_a + its_b + its_3, u_exprb4, 6 + its_a + its_b + its_3 + its_4

##############################################################################
##############################################################################

### 2 matrices

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
    u_exprb43   : 1D vector u (output) after time dt (3rd and 4th order)
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
