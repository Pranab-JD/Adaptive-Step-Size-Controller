"""
Created on Mon Oct 26 18:24:14 2020

@author: Pranab JD

Description: 
    This code solves the viscous Burgers' equation:
    du/dt = d^2(u^2)/dx^2 + eta * du/dx (1D)
    using different exponential time integrators.
    Advective term - 3rd order upwind scheme
    Adaptive step size is implemented.
"""

import numpy as np
from Class_Controller import *
from Leja_Interpolation import *

##############################################################################

class Diffusion_Advection_1D(Cost_Controller):
    
    def __init__(self, N, tmax, eta, error_tol):
        super().__init__(N, tmax, eta, error_tol)
        self.sigma_init = 1.4 * 1e-3                        # Initial amplitude of Gaussian
        self.initialize_U()
        self.eigen_linear_operators()
    
	### Initial distribution
    def initialize_U(self):
        u0 = np.exp((-(self.X - 0.5)**2)/(2 * self.sigma_init**2))
        self.u = u0.copy()
        
    def initialize_matrices(self):
        super().initialize_matrices()
        self.A = self.A_dif + self.A_adv

    def eigen_linear_operators(self):
        global eigen_min, eigen_max, eigen_imag, c_real, Gamma_real, c_imag, Gamma_imag
        eigen_min = 0
        eigen_max, eigen_imag = Gershgorin(self.A)      # Max real eigen value is negative (Parabolic DE)
        c_real = 0.5 * (eigen_max + eigen_min)          
        Gamma_real = 0.25 * (eigen_min - eigen_max)
        c_imag = 0
        Gamma_imag = 0.25 * (eigen_imag - (- eigen_min))
        

    ##############################################################################    
        
    def Solution(self, u, dt):
        """
        Parameters
        ----------
        u           : 1D vector u (input)
        dt          : dt

        Returns
        -------
        u_sol       : 1D vector u (output) after time dt using the preferred method
        u_ref       : 1D vector u (output) after time dt using the reference method
        u           : 1D vector u (input)
        its_mat_vec : Number of matrix-vector products

        """

        ############## --------------------- ##############

        ### u_sol, its_sol: Solution and the number of iterations needed to get the solution
        ### u_ref, its_ref: Reference solution and the number of iterations needed to get that

        u_sol, its_sol = real_Leja_exp(self.A, u, dt, c_real, Gamma_real)
        # u_sol, its_sol = imag_Leja_exp(self.A, u, dt, c_imag, Gamma_imag)
        
        its_ref = 0

        ## No. of matrix-vector products
        its_mat_vec = its_sol + its_ref

        return u_sol, u, its_mat_vec
    
    ##############################################################################