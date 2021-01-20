"""
Created on Mon Oct 26 18:24:14 2020

@author: Pranab JD

Description: 
    This code solves the linear diffusion-advection equation:
    du/dt = d^2(u^2)/dx^2 + eta * du/dx (1D)
    using exponential time integration (ETD).
    Advective term - 3rd order upwind scheme
    Adaptive step size is implemented.
"""

import numpy as np
from Class_Controller import *
from Leja_Interpolation import *

##############################################################################

class Diffusion_Advection_1D(Cost_Controller_1D):
    
    def __init__(self, N, tmax, eta, error_tol):
        super().__init__(N, tmax, eta, error_tol)
        self.sigma_init = 1.4 * 1e-3    # Initial amplitude of Gaussian
        self.initialize_U()
        self.eigen_linear_operators()
    
	### Initial distribution
    def initialize_U(self):
        u0 = np.exp((-(self.X - 0.5)**2)/(2 * self.sigma_init**2))
        self.u = u0.copy()
        
    def initialize_matrices(self):
        super().initialize_matrices()
        self.A = self.A_adv + self.A_dif

    def eigen_linear_operators(self):
        global eigen_min, eigen_max, eigen_imag, c_real, Gamma_real, c_imag, Gamma_imag
        eigen_min = 0
        eigen_max, eigen_imag = Gershgorin(self.A)      # Max real eigen value is negative (Parabolic DE)
        c_real = 0.5 * (eigen_max + eigen_min)          
        Gamma_real = 0.25 * (eigen_min - eigen_max)
        c_imag = 0
        Gamma_imag = 0.25 * (eigen_imag - (- eigen_min))
        
    def Solution(self, u, dt):
        """
        Parameters
        ----------
        u           : 1D vector u (input)
        dt          : dt

        Returns
        -------
        u_sol       : 1D vector u (output) after time dt using the preferred method
        u           : 1D vector u (input)
        its_sol     : Number of matrix-vector products

        """

        ### u_sol, its_sol: Solution and the number of iterations needed to get the solution

        u_sol, its_sol = real_Leja_exp(self.A, u, dt, c_real, Gamma_real)
        
        ### Richardson Extrapolation
        # u_ref_1, its_ref_1 = real_Leja_exp(self.A, u, dt/2, c_real, Gamma_real)
        # u_ref, its_ref_2 = real_Leja_exp(self.A, u_ref_1, dt/2, c_real, Gamma_real)
        
        # u_sol, its_sol = imag_Leja_exp(self.A, u, dt, c_imag, Gamma_imag)

        return u_sol, u, its_sol
    
##############################################################################
    
class Diffusion_Advection_2D(Cost_Controller_2D):
    
    def __init__(self, N_x, N_y, tmax, eta_x, eta_y, error_tol):
        super().__init__(N_x, N_y, tmax, eta_x, eta_y, error_tol)
        self.sigma_init = 1.4 * 1e-3    # Initial amplitude of Gaussian
        self.initialize_U()
        self.eigen_linear_operators()
    
	### Initial distribution
    def initialize_U(self):
        u0 = np.exp((-(self.X - 0.5)**2 - (self.Y - 0.5)**2)/(2 * self.sigma_init**2))
        self.u = u0.copy()
        
    def initialize_matrices(self):
        super().initialize_matrices()
        self.A = self.A_adv + self.A_dif

    def eigen_linear_operators(self):
        global eigen_min, eigen_max, eigen_imag, c_real, Gamma_real, c_imag, Gamma_imag
        eigen_min = 0
        eigen_max, eigen_imag = Gershgorin(self.A)      # Max real eigen value is negative (Parabolic DE)
        c_real = 0.5 * (eigen_max + eigen_min)          
        Gamma_real = 0.25 * (eigen_min - eigen_max)
        c_imag = 0
        Gamma_imag = 0.25 * (eigen_imag - (- eigen_min))
        
    def Solution(self, u, dt):
        """
        Parameters
        ----------
        u           : 1D vector u (input)
        dt          : dt

        Returns
        -------
        u_sol       : 1D vector u (output) after time dt using the preferred method
        u           : 1D vector u (input)
        its_sol     : Number of matrix-vector products

        """

        ### u_sol, its_sol: Solution and the number of iterations needed to get the solution

        u_sol, its_sol = real_Leja_exp(self.A, u, dt, c_real, Gamma_real)
        
        # u_ref_1, its_ref_1 = real_Leja_exp(self.A, u, dt/2, c_real, Gamma_real)
        # u_ref, its_ref_2 = real_Leja_exp(self.A, u_ref_1, dt/2, c_real, Gamma_real)
        
        # u_sol, its_sol = imag_Leja_exp(self.A, u, dt, c_imag, Gamma_imag)

        return u_sol, u, its_sol
    
##############################################################################