"""
Created on Mon Oct 26 18:24:14 2020

@author: Pranab JD

Description:
    Inviscid Burgers' equation:
    
    du/dt = eta * d(u^2)/dx (1D)
    
    du/dt = eta_x * d(u^2)/dx
          + eta_y * d(u^2)/dy (2D)   
          
"""

import numpy as np
from Class_Controller import *
from Leja_Interpolation import *
from Integrators_1_matrix import *

##############################################################################

class Inviscid_Burgers_1D(Cost_Controller_1D):

    def __init__(self, N, tmax, eta, error_tol):
        super().__init__(N, tmax, eta, error_tol)
        self.epsilon_1 = 0.01           # Amplitude of 1st sine wave
        self.epsilon_2 = 0.01           # Amplitude of 2nd sine wave
        self.initialize_U()

	### Initial distribution
    def initialize_U(self):
        u0 = 2 + self.epsilon_1 * np.sin(2 * np.pi * self.X) + self.epsilon_2 * np.sin(8 * np.pi * self.X + 0.3)
        self.u = u0.copy()

    def initialize_parameters(self):
        super().initialize_parameters()
        self.dt = 1 * self.adv_cfl      # N * CFL condition

    def initialize_matrices(self):
        super().initialize_matrices()
        self.A_adv = 0.5 * self.A_adv   # 1/2, since conservative form of Burgers' equation
        self.A_dif = 0                  # Inviscid equation

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

        ## Eigen values (Advection)
        eigen_min_adv = 0
        eigen_max_adv, eigen_imag_adv, its_power = Power_iteration(self.A_adv, u, 2)   # Max real, imag eigen value
        eigen_max_adv = eigen_max_adv * 1.25                                           # Safety factor
        eigen_imag_adv = eigen_imag_adv * 1.125                                         # Safety factor

        ## c and gamma
        c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
        Gamma_real_adv = 0.25 * (eigen_min_adv - eigen_max_adv)
        c_imag_adv = 0
        Gamma_imag_adv = 0.25 * (eigen_imag_adv - (- eigen_imag_adv))

        ############## --------------------- ##############

        ### u_sol, its_sol: Solution and the number of iterations needed to get the solution
        ### u_ref, its_ref: Reference solution and the number of iterations needed to get that

        # c, Gamma = c_real_adv, Gamma_real_adv
        c, Gamma = c_imag_adv, Gamma_imag_adv

        ### ------------------------------------------------------ ###

        # u_sol, its_sol, u_ref, its_ref = EXPRB32(self.A_adv, 2, u, dt, c, Gamma, 0)

        u_sol, its_sol, u_ref, its_ref = EXPRB43(self.A_adv, 2, u, dt, c, Gamma, 1)

        # u_sol, its_sol, u_ref, its_ref = RKF45(self.A_adv, 2, u, dt)

        ### ------------------------------------------------------ ###

        ### No. of matrix-vector products

        ## its_sol included in its_ref
        its_mat_vec = its_ref + its_power

        return u_sol, u_ref, u, its_mat_vec

##############################################################################

class Inviscid_Burgers_2D(Cost_Controller_2D):
    
    def __init__(self, N_x, N_y, tmax, eta_x, eta_y, error_tol):
        super().__init__(N_x, N_y, tmax, eta_x, eta_y, error_tol)
        self.epsilon_1 = 0.01           # Amplitude of 1st sine wave
        self.epsilon_2 = 0.01           # Amplitude of 2nd sine wave
        self.initialize_U()

	### Initial distribution
    def initialize_U(self):
        u0 = 2 + self.epsilon_1 * (np.sin(2 * np.pi * self.X) + np.sin(2 * np.pi * self.Y)) \
        	   + self.epsilon_2 * (np.sin(8 * np.pi * self.X + 0.3) + np.sin(8 * np.pi * self.Y + 0.3))
        self.u = u0.copy()

    def initialize_parameters(self):
        super().initialize_parameters()
        self.dt = 2 * self.adv_cfl      # N * CFL condition

    def initialize_matrices(self):
        super().initialize_matrices()
        self.A_adv = 0.5 * self.A_adv   # 1/2, since conservative form of Burgers' equation
        self.A_dif = 0                  # Inviscid equation

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

        ## Eigen values (Advection)
        eigen_min_adv = 0
        eigen_max_adv, eigen_imag_adv, its_power = Power_iteration(self.A_adv, u, 2)   # Max real, imag eigen value
        eigen_max_adv = eigen_max_adv * 1.25                                           # Safety factor
        eigen_imag_adv = eigen_imag_adv * 1.25                                         # Safety factor

        ## c and gamma
        c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
        Gamma_real_adv = 0.25 * (eigen_min_adv - eigen_max_adv)
        c_imag_adv = 0
        Gamma_imag_adv = 0.25 * (eigen_imag_adv - (- eigen_imag_adv))

        ############## --------------------- ##############

        ### u_sol, its_sol: Solution and the number of iterations needed to get the solution
        ### u_ref, its_ref: Reference solution and the number of iterations needed to get that

        # c, Gamma = c_real_adv, Gamma_real_adv
        c, Gamma = c_imag_adv, Gamma_imag_adv

        ### ------------------------------------------------------ ###

        # u_sol, its_sol, u_ref, its_ref = EXPRB32(self.A_adv, 2, u, dt, c, Gamma, 0)

        u_sol, its_sol, u_ref, its_ref = EXPRB43(self.A_adv, 2, u, dt, c, Gamma, 1)

        # u_sol, its_sol, u_ref, its_ref = RKF45(self.A_adv, 2, u, dt)

        ### ------------------------------------------------------ ###

        ### No. of matrix-vector products

        ## its_sol included in its_ref
        its_mat_vec = its_ref + its_power

        return u_sol, u_ref, u, its_mat_vec

##############################################################################