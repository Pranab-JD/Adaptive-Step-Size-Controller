"""
Created on Mon Oct 26 18:24:14 2020

@author: Pranab JD

Description: 
    This code solves the inviscid Burgers' equation:
    du/dt = d^2(u^2)/dx^2 + eta * du/dx (1D)
    using different exponential time integrators.
    Advective term - 3rd order upwind scheme
    Adaptive step size is implemented.
"""

import numpy as np
from Class_Controller import *
from Leja_Interpolation import *
from Integrators_1_matrix import *

##############################################################################

class Inviscid_Burgers_1D(Cost_Controller):
    
    def __init__(self, N, tmax, eta, error_tol):
        super().__init__(N, tmax, eta, error_tol)
        self.epsilon_1 = 0.01           # Amplitude of 1st sine wave
        self.epsilon_2 = 0.01           # Amplitude of 2nd sine wave
        self.initialize_U()
    
	### Initial distribution
    def initialize_U(self):
        u0 = 2 + self.epsilon_1 * np.sin(2 * np.pi * self.X) + self.epsilon_2 * np.sin(8 * np.pi * self.X + 0.3)
        self.u = u0.copy()

    def initialize_matrices(self):
        super().initialize_matrices()
        self.A_adv = 0.5 * self.A_adv   # 1/2, since conservative form of Burgers' equation
        self.A_dif = 0                  # Inviscid equation

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

        # u_sol, its_sol, u_ref, its_ref = EXPRB32(self.A_adv, 2, u, dt, c, Gamma, 1)

        u_sol, its_sol, u_ref, its_ref = EXPRB43(self.A_adv, 2, u, dt, c, Gamma, 1)

        # u_ref, its_ref = RK4(self.A_adv, 2, self.A_dif, 1, u, dt)

        ### ------------------------------------------------------ ###

        ## No. of matrix-vector products
        its_mat_vec = its_sol + its_ref + its_power

        return u_sol, u_ref, u, its_mat_vec
    
    ##############################################################################