"""
Created on Thu Aug  6 17:29:51 2020

@author: Pranab JD

Description: -
    This code solves the viscous Burgers' equation:
    du/dt = d^2u/dx^2 + eta * d(u^2)/dx (1D)
    using RK4, RKF45(embedded), ETD, and ETDRK2.
    Advective term - 3rd order upwind scheme
    Adaptive step size is implemented.
"""

import os
import math
import numpy as np
from decimal import *
from Leja_Header import *
import matplotlib.pyplot as plt
from Adaptive_Step_Size import *
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix

from datetime import datetime

startTime = datetime.now()
getcontext().prec = 16                  # Precision for decimal numbers

##############################################################################

class Viscous_Burgers_1D_Adaptive_h:

    def __init__(self, N, tmax, eta, error_tol):
        self.N = int(N)                 # Number of points along X
        self.xmin = 0                   # Min value of X
        self.xmax = 1                   # Max value of X
        self.tmax = tmax                # Maximum time
        self.error_tol = error_tol      # Maximum error permitted
        self.eta = eta                  # Peclet number
        self.sigma = 0.02          		# Amplitude of Gaussian
        self.x_0 = 0.9           		# Center of the Gaussian
        self.gamma = (3 + 3**0.5)/6     # gamma for SDIRK methods
        self.initialize_spatial_domain()
        self.initialize_U()
        self.initialize_parameters()
        self.initialize_matrices()

	### Discretize the spatial domain
    def initialize_spatial_domain(self):
        self.dx = (self.xmax - self.xmin)/(self.N)
        self.X = np.linspace(self.xmin, self.xmax, self.N, endpoint = False)

	### Initial distribution
    def initialize_U(self):
        u0 = 1 + (np.exp(1 - (1/(1 - (2 * self.X - 1)**2)))) + 1./2. * np.exp(-(self.X - self.x_0)**2/(2 * self.sigma**2))
        self.u = u0.copy()

    ### Parameters
    def initialize_parameters(self):
        self.adv_cfl = self.dx/abs(self.eta)
        self.dif_cfl = self.dx**2/2
        print('Advection CFL: ', self.adv_cfl)
        print('Diffusion CFL: ', self.dif_cfl)
        self.dt = 1.5 * min(self.adv_cfl, self.dif_cfl)  # N * CFL condition
        self.nsteps = int(np.ceil(self.tmax/self.dt))    # number of time steps
        self.R = 1./6. * self.eta/self.dx
        self.F = 1/self.dx**2                            # Fourier mesh number

    ### Operator matrices
    def initialize_matrices(self):
        self.A_adv = np.zeros((self.N, self.N))
        self.A_dif = np.zeros((self.N, self.N))

        ## Factor of 1/2 - conservative Burger's equation
        for ij in range(self.N):
            self.A_adv[ij, int(ij + 2) % self.N] = - self.R/2
            self.A_adv[ij, int(ij + 1) % self.N] = 6 * self.R/2
            self.A_adv[ij, ij % self.N] = -3 * self.R/2
            self.A_adv[ij, int(ij - 1) % self.N] = -2 * self.R/2

            self.A_dif[ij, int(ij + 1) % self.N] = self.F
            self.A_dif[ij, ij % self.N] = -2 * self.F
            self.A_dif[ij, int(ij - 1) % self.N] = self.F

        self.A_adv = csr_matrix(self.A_adv)
        self.A_dif = csr_matrix(self.A_dif)
        
    ##############################################################################

    def RK4(self, u, dt):
        """
        Parameters
        ----------
        u       : Vector u (Input)
        dt      : dt
    
        Returns
        -------
        u_rk4 : Vector u (Output) after time dt
    
        """

        k1 = dt * (self.A_adv.dot(u**2) + self.A_dif.dot(u))
        k2 = dt * (self.A_adv.dot((u + k1/2)**2) + self.A_dif.dot(u + k1/2))
        k3 = dt * (self.A_adv.dot((u + k2/2)**2) + self.A_dif.dot(u + k2/2))
        k4 = dt * (self.A_adv.dot((u + k3)**2) + self.A_dif.dot(u + k3))

        ## Solution
        u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

        return u_rk4, dt
    
    
    ##############################################################################
    
    def RKF45(self, u):
        """
        Parameters
        ----------
        u       : Vector u (Input)
        dt      : dt
    
        Returns
        -------
        u_rk4 : Vector u (Output) after time dt
    
        """
        
        ## Max. number of iters to achieve tolerance in a single time loop
        n_iters = 1000
        
        for mm in range(n_iters):

            k1 = self.dt * (self.A_adv.dot(u**2) + self.A_dif.dot(u))
            
            k2 = self.dt * (self.A_adv.dot((u + k1/4)**2) + self.A_dif.dot(u + k1/4))
            
            k3 = self.dt * (self.A_adv.dot((u + 3./32.*k1 + 9./32.*k2)**2) + self.A_dif.dot(u + 3./32.*k1 + 9./32.*k2))
            
            k4 = self.dt * (self.A_adv.dot((u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**2) \
                         + self.A_dif.dot(u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3))
            
            k5 = self.dt * (self.A_adv.dot((u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**2) \
                         + self.A_dif.dot(u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4))
            
            k6 = self.dt * (self.A_adv.dot((u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**2) \
                         + self.A_dif.dot(u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)) 
             
            ### Solutions
            u_rkf4 = u + (25./216*k1 + 1408./2565.*k3 + 2197./4104.*k4 - 1./5.*k5)
            u_rkf5 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)
            
            ### Error estimate ###
            error_rkf45 = np.mean(abs(u_rkf4 - u_rkf5))
            
            ### Step size controller ###
            new_dt = self.dt * (self.error_tol/error_rkf45)**(1/5)
            self.dt = 0.8 * new_dt          # Safety factor

            if error_rkf45 <= self.error_tol:
                # print('Error within limits. dt accepted!! Error = ', error_rkf45)
                break
            
            ## Error alert
            if mm == (n_iters - 1):
                print('Condition reached. Check parameters!!!')

        return u_rkf4, u_rkf5, error_rkf45, self.dt
    
    
    ##############################################################################

    def ETDRK2(self, u, dt):

        # path = './ETDRK2/06/IR/'
        # os.makedirs(os.path.dirname(path), exist_ok = True)
        
        ### Write simulation paramters to a file
        # file_param = open(path + 'Simulation_Parameters.txt', 'w+')
        # file_param.write('N = %d' % self.N + '\n')
        # file_param.write('eta = %f' % self.eta + '\n')
        # file_param.write('Advection CFL = %.15f' % self.adv_cfl + '\n')
        # file_param.write('Diffusion CFL = %.15f' % self.dif_cfl + '\n')
        # file_param.write('dt = %.15f' % self.dt + '\n')
        # file_param.write('Simulation time = %f' % self.tmax + '\n')
        # file_param.write('Advection: Imag Leja' + '\n')
        # file_param.write('Diffusion: Real Leja')
        # file_param.close()

        ## Leja points
        Leja_X = Leja_Points()
        
        epsilon = 1e-7                               		        # Amplitude of perturbation
        
        ## Eigen values (Diffusion)
        eigen_min_dif = 0
        eigen_max_dif, eigen_imag_dif = Gershgorin(self.A_dif)      # Max real, imag eigen value

        ############## --------------------- ##############
        
        ## Eigen values (Advection)
        eigen_min_adv = 0
        eigen_max_adv, eigen_imag_adv = Power_iteration(self.A_adv, u, 2)   # Max real, imag eigen value
        eigen_max_adv = eigen_max_adv * 1.2                                             # Safety factor
        eigen_imag_adv = eigen_imag_adv * 1.125                                         # Safety factor
        
        ## c and gamma
        c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
        Gamma_real_adv = 0.25 * (eigen_max_adv - eigen_min_adv)
        c_imag_adv = 0
        Gamma_imag_adv = 0.25 * (eigen_imag_adv - 0)
        c_real_dif = 0.5 * (eigen_max_dif + eigen_min_dif)
        Gamma_real_dif = 0.25 * (eigen_max_dif - eigen_min_dif)

        ################### Advective Term ###################
        
        ### ETD1 ###
        A_dot_u_1 = self.A_adv.dot(u**2)
        
        ### J(u) * u
        Linear_u = (self.A_adv.dot((u + (epsilon * u))**2) - A_dot_u_1)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u = A_dot_u_1 - Linear_u
        
        ## Linear Term
        # u_lin_adv = real_Leja_exp(self.A_adv, self.u_etdrk2, 2, self.dt, Leja_X, c_real_adv, Gamma_real_adv)
        u_lin_adv = imag_Leja_exp(self.A_adv, u, 2, dt, Leja_X, c_imag_adv, Gamma_imag_adv)
        
        ## Nonlinear Term 
        # u_nl_adv = real_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_real_adv, Gamma_real_adv) * self.dt
        u_nl_adv = imag_Leja_phi(phi_1, Nonlin_u, dt, Leja_X, c_imag_adv, Gamma_imag_adv) * dt

        ############## --------------------- ##############

        ### ETD1 Solution ###
        a_n = u_lin_adv + u_nl_adv
        
        ############## --------------------- ##############

        ### ETDRK2 ###
        A_dot_u_2 = self.A_adv.dot(a_n**2)
        
        ### J(u) * u
        Linear_u2 = (self.A_adv.dot((a_n + (epsilon * a_n))**2) - A_dot_u_2)/epsilon

        ### F(u) - (J(u) * u)
        Nonlin_u2 = A_dot_u_2 - Linear_u2
        
        ## Nonlinear Term 
        # u_nl_2 = real_Leja_phi(phi_2, (Nonlin_u2 - Nonlin_u), self.dt, Leja_X, c_real_adv, Gamma_real_adv) * self.dt
        u_nl_2 = imag_Leja_phi(phi_2, (Nonlin_u2 - Nonlin_u), dt, Leja_X, c_imag_adv, Gamma_imag_adv) * dt
        
        ############## --------------------- ##############
        
        ### ETDRK2 Solution ###
        u_adv = a_n + u_nl_2
        
        ################### Diffusive Term ###################
        
        u_diff = real_Leja_exp(self.A_dif, u_adv, 1, dt, Leja_X, c_real_dif, Gamma_real_dif)
        
        ############## --------------------- ##############

        ### Full solution
        u_temp = u_diff
        
        ## Update u and t
        u_etdrk2 = u_temp.copy()
        
        return u_etdrk2, dt
    
    ##############################################################################
        
    def run(self):
        
        time = 0                                    # Time
        counter = 0                                 # Counter for # of iterations
        
        ## Time loop
        while (time < self.tmax):

            u, u_ref, error, dt = Richardson_Extrapolation(self.ETDRK2, 2, self.u, self.dt, self.error_tol)
            u, u_ref, error, dt = Higher_Order_Method(self.RK4, self.ETDRK2, 2, self.u, self.dt, self.error_tol)

            counter = counter + 1
            time = time + self.dt
            self.u = u.copy()
            self.dt = dt.copy()

            print('Time = ', time)
            print('dt = ', self.dt)

            ## Test plots
            plt.plot(self.X, u_ref, 'rd', label = 'Reference')
            plt.plot(self.X, u, 'b.', label = 'Data')
            plt.legend()
            plt.pause(self.dt)
            plt.clf()
    
    
    ##############################################################################        
        
    def run_RKF45(self):
        
        time = 0                                    # Time
        counter = 0                                 # Counter for # of iterations
        
        ## Time loop
        while (time < self.tmax):

            u, u_ref, error, dt = self.RKF45(self.u)

            counter = counter + 1
            time = time + self.dt
            self.u = u.copy()

            print('Time = ', time)
            print('dt = ', self.dt)

            ## Test plots
            plt.plot(self.X, u_ref, 'rd', label = 'Reference')
            plt.plot(self.X, u, 'b.', label = 'Data')
            plt.legend()
            plt.pause(self.dt)
            plt.clf()
    
        
##############################################################################

# Assign values for N, tmax, and eta
N = 100
t_max = 5*1e-3
eta = 100
error_tol = 1e-4

def main():
    sim = Viscous_Burgers_1D_Adaptive_h(N, t_max, eta, error_tol)
    # sim.RK4()
    # sim.ETD()
    # sim.ETDRK2()
    sim.run_1()
    # sim.run_RKF45()
    plt.show()

if __name__ == "__main__":
    main()

##################################################################################

print('Time Elapsed = ', datetime.now() - startTime)