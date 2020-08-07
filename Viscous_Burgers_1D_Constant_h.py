"""
Created on Mon Aug  3 11:29:22 2020

@author: Pranab JD

Description: -
    This code solves the viscous Burgers' equation:
    du/dt = d^2u/dx^2 + eta * d(u^2)/dx (1D)
    using RK4, ETD, and ETDRK2.
    Advective term - 3rd order upwind scheme
    Step size is constant.

"""

import os
import math
import numpy as np
from decimal import *
from Leja_Header import *
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix

from datetime import datetime

startTime = datetime.now()
getcontext().prec = 10                  # Precision for decimal numbers

##############################################################################

class Viscous_Burgers_1D_Constant_h:

    def __init__(self, N, tmax, eta):
        self.N = int(N)                 # Number of points along X
        self.xmin = 0                   # Min value of X
        self.xmax = 1                   # Max value of X
        self.tmax = tmax                # Maximum time
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
        self.u_rk4 = u0.copy()                           # RK4
        self.u_etd = u0.copy()                           # ETD1
        self.u_etdrk2 = u0.copy()                        # ETDRK2

    ### Parameters
    def initialize_parameters(self):
        self.adv_cfl = self.dx/abs(self.eta)
        self.dif_cfl = self.dx**2/2
        print('Advection CFL: ', self.adv_cfl)
        print('Diffusion CFL: ', self.dif_cfl)
        self.dt = 0.1 * min(self.adv_cfl, self.dif_cfl)  # N * CFL condition
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

    def RK4(self):

        t = Decimal(0.0)

        ## Create directory
        path = './Reference Data/01/'
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        ### Write simulation paramters to a file
        file_param = open(path + 'Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %f' % self.eta + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax)
        file_param.close()
        
        file_rk4 = open(path + "u_rk4.txt", 'w+')
        file_time = open(path + "time.txt", 'w+')
        file_rk4.write(' '.join(map(str, self.u_rk4)) % self.u_rk4 + '\n')
        file_time.write('{}'.format(t) + '\n')

        ## Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)

            k1 = self.dt * (self.A_adv.dot(self.u_rk4**2) + self.A_dif.dot(self.u_rk4))
            k2 = self.dt * (self.A_adv.dot((self.u_rk4 + k1/2)**2) + self.A_dif.dot(self.u_rk4 + k1/2))
            k3 = self.dt * (self.A_adv.dot((self.u_rk4 + k2/2)**2) + self.A_dif.dot(self.u_rk4 + k2/2))
            k4 = self.dt * (self.A_adv.dot((self.u_rk4 + k3)**2) + self.A_dif.dot(self.u_rk4 + k3))

            ## Solution
            u_temp = self.u_rk4 + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

            ## Update u and t
            self.u_rk4 = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)
            
            ############## --------------------- ##############
            
            if nn % 100 == 0:
                print('Time = ', float(t))
                ## Write data to files
                file_rk4.write(' '.join(map(str, self.u_rk4)) % self.u_rk4 + '\n')
                file_time.write('{}'.format(t) + '\n')
                print('------------------------------------------------------------')

        ### Write final data to file
        file_f = open(path + "Final_data.txt", 'w+')
        file_f.write(' '.join(map(str, self.u_rk4)) % self.u_rk4 + '\n')
        file_f.close()

        print('Final time = ', t)
        file_rk4.close()
        file_time.close()
        

    ##############################################################################

    def ETD(self):

        ## Create directory
        path = './ETD/04/RR/'
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        ### Write simulation paramters to a file
        file_param = open(path + 'Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %f' % self.eta + '\n')
        file_param.write('Advection CFL = %.15f' % self.adv_cfl + '\n')
        file_param.write('Diffusion CFL = %.15f' % self.dif_cfl + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.write('Advection: Real Leja' + '\n')
        file_param.write('Diffusion: Real Leja')
        file_param.close()

        file = open(path + "u_etd.txt", 'w+')
        file.write(' '.join(map(str, self.u_etd)) % self.u_etd + '\n')

        ## Leja points
        Leja_X = Leja_Points()
        
        ## Eigen values (Diffusion)
        eigen_min_dif = 0
        eigen_max_dif, eigen_imag_dif = Gershgorin(self.A_dif)      # Max real, imag eigen value
        
        ############## --------------------- ##############

        epsilon = 1e-7                               		        # Amplitude of perturbation
        t = Decimal(0.0)
        print('dt =', self.dt)
        
        ### Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
                    
            ############## --------------------- ##############
            
            ## Eigen values (Advection)
            eigen_min_adv = 0
            eigen_max_adv, eigen_imag_adv = Power_iteration(self.A_adv, self.u_etd, 2)      # Max real, imag eigen value
            eigen_max_adv = eigen_max_adv * 1.2                                             # Safety factor
            eigen_imag_adv = eigen_imag_adv * 1.125                                         # Safety factor
            
            ## c and gamma
            c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
            Gamma_real_adv = 0.25 * (eigen_max_adv - eigen_min_adv)
            c_imag_adv = 0
            Gamma_imag_adv = 0.25 * (eigen_imag_adv - (-eigen_imag_adv))
            c_real_dif = 0.5 * (eigen_max_dif + eigen_min_dif)
            Gamma_real_dif = 0.25 * (eigen_max_dif - eigen_min_dif)
            
            ################### Advective Term ###################
            
            ### Matrix-vector product
            A_dot_u_1 = self.A_adv.dot(self.u_etd**2)

            ### J(u) * u
            Linear_u = (self.A_adv.dot((self.u_etd + (epsilon * self.u_etd))**2) - A_dot_u_1)/epsilon

            ### F(u) - (J(u) * u)
            Nonlin_u = A_dot_u_1 - Linear_u
            
            ## Linear Term
            u_lin_adv = real_Leja_exp(self.A_adv, self.u_etd, 2, self.dt, Leja_X, c_real_adv, Gamma_real_adv)
            # u_lin_adv = imag_Leja_exp(self.A_adv, self.u_etd, 2, self.dt, Leja_X, c_imag_adv, Gamma_imag_adv)

            ## Nonlinear Term 
            u_nl_adv = real_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_real_adv, Gamma_real_adv) * self.dt
            # u_nl_adv = imag_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_imag_adv, Gamma_imag_adv) * self.dt
            
            ## Advection solution
            u_adv = u_lin_adv + u_nl_adv
            
            ################### Diffusive Term ###################
            
            u_diff = real_Leja_exp(self.A_dif, u_adv, 1, self.dt, Leja_X, c_real_dif, Gamma_real_dif)

            ############## --------------------- ##############

            ### Full Solution ###
            u_var = u_diff

            ## Update u and t
            self.u_etd = u_var.copy()
            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############

            # plt.plot(self.X, self.u_etd, 'b.')
            # plt.pause(self.dt)
            # plt.clf()

            ############## --------------------- ##############

            ## Write data to files
            file.write(' '.join(map(str, self.u_etd)) % self.u_etd + '\n')
            
            if nn % 200 == 0:
                print('Time = ', float(t))      
                print('------------------------------------------------------------')

        print('Final time = ', t)
        file.close()


    ##############################################################################

    def ETDRK2(self):

        path = './ETDRK2/06/IR/'
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        ### Write simulation paramters to a file
        file_param = open(path + 'Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %f' % self.eta + '\n')
        file_param.write('Advection CFL = %.15f' % self.adv_cfl + '\n')
        file_param.write('Diffusion CFL = %.15f' % self.dif_cfl + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.write('Advection: Imag Leja' + '\n')
        file_param.write('Diffusion: Real Leja')
        file_param.close()

        file = open(path + "u_etdrk2.txt", 'w+')
        file.write(' '.join(map(str, self.u_etdrk2)) % self.u_etdrk2 + '\n')

        ## Leja points
        Leja_X = Leja_Points()
        
        ## Eigen values (Diffusion)
        eigen_min_dif = 0
        eigen_max_dif, eigen_imag_dif = Gershgorin(self.A_dif)      # Max real, imag eigen value

        ############## --------------------- ##############

        epsilon = 1e-7                               		        # Amplitude of perturbation
        t = Decimal(0.0)
        print('dt =', self.dt)

        ### Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
                    
            ############## --------------------- ##############
            
            ## Eigen values (Advection)
            eigen_min_adv = 0
            eigen_max_adv, eigen_imag_adv = Power_iteration(self.A_adv, self.u_etdrk2, 2)   # Max real, imag eigen value
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
            A_dot_u_1 = self.A_adv.dot(self.u_etdrk2**2)
            
            ### J(u) * u
            Linear_u = (self.A_adv.dot((self.u_etdrk2 + (epsilon * self.u_etdrk2))**2) - A_dot_u_1)/epsilon

            ### F(u) - (J(u) * u)
            Nonlin_u = A_dot_u_1 - Linear_u
            
            ## Linear Term
            # u_lin_adv = real_Leja_exp(self.A_adv, self.u_etdrk2, 2, self.dt, Leja_X, c_real_adv, Gamma_real_adv)
            u_lin_adv = imag_Leja_exp(self.A_adv, self.u_etdrk2, 2, self.dt, Leja_X, c_imag_adv, Gamma_imag_adv)
            
            ## Nonlinear Term 
            # u_nl_adv = real_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_real_adv, Gamma_real_adv) * self.dt
            u_nl_adv = imag_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_imag_adv, Gamma_imag_adv) * self.dt

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
            u_nl_2 = imag_Leja_phi(phi_2, (Nonlin_u2 - Nonlin_u), self.dt, Leja_X, c_imag_adv, Gamma_imag_adv) * self.dt
            
            ############## --------------------- ##############
            
            ### ETDRK2 Solution ###
            u_adv = a_n + u_nl_2
            
            ################### Diffusive Term ###################
            
            u_diff = real_Leja_exp(self.A_dif, u_adv, 1, self.dt, Leja_X, c_real_dif, Gamma_real_dif)
            
            ############## --------------------- ##############

            ### Full solution
            u_temp = u_diff
            
            ## Update u and t
            self.u_etdrk2 = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############
            
            # plt.plot(self.X, self.u_etdrk2, 'b.')
            # plt.pause(self.dt)
            # plt.clf()

            ############## --------------------- ##############

            ## Write data to files
            file.write(' '.join(map(str, self.u_etdrk2)) % self.u_etdrk2 + '\n')
            
            if nn % 200 == 0:
                print('Time = ', float(t))
                print('------------------------------------------------------------')

        print('Final time = ', t)
        file.close()


##############################################################################

# Assign values for N, tmax, and eta
N = 100
t_max = 5 * 1e-3
eta = 100

def main():
    sim = Viscous_Burgers_1D_Constant_h(N, t_max, eta)
    sim.RK4()
    # sim.ETD()
    # sim.ETDRK2()

if __name__ == "__main__":
    main()

##################################################################################

print('Time Elapsed = ', datetime.now() - startTime)
