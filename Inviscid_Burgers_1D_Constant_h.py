"""
Created on Tue Apr 07 23:10:04 2020

@author: Pranab JD

Description: -
    This code solves the inviscid Burgers' Equation:
    du/dt = eta * d(u^2)/dx + g(u) (1D)
    using different exponential time integrators.
    Advective term - 3rd order upwind scheme
    Step size is constant.

"""

import os
import shutil
import numpy as np
from decimal import *
from Leja_Header import *
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from Integrators_1_matrix import *

from datetime import datetime

startTime = datetime.now()
getcontext().prec = 16                  # Precision for decimal numbers

##############################################################################

class Inviscid_Burgers_1D_Constant_h:

    def __init__(self, N, tmax, eta):
        self.N = int(N)                 # Number of points along X
        self.xmin = 0                   # Min value of X
        self.xmax = 1                   # Max value of X
        self.tmax = tmax                # Maximum time
        self.eta = eta                  # Peclet number
        self.epsilon_1 = 0.01           # Amplitude of 1st sine wave
        self.epsilon_2 = 0.01           # Amplitude of 2nd sine wave
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
        u0 = 2 + self.epsilon_1 * np.sin(2 * np.pi * self.X) + self.epsilon_2 * np.sin(8 * np.pi * self.X + 0.3)
        self.u = u0.copy()                           
        self.u_rk4 = u0.copy()                           # Reference solution

    ### Parameters
    def initialize_parameters(self):
        self.adv_cfl = self.dx/abs(self.eta)
        print('CFL time: ', self.adv_cfl)
        self.dt = 0.1 * self.adv_cfl                     # N * CFL condition
        self.nsteps = int(np.ceil(self.tmax/self.dt))    # number of time steps
        self.R = 1./6. * self.eta/self.dx    

    ### Operator matrices
    def initialize_matrices(self):
        self.A = np.zeros((self.N, self.N))

        ## Factor of 1/2 - conservative Burgers' equation
        for ij in range(self.N):
            self.A[ij, int(ij + 2) % self.N] = - self.R/2
            self.A[ij, int(ij + 1) % self.N] = 6 * self.R/2
            self.A[ij, ij % self.N] = -3 * self.R/2
            self.A[ij, int(ij - 1) % self.N] = -2 * self.R/2

        self.A = csr_matrix(self.A)

    ##############################################################################

    def Reference_Solution(self):
        
        ### Create directory
        path = os.path.expanduser("~/PrJD/Burgers' Equation/1D/Inviscid/Constant/C - 100/RK4/dt 0.1/")
        
        if os.path.exists(path):
            shutil.rmtree(path)                         # remove previous directory with same name
        os.makedirs(path, 0o777)                        # create directory with access rights

        ### Write simulation parameters to a file
        file_param = open(path + 'Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %f' % self.eta + '\n')
        file_param.write('CFL time = %.15f' % self.adv_cfl + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax)
        file_param.close()
        
        file = open(path + "u_ref.txt", 'w+')
        file.write(' '.join(map(str, self.u)) % self.u + '\n')
        
        t = Decimal(0.0)
        print('dt =', self.dt)
        
        ### Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
                    
            ############## --------------------- ##############
        
            u_rk4 = RK4(self.A, 2, self.u, self.dt)[0]
            self.u = u_rk4.copy()
            t = Decimal(t) + Decimal(self.dt)
            
            ############## --------------------- ##############

            # plt.plot(self.X, u_rk4, 'b.')
            # plt.pause(self.dt/2)
            # plt.clf()

            ############## --------------------- ##############
        
            ## Write data to files
            file.write(' '.join(map(str, self.u)) % self.u + '\n')
            
            if nn % 200 == 0:
                print('Time = ', float(t))      
                print('------------------------------------------------------------')
             
        ### Write final data to separate file   
        file_final = open(path + "Final_data.txt", 'w+')
        file_final.write(' '.join(map(str, self.u)) % self.u + '\n')
        file_final.close()
        file.close()
        print('Final time = ', t)
    
    
    ##############################################################################

    def Solution(self, u, dt):
        
        ## Leja points
        Leja_X = Leja_Points()
    
        ############## --------------------- ##############
        
        ## Eigen values (Advection)
        eigen_min_adv = 0
        eigen_max_adv, eigen_imag_adv, its_1 = Power_iteration(self.A, u, 2)       # Max real, imag eigen value
        eigen_max_adv = eigen_max_adv * 1.25                                       # Safety factor
        eigen_imag_adv = eigen_imag_adv * 1.125                                    # Safety factor
        
        ## c and gamma
        c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
        Gamma_real_adv = 0.25 * (eigen_max_adv - eigen_min_adv)
        c_imag_adv = 0
        Gamma_imag_adv = 0.25 * (eigen_imag_adv - (- eigen_imag_adv))
        
        ############## --------------------- ##############
        
        ### Matrix-vector function
        f_u = self.A.dot(self.u**2)
        
        u_temp = EXPRB42(self.A, 2, u, dt, Leja_X, c_imag_adv, Gamma_imag_adv)[0]
        
        ############## --------------------- ##############
        
        ## Update u
        u = u_temp.copy()
        
        return u
    
    ##############################################################################
    
    def run(self):
        
        ### Create directory
        path = os.path.expanduser("~/PrJD/Burgers' Equation/1D/Inviscid/Constant/C - 100/EXPRB42/dt 0.1/")
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        if os.path.exists(path):
            shutil.rmtree(path)                         # remove previous directory with same name
        os.makedirs(path, 0o777)                        # create directory with access rights
        
        ### Write simulation parameters to a file
        file_param = open(path + '/Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %f' % self.eta + '\n')
        file_param.write('Advection CFL = %.5e' % self.adv_cfl + '\n')
        file_param.write('Simulation time = %e' % self.tmax)
        file_param.close()
        
        ## Create files
        file = open(path + "u.txt", 'w+')
        
        ## Write initial value of u to files
        file.write(' '.join(map(str, self.u)) % self.u + '\n')
        
        t = Decimal(0.0)                            # Time
        
        ### Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
                    
            ############## --------------------- ##############
                    
            u = self.Solution(self.u, self.dt)

            t = Decimal(t) + Decimal(self.dt)
            self.u = u.copy()
            
            ### Write data to files
            file.write(' '.join(map(str, u)) % u + '\n')
            
            ############## --------------------- ##############

            ### Test plots
            # plt.plot(self.X, self.u, 'b.')
            # plt.pause(self.dt/2)
            # plt.clf()
            
            ############## --------------------- ##############
        
        ### Write final data to separate file   
        file_final = open(path + "Final_data.txt", 'w+')
        file_final.write(' '.join(map(str, self.u)) % self.u + '\n')
        file_final.close()
        file.close()
        print('Final time = ', t)
        
##############################################################################

### Assign values for N, tmax, and eta
N = 100
t_max = 5 * 1e-3
eta = 100

def main():
    sim = Inviscid_Burgers_1D_Constant_h(N, t_max, eta)
    # sim.Reference_Solution()
    # sim.run()
    
if __name__ == "__main__":
    main()

##################################################################################

print('Time Elapsed = ', datetime.now() - startTime)