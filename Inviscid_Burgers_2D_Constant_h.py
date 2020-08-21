"""
Created on Tue Apr 07 23:10:04 2020

@author: Pranab JD

Description: -
    This code solves the inviscid Burgers' Equation:
    du/dt = eta_x * d(u^2)/dx + eta_y * d(u^2)/dy (2D)
    using RK4, Crank-Nicolson, SDIRK23, ETD,
    and ETDRK2. 
    Advective term - 1st order upwind scheme
    Step size is constant.

"""

import os
import shutil
import numpy as np
from decimal import *
from Leja_Header import *
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, identity

from datetime import datetime

startTime = datetime.now()
getcontext().prec = 8                   # Precision for decimal numbers

##############################################################################

class Inviscid_Burgers_2D_Constant_h:
    
    def __init__(self, N_x, N_y, tmax, eta_x, eta_y):
        self.N_x = N_x                          # Number of points along X
        self.N_y = N_y                          # Number of points along Y
        self.xmin = 0                           # Min value of X
        self.xmax = 1                           # Max value of X
        self.ymin = 0                           # Min value of Y
        self.ymax = 1                           # Max value of Y
        self.eta_x = eta_x                      # Peclet number along X
        self.eta_y = eta_y                      # Peclet number along Y
        self.tmax = tmax                        # Maximum time
        self.epsilon_1 = 0.01                   # Amplitude of 1st sine wave
        self.epsilon_2 = 0.01                   # Amplitude of 2nd sine wave
        self.gamma = (3 + 3**0.5)/6             # gamma for SDIRK23
        self.initialize_spatial_domain()
        self.initialize_U()
        self.initialize_parameters()
        self.initialize_matrices()
        
    ### Discretize the spatial domain
    def initialize_spatial_domain(self):
        self.dx = (self.xmax - self.xmin)/self.N_x
        self.dy = (self.ymax - self.ymin)/self.N_y
        self.X = np.linspace(self.xmin, self.xmax, self.N_x, endpoint = False)
        self.Y = np.linspace(self.ymin, self.ymax, self.N_y, endpoint = False)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        
    ### Initial distribution    
    def initialize_U(self):
        u0 = 2 + self.epsilon_1 * (np.sin(2 * np.pi * self.X) + np.sin(2 * np.pi * self.Y)) \
        	   + self.epsilon_2 * (np.sin(8 * np.pi * self.X + 0.3) + np.sin(8 * np.pi * self.Y + 0.3))
        self.u_rk4 = u0.copy()          # RK4
        self.u_CN = u0.copy()           # Crank-Nicolson
        self.u_23 = u0.copy()           # SDIRK23
        self.u_etd = u0.copy()          # ETD
        self.u_etdrk2 = u0.copy()       # ETDRK2
        
    ### Parameters  
    def initialize_parameters(self):
        self.dt = 0.25 * ((self.dx * self.dy)/(self.eta_y*self.dx + self.eta_x*self.dy))
        self.nsteps = int(np.ceil(self.tmax/self.dt))            	# Number of time steps
        self.Rx = self.eta_x/self.dx
        self.Ry = self.eta_y/self.dy
        
    ### Operator matrices    
    def initialize_matrices(self):    
        self.Adv_x = np.zeros((self.N_x, self.N_x))         # Advection (X)
        self.Adv_y = np.zeros((self.N_y, self.N_y))         # Advection (Y)

        for ij in range(self.N_x):
            self.Adv_x[ij, int(ij + 1) % int(self.N_x)] = self.Rx/2                       
            self.Adv_x[ij, ij % int(self.N_x)] = - self.Rx/2                          
        
        for ij in range(self.N_y):    
            self.Adv_y[ij, int(ij + 1) % int(self.N_y)] = self.Ry/2                         
            self.Adv_y[ij, ij % int(self.N_y)] = - self.Ry/2                          
            
        self.A = kron(identity(self.N_y), self.Adv_x) + kron(self.Adv_y, identity(self.N_x))
        
       
    ##############################################################################
        
    def RK4(self):
        
        ### Create directory
        path = './Reference Data/05/Data Files/'
        
        if os.path.exists(path):
            shutil.rmtree(path)                         # remove previous directory with same name
        os.makedirs(path, 0o777)                        # create directory with access rights
        
        ### Write simulation paramters to a file
        file_param = open('./Reference Data/05/Simulation_Parameters.txt', 'w+')
        file_param.write('N_x = %d' % self.N_x + '\n')
        file_param.write('N_y = %d' % self.N_y + '\n')
        file_param.write('eta_x = %f' % self.eta_x + '\n')
        file_param.write('eta_y = %f' % self.eta_y + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.close()

        ## Reshape u into 1D array 
        self.u_rk4 = self.u_rk4.reshape(self.N_x * self.N_y)
 
        t = Decimal(0.0)
        print('dt = ', self.dt)
              
        ## Time loop     
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
                
            k1 = self.dt * self.A.dot(self.u_rk4**2)         
            k2 = self.dt * self.A.dot((self.u_rk4 + k1/2)**2)
            k3 = self.dt * self.A.dot((self.u_rk4 + k2/2)**2)           
            k4 = self.dt * self.A.dot((self.u_rk4 + k3)**2)
            
            ## Solution
            u_var = self.u_rk4 + 1./6.*(k1 + 2*k2 + 2*k3 + k4)
            
            ## Update u and t
            self.u_rk4 = u_var.copy()
            t = Decimal(t) + Decimal(self.dt)
            
            if nn % 100 == 0:
                
                print('Time = ', float(t))
                
                ### Write matrix [u(X, Y)] to file
                file = open(path + '/%d.txt' % nn, 'w+')
                np.savetxt(file, self.u_rk4.reshape(self.N_y, self.N_x), fmt = '%.25f')
                file.close()
                
                file_time.write('{}'.format(t) + '\n')
                
                print('------------------------------------------------------------')
            
            # plt.imshow(self.u_rk4.reshape(self.N_y, self.N_x), cmap = cm.jet, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt/2)
            
        ### Write final data to file
        file = open('./Reference Data/05/Final_data.txt', 'w+')
        np.savetxt(file, self.u_rk4.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file.close()
        
        print('Final time = ', float(t))


    ##############################################################################
    
    def ETD(self):
        
        ## Create directory
        path = './ETD/05/Real/'
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        ### Write simulation paramters to a file
        file_param = open('./ETD/05/Simulation_Parameters.txt', 'w+')
        file_param.write('N_x = %d' % self.N_x + '\n')
        file_param.write('N_y = %d' % self.N_y + '\n')
        file_param.write('eta_x = %f' % self.eta_x + '\n')
        file_param.write('eta_y = %f' % self.eta_y + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.close()
        
        print('------------------------------------------------------------')        

        self.u_etd = self.u_etd.reshape(self.N_x * self.N_y)
        print('Shape of A = ', np.shape(self.A))
        print('Shape of u = ', np.shape(self.u_etd))
        
        print('------------------------------------------------------------')

        ## Leja points
        Leja_X = Leja_Points()
        
        ############## --------------------- ############## 
        
        epsilon = 1e-7                               		          # Amplitude of perturbation
        t = Decimal(0.0)
        print('dt = ', self.dt)
        
        ### Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
            
            ############## --------------------- ##############
            
            ## Eigen Values
            eigen_min = 0
            eigen_max, eigen_imag = Power_iteration(self.A, self.u_etd)            # Max real, imag eigen value
            eigen_max = eigen_max * 1.2
            eigen_imag = eigen_imag * 1.125
            
            c_real = 0.5 * (eigen_max + eigen_min)
            Gamma_real = 0.25 * (eigen_max - eigen_min)
            c_imag = 0
            Gamma_imag = 0.25 * (eigen_imag - (-eigen_imag))      
            
            ############## --------------------- ##############

            ### J(u) * u
            Linear_u = (self.A.dot((self.u_etd + (epsilon * self.u_etd))**2) - (self.A.dot(self.u_etd**2)))/epsilon

            ### F(u) - (J(u) * u)
            Nonlin_u = self.A.dot(self.u_etd**2) - Linear_u

            ################### Linear Term ###################

            u_lin_temp = real_Leja_exp(self, Leja_X, c_real, Gamma_real, self.u_etd)
            # u_lin_temp = imag_Leja_exp(self, Leja_X, c_imag, Gamma_imag, self.u_etd)
            u_lin = u_lin_temp.copy()

            ################# Nonlinear Term ##################

            u_nl_temp = real_Leja_phi(self, phi_1, Leja_X, c_real, Gamma_real, Nonlin_u) * self.dt
            # u_nl_temp = imag_Leja_phi(self, phi_1, Leja_X, c_imag, Gamma_imag, Nonlin_u) * self.dt
            u_nl = u_nl_temp.copy()

            ############## --------------------- ##############

            ### Full Solution ###
            u_var = u_lin + u_nl

            ## Update u and t
            self.u_etd = u_var.copy()
            t = Decimal(t) + Decimal(self.dt)
            
            # plt.imshow(self.u_etd.reshape(self.N_y, self.N_x), cmap = cm.jet, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt/3)
            
            if nn % 100 == 0:
                print('Time = ', float(t))
                print('------------------------------------------------------------')
                
        ### Write final data to file
        file = open(path + 'Final_data.txt', 'w+')
        np.savetxt(file, self.u_etd.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file.close()
        
        print('Final time = ', float(t))
            
            
    ##############################################################################
    
    def ETDRK2(self):
        
        ## Create directory
        path = './ETDRK2/05/Real/'
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        ### Write simulation paramters to a file
        file_param = open('./ETDRK2/05/Simulation_Parameters.txt', 'w+')
        file_param.write('N_x = %d' % self.N_x + '\n')
        file_param.write('N_y = %d' % self.N_y + '\n')
        file_param.write('eta_x = %f' % self.eta_x + '\n')
        file_param.write('eta_y = %f' % self.eta_y + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.close()
        
        print('------------------------------------------------------------')        

        self.u_etdrk2 = self.u_etdrk2.reshape(self.N_x * self.N_y)
        print('Shape of A = ', np.shape(self.A))
        print('Shape of u = ', np.shape(self.u_etdrk2))
        
        print('------------------------------------------------------------')

        ## Leja points
        Leja_X = Leja_Points()
                
        ############## --------------------- ##############
            
        epsilon = 1e-7                               		          # Amplitude of perturbation
        t = Decimal(0.0)
        print('dt = ', self.dt)
        
        ### Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
                    print('Final time = ', float(t) + self.dt)
        
            ############## --------------------- ##############
            
            ## Eigen Values
            eigen_min = 0
            eigen_max, eigen_imag = Power_iteration(self.A, self.u_etdrk2)            # Max real, imag eigen value
            eigen_max = eigen_max * 1.2
            eigen_imag = eigen_imag * 1.125
            
            c_real = 0.5 * (eigen_max + eigen_min)
            Gamma_real = 0.25 * (eigen_max - eigen_min)
            c_imag = 0
            Gamma_imag = 0.25 * (eigen_imag - (-eigen_imag))      
            
            ############## --------------------- ##############

            ### J(u) * u
            Linear_u = (self.A.dot((self.u_etdrk2 + (epsilon * self.u_etdrk2))**2) - self.A.dot(self.u_etdrk2**2))/epsilon

            ### F(u) - (J(u) * u)
            Nonlin_u = self.A.dot(self.u_etdrk2**2) - Linear_u

            ################### Linear Term ###################

            u_lin_temp = real_Leja_exp(self, Leja_X, c_real, Gamma_real, self.u_etdrk2)
            # u_lin_temp = imag_Leja_exp(self, Leja_X, c_imag, Gamma_imag, self.u_etdrk2)
            u_lin = u_lin_temp.copy()

            ################# Nonlinear Term 1 ################

            u_nl_temp = real_Leja_phi(self, phi_1, Leja_X, c_real, Gamma_real, Nonlin_u) * self.dt
            # u_nl_temp = imag_Leja_phi(self, phi_1, Leja_X, c_imag, Gamma_imag, Nonlin_u) * self.dt
            u_nl = u_nl_temp.copy()

            ############## --------------------- ##############

            ### ETD1 Solution ###
            a_n = u_lin + u_nl

            ############## --------------------- ##############
            
            ### J(u) * u
            Linear_u2 = (self.A.dot((a_n + (epsilon * a_n))**2) - self.A.dot(a_n**2))/epsilon

            ### F(u) - (J(u) * u)
            Nonlin_u2 = self.A.dot(a_n**2) - Linear_u2
            
            ################# Nonlinear Term 2 ################

            u_nl_temp_2 = real_Leja_phi(self, phi_2, Leja_X, c_real, Gamma_real, (Nonlin_u2 - Nonlin_u)) * self.dt
            # u_nl_temp_2 = imag_Leja_phi(self, phi_2, Leja_X, c_imag, Gamma_imag, (Nonlin_u2 - Nonlin_u)) * self.dt
            u_nl_2 = u_nl_temp_2.copy()
            
            ############## --------------------- ##############

            ### Full solution
            u_temp = a_n + u_nl_2
            
            ## Update u and t
            self.u_etdrk2 = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############

            # plt.imshow(self.u_etdrk2.reshape(self.N_y, self.N_x), cmap = cm.jet, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt/2)
            
            if nn % 100 == 0:
                print('Time = ', float(t))
                print('------------------------------------------------------------')
                
        ### Write final data to file
        file = open(path + 'Final_data.txt', 'w+')
        np.savetxt(file, self.u_etdrk2.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file.close()
        
        print('Final time = ', float(t))

              
##############################################################################

## Assign values for N, tmax, and eta
N_x = 300
N_y = 300
tmax = 3 * 1e-3
eta_x = 200
eta_y = 200

### 1/(N_x - 1) * eta_x = 1/(N_y - 1) * eta_y for same numerical diffusion along X and Y

def main():
    sim = Inviscid_Burgers_2D_Constant_h(N_x, N_y, tmax, eta_x, eta_y)
    # sim.RK4()
    # sim.Crank_Nicolson()
    # sim.SDIRK23()
    # sim.ETD()
    sim.ETDRK2()
    
     
if __name__ == "__main__":
    main()

##################################################################################

print('Time Elapsed = ', datetime.now() - startTime)
