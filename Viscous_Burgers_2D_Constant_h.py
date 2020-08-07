"""
Created on Wed Aug  5 16:45:54 2020

@author: Pranab JD

Description: -
    This code solves the viscous Burgers' equation:
    du/dt = d^2u/dx^2 + d^2u/dy^2
    + eta_x * d(u^2)/dx + eta_y * d(u^y)/dx (2D)
    using RK4, ETD, and ETDRK2.
    Advective term - 3rd order upwind scheme
    Step size is constant.
"""

import os
import shutil
import numpy as np
from decimal import *
from Leja_Header import *
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.sparse import kron, identity

from datetime import datetime

startTime = datetime.now()
getcontext().prec = 12                   # Precision for decimal numbers

##############################################################################

class Viscous_Burgers_2D_Constant_h:
    
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
        self.sigma = 0.02          		        # Amplitude of Gaussian
        self.x_0 = 0.9           		        # Center of the Gaussian (X)
        self.y_0 = 0.9           		        # Center of the Gaussian (Y)
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
        u0 =  1 + (np.exp(1 - (1/(1 - (2 * self.X - 1)**2)))) + (np.exp(1 - (1/(1 - (2 * self.Y - 1)**2)))) \
                + 1./2. * np.exp(-((self.X - self.x_0)**2 + (self.Y - self.y_0)**2)/(2 * self.sigma**2))
        self.u_rk4 = u0.copy()          # RK4
        self.u_etd = u0.copy()          # ETD
        self.u_etdrk2 = u0.copy()       # ETDRK2

        
    ### Parameters  
    def initialize_parameters(self):
        self.adv_cfl = ((self.dx * self.dy)/(self.eta_y*self.dx + self.eta_x*self.dy))
        self.dif_cfl = ((self.dx**2 * self.dy**2)/(self.dx**2 + self.dy**2))
        print('Advection CFL: ', self.adv_cfl)
        print('Diffusion CFL: ', self.dif_cfl)
        self.dt = 0.01 * min(self.adv_cfl, self.dif_cfl)  # N * CFL condition
        self.nsteps = int(np.ceil(self.tmax/self.dt))     # Number of time steps
        self.Rx = 1./6. * self.eta_x/self.dx
        self.Ry = 1./6. * self.eta_y/self.dy
        self.Fx = 1/self.dx**2                            # Fourier mesh number
        self.Fy = 1/self.dy**2                            # Fourier mesh number
        
        
    ### Operator matrices    
    def initialize_matrices(self):    
        self.Adv_x = np.zeros((self.N_x, self.N_x))       # Advection (X)
        self.Adv_y = np.zeros((self.N_y, self.N_y))       # Advection (Y)
        self.Dif_x = np.zeros((self.N_x, self.N_x))       # Diffusion (X)
        self.Dif_y = np.zeros((self.N_y, self.N_y))       # Diffusion (Y)
    
        for ij in range(self.N_x):
            self.Adv_x[ij, int(ij + 2) % self.N_x] = - self.Rx/2
            self.Adv_x[ij, int(ij + 1) % self.N_x] = 6 * self.Rx/2
            self.Adv_x[ij, ij % self.N_x] = -3 * self.Rx/2
            self.Adv_x[ij, int(ij - 1) % self.N_x] = -2 * self.Rx/2
            
            self.Dif_x[ij, int(ij + 1) % int(self.N_x)] = self.Fx
            self.Dif_x[ij, ij % self.N_x] = -(2 * self.Fx)
            self.Dif_x[ij, int(ij - 1) % int(self.N_x)] = self.Fx
        
        for ij in range(self.N_y):    
            self.Adv_y[ij, int(ij + 2) % self.N_y] = - self.Ry/2
            self.Adv_y[ij, int(ij + 1) % self.N_y] = 6 * self.Ry/2
            self.Adv_y[ij, ij % self.N_y] = -3 * self.Ry/2
            self.Adv_y[ij, int(ij - 1) % self.N_y] = -2 * self.Ry/2
            
            self.Dif_y[ij, int(ij + 1) % int(self.N_y)] = self.Fy
            self.Dif_y[ij, ij % self.N_y] = -(2 * self.Fy)
            self.Dif_y[ij, int(ij - 1) % int(self.N_y)] = self.Fy
            
        self.A_adv = kron(identity(self.N_y), self.Adv_x) + kron(self.Adv_y, identity(self.N_x))
        self.A_dif = kron(identity(self.N_y), self.Dif_x) + kron(self.Dif_y, identity(self.N_x))


    ##############################################################################

    def RK4(self):

        t = Decimal(0.0)

        ## Create directory
        path = './Reference Data/02/'
        path_files = './Reference Data/02/Data Files'
        
        if os.path.exists(path_files):
            shutil.rmtree(path_files)                         # remove previous directory with same name
        os.makedirs(path_files, 0o777)                        # create directory with access rights
        
        ### Write simulation paramters to a file
        file_param = open(path + 'Simulation_Parameters.txt', 'w+')
        file_param.write('N_x = %d' % self.N_x + '\n')
        file_param.write('N_y = %d' % self.N_y + '\n')
        file_param.write('eta_x = %f' % self.eta_x + '\n')
        file_param.write('eta_y = %f' % self.eta_y + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax)
        file_param.close()
        
        file_time = open(path + "time.txt", 'w+')
        file_time.write('{}'.format(t) + '\n')
        
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
            
            if nn % 100 == 0 or nn == self.nsteps - 1:
                
                print('Time = ', float(t))
                
                ### Write matrix [u(X, Y)] to file
                file = open(path_files + '/%d.txt' % nn, 'w+')
                np.savetxt(file, self.u_rk4.reshape(self.N_y, self.N_x), fmt = '%.25f')
                file.close()
                
                file_time.write('{}'.format(t) + '\n')
                
                print('------------------------------------------------------------')
                
            # plt.imshow(self.u_rk4.reshape(self.N_y, self.N_x), cmap = cm.jet, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt/2)

        ### Write final data to file
        file_f = open(path + "Final_data.txt", 'w+')
        file_f.write(' '.join(map(str, self.u_rk4)) % self.u_rk4 + '\n')
        file_f.close()

        file_time.close()
        print('Final time = ', t)
        
        
    ##############################################################################





















##############################################################################

## Assign values for N, tmax, and eta
N_x = 200
N_y = 100
tmax = 5 * 1e-3
eta_x = 500
eta_y = 250

### 1/(N_x - 1) * eta_x = 1/(N_y - 1) * eta_y for same numerical diffusion along X and Y

def main():
    sim = Viscous_Burgers_2D_Constant_h(N_x, N_y, tmax, eta_x, eta_y)
    sim.RK4()
    # sim.ETD()
    # sim.ETDRK2()

if __name__ == "__main__":
    main()

##################################################################################

print('Time Elapsed = ', datetime.now() - startTime)
