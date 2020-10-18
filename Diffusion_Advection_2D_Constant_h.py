"""
Created on Thu Jul 16 20:11:19 2020

@author: Pranab JD

Description: -
    This code solves the linear Diffusion-Advection equation:
    du/dt = eta_x * du/dx + d^2u/dx^2 + eta_y * du/dy + d^2u/dy^2 (2D)
    using different exponential time integrators.
    Advective term - 1st order upwind scheme
    Step size is constant.
"""

import os
import shutil
import numpy as np
from decimal import *
from matplotlib import cm
import matplotlib.pyplot as plt
from Leja_Interpolation import *
import scipy.sparse.linalg as spla
from scipy.sparse import kron, identity

from datetime import datetime

startTime = datetime.now()

##############################################################################

class Diffusion_Advection_2D_Constant_h:

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
        self.sigma_init = 1.4 * 1e-3            # Initial amplitude of Gaussian
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
        u0 = np.exp((-(self.X - 0.5)**2 - (self.Y - 0.5)**2)/(2 * self.sigma_init**2))
        self.u = u0.copy()           	

    ### Parameters
    def initialize_parameters(self):
        self.dt = 0.5 * min(((self.dx**2 * self.dy**2)/(self.dx**2 + self.dy**2)), ((self.dx * self.dy)/(self.eta_y*self.dx + self.eta_x*self.dy)))
        self.nsteps = int(np.ceil(self.tmax/self.dt))           	    # number of time steps
        self.Rx = self.eta_x/self.dx
        self.Ry = self.eta_y/self.dy
        self.Fx = 1/self.dx**2
        self.Fy = 1/self.dy**2


    ### Operator matrices
    def initialize_matrices(self):
        self.Adv_x = np.zeros((self.N_x, self.N_x))         # Advection (X)
        self.Adv_y = np.zeros((self.N_y, self.N_y))         # Advection (Y)
        self.Dif_x = np.zeros((self.N_x, self.N_x))         # Diffusion (X)
        self.Dif_y = np.zeros((self.N_y, self.N_y))         # Diffusion (Y)


        for ij in range(self.N_x):
            self.Adv_x[ij, int(ij + 1) % int(self.N_x)] = self.Rx
            self.Adv_x[ij, ij % self.N_x] = - self.Rx

            self.Dif_x[ij, int(ij + 1) % int(self.N_x)] = self.Fx
            self.Dif_x[ij, ij % self.N_x] = -(2 * self.Fx)
            self.Dif_x[ij, int(ij - 1) % int(self.N_x)] = self.Fx

        for ij in range(self.N_y):
            self.Adv_y[ij, int(ij + 1) % int(self.N_y)] = self.Ry
            self.Adv_y[ij, ij % self.N_y] = - self.Ry

            self.Dif_y[ij, int(ij + 1) % int(self.N_y)] = self.Fy
            self.Dif_y[ij, ij % self.N_y] = -(2 * self.Fy)
            self.Dif_y[ij, int(ij - 1) % int(self.N_y)] = self.Fy
            

        self.A = kron(identity(self.N_y), self.Adv_x) + kron(self.Adv_y, identity(self.N_x)) + \
                 kron(identity(self.N_y), self.Dif_x) + kron(self.Dif_y, identity(self.N_x))
    
    
    ##############################################################################

    def RK4(self):

        ### Create directory
        path = './Reference Data/05/Data Files/'

        if os.path.exists(path):
            shutil.rmtree(path)                         # remove previous directory with same name
        os.makedirs(path, 0o777)                        # create directory with access rights

        ### Write simulation parameters to a file
        file_param = open('./Reference Data/05/Simulation_Parameters.txt', 'w+')
        file_param.write('N_x = %d' % self.N_x + '\n')
        file_param.write('N_y = %d' % self.N_y + '\n')
        file_param.write('eta_x = %f' % self.eta_x + '\n')
        file_param.write('eta_y = %f' % self.eta_y + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.close()

        print('------------------------------------------------------------')

        self.u_rk4 = self.u_rk4.reshape(self.N_x * self.N_y)
        print('Shape of A = ', np.shape(self.A))
        print('Shape of u = ', np.shape(self.u_rk4))

        print('------------------------------------------------------------')

        ### Time loop
        t = Decimal(0.0)
        print('dt = ', self.dt)

        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
                    print('Final time = ', float(t) + self.dt, '\n', nn)

            k1 = self.dt * self.A.dot(self.u_rk4)
            k2 = self.dt * self.A.dot(self.u_rk4 + k1/2)
            k3 = self.dt * self.A.dot(self.u_rk4 + k2/2)
            k4 = self.dt * self.A.dot(self.u_rk4 + k3)

            ## Solution
            u_var = self.u_rk4 + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

            ## Update u and t
            self.u_rk4 = u_var.copy()
            t = Decimal(t) + Decimal(self.dt)

            if nn % 50 == 0:
                            
                ### Write matrix [u(X, Y)] to file
                file = open(path + '/%d.txt' % nn, 'w+')
                np.savetxt(file, self.u_rk4.reshape(self.N_y, self.N_x), fmt = '%.25f')
                file.close()
                
                print('Time = ', float(t))
                print('------------------------------------------------------------')

            # plt.imshow(self.u_rk4.reshape(self.N_y, self.N_x), cmap = cm.hot, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt/3)

        ### Write final data to file
        file = open('./Reference Data/05/Final_data.txt', 'w+')
        np.savetxt(file, self.u_rk4.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file.close()
        
        print('Final time = ', float(t))

    ##############################################################################

    def Exponential_Integrator(self):

        ### Create directory
        path = './Exponential Integrator/05/Imag Leja/'
        os.makedirs(os.path.dirname(path), exist_ok = True)

        ### Write simulation parameters to a file
        file_param = open('./Exponential Integrator/05/Simulation_Parameters.txt', 'w+')
        file_param.write('N_x = %d' % self.N_x + '\n')
        file_param.write('N_y = %d' % self.N_y + '\n')
        file_param.write('eta_x = %f' % self.eta_x + '\n')
        file_param.write('eta_y = %f' % self.eta_y + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.close()

        ############## --------------------- ##############

        print('------------------------------------------------------------')

        self.u = self.u.reshape(self.N_x * self.N_y)
        print('Shape of A = ', np.shape(self.A))
        print('Shape of u = ', np.shape(self.u))

        print('------------------------------------------------------------')

        eigen_min = 0
        eigen_max, eigen_imag = Gershgorin(self.A)            # Max real & imag eigen values

        c_real = 0.5 * (eigen_max + eigen_min)
        Gamma_real = 0.25 * (eigen_max - eigen_min)
        c_imag = 0
        Gamma_imag = 0.25 * (eigen_imag - (-eigen_imag))

        ############## --------------------- ##############

        ### Time loop
        t = Decimal(0.0)
        print('dt = ', self.dt)

        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)
                    print('Final time = ', float(t) + self.dt)
                    
            ### Solution
            # u_sol, its_sol = real_Leja_exp(self.A, self.u, self.dt, c_real, Gamma_real)
            u_sol, its_sol = imag_Leja_exp(self.A, self.u, self.dt, c_imag, Gamma_imag)

            ### Update t and u
            self.u = u_sol.copy()
            t = Decimal(t) + Decimal(self.dt)

            # plt.imshow(self.u.reshape(self.N_y, self.N_x), cmap = cm.hot, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt/3)

            if nn % 100 == 0:
                print('Time = ', float(t))
                print('------------------------------------------------------------')

        ### Write final data to file
        file = open(path + 'Final_data.txt', 'w+')
        np.savetxt(file, self.u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file.close()
        
        print('Final time = ', float(t))

##############################################################################

# Assign values for N, tmax, and eta
N_x = 100
N_y = 100
tmax = 1e-3
eta_x = 100
eta_y = 300

def main():
    sim = Diffusion_Advection_2D_Constant_h(N_x, N_y, tmax, eta_x, eta_y)
    # sim.RK4()
    # sim.Crank_Nicolson()
    sim.Exponential_Integrator()
    # sim.SDIRK23()

if __name__ == "__main__":
    main()

##############################################################################

print('Time Elapsed = ', datetime.now() - startTime)
