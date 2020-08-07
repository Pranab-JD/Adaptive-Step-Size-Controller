"""
Created on Thu Jul 16 20:11:19 2020

@author: Pranab JD

Description: -
    This code solves the linear Diffusion-Advection equation:
    du/dt = eta_x * du/dx + d^2u/dx^2 + eta_y * du/dy + d^2u/dy^2 (2D)
    using RK4, Crank-Nicolson, SDIRK23, and ETD schemes.
    Advective term - 1st order upwind scheme
    Step size is constant.
"""

import os
import shutil
import numpy as np
from decimal import *
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, identity

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
        self.u_CN = u0.copy()           	# Crank-Nicolson
        self.u_EI = u0.copy()               # Exponential Integration
        self.u_23 = u0.copy()           	# SDIRK23
        self.u_rk4 = u0.copy()           	# RK4


    ### Parameters
    def initialize_parameters(self):
        self.dt = 0.25 * min(((self.dx**2 * self.dy**2)/(self.dx**2 + self.dy**2)), ((self.dx * self.dy)/(self.eta_y*self.dx + self.eta_x*self.dy)))
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

        def Leja_Points():
            """
            Load Leja points from binary file

            """
            dt = np.dtype("f8")
            return np.fromfile('real_leja_d.bin', dtype = dt)

        Leja_X = Leja_Points()

        def Divided_Difference(X, func):
            """
            Parameters
            ----------
            X       : Leja points
            func    : func(X)

            Returns
            -------
            div_diff : Polynomial coefficients

            """

            div_diff = func(X)

            for ii in range(1, int(len(X)/10)):
                for jj in range(ii):
                    div_diff[ii] = (div_diff[jj] - div_diff[ii])/(X[jj] - X[ii])

            return div_diff

        def Gershgorin(A):
            """
            Parameters
            ----------
            A       : N x N matrix A
            Returns
            -------
            eig_real : Largest real eigen value
            eig_imag : Largest imaginary eigen value
            """
        
            A_Herm = (A + A.T.conj())/2
            A_SkewHerm = (A - A.T.conj())/2
        
            row_sum_real = np.zeros(np.shape(A)[0])
            row_sum_imag = np.zeros(np.shape(A)[0])
        
            for ii in range(len(row_sum_real)):
                row_sum_real[ii] = np.sum(abs(A_Herm[ii, :]))
                row_sum_imag[ii] = np.sum(abs(A_SkewHerm[ii, :]))
        
            eig_real = np.max(row_sum_real)
            eig_imag = np.max(row_sum_imag)
        
            return eig_real, eig_imag

        ############## --------------------- ##############

        def real_Leja(u):
            """
            Parameters
            ----------
            u                : Vector u
            
            Returns
            -------
            np.real(u_real)  : Polynomial interpolation of u
                               at real Leja points
            
            """

            def func(xx):
                return np.exp(self.dt * (c_real + Gamma_real*xx))

            ## Polynomial coefficients
            coeffs = Divided_Difference(Leja_X, func)

            ## a_0 term
            poly = u.copy()
            poly = coeffs[0] * poly
            
            ## a_1, a_2 .... a_n terms
            max_Leja_pts = 400
            y = u.copy()
            poly_tol = 1e-7
            scale_fact = 1/Gamma_real                                    # Re-scaling factor

            for ii in range(1, max_Leja_pts):

                shift_fact = -c_real * scale_fact - Leja_X[ii - 1]       # Re-shifting factor

                u_temp = y.copy()
                y = y * shift_fact
                y = y + scale_fact * self.A.dot(u_temp)
                poly = poly + coeffs[ii] * y

                ## If new number (next order) to be added < tol, ignore it
                if (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii]) < poly_tol:
                    # print('No. of Leja points used = ', ii)
                    break

                if ii >= max_Leja_pts - 1:
                    print('ERROR: max number of Leja iterations reached')

            ## Solution
            u_real = poly.copy()

            return np.real(u_real)

        def imag_Leja(u):
            """
            Parameters
            ----------
            u                : Vector u
            
            Returns
            ----------
            np.real(u_imag)  : Polynomial interpolation of u
                               at imaginary Leja points
            
            """

            def func(xx):
                return np.exp(1j * self.dt * (c_imag + Gamma_imag*xx))

            ## Polynomial coefficients
            coeffs = Divided_Difference(Leja_X, func)

            ## a_0 term
            poly = u.copy() + 0 * 1j
            poly = coeffs[0] * poly

            ## a_1, a_2 .... a_n terms
            max_Leja_pts = 400
            y = u.copy() + 0 * 1j
            poly_tol = 1e-7
            scale_fact = 1/Gamma_imag                                    # Re-scaling factor

            for ii in range(1, max_Leja_pts):

                shift_fact = -c_imag * scale_fact - Leja_X[ii - 1]       # Re-shifting factor

                u_temp = y.copy()
                y = y * shift_fact
                y = y + scale_fact * self.A.dot(u_temp) * (-1j)
                poly = poly + coeffs[ii] * y

                ## If new number (next order) to be added < tol, ignore it
                if (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii]) < poly_tol:
                    # print('No. of Leja points used = ', ii)
                    break

                if ii >= max_Leja_pts - 1:
                    print('ERROR: max number of Leja iterations reached')

            ## Solution
            u_imag = poly.copy()

            return np.real(u_imag)

        print('------------------------------------------------------------')

        self.u_EI = self.u_EI.reshape(self.N_x * self.N_y)
        print('Shape of A = ', np.shape(self.A))
        print('Shape of u = ', np.shape(self.u_EI))

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
            # u_sol = real_Leja(self.u_EI)
            u_sol = imag_Leja(self.u_EI)

            ### Update t and u
            self.u_EI = u_sol.copy()
            t = Decimal(t) + Decimal(self.dt)

            # plt.imshow(self.u_EI.reshape(self.N_y, self.N_x), cmap = cm.hot, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt/3)

            if nn % 100 == 0:
                print('Time = ', float(t))
                print('------------------------------------------------------------')

        ### Write final data to file
        file = open(path + 'Final_data.txt', 'w+')
        np.savetxt(file, self.u_EI.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file.close()
        
        print('Final time = ', float(t))

##############################################################################

# Assign values for N, tmax, and eta
N_x = 100
N_y = 100
tmax = 3 * 1e-3
eta_x = 100
eta_y = 100

def main():
    sim = Diffusion_Advection_2D_Constant_h(N_x, N_y, tmax, eta_x, eta_y)
    # sim.RK4()
    # sim.Crank_Nicolson()
    # sim.Exponential_Integrator()
    # sim.SDIRK23()

if __name__ == "__main__":
    main()

##############################################################################

print('Time Elapsed = ', datetime.now() - startTime)
