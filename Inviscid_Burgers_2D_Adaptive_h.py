"""
Created on Wed Sep 06 20:13:46 2020

@author: Pranab JD

Description: -
    This code solves the inviscid Burgers' equation:
    du/dt = eta_x * d(u^2)/dx + eta_y * d(u^2)/dy (2D)
    using different time integrators.
    Advective term - 3rd order upwind scheme
    Adaptive step size is implemented.
    
"""

import os
import shutil
import numpy as np
from decimal import *
from Leja_Interpolation import *
from matplotlib import cm
import matplotlib.pyplot as plt
from Adaptive_Step_Size import *
from Integrators_1_matrix import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import kron, identity

from datetime import datetime

startTime = datetime.now()
getcontext().prec = 16                          # Precision for decimal numbers

##############################################################################

class Inviscid_Burgers_2D_Adaptive_h:
    
    def __init__(self, N_x, N_y, tmax, eta_x, eta_y, error_tol):
        self.N_x = N_x                          # Number of points along X
        self.N_y = N_y                          # Number of points along Y
        self.xmin = 0                           # Min value of X
        self.xmax = 1                           # Max value of X
        self.ymin = 0                           # Min value of Y
        self.ymax = 1                           # Max value of Y
        self.eta_x = eta_x                      # Peclet number along X
        self.eta_y = eta_y                      # Peclet number along Y
        self.tmax = tmax                        # Maximum time
        self.error_tol = error_tol              # Maximum error permitted
        self.epsilon_1 = 0.01                   # Amplitude of 1st sine wave
        self.epsilon_2 = 0.01                   # Amplitude of 2nd sine wave
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
        self.u = u0.copy()          

    ### Parameters  
    def initialize_parameters(self):
        self.adv_cfl = (self.dx * self.dy)/(self.eta_y*self.dx + self.eta_x*self.dy)
        print('CFL time: ', self.adv_cfl)
        print('Tolerance:', self.error_tol)
        self.dt = 0.1 * self.adv_cfl                      # N * CFL condition
        self.nsteps = int(np.ceil(self.tmax/self.dt))     # Number of time steps
        self.Rx = 1./6. * self.eta_x/self.dx
        self.Ry = 1./6. * self.eta_y/self.dy
        
    ### Operator matrices    
    def initialize_matrices(self):    
        self.Adv_x = np.zeros((self.N_x, self.N_x))       # Advection (X)
        self.Adv_y = np.zeros((self.N_y, self.N_y))       # Advection (Y)
    
        for ij in range(self.N_x):
            self.Adv_x[ij, int(ij + 2) % self.N_x] = - self.Rx/2
            self.Adv_x[ij, int(ij + 1) % self.N_x] = 6 * self.Rx/2
            self.Adv_x[ij, ij % self.N_x] = -3 * self.Rx/2
            self.Adv_x[ij, int(ij - 1) % self.N_x] = -2 * self.Rx/2
        
        for ij in range(self.N_y):    
            self.Adv_y[ij, int(ij + 2) % self.N_y] = - self.Ry/2
            self.Adv_y[ij, int(ij + 1) % self.N_y] = 6 * self.Ry/2
            self.Adv_y[ij, ij % self.N_y] = -3 * self.Ry/2
            self.Adv_y[ij, int(ij - 1) % self.N_y] = -2 * self.Ry/2
            
        self.A_adv = kron(identity(self.N_y), self.Adv_x) + kron(self.Adv_y, identity(self.N_x))
        

    ##############################################################################
    
    def Solution(self, u, dt):
        
        ## Leja points
        Leja_X = Leja_Points()
    
        ############## --------------------- ##############
        
        ## Eigen values (Advection)
        eigen_min_adv = 0
        eigen_max_adv, eigen_imag_adv, its_power = Power_iteration(self.A_adv, u, 2)    # Max real, imag eigen value
        eigen_max_adv = eigen_max_adv * 1.2                                             # Safety factor
        eigen_imag_adv = eigen_imag_adv * 1.125                                         # Safety factor
        
        ## c and gamma
        c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
        Gamma_real_adv = 0.25 * (eigen_max_adv - eigen_min_adv)
        c_imag_adv = 0
        Gamma_imag_adv = 0.25 * (eigen_imag_adv - (- eigen_imag_adv))
        
        ############## --------------------- ##############
        
        ### Matrix-vector function
        f_u = self.A_adv.dot(self.u**2)
        
        u_temp, its_method = EXPRB43(self.A_adv, 2, u, dt, Leja_X, c_imag_adv, Gamma_imag_adv)
        
        ############## --------------------- ##############

        ## Update u
        u = u_temp.copy()
        
        return u, dt, its_power + its_method + 2
    
    ##############################################################################
        
    def run(self):
        
        ### Create directory
        emax = '{:5.1e}'.format(self.error_tol)
        path = os.path.expanduser("~/PrJD/Burgers' Equation/2D/Inviscid/Adaptive/A/N_100_100/eta_500_500/" + "/tol " + str(emax) + "/EXPRB43/3rd order/")
        path_sim = os.path.expanduser("~/PrJD/Burgers' Equation/2D/Inviscid/Adaptive/A/N_100_100/eta_500_500/")
        
        if os.path.exists(path):
            shutil.rmtree(path)                     # remove previous directory with same name
        os.makedirs(path, 0o777)                    # create directory with access rights
        
        ### Write simulation parameters to a file
        file_param = open(path_sim + '/Simulation_Parameters.txt', 'w+')
        file_param.write('N_x = %d' % self.N_x + '\n')
        file_param.write('N_y = %d' % self.N_y + '\n')
        file_param.write('eta_x = %f' % self.eta_x + '\n')
        file_param.write('eta_y = %f' % self.eta_y + '\n')
        file_param.write('CFL time = %.5e' % self.adv_cfl + '\n')
        file_param.write('Simulation time = %e' % self.tmax + '\n')
        file_param.write('Max. error = %e' % self.error_tol)
        file_param.close()
        
        ## Create files
        file_dt = open(path + "dt.txt", 'w+')
        file_er = open(path + "error.txt", 'w+')
        
        t = Decimal(0.0)                            # Time
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products
        
        self.u = self.u.reshape(self.N_x * self.N_y)
        
        ## Time loop
        while (float(t) < self.tmax):
            
            if float(t) + self.dt >= self.tmax:
                self.dt = self.tmax - float(t)
        
            u, u_ref, error, dt, num_mv = Higher_Order_Method(3, RK4, self.Solution, self.A_adv, self.u, self.dt, self.error_tol)

            counter = counter + 1
            count_mv = count_mv + num_mv
            t = Decimal(t) + Decimal(self.dt)
            self.u = u.copy()
            self.dt = dt.copy()
            
            ### Write data to files
            file_dt.write('%.15f' % self.dt + '\n')
            file_er.write('%.15f' % error + '\n')
            
            ############## --------------------- ##############

            ## Test plots
            # ax = fig.gca(projection = '3d')
            # surf = ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = cm.plasma, linewidth = 0, antialiased=False)
            # plt.gca().invert_xaxis()
            # plt.title('Data')
            # plt.pause(self.dt/4)
            # plt.clf()
            
            ############## --------------------- ##############

        print('Number of iterations = ', counter)
        print('Number of matrix_vector products = ', count_mv)
        
        ### Write simulation results to file
        file_res = open(path + 'Results.txt', 'w+')
        file_res.write('Number of iterations needed to reach tmax = %d' % counter + '\n')
        file_res.write('Number of matrix-vector products = %d' % count_mv)
        file_res.close()
        
        ### Write final data to file
        data = open(path + 'Final_data.txt', 'w+')
        ref = open(path + 'Final_data_ref.txt', 'w+')
        np.savetxt(data, self.u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        np.savetxt(ref, u_ref.reshape(self.N_y, self.N_x), fmt = '%.25f')
        data.close()
        ref.close()
         
        ### Close files
        file_dt.close()
        file_er.close()


##############################################################################

## Assign values for N, tmax, and eta
N_x = 100
N_y = 100
tmax = 5*1e-2
eta_x = 500
eta_y = 500
error_tol = 1e-8

### 1/(N_x - 1) * eta_x = 1/(N_y - 1) * eta_y for equal numerical diffusion along X and Y

def main():
    sim = Inviscid_Burgers_2D_Adaptive_h(N_x, N_y, tmax, eta_x, eta_y, error_tol)
    sim.run()

if __name__ == "__main__":
    main()

##################################################################################

print('Time Elapsed = ', datetime.now() - startTime)
