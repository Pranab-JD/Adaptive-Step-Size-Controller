"""
Created on Tue Apr 10 17:43:41 2020

@author: Pranab JD

Description: -
    This code solves the linear Diffusion-Advection equation:
    du/dt = eta * du/dx + d^2u/dx^2 (1D)
    using Crank-Nicolson, RKF4, SDIRK23,
    ETD, and ETDRK2 schemes. 
    Advective term - 1st order upwind scheme
    Step size is adaptive.

"""

import os
import shutil
import numpy as np
import scipy.sparse.linalg as spla
from decimal import *
import matplotlib.pyplot as plt
from datetime import datetime

startTime = datetime.now()
getcontext().prec = 8                   # Precision for decimal numbers

##############################################################################

class Diffusion_Advection_1D_Adaptive_h:

    def __init__(self, N, tmax, eta, e_max):
        self.N = int(N)                 # Number of points along X
        self.xmin = 0                   # Min value of X
        self.xmax = 1                   # Max value of X
        self.tmax = tmax                # Maximum time
        self.e_max = e_max              # Maximum error permitted
        self.eta = eta                  # Peclet number
        self.sigma_init = 1.4 * 1e-2    # Initial amplitude of Gaussian
        self.epsilon_1 = 0.01           # Amplitude of 1st sine wave
        self.epsilon_2 = 0.01           # Amplitude of 2nd sine wave
        self.gamma = (3 + 3**0.5)/6     # gamma for SDIRK methods
        self.initial_values_CN = False
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
        u0 = np.exp((-(self.X - 0.5)**2)/(2 * self.sigma_init**2))
        # u0 = 2 + self.epsilon_1 * np.sin(2 * np.pi * self.X) + self.epsilon_2 * np.sin(8 * np.pi * self.X + 0.3)
        self.u = u0.copy()
        self.u_RKF5 = u0.copy()       		# RKF5
        self.u_RKF4 = u0.copy()         	# RKF4
        self.u_etd = u0.copy()           	# EI
        self.u_CN = u0.copy()           	# CN
        self.u_23 = u0.copy()           	# SDIRK23
        self.u_rk4 = u0.copy()              # RK4


    ### Parameters
    def initialize_parameters(self):
        self.adv_cfl = self.dx/abs(self.eta)                # Advection CFL condition
        self.diff_cfl = self.dx**2/2                        # Diffusion CFL condition
        if self.eta == 0:
            self.dt = 1 * self.diff_cfl                      
        else:
            self.dt = 1e-3 * min(self.adv_cfl, self.diff_cfl)                        
        self.nsteps = int(self.tmax/self.dt)           	    # number of time steps
        self.R = self.eta/self.dx              	            # R = eta * dt/dx
        self.F = 1/self.dx**2							    # Fourier mesh number

    ### Matrices
    def initialize_matrices(self):
        self.A = np.zeros((self.N, self.N))                                      # LHS matrix
        self.b = np.zeros(self.N)                                                # RHS array
        
        for jj in range(self.N):
            self.A[jj, int(jj + 1) % int(self.N)] = self.F + self.R             # ii + 1
            self.A[jj, jj % self.N] = -2*self.F - self.R                        # ii
            self.A[jj, int(jj - 1) % int(self.N)] = self.F                      # ii - 1

    ##############################################################################

    ### RHS of Diffusion-Advection equation
    def dudt(self, u_temp):

        u_var = np.zeros(self.N)
        for ii in range(self.N):
            u_var[ii] = self.eta * (u_temp[int(ii + 1) % int(self.N)] - u_temp[ii])/self.dx \
                      + (u_temp[int(ii + 1) % int(self.N)] - 2*u_temp[ii] + u_temp[int(ii - 1) % int(self.N)])/self.dx**2

        return u_var

    ##############################################################################

    ### RKF5
    def RKF5(self, u_temp):
        
        u_RKF5 = np.zeros(self.N)
        
        k1 = self.dudt(u_temp)
        
        k2 = self.dudt(u_temp + k1*self.dt/4)
        
        k3 = self.dudt(u_temp + 3/32*k1*self.dt + 9/32*k2*self.dt)
        
        k4 = self.dudt(u_temp + 1932/2197*k1*self.dt - 7200/2197*k2*self.dt + 7296/2197*k3*self.dt)

        k5 = self.dudt(u_temp + 439/216*k1*self.dt - 8*k2*self.dt + 3680/513*k3*self.dt - 845/4104*k4*self.dt)

        k6 = self.dudt(u_temp - 8/27*k1*self.dt + 2*k2*self.dt - 3544/2565*k3*self.dt + 1859/4140*k4*self.dt - 11/40*k5*self.dt)

        ## Solution
        u_RKF5 = u_temp + self.dt * (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

        # Periodic boundary conditions
        u_RKF5[self.N - 1] = u_RKF5[0]
        
        return u_RKF5
       
    ##############################################################################

    ### Error between 4th and 5th order Runge-Kutta-Fehlberg
    def error_RKF45(self):

        n_iters = 1000
        
        for mm in range(n_iters):

            k1 = self.dudt(self.u_RKF5)

            k2 = self.dudt(self.u_RKF5 + k1*self.dt/4)

            k3 = self.dudt(self.u_RKF5 + 3/32*k1*self.dt + 9/32*k2*self.dt)

            k4 = self.dudt(self.u_RKF5 + 1932/2197*k1*self.dt - 7200/2197*k2*self.dt + 7296/2197*k3*self.dt)

            k5 = self.dudt(self.u_RKF5 + 439/216*k1*self.dt - 8*k2*self.dt + 3680/513*k3*self.dt - 845/4104*k4*self.dt)

            k6 = self.dudt(self.u_RKF5 - 8/27*k1*self.dt + 2*k2*self.dt - 3544/2565*k3*self.dt + 1859/4140*k4*self.dt - 11/40*k5*self.dt)

            ## Solutions
            self.u_RKF4 = self.u_RKF5 + self.dt * (25./216*k1 + 1408./2565.*k3 + 2197./4104.*k4 - 1./5.*k5)
            self.u_RKF5_temp = self.u_RKF5 + self.dt * (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

            ## Update u for RKF5
            self.u_RKF5 = self.u_RKF5_temp.copy()

            # Periodic boundary conditions
            self.u_RKF5[self.N - 1] = self.u_RKF5[0]
            self.u_RKF4[self.N - 1] = self.u_RKF4[0]

            ## Error estimate ###
            error_rkf45 = np.mean(abs(self.u_RKF4 - self.u_RKF5))
            new_dt = self.dt * (self.e_max/error_rkf45)**(1/5)

            ### Step size controller ###
            self.dt = 0.8 * new_dt

            if error_rkf45 <= self.e_max:
                print('Error within limits. dt accepted!! Error = ', error_rkf45)
                break

            ## Error alert
            if ii == (n_iters - 1):
                print('Condition reached. Check parameters!!!')

        return self.u_RKF4, self.u_RKF5, error_rkf45, self.dt

    ###------------------------------------------------------------------------###

    ### Adaptive stepping for 4th order Runge-Kutta-Fehlberg
    def RKF45(self):

        time = 0                                    # Time
        counter = 0                                 # Counter for # of iterations

        ## Make directory
        emax = '{:5.1e}'.format(self.e_max)
        path = './RKF4/N_' + str(self.N) + '/eta_' + str(self.eta) + '/emax_' + str(emax) + '/'

        if os.path.exists(path):
            shutil.rmtree(path)                     # remove previous directory with same name
        os.makedirs(path, 0o777)                    # create directory with access rights

        ## Create files
        file_1 = open(path + "u.txt", 'w+')
        file_2 = open(path + "u_RKF5.txt", 'w+')
        file_3 = open(path + "dt.txt", 'w+')
        file_4 = open(path + "error.txt", 'w+')

        ## Write initial value of u to files
        file_1.write(' '.join(map(str, self.u)) % self.u + '\n')
        file_2.write(' '.join(map(str, self.u)) % self.u + '\n')

        ## Time loop
        while (time < self.tmax):

            u4, u5, error, dt = self.error_RKF45()

            counter = counter + 1
            time = time + self.dt

            print('Time = ', time)
            print('dt = ', self.dt)

            ## Test plots
            plt.plot(self.X, u5, 'rd-')
            plt.plot(self.X, u4, 'b.-')
            plt.pause(self.dt)
            plt.clf()

            # Write data to files
            file_1.write(' '.join(map(str, u4)) % u4 + '\n')
            file_2.write(' '.join(map(str, u5)) % u5 + '\n')
            file_3.write('%.12f' % self.dt + '\n')
            file_4.write('%.12f' % error + '\n')

        print('Number of iterations = ', counter)

        ## Write number of iteration to 'dt' file
        file_3.write('Number of iterations = \n')
        file_3.write('%d' % counter + '\n')

        ## Close files
        file_1.close()
        file_2.close()
        file_3.close()
        file_4.close()

    ##############################################################################

    ### Error between 2nd order Crank-Nicolson and 5th order Runge-Kutta-Fehlberg
    def error_CN(self):
        
        ## Exact solution            
        u5 = self.RKF5(self.u_CN)

        niters = 1
        
        for mm in range(niters):

            ## Update LHS matrix
            for ij in range(self.N):

                self.A[ij, int(ij + 1) % int(self.N)] = - (self.F + self.R)/2       # ii + 1
                self.A[ij, ij % N] = 1 + self.F + self.R/2                          # ii
                self.A[ij, int(ij - 1) % int(self.N)] = - self.F/2                  # ii - 1

            ## Update RHS array
            for jj in range(self.N):

                self.b[jj] = (self.F + self.R)/2 * self.u_CN[int(jj + 1) % int(self.N)] + (1 - self.F - self.R/2) * self.u_CN[jj] \
                           + self.F/2 * self.u_CN[int(jj - 1) % int(self.N)]

            ## Number of iterations for gmres to converge
            num_iters = 1
            # def callback(xk):
            #     nonlocal num_iters
            #     num_iters = num_iters + 1

            ### Solution ###
            self.u_CN_temp, result = scipy.sparse.linalg.gmres(self.A, self.b, tol = 1e-10)
            # print('Number of iterations taken by gmres = ', num_iters)

            ## Periodic boundary conditions
            self.u_CN_temp[self.N - 1] = self.u_CN_temp[0]

            ## Convert array of arrays into array of floats
            self.u_CN_temp = np.array([i[0] if isinstance(i, np.ndarray) else i for i in self.u_CN_temp])
            
            ### Error estimate ###
            error_cn = np.mean(abs(self.u_CN_temp - u5))
            # print(error_cn)
            new_dt = self.dt * (self.e_max/error_cn)**(1/3)
            # print(new_dt)
            # 
            # ### Step size controller ###
            # self.dt = 0.8 * new_dt
            # 
            if error_cn <= self.e_max:
                print('Error within limits. dt accepted!! Error = ', error_cn)
                break

        ## Update u for CN and AM3
        self.u_CN = self.u_CN_temp.copy()

        return self.u_CN, u5, self.dt

    ###------------------------------------------------------------------------###

    ### Adaptive stepping for 2nd order Crank-Nicolson
    def Crank_Nicolson(self):

        max_iters = 15                              # Arbitrary # of max iterations
        time = 0                                    # Time
        counter = 0                                 # Counter for # of iterations

        ## Time loop
        while (counter < max_iters and time < self.tmax):

            u_cn, u5, dt = self.error_CN()
            counter = counter + 1

            time = time + self.dt

            # print('Time = ', time)
            print('dt = ', self.dt)

            ## Test plots
            plt.plot(self.X, u5, 'rd-')
            plt.plot(self.X, u_cn, 'b.-')           
            plt.pause(self.dt)
            plt.clf()

        print('Number of iterations = ', counter)
        
    ##############################################################################
       
    def RK4(self):
        
        path = './Leja/'  
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        file_rk4 = open(path + "u_rk4.txt", 'w+')
        file_rk4.write(' '.join(map(str, self.u_rk4)) % self.u_rk4 + '\n')
              
        t = Decimal(0.0)         
              
        ## Time loop
        for nn in range(self.nsteps + 1):
            
            if nn == self.nsteps:
                self.dt = float(Decimal(self.tmax) - t)
 
            k1 = self.dt * np.dot(self.A, self.u_rk4)
          
            k2 = self.dt * np.dot(self.A, (self.u_rk4 + k1/2))

            k3 = self.dt * np.dot(self.A, (self.u_rk4 + k2/2))
            
            k4 = self.dt * np.dot(self.A, (self.u_rk4 + k3))
            
            ## Solution
            u_temp = self.u_rk4 + 1./6.*(k1 + 2*k2 + 2*k3 + k4)
            
            ## Update u and t
            self.u_rk4 = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)
            
            ############## --------------------- ##############
            
            ## Test plots
            # plt.plot(self.X, self.u_rk4, 'b.-')           
            # plt.pause(self.dt)
            # plt.clf()
                     
            ## Write data to files
            file_rk4.write(' '.join(map(str, self.u_rk4)) % self.u_rk4 + '\n')
               
        print(t)       
        file_rk4.close()
        
    ##############################################################################
       
    def RK2(self):
        
        path = './Error Calculation/dt 0.075 1000/'  
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        file_rk2 = open(path + "u_rk2.txt", 'w+')
        file_rk2.write(' '.join(map(str, self.u_rk2)) % self.u_rk2 + '\n')
              
        t = Decimal(0.0)      
              
        ## Time loop
        for nn in range(self.nsteps + 1):
            
            if nn == self.nsteps:
                self.dt = float(Decimal(self.tmax) - t)
            
            k1 = self.dt * np.dot(self.A, self.u_rk2)
          
            k2 = self.dt * np.dot(self.A, (self.u_rk2 + k1))
            
            ## Solution
            u_temp = self.u_rk2 + 1./2.*(k1 + k2)
            
            ## Update u and t
            self.u_rk2 = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)
            
            ############## --------------------- ##############
                     
            ## Write data to files
            file_rk2.write(' '.join(map(str, self.u_rk2)) % self.u_rk2 + '\n')
               
        print(t)       
        file_rk2.close()
        
    ##############################################################################    
        
    def ETD(self):
        
        path = './Leja/'       
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        file = open(path + "u_ETD.txt", 'w+')
        file.write(' '.join(map(str, self.u_etd)) % self.u_etd + '\n')
        
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
        
            for ii in range(1, len(X)):
                for jj in range(ii):
                    div_diff[ii] = (div_diff[jj] - div_diff[ii])/(X[jj] - X[ii])
        
            return div_diff
        
        def real_Leja():
            """
            Real (negative) eigen values
            
            """
            
            def func(xx):
                return np.exp(self.dt * (c_real + Gamma_real*xx))
                
            ## Polynomial coefficients
            coeffs = Divided_Difference(Leja_X, func)
            
            ## a_0 term
            poly = self.u_etd.copy()
            poly = coeffs[0] * poly
            
            ## a_1, a_2 .... a_n terms
            max_Leja_pts = 400
            y = self.u_etd.copy()
            poly_tol = 1e-7
            scale_fact = 1/Gamma_real                                    # Re-scaling factor
                      
            for ii in range(1, max_Leja_pts):
                
                shift_fact = -c_real * scale_fact - Leja_X[ii - 1]       # Re-shifting factor
                
                u_temp = y.copy()
                y = y * shift_fact
                y = y + scale_fact * np.dot(self.A, u_temp)
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
            
        def imag_Leja():
            """
            Imaginary eigen values
            
            """
            
            def func(xx):
                return np.exp(1j * self.dt * (c_imag + Gamma_imag*xx))
            
            ## Polynomial coefficients
            coeffs = Divided_Difference(Leja_X, func)
            
            ## a_0 term
            poly = self.u_etd.copy() + 0 * 1j
            poly = coeffs[0] * poly
            
            ## a_1, a_2 .... a_n terms
            max_Leja_pts = 400
            y = self.u_etd.copy() + 0 * 1j
            poly_tol = 1e-7
            scale_fact = 1/Gamma_imag                                    # Re-scaling factor
            
            for ii in range(1, max_Leja_pts):
                
                shift_fact = -c_imag * scale_fact - Leja_X[ii - 1]       # Re-shifting factor            
 
                u_temp = y.copy()
                y = y * shift_fact
                y = y + scale_fact * np.dot(self.A, u_temp) * (-1j)
                poly = poly + coeffs[ii] * y
                
                ## If new number (next order) to be added < tol, ignore it   
                if (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii]) < poly_tol:
                    # print('No. of Leja points used = ', ii)
                    break
            
            ## Solution 
            u_imag = poly.copy()
            
            return np.real(u_imag)
         
        ############## --------------------- ##############    
     
        ### Polynomial interpolation
        ''' Use Gershgorin's theorem '''
        d, P = np.linalg.eig(self.A)                  # d: eigen values, P: matrix of eigen vectors
        
        eigen_min = 0                                           # Min value
        eigen_max = np.max(abs(np.real(d)))                     # Max value 
        eigen_imag = np.max(abs(np.imag(d)))
        
        c_real = 0.5 * (eigen_max + eigen_min)
        Gamma_real = 0.25 * (eigen_max - eigen_min)
        c_imag = 0
        Gamma_imag = 0.25 * (eigen_imag - (-eigen_imag))
        
        ############## --------------------- ##############
        
        ### Time loop
        t = Decimal(0.0)
        
        print('dt =', self.dt)
        print('Advection CFL =', '%.e' % Decimal(self.adv_cfl))
        print('Diffusion CFL =', '%.e' % Decimal(self.diff_cfl))
        
        for nn in range(self.nsteps + 1):
              
            if nn == self.nsteps:
                if float(Decimal(self.tmax) - t) >= 1e-12:
                    self.dt = float(Decimal(self.tmax) - t)
                    print('Final dt = ', self.dt)
                else:
                    break
            
            u_temp = real_Leja()
            # u_temp = imag_Leja()

            ## Update u and t
            self.u_etd = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)
            
            ############## --------------------- ##############
            
            # plt.plot(self.X, self.u_etd, 'b.')
            # plt.pause(self.dt)
            # plt.clf()
            
            ############## --------------------- ##############
            
            ## Write data to files
            file.write(' '.join(map(str, self.u_etd)) % self.u_etd + '\n')   
                         
        file.close()  
        
##############################################################################

# Assign values for N, tmax, and eta
N = 100
t_max = 1e-3
eta = 100
e_max = 1e-5


def main():
    sim = Diffusion_Advection_1D_Adaptive_h(N, t_max, eta, e_max)
    # sim.RKF45()
    # sim.Crank_Nicolson()
    # sim.RK2()
    sim.RK4()
    # sim.ETD()
    # plt.show()

if __name__ == "__main__":
    main()

##############################################################################

print('Time Elapsed = ', datetime.now() - startTime)
