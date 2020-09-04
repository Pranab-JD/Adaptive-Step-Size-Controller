"""
Created on Tue Apr 07 23:10:04 2020

@author: Pranab JD

Description: -
    This code solves the inviscid Burgers' Equation:
    du/dt = eta * d(u^2)/dx + g(u) (1D)
    using RK4, RKF5, Crank-Nicolson, SDIRK23,
    ETD, and ETDRK2.
    Advective term - 1st order upwind scheme
    Step size is constant.

"""

import os
import math
import numpy as np
from decimal import *
from Leja_Header import *
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, identity

from datetime import datetime

startTime = datetime.now()
getcontext().prec = 8                   # Precision for decimal numbers

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
        u0 = 2 + self.epsilon_1 * np.sin(2 * np.pi * self.X) + self.epsilon_2 * np.sin(8 * np.pi * self.X + 0.3)
        self.u_RKF = u0.copy()                           # RKF5
        self.u_rk4 = u0.copy()                           # RK4
        self.u_CN = u0.copy()                            # Crank-Nicolson
        self.u_23 = u0.copy()                            # SDIRK23
        self.u_etd = u0.copy()                           # ETD1
        self.u_etdrk2 = u0.copy()                        # ETDRK2


    ### Parameters
    def initialize_parameters(self):
        self.dt = 0.1 * self.dx/abs(self.eta)            # N * CFL condition
        self.nsteps = int(np.ceil(self.tmax/self.dt))    # number of time steps
        self.R = self.eta/self.dx


    ### Operator matrices
    def initialize_matrices(self):
        self.A = np.zeros((self.N, self.N))

        ## Factor of 1/2 - conservative Burger's equation
        for ij in range(self.N):
            # self.A[ij, int(ij + 2) % self.N] = - self.R/2
            self.A[ij, int(ij + 1) % self.N] =  self.R/2
            self.A[ij, ij % self.N] = - self.R/2
            # self.A[ij, int(ij - 1) % self.N] = -2 * self.R/2

        self.A = csr_matrix(self.A)

    ##############################################################################

    ### RHS of Burgers' equation
    def dudt(self, u_temp):

        u_var = np.zeros(self.N)
        for ii in range(self.N):
            u_var[ii] = 1/2 * self.eta * (u_temp[int(ii + 1) % int(self.N)]**2 - u_temp[ii]**2)/self.dx

        return u_var

    ##############################################################################

    def RKF5(self):

        path = './dt 0.1/'
        os.makedirs(os.path.dirname(path), exist_ok = True)

        file_RKF = open(path + "u_RKF5.txt", 'w+')
        file_RKF.write(' '.join(map(str, self.u_RKF)) % self.u_RKF + '\n')

        t = Decimal(0.0)

        ## Time loop
        for nn in range(self.nsteps + 1):

            if nn == self.nsteps:
                self.dt = float(Decimal(self.tmax) - t)

            k1 = self.dt * np.dot(self.A, self.u_RKF**2)

            k2 = self.dt * np.dot(self.A, (self.u_RKF + k1/4)**2)

            k3 = self.dt * np.dot(self.A, (self.u_RKF + 3./32.*k1 + 9/32*k2)**2)

            k4 = self.dt * np.dot(self.A, (self.u_RKF + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**2)

            k5 = self.dt * np.dot(self.A, (self.u_RKF + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**2)

            k6 = self.dt * np.dot(self.A, (self.u_RKF - 8./27.*k1 + 2.*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**2)

            ## Solution
            u_temp = self.u_RKF + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

            ## Update u and t
            self.u_RKF = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############

            ## Write data to files
            file_RKF.write(' '.join(map(str, self.u_RKF)) % self.u_RKF + '\n')

        print(t)
        file_RKF.close()


    ##############################################################################

    def RK4(self):

        t = Decimal(0.0)

        path = './Reference Data/t_max 0.004/'
        os.makedirs(os.path.dirname(path), exist_ok = True)

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

            k1 = self.dt * self.A.dot(self.u_rk4**2)
            k2 = self.dt * self.A.dot((self.u_rk4 + k1/2)**2)
            k3 = self.dt * self.A.dot((self.u_rk4 + k2/2)**2)
            k4 = self.dt * self.A.dot((self.u_rk4 + k3)**2)

            ## Solution
            u_temp = self.u_rk4 + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

            ## Update u and t
            self.u_rk4 = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############

            ## Write data to files
            file_rk4.write(' '.join(map(str, self.u_rk4)) % self.u_rk4 + '\n')
            file_time.write('{}'.format(t) + '\n')

            if nn % 50 == 0:
                print('Time = ', float(t))
                print('------------------------------------------------------------')

        ### Write final data to file
        file_f = open(path + "Final_data.txt", 'w+')
        file_f.write(' '.join(map(str, self.u_rk4)) % self.u_rk4 + '\n')
        file_f.close()

        print('Final time = ', t)
        file_rk4.close()
        file_time.close()


    ##############################################################################

    def Crank_Nicolson(self):

        A = self.A * self.dt
        b = np.zeros(self.N)

        path = './dt 0.075/'
        os.makedirs(os.path.dirname(path), exist_ok = True)

        file_CN = open(path + "u_CN.txt", 'w+')
        file_CN.write(' '.join(map(str, self.u_CN)) % self.u_CN + '\n')

        ## Parameters for linearization
        u_temp, u_n2, u_n1, u_n0 = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)
        w_lin1, w_lin2, w_matrix = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)

        u_n2 = u_n1.copy()                                 # u_(n-2)
        u_n1 = u_n0.copy()                                 # u_(n-1)
        u_n0 = self.u_CN.copy()                            # u_n

        ###------------------------------------------------------------------------------###

        t = Decimal(0.0)

        ## First 2 time steps (Forward Euler)
        for nn in range(2):

            u_temp = self.u_CN + self.dudt(self.u_CN) * self.dt

            ## Update u and t
            self.u_CN = u_temp.copy()
            u_n2 = u_n1.copy()                             # u_(n-2)
            u_n1 = u_n0.copy()                             # u_(n-1)
            u_n0 = self.u_CN.copy()                        # u_n
            t = Decimal(t) + Decimal(self.dt)

            ## Write data to files
            file_CN.write(' '.join(map(str, self.u_CN)) % self.u_CN + '\n')

        ###------------------------------------------------------------------------------###

        ## Time loop (3rd step onwards)
        for nn in range(2, self.nsteps + 1):

            if nn == self.nsteps:
                self.dt = float(Decimal(self.tmax) - t)

            print(self.dt)
            ## Linear extrapolation
            for ij in range(self.N):
                w_lin1[ij] = (2 * u_n1[ij]) - u_n2[ij]                              # w_n
                w_lin2[ij] = (2 * u_n0[ij]) - u_n1[ij]                              # w_(n+1)

            ## Linearization term (taking into account the effect of matrix wrapping)
            # w = 1/2 * (w_i+1^n+1 + w_i^n+1 + w_i+1^n + w_i^n)
            for ij in range(self.N):
                w_matrix[ij] = 1/2 * (w_lin1[int(ij + 1) % self.N] + w_lin1[ij] + w_lin2[int(ij + 1) % self.N] + w_lin2[ij])

            ## Update LHS matrix
            for ii in range(self.N):
                A[ii, int(ii + 1) % self.N] = - self.R_dt/4 * w_matrix[ii]        # ii + 1
                A[ii, ii % self.N] = 1 + (self.R_dt/4 * w_matrix[ii])                       # ii

            ## Update RHS array
            for jj in range(self.N):
                b[jj] = (self.R_dt/4 * w_matrix[jj] * self.u_CN[int(jj + 1) % self.N]) + ((1 - (self.R_dt/4 * w_matrix[jj])) * self.u_CN[jj])

            ## Solve system of linear equations
            u_temp, result = spla.gmres(A, b, tol = 1e-5)

            ## Update u and t
            self.u_CN = u_temp.copy()
            u_n2 = u_n1.copy()                             # u_(n-2)
            u_n1 = u_n0.copy()                             # u_(n-1)
            u_n0 = self.u_CN.copy()                        # u_n
            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############

            # plt.plot(self.X, self.u_CN, 'bo')
            # plt.pause(self.dt)
            # plt.clf()

            ############## --------------------- ##############

            ## Write data to files
            file_CN.write(' '.join(map(str, self.u_CN)) % self.u_CN + '\n')

        print(t)
        file_CN.close()


    ##############################################################################

    def SDIRK23(self):

        ## Parameters
        p = 0                       # k1, k2
        epsilon = 1e-7              # amplitude of perturbation
        Jac_1 = np.zeros((self.N, self.N))
        Jac_2 = np.zeros((self.N, self.N))

        path = './Leja/'
        os.makedirs(os.path.dirname(path), exist_ok = True)

        file = open(path + "u_23.txt", 'w+')
        file.write(' '.join(map(str, self.u_23)) % self.u_23 + '\n')

        t = Decimal(0.0)

        ## Time loop
        for nn in range(self.nsteps + 1):

            if nn == self.nsteps:
                if float(Decimal(self.tmax) - t) >= 1e-12:
                    self.dt = float(Decimal(self.tmax) - t)
                else:
                    break

            ## Calculation of k1
            ij = 0

            while ij < 100:

                ## Unperturbed function
                f1 = np.dot(self.A, (self.u_23 + (self.dt * self.gamma * p))**2)

                ## F(p) = p - f1(p) = 0
                F1 = p - f1

                ## Jacobian
                for jj in range(self.N):

                    ## Unit vector
                    vector = np.zeros(self.N)
                    vector[jj] = 1

                    Jac_1[:, jj] = np.dot(self.A, (self.u_23 + (self.dt * self.gamma * p) + (epsilon * vector))**2) - f1

                Jacobian = np.identity(N) - (self.dt * self.gamma * Jac_1)

                # Calculate L = Jac^{-1} . F; Jac . L = F
                L, result = spla.gmres(Jacobian, F1, tol = 1e-5)
                ''' Include callback function '''

                # Calculate new value of p (Newton iterations)
                p_new = p - L

                # Check if new value of p satisfies required tolerance
                if all((abs(p_new - p)) <= 1e-5):
                    break
                else:
                    p = p_new

                ij = ij + 1

            # print('No. of Newton iterations for k1 = ', ij)
            k1 = p_new

            ## Calculation of k2
            ij = 0

            while ij < 100:

                ## Unperturbed function
                f2 = np.dot(self.A, (self.u_23 + ((1 - (2 * self.gamma)) * self.dt * k1) + (self.dt * self.gamma * p))**2)

                ## F(p) = p - f2(p) = 0
                F2 = p - f2

                ## Jacobian
                for jj in range(self.N):

                    ## Unit vector
                    vector = np.zeros(self.N)
                    vector[jj] = 1

                    Jac_2[:, jj] = np.dot(self.A, (self.u_23 + ((1 - (2 * self.gamma)) * self.dt * k1) + (self.dt * self.gamma * p) + (epsilon * vector))**2) - f1

                Jacobian = np.identity(N) - (self.dt * self.gamma * Jac_2)

                # Calculate L = Jac^{-1} . F; Jac . L = F
                L, result = spla.gmres(Jacobian, F2, tol = 1e-5)
                ''' Include callback function '''

                # Calculate new value of p (Newton iterations)
                p_new = p - L

                # Check if new value of p satisfies required tolerance
                if all((abs(p_new - p)) <= 1e-5):
                    break
                else:
                    p = p_new

                ij = ij + 1

            # print('No. of Newton iterations for k2 = ', ij)
            k2 = p_new

            ## Update u
            u_23_temp = self.u_23 + (self.dt * (k1 + k2)/2)
            self.u_23 = u_23_temp.copy()

            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############

            ## Write data to files
            file.write(' '.join(map(str, self.u_23)) % self.u_23 + '\n')

        print(t)
        file.close()


    ##############################################################################

    def ETD(self):

        path = './ETD/Eigen Values/'
        os.makedirs(os.path.dirname(path), exist_ok = True)

        ### Write simulation paramters to a file
        file_param = open(path + 'Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %f' % self.eta + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.write('Imag Leja')
        file_param.close()

        file = open(path + "u_etd.txt", 'w+')
        file.write(' '.join(map(str, self.u_etd)) % self.u_etd + '\n')

        ## Leja points
        Leja_X = Leja_Points()

        ############## --------------------- ##############

        epsilon = 1e-7                               		      # Amplitude of perturbation
        t = Decimal(0.0)
        print('dt =', self.dt)

        ### Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)

            ############## --------------------- ##############

            ## Eigen Values
            eigen_min = 0
            eigen_max, eigen_imag = Power_iteration(self.A, self.u_etd, 2)      # Max real, imag eigen value
            eigen_max = eigen_max * 1.2                                         # Safety factor
            eigen_imag = eigen_imag * 1.125                                     # Safety factor

            c_real = 0.5 * (eigen_max + eigen_min)
            Gamma_real = 0.25 * (eigen_max - eigen_min)
            c_imag = 0
            Gamma_imag = 0.25 * (eigen_imag - (-eigen_imag))

            ############## --------------------- ##############

            ### Matrix-vector product
            A_dot_u_1 = self.A.dot(self.u_etd**2)

            ### J(u) * u
            Linear_u = (self.A.dot((self.u_etd + (epsilon * self.u_etd))**2) - A_dot_u_1)/epsilon

            ### F(u) - (J(u) * u)
            Nonlin_u = A_dot_u_1 - Linear_u

            ## Linear Term
            # u_lin = real_Leja_exp(self.A, self.u_etd, 2, self.dt, Leja_X, c_real, Gamma_real)
            u_lin = imag_Leja_exp(self.A, self.u_etd, 2, self.dt, Leja_X, c_imag, Gamma_imag)

            ## Nonlinear Term
            # u_nl = real_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_real, Gamma_real) * self.dt
            u_nl = imag_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_imag, Gamma_imag) * self.dt

            ############## --------------------- ##############

            ### Full Solution ###
            u_var = u_lin + u_nl

            ## Update u and t
            self.u_etd = u_var.copy()
            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############

            # plt.plot(self.X, self.u_etd, 'bo')
            # plt.pause(self.dt)
            # plt.clf()

            ############## --------------------- ##############

            ## Write data to files
            file.write(' '.join(map(str, self.u_etd)) % self.u_etd + '\n')

            if nn % 100 == 0:
                print('Time = ', float(t))
                print('------------------------------------------------------------')

        print('Final time = ', t)
        file.close()


    ##############################################################################

    def ETDRK2(self):

        path = './ETDRK2/dt 0.1/Real/'
        os.makedirs(os.path.dirname(path), exist_ok = True)

        ### Write simulation paramters to a file
        file_param = open(path + 'Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %f' % self.eta + '\n')
        file_param.write('dt = %.15f' % self.dt + '\n')
        file_param.write('Simulation time = %f' % self.tmax + '\n')
        file_param.write('Real Leja')
        file_param.close()

        file = open(path + "u_ETDRK2.txt", 'w+')
        file.write(' '.join(map(str, self.u_etdrk2)) % self.u_etdrk2 + '\n')

        ## Leja points
        Leja_X = Leja_Points()

        ############## --------------------- ##############

        epsilon = 1e-7                               		      # Amplitude of perturbation
        t = Decimal(0.0)
        print('dt =', self.dt)

        ### Time loop
        for nn in range(self.nsteps):

            if  float(t) + self.dt > self.tmax:
                self.dt = self.tmax - float(t)
                if self.dt >= 1e-12:
                    print('Final dt = ', self.dt)

            ############## --------------------- ##############

            ## Eigen Values
            eigen_min = 0
            eigen_max, eigen_imag = Power_iteration(self.A, self.u_etdrk2, 2)       # Max real, imag eigen value
            eigen_max = eigen_max * 1.2                                             # Safety factor
            eigen_imag = eigen_imag * 1.125                                         # Safety factor

            c_real = 0.5 * (eigen_max + eigen_min)
            Gamma_real = 0.25 * (eigen_max - eigen_min)
            c_imag = 0
            Gamma_imag = 0.25 * (eigen_imag - (-eigen_imag))

            ############## --------------------- ##############

            ### ETD1 ###
            A_dot_u_1 = self.A.dot(self.u_etdrk2**2)

            ### J(u) * u
            Linear_u = (self.A.dot((self.u_etdrk2 + (epsilon * self.u_etdrk2))**2) - A_dot_u_1)/epsilon

            ### F(u) - (J(u) * u)
            Nonlin_u = A_dot_u_1 - Linear_u

            ## Linear Term
            u_lin = real_Leja_exp(self.A, self.u_etdrk2, 2, self.dt, Leja_X, c_real, Gamma_real)
            # u_lin = imag_Leja_exp(self.A, self.u_etdrk2, 2, self.dt, Leja_X, c_imag, Gamma_imag)

            ## Nonlinear Term
            u_nl = real_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_real, Gamma_real) * self.dt
            # u_nl = imag_Leja_phi(phi_1, Nonlin_u, self.dt, Leja_X, c_imag, Gamma_imag) * self.dt

            ############## --------------------- ##############

            ### ETD1 Solution ###
            a_n = u_lin + u_nl

            ############## --------------------- ##############

            ### ETDRK2 ###
            A_dot_u_2 = self.A.dot(a_n**2)

            ### J(u) * u
            Linear_u2 = (self.A.dot((a_n + (epsilon * a_n))**2) - A_dot_u_2)/epsilon

            ### F(u) - (J(u) * u)
            Nonlin_u2 = A_dot_u_2 - Linear_u2

            ## Nonlinear Term
            u_nl_2 = real_Leja_phi(phi_2, (Nonlin_u2 - Nonlin_u), self.dt, Leja_X, c_real, Gamma_real) * self.dt
            # u_nl_2 = imag_Leja_phi(phi_2, (Nonlin_u2 - Nonlin_u), self.dt, Leja_X, c_imag, Gamma_imag) * self.dt

            ############## --------------------- ##############

            ### Full solution
            u_temp = a_n + u_nl_2

            ## Update u and t
            self.u_etdrk2 = u_temp.copy()
            t = Decimal(t) + Decimal(self.dt)

            ############## --------------------- ##############

            # plt.plot(self.X, self.u_etdrk2, 'bo')
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
t_max = 4 * 1e-3
eta = 100

def main():
    sim = Inviscid_Burgers_1D_Constant_h(N, t_max, eta)
    # sim.RKF5()
    # sim.RK4()
    # sim.Crank_Nicolson()
    # sim.SDIRK23()
    # sim.ETD()
    sim.ETDRK2()

if __name__ == "__main__":
    main()

##################################################################################

print('Time Elapsed = ', datetime.now() - startTime)
