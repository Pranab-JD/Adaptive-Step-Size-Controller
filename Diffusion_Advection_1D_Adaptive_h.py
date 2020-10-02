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
import matplotlib.pyplot as plt
from Leja_Interpolation import *
from Adaptive_Step_Size import *
from Integrators_1_matrix import *
from scipy.sparse import csr_matrix

from datetime import datetime

startTime = datetime.now()

##############################################################################

class Diffusion_Advection_1D_Adaptive_h:

    def __init__(self, N, tmax, eta, error_tol):
        self.N = int(N)                 # Number of points along X
        self.xmin = 0                   # Min value of X
        self.xmax = 1                   # Max value of X
        self.tmax = tmax                # Maximum time
        self.error_tol = error_tol      # Maximum error permitted
        self.eta = eta                  # Peclet number
        self.sigma_init = 1.4 * 1e-3    # Initial amplitude of Gaussian
        self.epsilon_1 = 0.01           # Amplitude of 1st sine wave
        self.epsilon_2 = 0.01           # Amplitude of 2nd sine wave
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
        self.u = u0.copy()

    ### Parameters
    def initialize_parameters(self):
        self.adv_cfl = self.dx/abs(self.eta)                
        self.diff_cfl = self.dx**2/2                        
        self.dt = 0.5 * min(self.adv_cfl, self.diff_cfl)   # N * CFL condition                     
        print('CFL time: ', self.adv_cfl)
        print('Tolerance:', self.error_tol)
        self.R = 1./6. * self.eta/self.dx      	            # R = eta * dt/dx
        self.F = 1/self.dx**2							    # Fourier mesh number

    ### Matrices
    def initialize_matrices(self):
        self.A = np.zeros((self.N, self.N))
        
        for ij in range(self.N):
            self.A[ij, int(ij + 2) % self.N] = -self.R/2
            self.A[ij, int(ij + 1) % self.N] = 6*self.R/2 + self.F
            self.A[ij, ij % self.N] = -3*self.R/2 - 2*self.F
            self.A[ij, int(ij - 1) % self.N] = -2*self.R/2 + self.F
            
        self.A = csr_matrix(self.A)
        
        ## Eigen values (Diffusion)
        global eigen_min, eigen_max, eigen_imag, c_real, Gamma_real, c_imag, Gamma_imag
        eigen_min = 0
        eigen_max, eigen_imag = Gershgorin(self.A)                              # Max real, imag eigen value
        c_real = 0.5 * (eigen_max + eigen_min)
        Gamma_real = 0.25 * (eigen_max - eigen_min)
        c_imag = 0
        Gamma_imag = 0.25 * (eigen_imag - (- eigen_imag)) 

    ##############################################################################
    
    def Solution(self, u, dt):
        """
        Parameters
        ----------
        u       : 1D vector u (input)
        dt      : dt
    
        Returns
        -------
        u_sol       : 1D vector u (output) after time dt
        u           : 1D vector u (input)
        its_sol     : Number of matrix-vector products
    
        """
    
        # u_sol, its_sol = real_Leja_exp(self.A, u, dt, c_real, Gamma_real)
        u_sol, its_sol = imag_Leja_exp(self.A, u, dt, c_imag, Gamma_imag)

        return u_sol, u, its_sol
    
    ##############################################################################
    
    def run(self):

        time = 0                                    # Time
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products
        
        mat_vec_prod = []  

        ############## --------------------- ##############

        ### Time loop ###
        while (time < self.tmax):

            if counter < 2:

                u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Higher_Order_Method_1(self.A, 1, self.Solution, Method_order, \
                                                                                            Ref_integrator, u_sol, u, dt, self.error_tol)
                
                cost_trad = num_mv_sol + num_mv_ric
                cost_cont = 0
                trad_iter = trad_iter + 1
                
                print('# of matrix vector products using trad controller', num_mv_sol + num_mv_ric)
                
                DT_cost.append(dt_used)
                DT_trad_temp[counter] = dt_used
                DT_trad_temp[counter + 1] = dt_trad

            elif counter >= 2:
                
                ############## --------------------- ##############
                
                ### Cost controller
                mat_vec_prod_n = mat_vec_prod[counter - 1]; mat_vec_prod_n_1 = mat_vec_prod[counter - 2]
                dt_temp_n = dt_temp[counter - 1]; dt_temp_n_1 = dt_temp[counter - 2]
                
                dt_controller = Step_Size_Controller(mat_vec_prod_n, dt_temp_n, mat_vec_prod_n_1, dt_temp_n_1)
                DT_cost.append(dt_controller)
                
                ############## --------------------- ##############
                
                dt = min(dt_controller, dt_trad)
                # dt = dt_controller
                
                ############## --------------------- ##############
                
                ### Solve with dt
                u_sol, u, num_mv_sol = self.Solution(self.u, dt)
                
                ### Reference Solution and error
                u_ref, its_ref_1 = RKF5(self.A_adv, 2, self.A_dif, 1, self.u, dt)
                error = np.mean(abs(u_sol - u_ref))
                
                ############## --------------------- ##############
                
                if error > self.error_tol or dt == dt_trad:
                
                    print('Error > Tolerance. Reducing dt')
                    
                    ### Traditional controller
                    u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Higher_Order_Method_2(4, self.A_adv, self.A_dif, self.Solution, RKF5, u_sol, u, dt, self.error_tol)
                    
                    trad_iter = trad_iter + 1
                    
                    DT_trad_temp[counter] = dt_used
                    DT_trad_temp[counter + 1] = dt_trad         
                    
                    cost_trad = num_mv_sol + num_mv_trad
                    cost_cont = 0
                    
                    print('# of matrix vector products using trad controller', cost_trad)
                        
                else:
                
                    ### Cost controller
                    cost_cont = num_mv_sol + its_ref_1
                    cost_trad = 0
                    cost_iter = cost_iter + 1
                    
                    ############## --------------------- ##############
                                  
                    ### Estimate of dt for next time step using traditional controller ###
                    new_dt = dt * (self.error_tol/error)**(1/(4 + 1))
                    dt_trad = 0.875 * new_dt          # Safety factor
                    
                    dt_used = dt
                    DT_trad_temp[counter + 1] = dt_trad
                    
                    print('# of matrix vector products using cost controller', cost_cont)

            ############## --------------------- ##############     
                                        
            ### dt for final time step
            if time + dt_trad >= self.tmax:
                dt_used = self.tmax - time
            
                print('Last step:', time, dt_used)
            
                u_sol, u, num_mv_final = self.Solution(self.u, dt_used)
            
                DT_cost[-1] = dt_used
                DT_trad_temp[counter] = dt_used
                
                cost_cont = cost_trad = num_mv_final
                 

            ############## --------------------- ##############

            ### Update variables

            counter = counter + 1
            count_mv_iter = max(cost_cont, cost_trad)
            count_mv = count_mv + count_mv_iter
            mat_vec_prod.append(count_mv_iter)
            
            print(counter)


            self.u = u_sol.copy()
            self.dt = dt_used
            time = time + self.dt
            dt_temp.append(self.dt)
            print('dt used =', dt_used)

            Time.append(time)
            
            
        print('Cost controller used in ', cost_iter, 'time steps')
        print('Traditional controller used in ', trad_iter, 'time steps')
        print('Number of time steps = ', counter)
        print('Total number of matrix-vector products = ', count_mv)
        print('Sum of dt = ', sum(dt_temp))
        

        ### Convert dict into list
        DT_trad = []
        for key, value in DT_trad_temp.items():
            DT_trad.append(value)
        
        DT_trad = DT_trad[:-1]

    
    
    
    ##############################################################################
    
        Nn = np.arange(0, counter, 1)
        plt.figure(figsize = (8, 6), dpi = 200)
        
        plt.semilogy(Time[0:-1], dt_temp[0:-1], 'g.', label = 'dt used')
        plt.semilogy(Time[0:-1], DT_cost[0:-1], 'r:', label = 'dt Cost controller')
        plt.semilogy(Time[0:-1], DT_trad[0:-1], 'b-.', label = 'dt Trad controller')
        plt.title('dt')
        
        
        
        plt.legend()

        # plt.figure(figsize = (8, 6), dpi = 200)
        
        # plt.semilogy(Nn, cost_cost, 'r*', label = 'Cost')
        # plt.semilogy(Nn, cost_Trad, 'b.', label = 'Trad')
        # plt.title('Cost')
        # 
        # plt.legend()
        
        
        plt.show()


        
##############################################################################

# Assign values for N, tmax, and eta
N = 100
t_max = 9e-5
eta = 100
error_tol = 1e-4


def main():
    sim = Diffusion_Advection_1D_Adaptive_h(N, t_max, eta, error_tol)
    sim.run()
    # plt.show()

if __name__ == "__main__":
    main()

##############################################################################

print('Time Elapsed = ', datetime.now() - startTime)
