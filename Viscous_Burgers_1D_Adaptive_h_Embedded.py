"""
Created on Thu Aug  6 17:29:51 2020

@author: Pranab JD

Description: -
    This code solves the viscous Burgers' equation:
    du/dt = d^2u/dx^2 + eta * d(u^2)/dx (1D)
    using different time integrators.
    Advective term - 3rd order upwind scheme
    Adaptive step size is implemented.

"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from Leja_Interpolation import *
from Adaptive_Step_Size import *
from Integrators_2_matrices import *
from scipy.sparse import csr_matrix

from datetime import datetime

startTime = datetime.now()

##############################################################################

class Viscous_Burgers_1D_Adaptive_h:

    def __init__(self, N, tmax, eta, error_tol):
        self.N = int(N)                 # Number of points along X
        self.xmin = 0                   # Min value of X
        self.xmax = 1                   # Max value of X
        self.tmax = tmax                # Maximum time
        self.error_tol = error_tol      # Maximum error permitted
        self.eta = eta                  # Peclet number
        self.sigma = 0.02          		# Amplitude of Gaussian
        self.x_0 = 0.9           		# Center of the Gaussian
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
        self.u = u0.copy()

    ### Parameters
    def initialize_parameters(self):
        self.adv_cfl = self.dx/abs(self.eta)
        self.dif_cfl = self.dx**2/2
        print('Advection CFL: ', self.adv_cfl)
        print('Diffusion CFL: ', self.dif_cfl)
        print('Tolerance:', self.error_tol)
        self.dt = 1.25 * min(self.adv_cfl, self.dif_cfl)  # N * CFL condition
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

        ## Eigen values (Diffusion)
        global eigen_min_dif, eigen_max_dif, eigen_imag_dif, c_real_dif, Gamma_real_dif
        eigen_min_dif = 0
        eigen_max_dif, eigen_imag_dif = Gershgorin(self.A_dif)      # Max real, imag eigen value
        c_real_dif = 0.5 * (eigen_max_dif + eigen_min_dif)
        Gamma_real_dif = 0.25 * (eigen_max_dif - eigen_min_dif)

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
        2 + its_sol : Number of matrix-vector products
    
        """

        ## Eigen values (Advection)
        eigen_min_adv = 0
        eigen_max_adv, eigen_imag_adv, its_power = Power_iteration(self.A_adv, u, 2)   # Max real, imag eigen value
        eigen_max_adv = eigen_max_adv * 1.25                                           # Safety factor
        eigen_imag_adv = eigen_imag_adv * 1.25                                         # Safety factor

        ## c and gamma
        c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
        Gamma_real_adv = 0.25 * (eigen_max_adv - eigen_min_adv)
        c_imag_adv = 0
        Gamma_imag_adv = 0.25 * (eigen_imag_adv - (- eigen_imag_adv)) 
        
        ############## --------------------- ##############
    
        ### RHS of PDE (in the form of matrix-vector product)
        f_u = self.A_adv.dot(u**2) + self.A_dif.dot(u)
        
        ### Change integrator as needed
        u_sol_3, its_sol_3, u_sol_4, its_sol_4 = EXPRB43(self.A_adv, 2, self.A_dif, 1, u, dt, c_imag_adv, Gamma_imag_adv)
        
        global Ref_integrator, Method_order
        Ref_integrator = RKF5
        Method_order = 3

        return u_sol_3, u_sol_4, u, its_sol_3, its_sol_4

    ##############################################################################

    def run(self):
        
        ### Create directory
        emax = '{:5.1e}'.format(self.error_tol)
        n_val = '{:3.0f}'.format(self.N)
        path = os.path.expanduser("~/PrJD/Burgers' Equation/1D/Viscous/Adaptive/D - 100/N_" + str(n_val) + "/Traditional/Embedded/tol " + str(emax) + "/EXPRB43/")
        path_sim = os.path.expanduser("~/PrJD/Burgers' Equation/1D/Viscous/Adaptive/D - 100/N_" + str(n_val))
        
        if os.path.exists(path):
            shutil.rmtree(path)                     # remove previous directory with same name
        os.makedirs(path, 0o777)                    # create directory with access rights
        
        ### Write simulation parameters to a file
        file_param = open(path_sim + '/Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %f' % self.eta + '\n')
        file_param.write('CFL time = %.5e' % self.adv_cfl + '\n')
        file_param.write('Simulation time = %e' % self.tmax + '\n')
        file_param.write('Max. error = %e' % self.error_tol)
        file_param.close()
        
        ## Create files
        file_1 = open(path + "u.txt", 'w+')
        file_2 = open(path + "u_ref.txt", 'w+')
        file_3 = open(path + "dt.txt", 'w+')
        
        ## Write initial value of u to files
        file_1.write(' '.join(map(str, self.u)) % self.u + '\n')
        file_2.write(' '.join(map(str, self.u)) % self.u + '\n')

        time = 0                                    # Time
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products
        
        mat_vec_prod = []
        dt_temp = []

        cost_exprb3 = 0
        cost_exprb4 = 0
        cost_iter = 0
        trad_iter = 0
        dt_trad = self.dt

        ############## --------------------- ##############

        ### Time loop ###
        while (time < self.tmax):
            
            if counter < 2:

                ### Traditional Controller
                u_sol_3, u_sol_4, u, num_mv_sol_3, num_mv_sol_4 = self.Solution(self.u, dt_trad)
                u_sol_3, u_sol_4, dt_inp, dt_used, dt_trad, num_mv_trad_3, num_mv_trad_4 = Trad_Controller(self.Solution, Method_order, \
                                                                                                            u_sol_3, u_sol_4, u, dt_trad, self.error_tol)
                
                cost_trad = num_mv_sol_3 + num_mv_sol_4 + num_mv_trad_3 + num_mv_trad_4
                cost_cont = 0
                trad_iter = trad_iter + 1
            
            ############## --------------------- ##############     
                                        
            ### dt for final time step
            elif time + dt_trad >= self.tmax:
       
                dt_final = self.tmax - time
                
                ### Solution 
                u_sol_3, u_sol_4, u, num_mv_sol_3, num_mv_sol_4 = self.Solution(self.u, dt_final)
                
                ### Error
                error = np.mean(abs(u_sol_3 - u_sol_4))
                
                ############## --------------------- ##############
                
                counts_3 = 0; counts_4 = 0; n_iters = 1000    
                    
                if error > self.error_tol:
                
                    u_sol_3, u_sol_4, dt_inp, dt_used, dt_trad, num_mv_trad_3, num_mv_trad_4 = Trad_Controller(self.Solution, Method_order, \
                                                                                                                u_sol_3, u_sol_4, u, dt_final, self.error_tol)
                            
                    ## its_ref_1 added in num_mv_trad
                    cost_trad = num_mv_sol_3 + num_mv_sol_4 + num_mv_trad_3 + num_mv_trad_4
                    cost_cont = 0    
                    trad_iter = trad_iter + 1
                
                else:
                
                    cost_cont = cost_trad = num_mv_sol_3 + num_mv_sol_4    
                    print('Last time step:', time, dt_final, time + dt_final)  

            
            ############## --------------------- ############## 
            
            else:
                
                if dt_trad <= 1e-9:
                    dt_trad = 1.25 * min(self.adv_cfl, self.dif_cfl)

                ## Traditional Controller
                u_sol_3, u_sol_4, u, num_mv_sol_3, num_mv_sol_4 = self.Solution(self.u, dt_trad)
                u_sol_3, u_sol_4, dt_inp, dt_used, dt_trad, num_mv_trad_3, num_mv_trad_4 = Trad_Controller(self.Solution, Method_order, \
                                                                                                          u_sol_3, u_sol_4, u, dt_trad, self.error_tol)
                
                cost_trad = num_mv_sol_3 + num_mv_trad_3 + num_mv_sol_4 + num_mv_trad_4
                cost_cont = 0
                trad_iter = trad_iter + 1
                
                ############## --------------------- ##############
                
                # ### Cost controller
                # mat_vec_prod_n = mat_vec_prod[counter - 1]; mat_vec_prod_n_1 = mat_vec_prod[counter - 2]
                # dt_temp_n = dt_temp[counter - 1]; dt_temp_n_1 = dt_temp[counter - 2]
                # 
                # dt_controller = Step_Size_Controller(mat_vec_prod_n, dt_temp_n, mat_vec_prod_n_1, dt_temp_n_1)
                # 
                # dt = min(dt_controller, dt_trad)
                # 
                # ############## --------------------- ##############
                # 
                # ### Solve with dt
                # u_sol_3, u_sol_4, u, num_mv_sol_3, num_mv_sol_4 = self.Solution(self.u, dt)
                # 
                # ### Reference Solution and error
                # error = np.mean(abs(u_sol_3 - u_sol_4))
                # 
                # ############## --------------------- ##############
                # 
                # if error > self.error_tol or dt == dt_trad:
                #     
                #     u_sol_3, u_sol_4, dt_inp, dt_used, dt_trad, num_mv_trad_3, num_mv_trad_4 = Trad_Controller(self.Solution, Method_order, \
                #                                                                                                 u_sol_3, u_sol_4, u, dt_trad, self.error_tol)
                #             
                #     ## its_ref_1 added in num_mv_trad
                #     cost_trad = num_mv_sol_3 + num_mv_sol_4 + num_mv_trad_3 + num_mv_trad_4
                #     cost_cont = 0    
                #     trad_iter = trad_iter + 1
                #         
                # else:
                # 
                #     ### Cost controller
                #     cost_cont = num_mv_sol_3 + num_mv_sol_4
                #     cost_trad = 0
                #     cost_iter = cost_iter + 1
                #     
                #     ## dt used in this time step
                #     dt_used = dt
                #     
                #     ############## --------------------- ##############
                #                   
                #     ### Estimate of dt for next time step using traditional controller ###
                #     new_dt = dt * (self.error_tol/error)**(1/(Method_order + 1))
                #     dt_trad = 0.875 * new_dt          # Safety factor

            ############## --------------------- ##############     

            ### Update variables
            counter = counter + 1
            count_mv_iter = max(cost_cont, cost_trad)           # No. of matrix-vector products at each time step
            count_mv = count_mv + count_mv_iter                 # Total no. of matrix-vector products
            mat_vec_prod.append(count_mv_iter)                  # List of no. of matrix-vector products at each time step
            dt_temp.append(dt_used)                             # List of no. of dt at each time step
            cost_exprb3 = cost_exprb3 + num_mv_sol_3 + num_mv_trad_3
            cost_exprb4 = cost_exprb3 + num_mv_sol_4 + num_mv_trad_4

            self.u = u_sol_3.copy()
            self.dt = dt_used
            time = time + self.dt
            print(self.dt, counter)
            if self.dt <= 1e-10:
                print('Error!! Error!!!', counter, 'dt = ', self.dt, 'N = ', self.N, 'eta = ', self.eta)
            
            ### Write data to files
            file_1.write(' '.join(map(str, u_sol_3)) % u_sol_3 + '\n')
            file_2.write(' '.join(map(str, u_sol_4)) % u_sol_4 + '\n')
            file_3.write('%.15f' % self.dt + '\n')
            
            ############# --------------------- ##############

            ### Test plots
            # plt.plot(self.X, u_sol_4, 'rd', label = 'Reference')
            # plt.plot(self.X, u_sol_3, 'b.', label = 'Data')
            # plt.legend()
            # plt.pause(self.dt/4)
            # plt.clf()

            ############## --------------------- ##############

        print('Cost controller used in ', cost_iter, 'time steps')
        print('Traditional controller used in ', trad_iter, 'time steps')
        print('Number of time steps = ', counter)
        print('Total number of matrix-vector products = ', count_mv, 'EXPRB3 = ', cost_exprb3, 'EXPRB4 = ', cost_exprb4)

        ############## --------------------- ##############
        
        ### Write simulation results to file
        file_res = open(path + 'Results.txt', 'w+')
        file_res.write('Number of time steps = %d' % counter + '\n')
        file_res.write('Number of matrix-vector products = %d' % count_mv + '\n')
        file_res.write('Number of matrix-vector products (3rd order) = %d' % cost_exprb3 + '\n')
        file_res.write('Number of matrix-vector products (4th order) = %d' % cost_exprb4 + '\n')
        file_res.write('Cost controller used in %d' % cost_iter + ' time steps')
        file_res.close()
        
        ### Close files
        file_1.close()
        file_2.close()
        file_3.close()


##############################################################################

error_list_1 = [1e-4, 1e-5, 1e-6, 1e-7, 5e-5, 5e-6, 5e-7, 5e-8]
error_list_2 = [1e-4]

## Assign values for N, tmax, tol, and eta
for ii in error_list_2:

    loopTime = datetime.now()

    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')

    N = 300
    t_max = 1e-2
    eta = 100
    error_tol = ii

    def main():
        sim = Viscous_Burgers_1D_Adaptive_h(N, t_max, eta, error_tol)
        sim.run()
        # plt.show()

    if __name__ == "__main__":
        main()

    print('Time for given tolerance = ', datetime.now() - loopTime)

##################################################################################

print('Total Time Elapsed = ', datetime.now() - startTime)
