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
from Leja_Interpolation import *
from Adaptive_Step_Size import *
from Integrators_1_matrix import *
from scipy.sparse import kron, identity

from datetime import datetime

startTime = datetime.now()

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
        self.dt = 1.25 * self.adv_cfl                      # N * CFL condition
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
            
        self.A = kron(identity(self.N_y), self.Adv_x) + kron(self.Adv_y, identity(self.N_x))
        
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
        1 + its_sol : Number of matrix-vector products
    
        """

        ## Eigen values (Advection)
        eigen_min_adv = 0
        eigen_max_adv, eigen_imag_adv, its_power = Power_iteration(self.A, u, 2)   # Max real, imag eigen value
        eigen_max_adv = eigen_max_adv * 1.25                                       # Safety factor
        eigen_imag_adv = eigen_imag_adv * 1.25                                     # Safety factor

        ## c and gamma
        c_real_adv = 0.5 * (eigen_max_adv + eigen_min_adv)
        Gamma_real_adv = 0.25 * (eigen_max_adv - eigen_min_adv)
        c_imag_adv = 0
        Gamma_imag_adv = 0.25 * (eigen_imag_adv - (- eigen_imag_adv)) 
        
        ############## --------------------- ##############
    
        ### RHS of PDE (in the form of matrix-vector product)
        f_u = self.A.dot(u**2)
        
        ### Change integrator as needed
        u_sol, its_sol = EXPRB43(self.A, 2, u, dt, c_imag_adv, Gamma_imag_adv)[2:4]
        
        global Ref_integrator, Method_order
        Ref_integrator = RKF5
        Method_order = 4

        return u_sol, u, 1 + its_sol   
    
    ##############################################################################
    
    def Traditional_Controller(self, dt):
        """
        If dt_used = dt_inp, then dt_new > dt_inp.
        If dt_used = dt_new, then dt_new < dt_inp.
        """

        u_sol, u, num_mv_sol = self.Solution(self.u, dt)
        u_sol, u_ref, dt_inp, dt_used, dt_new, num_mv_trad = Higher_Order_Method_1(self.A, 2, self.Solution, Method_order, Ref_integrator, u_sol, u, dt, self.error_tol)

        return u_sol, u_ref, dt_inp, dt_used, dt_new, num_mv_sol, num_mv_trad

    ##############################################################################
        
    def run(self):
        
        ### Create directory
        emax = '{:5.1e}'.format(self.error_tol)
        nx = '{:2.0f}'.format(self.N_x); ny = '{:2.0f}'.format(self.N_y)
        vx = '{:2.0f}'.format(self.eta_x); vy = '{:2.0f}'.format(self.eta_y)
        path = os.path.expanduser("~/PrJD/Burgers' Equation/2D/Inviscid/Adaptive/D/N_" + str(nx) + "_" + str(ny) + \
                                  "/eta_" + str(vx) + "_" + str(vy) + "/Traditional/tol " + str(emax) + "/EXPRB43/4th order/")
        path_sim = os.path.expanduser("~/PrJD/Burgers' Equation/2D/Inviscid/Adaptive/D/N_" + str(nx) + "_" + str(ny) + "/eta_" + str(vx) + "_" + str(vy))
        
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
        
        time = 0                                    # Time
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products
        
        mat_vec_prod = []
        dt_temp = []

        cost_iter = 0
        trad_iter = 0
        dt_trad = self.dt
        
        self.u = self.u.reshape(self.N_x * self.N_y)
        
        ## Time loop
        while (time < self.tmax):
            
            if counter < 2:

                ### Traditional Controller
                u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_sol, num_mv_trad = self.Traditional_Controller(dt_trad)
                
                cost_trad = num_mv_sol + num_mv_trad
                cost_cont = 0
                trad_iter = trad_iter + 1
                
            ############## --------------------- ##############
            
            ### dt for final time step
            elif time + dt_trad >= self.tmax:
       
                dt_used = self.tmax - time
                
                ### Solution 
                u_sol, u, num_mv_final = self.Solution(self.u, dt_used)
                
                ### Reference Solution and error
                u_ref, its_ref_1 = Ref_integrator(self.A, 2, self.u, dt_used)
                error = np.mean(abs(u_sol - u_ref))
                
                ############## --------------------- ##############
                
                if error > self.error_tol:
                    
                    print('Error > tol in the final time step!! Reducing dt.......')
                    
                    ### Traditional controller
                    u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Higher_Order_Method_1(self.A, 2, self.Solution, Method_order, \
                                                                                                Ref_integrator, u_sol, u, dt_used, self.error_tol)
                    
                    ## its_ref_1 added in num_mv_trad
                    cost_trad = num_mv_final + num_mv_trad
                    cost_cont = 0    
                    trad_iter = trad_iter + 1
          
                else:
                    cost_cont = cost_trad = num_mv_final       
                    print('Last time step:', time, dt_used)            
            
            ############## --------------------- ############## 
        
            else:
                
                ### Traditional Controller
                u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_sol, num_mv_trad = self.Traditional_Controller(dt_trad)
                
                cost_trad = num_mv_sol + num_mv_trad
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
                # u_sol, u, num_mv_sol = self.Solution(self.u, dt)
                # 
                # ### Reference Solution and error
                # u_ref, its_ref_1 = Ref_integrator(self.A, 2, self.u, dt)
                # error = np.mean(abs(u_sol - u_ref))
                # 
                # ############## --------------------- ##############
                # 
                # if error > self.error_tol or dt == dt_trad:
                #     
                #     ### Traditional controller
                #     u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Higher_Order_Method_1(self.A, 2, self.Solution, Method_order, \
                #                                                                                 Ref_integrator, u_sol, u, dt, self.error_tol)
                #     
                #     ## its_ref_1 added in num_mv_trad
                #     cost_trad = num_mv_sol + num_mv_trad
                #     cost_cont = 0
                #     trad_iter = trad_iter + 1
                #         
                # else:
                # 
                #     ### Cost controller
                #     cost_cont = num_mv_sol + its_ref_1
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

            self.u = u_sol.copy()
            self.dt = dt_used
            time = time + self.dt
            
            ### Write data to files
            file_dt.write('%.15f' % self.dt + '\n')
            
            ############## --------------------- ##############

            ## Test plots
            # ax = fig.gca(projection = '3d')
            # surf = ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = cm.plasma, linewidth = 0, antialiased=False)
            # plt.gca().invert_xaxis()
            # plt.title('Data')
            # plt.pause(self.dt/4)
            # plt.clf()
            
            ############## --------------------- ##############

        print('Cost controller used in ', cost_iter, 'time steps')
        print('Traditional controller used in ', trad_iter, 'time steps')
        print('Number of time steps = ', counter)
        print('Final time = ', time)
        print('Total number of matrix-vector products = ', count_mv)
        
        ### Write simulation results to file
        file_res = open(path + 'Results.txt', 'w+')
        file_res.write('Number of time steps = %d' % counter + '\n')
        file_res.write('Number of matrix-vector products = %d' % count_mv + '\n')
        file_res.write('Cost controller used in %d' % cost_iter + ' time steps')
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

##############################################################################

error_list_1 = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 5e-4, 5e-5, 5e-6, 5e-7, 5e-8]
error_list_2 = [1e-4]

## Assign values for N, tmax, and eta
for ii in error_list_1:
    
    print('--------------------------------------------------------------')
    print('--------------------------------------------------------------')

    loopTime = datetime.now()
    
    N_x = 10
    N_y = 40
    tmax = 5e-2
    eta_x = 10
    eta_y = 40
    error_tol = ii
    
    ### 1/(N_x - 1) * eta_x = 1/(N_y - 1) * eta_y for equal numerical diffusion along X and Y
    
    def main():
        sim = Inviscid_Burgers_2D_Adaptive_h(N_x, N_y, tmax, eta_x, eta_y, error_tol)
        sim.run()
    
    if __name__ == "__main__":
        main()
        
    print('Time for given tolerance = ', datetime.now() - loopTime)

##################################################################################

print('Total Time Elapsed = ', datetime.now() - startTime)