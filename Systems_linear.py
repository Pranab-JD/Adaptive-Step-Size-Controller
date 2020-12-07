"""
Created on Mon Nov 30 22:40:08 2020

@author: Pranab JD

Description: 
"""

import os
import shutil
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from Adaptive_Step_Size import *
import Diffusion_Advection as DA

##############################################################################

System_1D = DA.Diffusion_Advection_1D

class Run_1D_Systems(System_1D):

    def adaptive_h(self):

        ### Create directory
        n_val = '{:3.0f}'.format(self.N)
        eta_val = '{:2.0f}'.format(self.eta)
        emax = '{:5.1e}'.format(self.error_tol)
        direc_cost_control = os.path.expanduser("~/PrJD/Cost Controller Data Sets/Diffusion Advection/1D/Adaptive")
        path = os.path.expanduser(direc_cost_control + "/eta_" + str(eta_val) + "/N_" + str(n_val) + \
                                  "/Penalized/tol " + str(emax) + "/Real Leja/")
        path_sim = os.path.expanduser(direc_cost_control + "/eta_" + str(eta_val) + "/N_" + str(n_val))

        if os.path.exists(path):
            shutil.rmtree(path)                     # remove previous directory with same name
        os.makedirs(path, 0o777)                    # create directory with access rights

        ### Write simulation parameters to a file
        file_param = open(path_sim + '/Simulation_Parameters.txt', 'w+')
        file_param.write('N = %d' % self.N + '\n')
        file_param.write('eta = %d' % self.eta + '\n')
        file_param.write('Adv. CFL time = %.5e' % self.adv_cfl + '\n')
        file_param.write('Diff. CFL time = %.5e' % self.dif_cfl + '\n')
        file_param.write('Simulation time = %e' % self.tmax)
        file_param.close()

        ### Create files
        file_dt = open(path + "dt.txt", 'w+')

        time = 0                                    # Time
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products

        dt_temp = []
        mat_vec_prod = []
        time_arr = []
        dt = self.dt

        ############## --------------------- ##############

        ### Time loop ###
        while (time < self.tmax):

            if counter < 2:
                
                dt = 0.5 * dt
                
                ### Solution
                u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt)
                
                ### Error
                error = np.mean(abs(u_sol - u_ref))
                
                ## Total cost at each time step
                its_counts = 0

                while error > self.error_tol:
                    
                    dt = 0.5 * dt                                               # Reduce dt
                    u_sol, u_ref, u, num_mv_sol_2 = self.Solution(self.u, dt)   # Calculate with new dt
                    error = np.mean(abs(u_sol - u_ref))
                    
                    its_counts = its_counts + num_mv_sol_2                      # Total cost in this time step
                    
                    if error < self.error_tol:
                        break

                cost_cont = num_mv_sol + its_counts

            ############## --------------------- ##############
            
            else:
            
                ### Cost controller ###
                mat_vec_prod_n = mat_vec_prod[counter - 1]; mat_vec_prod_n_1 = mat_vec_prod[counter - 2]
                dt_temp_n = dt_temp[counter - 1]; dt_temp_n_1 = dt_temp[counter - 2]
                
                dt_controller = Cost_Controller(mat_vec_prod_n, dt_temp_n, mat_vec_prod_n_1, dt_temp_n_1, 1)
                
                dt = dt_controller

                ############## --------------------- ##############
                
                ### dt for final time step ###
                if time + dt >= self.tmax:

                    dt = self.tmax - time

                    ### Solution
                    u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt)
                    
                    ### Error
                    error = np.mean(abs(u_sol - u_ref))
                    
                    ## Total cost at each time step
                    its_counts = 0

                    while error > self.error_tol:
                        
                        dt = 0.5 * dt                                               # Reduce dt
                        u_sol, u_ref, u, num_mv_sol_2 = self.Solution(self.u, dt)   # Calculate with new dt
                        error = np.mean(abs(u_sol - u_ref))
                        
                        its_counts = its_counts + num_mv_sol_2                      # Total cost in this time step
                        
                        if error < self.error_tol:
                            break

                    cost_cont = num_mv_sol + its_counts

                ############## --------------------- ##############
                
                else:    
                    
                    ### Solution
                    u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt)
                    
                    ### Error
                    error = np.mean(abs(u_sol - u_ref))
                    
                    ## Total cost at each time step
                    its_counts = 0

                    while error > self.error_tol:
                        
                        dt = 0.5 * dt                                               # Reduce dt
                        u_sol, u_ref, u, num_mv_sol_2 = self.Solution(self.u, dt)   # Calculate with new dt
                        error = np.mean(abs(u_sol - u_ref))
                        
                        its_counts = its_counts + num_mv_sol_2                      # Total cost in this time step
                        
                        if error < self.error_tol:
                            break

                    cost_cont = num_mv_sol + its_counts
                
            ############## --------------------- ##############

            ### Update variables
            counter = counter + 1
            count_mv_iter = cost_cont                           # No. of matrix-vector products at each time step
            count_mv = count_mv + count_mv_iter                 # Total no. of matrix-vector products
            mat_vec_prod.append(count_mv_iter)                  # List of no. of matrix-vector products at each time step
            dt_temp.append(dt)                                  # List of no. of dt at each time step
            time_arr.append(time)

            self.u = u_ref.copy()
            self.dt = dt
            time = time + self.dt

            ############# --------------------- ##############

            ### Write data to files
            file_dt.write('%.15f' % self.dt + '\n')

            ############# --------------------- ##############

            # ### Test plots
            # plt.plot(self.X, u_sol, 'b.', label = 'Data')
            # plt.legend()
            # plt.pause(self.dt/2)
            # plt.clf()

            ############## --------------------- ##############

        print('Total number of time steps = ', counter)
        print('Total number of matrix-vector products = ', count_mv)
        print('Final time = ', time)

        ############## --------------------- ##############

        ### Write simulation results to file
        file_res = open(path + 'Results.txt', 'w+')
        file_res.write('Number of time steps = %d' % counter + '\n')
        file_res.write('Number of matrix-vector products = %d' % count_mv)
        file_res.close()

        ## Write final data to files
        file_final_sol = open(path + "Final_data_sol.txt", 'w+')
        file_final_ref = open(path + "Final_data_ref.txt", 'w+')
        file_final_sol.write(' '.join(map(str, u_sol)) % u_sol)
        file_final_ref.write(' '.join(map(str, self.u)) % self.u)
        file_final_sol.close()
        file_final_ref.close()
        
        ### Close files
        file_dt.close()

        ############# --------------------- ##############

        # ### Plot dt vs time
        # advCFL = np.ones(counter) * self.adv_cfl
        # difCFL = np.ones(counter) * self.dif_cfl

        # plt.figure()
        # plt.loglog(time_arr, dt_temp, 'b.:', label = 'dt used')
        # plt.loglog(time_arr, advCFL, 'r', label = 'Adv. CFL')
        # plt.loglog(time_arr, difCFL, 'g', label = 'Diff. CFL')
        # plt.legend()
        # plt.show()

##############################################################################

System_2D = DA.Diffusion_Advection_2D

class Run_2D_Systems(System_2D):

    def adaptive_h(self):

        ### Create directory
        n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        eta_x = '{:2.0f}'.format(self.eta_x); eta_y = '{:2.0f}'.format(self.eta_y)
        emax = '{:5.1e}'.format(self.error_tol)
        direc_cost_control = os.path.expanduser("~/PrJD/Cost Controller Data Sets/Diffusion Advection/2D/Adaptive")
        path = os.path.expanduser(direc_cost_control + "/eta_" + str(eta_x) + "_" + str(eta_y)  + "/N_" + str(n_x) + "_" + str(n_y) \
                                  + "/Non Penalized/tol " + str(emax) + "/Real Leja/")
        path_sim = os.path.expanduser(direc_cost_control + "/eta_" + str(eta_x) + "_" + str(eta_y)  + "/N_" + str(n_x) + "_" + str(n_y))

        if os.path.exists(path):
            shutil.rmtree(path)                     # remove previous directory with same name
        os.makedirs(path, 0o777)                    # create directory with access rights

        ### Write simulation parameters to a file
        file_param = open(path_sim + '/Simulation_Parameters.txt', 'w+')
        file_param.write('N_x = %d' % self.N_x + '\n')
        file_param.write('N_y = %d' % self.N_y + '\n')
        file_param.write('eta_x = %f' % self.eta_x + '\n')
        file_param.write('eta_y = %f' % self.eta_y + '\n')
        file_param.write('Adv. CFL time = %.5e' % self.adv_cfl + '\n')
        file_param.write('Diff. CFL time = %.5e' % self.dif_cfl + '\n')
        file_param.write('Simulation time = %e' % self.tmax)
        file_param.close()

        ### Create files
        file_dt = open(path + "dt.txt", 'w+')

        time = 0                                    # Time
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products

        dt_temp = []
        mat_vec_prod = []
        time_arr = []
        dt = self.dt
        
        ## Reshape into 1D
        self.u = self.u.reshape(self.N_x * self.N_y)

        ############## --------------------- ##############

        ### Time loop ###
        while (time < self.tmax):

            if counter < 2:
                
                dt = 0.5 * dt
                
                ### Solution
                u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt)
                
                ### Error
                error = np.mean(abs(u_sol - u_ref))
                
                ## Total cost at each time step
                its_counts = 0

                while error > self.error_tol:
                    
                    dt = 0.5 * dt                                               # Reduce dt
                    u_sol, u_ref, u, num_mv_sol_2 = self.Solution(self.u, dt)   # Calculate with new dt
                    error = np.mean(abs(u_sol - u_ref))
                    
                    its_counts = its_counts + num_mv_sol_2                      # Total cost in this time step
                    
                    if error < self.error_tol:
                        break

                cost_cont = num_mv_sol + its_counts

            ############## --------------------- ##############
            
            else:
            
                ### Cost controller ###
                mat_vec_prod_n = mat_vec_prod[counter - 1]; mat_vec_prod_n_1 = mat_vec_prod[counter - 2]
                dt_temp_n = dt_temp[counter - 1]; dt_temp_n_1 = dt_temp[counter - 2]
                
                dt_controller = Cost_Controller(mat_vec_prod_n, dt_temp_n, mat_vec_prod_n_1, dt_temp_n_1, 0)
                
                dt = dt_controller

                ############## --------------------- ##############
                
                ### dt for final time step ###
                if time + dt >= self.tmax:

                    dt = self.tmax - time

                    ### Solution
                    u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt)
                    
                    ### Error
                    error = np.mean(abs(u_sol - u_ref))
                    
                    ## Total cost at each time step
                    its_counts = 0

                    while error > self.error_tol:
                        
                        dt = 0.5 * dt                                               # Reduce dt
                        u_sol, u_ref, u, num_mv_sol_2 = self.Solution(self.u, dt)   # Calculate with new dt
                        error = np.mean(abs(u_sol - u_ref))
                        
                        its_counts = its_counts + num_mv_sol_2                      # Total cost in this time step
                        
                        if error < self.error_tol:
                            break

                    cost_cont = num_mv_sol + its_counts

                ############## --------------------- ##############
                
                else:    
                    
                    ### Solution
                    u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt)
                    
                    ### Error
                    error = np.mean(abs(u_sol - u_ref))
                    
                    ## Total cost at each time step
                    its_counts = 0

                    while error > self.error_tol:
                        
                        dt = 0.5 * dt                                               # Reduce dt
                        u_sol, u_ref, u, num_mv_sol_2 = self.Solution(self.u, dt)   # Calculate with new dt
                        error = np.mean(abs(u_sol - u_ref))
                        
                        its_counts = its_counts + num_mv_sol_2                      # Total cost in this time step
                        
                        if error < self.error_tol:
                            break

                    cost_cont = num_mv_sol + its_counts
                
            ############## --------------------- ##############

            ### Update variables
            counter = counter + 1
            count_mv_iter = cost_cont                           # No. of matrix-vector products at each time step
            count_mv = count_mv + count_mv_iter                 # Total no. of matrix-vector products
            mat_vec_prod.append(count_mv_iter)                  # List of no. of matrix-vector products at each time step
            dt_temp.append(dt)                                  # List of no. of dt at each time step
            time_arr.append(time)

            self.u = u_ref.copy()
            self.dt = dt
            time = time + self.dt

            ############# --------------------- ##############

            ### Write data to files
            file_dt.write('%.15f' % self.dt + '\n')

            ############# --------------------- ##############

            # ### Test plots
            # plt.imshow(self.u.reshape(self.N_y, self.N_x), cmap = cm.plasma, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt)
            
            ############## --------------------- ##############

        print('Total number of time steps = ', counter)
        print('Total number of matrix-vector products = ', count_mv)
        print('Final time = ', time)

        ############## --------------------- ##############

        ### Write simulation results to file
        file_res = open(path + 'Results.txt', 'w+')
        file_res.write('Number of time steps = %d' % counter + '\n')
        file_res.write('Number of matrix-vector products = %d' % count_mv)
        file_res.close()

        ## Write final data to files
        file_final_sol = open(path + "Final_data_sol.txt", 'w+')
        file_final_ref = open(path + "Final_data_ref.txt", 'w+')
        np.savetxt(file_final_sol, u_sol.reshape(self.N_y, self.N_x), fmt = '%.25f')
        np.savetxt(file_final_ref, self.u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file_final_sol.close()
        file_final_ref.close()
        
        ### Close files
        file_dt.close()

        ############# --------------------- ##############

        # ### Plot dt vs time
        # advCFL = np.ones(counter) * self.adv_cfl
        # difCFL = np.ones(counter) * self.dif_cfl

        # plt.figure()
        # plt.loglog(time_arr, dt_temp, 'b.:', label = 'dt used')
        # plt.loglog(time_arr, advCFL, 'r', label = 'Adv. CFL')
        # plt.loglog(time_arr, difCFL, 'g', label = 'Diff. CFL')
        # plt.legend()
        # plt.show()

##############################################################################