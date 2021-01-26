"""
Created on Mon Oct 26 18:24:14 2020

@author: Pranab JD

Description:
    Runs the code using designated integrators
    and step size controllers for the different
    equations under consideration.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from Adaptive_Step_Size import *
from stepsizectrlnnhist import cost_ctrler_nn
from Porous_Medium import Porous_Medium_1D
from Viscous_Burgers import Viscous_Burgers_1D
from Inviscid_Burgers import Inviscid_Burgers_1D
from Diffusion_Advection import Diffusion_Advection_1D

##############################################################################

System_1 = Porous_Medium_1D
System_2 = Viscous_Burgers_1D
System_3 = Inviscid_Burgers_1D
System_4 = Diffusion_Advection_1D

Process = System_2

class Run_1D_Systems(Process):

    def adaptive_h(self):

        ### Create directory
        n_val = '{:3.0f}'.format(self.N)
        eta_val = '{:2.0f}'.format(self.eta)
        emax = '{:5.1e}'.format(self.error_tol)
        direc_cost_control = os.path.expanduser("~/PrJD/Cost Controller Data Sets/Burgers' Equation/1D/Viscous/Adaptive/0.01")
        path = os.path.expanduser(direc_cost_control + "/eta_" + str(eta_val) + "/N_" + str(n_val) + "/Cost Controller 3N/tol " + str(emax) + \
                                    "/EXPRB43/")
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

        dt_history = []
        time_arr = []
        mat_vec_prod = []

        cost_iter = 0
        trad_iter = 0
        dt_trad = self.dt
        Method_order = 3

        ############## --------------------- ##############

        ### Time loop ###
        while (time < self.tmax):

            if counter < 3:

                ### Traditional Controller
                u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt_trad)

                error = np.mean(abs(u_sol - u_ref))

                if error > self.error_tol:
                    u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Traditional_Controller(self.Solution, self.u, dt_trad, \
                                                                                            Method_order, error, self.error_tol)
                else:
                    dt_used = dt_trad
                    num_mv_trad = 0

                    ### Estimate of dt for next time step if error < tol in the 1st try
                    new_dt = dt_used * (self.error_tol/error)**(1/(Method_order + 1))
                    dt_trad = 0.8 * new_dt          # Safety factor

                cost_trad = num_mv_sol + num_mv_trad
                cost_cont = 0
                trad_iter = trad_iter + 1

            ############## --------------------- ##############

            ### dt for final time step ###
            elif time + dt_trad >= self.tmax:

                dt_final = self.tmax - time

                ### Solution
                u_sol, u_ref, u, num_mv_final = self.Solution(self.u, dt_final)

                ### Error
                error = np.mean(abs(u_sol - u_ref))

                ############## --------------------- ##############

                if error > self.error_tol:

                    print('Error > tol in the final time step!! Reducing dt.......')

                    ### Traditional controller
                    u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Traditional_Controller(self.Solution, self.u, dt_trad, \
                                                                                            Method_order, error, self.error_tol)

                    cost_trad = num_mv_final + num_mv_trad
                    cost_cont = 0
                    trad_iter = trad_iter + 1

                else:
                    cost_cont = cost_trad = num_mv_final
                    dt_used = dt_final
                    print('Last time step:', time, dt_used, time + dt_used)
                    break

            ############## --------------------- ##############

            else:
                
                ### --------------xxxxxxxxxxxxxx Traditional Controller xxxxxxxxxxxxxx-------------- ###

                # u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt_trad)

                # error = np.mean(abs(u_ref - u_sol))

                # if error > self.error_tol:
                #     u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Traditional_Controller(self.Solution, self.u, dt_trad, \
                #                                                                             Method_order, error, self.error_tol)

                # else:
                #     dt_used = dt_trad
                #     num_mv_trad = 0

                #     ### Estimate of dt for next time step if error < tol in the 1st try
                #     new_dt = dt_used * (self.error_tol/error)**(1/(Method_order + 1))
                #     dt_trad = 0.8 * new_dt          # Safety factor

                # cost_trad = num_mv_sol + num_mv_trad
                # cost_cont = 0
                # trad_iter = trad_iter + 1
                
                ### ------------------------ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ------------------------ ###


                ### ---------- xxxxxxxxx ---------- Cost Controllers ---------- xxxxxxxxx ---------- ###
                
                def Cost_2Node():
                    """
                    Returns
                    -------
                    dt_controller   : dt estimate using 2 time steps' history

                    """

                    ### Cost and dt used from previous 2 time steps
                    mat_vec_prod_n = mat_vec_prod[counter - 1]; dt_history_n = dt_history[counter - 1]
                    mat_vec_prod_n_1 = mat_vec_prod[counter - 2]; dt_history_n_1 = dt_history[counter - 2]
                    
                    ### 0 = Non-penalized; 1 = Penalized
                    dt_controller = Cost_Controller_2N(mat_vec_prod_n, dt_history_n, mat_vec_prod_n_1, dt_history_n_1, 0)
                    
                    return dt_controller
                
                ############## --------------------- ##############
                
                def Cost_3Node():
                    """
                    Returns
                    -------
                    dt_controller   : dt estimate using 3 time steps' history
                    
                    """
                    
                    ### Call the 3-point cost controller
                    Cost_Controller_3N = cost_ctrler_nn()

                    ### Cost and dt used from previous 3 time steps
                    mat_vec_prod_n = mat_vec_prod[counter - 1]; dt_history_n = dt_history[counter - 1]
                    mat_vec_prod_n_1 = mat_vec_prod[counter - 2]; dt_history_n_1 = dt_history[counter - 2]
                    mat_vec_prod_n_2 = mat_vec_prod[counter - 3]; dt_history_n_2 = dt_history[counter - 3]

                    dt_controller = dt_used * Cost_Controller_3N.evaluate( \
                                                [mat_vec_prod_n_2, mat_vec_prod_n_1, mat_vec_prod_n], \
                                                [dt_history_n_2, dt_history_n_1, dt_history_n])

                    return dt_controller
                
                ############## --------------------- ##############
                
                ### Choose any 1 cost controller
                # dt_controller = Cost_2Node()
                dt_controller = Cost_3Node()
                
                ### Choose the minimum step size
                dt = min(dt_controller, dt_trad)
                
                ### Solve with dt
                u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt)
                
                ### Error
                error = np.mean(abs(u_sol - u_ref))
                
                ############## --------------------- ##############
                
                ### Chosen value of dt does not guarantee that error requirements are met
                if error > self.error_tol:
                
                    ### Traditional controller
                    u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Traditional_Controller(self.Solution, self.u, dt_trad, \
                                                                                            Method_order, error, self.error_tol)
                
                    cost_trad = num_mv_sol + num_mv_trad
                    cost_cont = 0
                    trad_iter = trad_iter + 1
                
                else:
                
                    if dt == dt_trad:
                
                        ### dt from traditional controller used
                        cost_trad = num_mv_sol
                        cost_cont = 0
                        trad_iter = trad_iter + 1
                
                    elif dt == dt_controller:
                
                        ### dt from cost controller used
                        cost_cont = num_mv_sol
                        cost_trad = 0
                        cost_iter = cost_iter + 1
                
                    else:
                        print('Error in selecting dt!! Unknown dt used!!!')
                
                    ############## --------------------- ##############
                
                    ## dt used in this time step
                    dt_used = dt
                
                    ### Estimate of dt for next time step using traditional controller ###
                    new_dt = dt_used * (self.error_tol/error)**(1/(Method_order + 1))
                    dt_trad = 0.8 * new_dt          # Safety factor
                    
            ############## --------------------- ##############

            ### Update variables
            counter = counter + 1
            count_mv_iter = max(cost_cont, cost_trad)           # No. of matrix-vector products at each time step
            count_mv = count_mv + count_mv_iter                 # Total no. of matrix-vector products
            mat_vec_prod.append(count_mv_iter)                  # List of no. of matrix-vector products at each time step
            dt_history.append(dt_used)                             # List of no. of dt at each time step

            self.u = u_ref.copy()
            self.dt = dt_used
            time = time + self.dt
            time_arr.append(time)

            ############# --------------------- ##############

            ### Write data to files
            file_dt.write('%.15f' % self.dt + '\n')

            ############# --------------------- ##############

            ### Test plots
            plt.plot(self.X, u_ref, 'rd', label = 'Reference')
            plt.plot(self.X, u_sol, 'b.', label = 'Data')
            plt.legend()
            plt.pause(self.dt/4)
            plt.clf()

            ############## --------------------- ##############

        print('Cost controller used in ', cost_iter, 'time steps')
        print('Traditional controller used in ', trad_iter, 'time steps')
        print('Total number of time steps = ', counter)
        print('Total number of matrix-vector products = ', count_mv)

        ############## --------------------- ##############

        ### Write simulation results to file
        file_res = open(path + 'Results.txt', 'w+')
        file_res.write('Number of time steps = %d' % counter + '\n')
        file_res.write('Cost controller used in %d' % cost_iter + ' time steps' + '\n')
        file_res.write('Number of matrix-vector products = %d' % count_mv)
        file_res.close()

        ## Write final data to files
        file_final_sol = open(path + "Final_data_sol.txt", 'w+')
        file_final_ref = open(path + "Final_data_ref.txt", 'w+')
        file_final_sol.write(' '.join(map(str, u_sol)) % u_sol)
        file_final_ref.write(' '.join(map(str, u_ref)) % u_ref)
        file_final_sol.close()
        file_final_ref.close()
        
        ### Close files
        file_dt.close()

        ############# --------------------- ##############

        ### Plot dt vs time
        advCFL = np.ones(counter) * self.adv_cfl
        difCFL = np.ones(counter) * self.dif_cfl

        plt.figure()
        plt.loglog(time_arr, dt_history, 'b.:', label = 'dt used')
        plt.loglog(time_arr, advCFL, 'r', label = 'Adv. CFL')
        plt.loglog(time_arr, difCFL, 'g', label = 'Diff. CFL')
        plt.legend()
        plt.show()

    ##############################################################################
    ##############################################################################

    def constant_h(self):

        ### Create directory
        n_val = '{:3.0f}'.format(self.N)
        eta_val = '{:2.0f}'.format(self.eta)
        dt_val = '{:5.1e}'.format(self.dt/(min(self.adv_cfl, self.dif_cfl)))
        direc_cost_control = os.path.expanduser("~/PrJD/Cost Controller Data Sets/Diffusion Advection/1D/Constant")
        path = os.path.expanduser(direc_cost_control + "/eta_" + str(eta_val) + "/N_" + str(n_val) + "/dt " + str(dt_val) + "/")
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
        file_param.write('Normalized dt = ' + str(dt_val) + '\n')
        file_param.write('Simulation time = %e' % self.tmax)
        file_param.close()
        
        ### Create files
        file_dt = open(path + "dt.txt", 'w+')
        
        time = 0
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products

        ############## --------------------- ##############

        ### Time loop ###
        while (time < self.tmax):

            if time + self.dt > self.tmax:

                if self.tmax - time >= 1e-10:
                    self.dt = self.tmax - time
                    print('Final step size:', self.dt)

                else:
                    print('Final step size:', self.dt)


            u_sol, u, num_mv_sol = self.Solution(self.u, self.dt)
            
            print(num_mv_sol)

            ### Update variables
            count_mv = count_mv + num_mv_sol
            counter = counter + 1

            self.u = u_sol.copy()
            time = time + self.dt

            ### Write data to files
            file_dt.write('%.15f' % self.dt + '\n')

            ############# --------------------- ##############

            # ### Test plots
            # plt.plot(self.X, u_sol, 'b.', label = 'Data')
            # plt.legend()
            # plt.pause(self.dt)
            # plt.clf()

            ############## --------------------- ##############

        print('Number of time steps = ', counter)
        print('Total number of matrix-vector products = ', count_mv)
        
        ### Write simulation results to file
        file_res = open(path + 'Results.txt', 'w+')
        file_res.write('Number of time steps = %d' % counter + '\n')
        file_res.write('Number of matrix-vector products = %d' % count_mv)
        file_res.close()

        ### Write final data to file
        file_final = open(path + "Final_data.txt", 'w+')
        file_final.write(' '.join(map(str, u_sol)) % u_sol + '\n')
        file_final.close()
        file_final.close()

##############################################################################