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
from matplotlib import cm
import matplotlib.pyplot as plt
from Adaptive_Step_Size import *
from Porous_Medium import Porous_Medium_2D
from stepsizectrlnnhist import cost_ctrler_nn
from Viscous_Burgers import Viscous_Burgers_2D
from Inviscid_Burgers import Inviscid_Burgers_2D
from Diffusion_Advection import Diffusion_Advection_2D

### Call the neural network cost controller
Cost_Controller_3N = cost_ctrler_nn()

##############################################################################

System_1 = Porous_Medium_2D
System_2 = Viscous_Burgers_2D
System_3 = Inviscid_Burgers_2D
System_4 = Diffusion_Advection_2D

Process = System_3

class Run_2D_Systems(Process):

    def adaptive_h(self):

        ### Create directory
        n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        eta_x = '{:2.0f}'.format(self.eta_x); eta_y = '{:2.0f}'.format(self.eta_y)
        emax = '{:5.1e}'.format(self.error_tol)
        direc_cost_control = os.path.expanduser("~/PrJD/Cost Controller Data Sets/Burgers' Equation/2D/Inviscid/Adaptive/t_0.0325")
        path = os.path.expanduser(direc_cost_control + "/eta_" + str(eta_x) + "_" + str(eta_y)  + "/N_" + str(n_x) + "_" + str(n_y) \
                                    + "/Penalized/tol " + str(emax) + "/EXPRB43/")
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
        file_dt = open(path + "dt.txt", 'w+')               # dt used at each time step
        file_dt_trad = open(path + "dt_trad.txt", 'w+')     # dt yielded by traditional controller

        time = 0                                            # Time
        counter = 0                                         # Counter for # of time steps
        count_mv = 0                                        # Counter for matrix-vector products

        dt_history = []                                     # Array - dt used
        time_arr = []                                       # Array - time elapsed after each time step
        mat_vec_prod = []                                   # Array - # of matrix-vector products

        cost_iter = 0                                       # Num. of times cost controller used
        trad_iter = 0                                       # Num. of times trad. controller used
        dt_trad = self.dt
        Method_order = 3                                    # Order of the time integrator (error estimator)

        ## Reshape into 1D
        self.u = self.u.reshape(self.N_x * self.N_y)

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

                ## dt estimated by trad controller in this time step
                dt_trad_est = dt_used

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

                    ## dt estimated by trad controller in this time step
                    dt_trad_est = dt_used 

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

                ### Chosen value of dt does not guarantee that error requirements are met
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

                ## dt estimated by trad controller in this time step
                # dt_trad_est = dt_used
                
                ### ------------------------ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ------------------------ ###


                ### ---------- xxxxxxxxx ---------- Cost Controllers ---------- xxxxxxxxx ----------- ###
                
                def Cost_2Node():
                    """
                    Returns
                    -------
                    dt_controller   : dt estimate using 2 time steps' history

                    """

                    ### Cost and dt used from previous 2 time steps
                    dt_history_n = dt_history[counter - 1]; cost_n = mat_vec_prod[counter - 1]/dt_history_n
                    dt_history_n_1 = dt_history[counter - 2]; cost_n_1 = mat_vec_prod[counter - 2]/dt_history_n_1
                    
                    ### 0 = Non-penalized; 1 = Penalized
                    dt_controller = Cost_Controller_2N(cost_n, dt_history_n, cost_n_1, dt_history_n_1, 0)
                    
                    return dt_controller
                
                ############## --------------------- ##############
                
                def Cost_3Node():
                    """
                    Returns
                    -------
                    dt_controller   : dt estimate using 3 time steps' history
                    
                    """

                    ### Cost and dt used from previous 3 time steps
                    dt_history_n = dt_history[counter - 1]; cost_n = mat_vec_prod[counter - 1]/dt_history_n
                    dt_history_n_1 = dt_history[counter - 2]; cost_n_1 = mat_vec_prod[counter - 2]/dt_history_n_1 
                    dt_history_n_2 = dt_history[counter - 3]; cost_n_2 = mat_vec_prod[counter - 3]/dt_history_n_2

                    dt_controller = dt_used * Cost_Controller_3N.evaluate( \
                                                [cost_n_2, cost_n_1, cost_n], \
                                                [dt_history_n_2, dt_history_n_1, dt_history_n])

                    return dt_controller
                
                ############## --------------------- ##############
                
                ### Choose any 1 cost controller
                dt_controller = Cost_2Node()
                # dt_controller = Cost_3Node()
                
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

                    ## dt estimated by trad controller in this time step
                    dt_trad_est = dt_trad 
                
                    ### Estimate of dt for next time step using traditional controller ###
                    new_dt = dt_used * (self.error_tol/error)**(1/(Method_order + 1))
                    dt_trad = 0.8 * new_dt          # Safety factor

            ############## --------------------- ##############

            ### Update variables
            counter = counter + 1
            count_mv_iter = max(cost_cont, cost_trad)           # No. of matrix-vector products at each time step
            count_mv = count_mv + count_mv_iter                 # Total no. of matrix-vector products
            mat_vec_prod.append(count_mv_iter)                  # List of no. of matrix-vector products at each time step
            dt_history.append(dt_used)                          # List of dt at each time step

            self.u = u_ref.copy()
            self.dt = dt_used
            time = time + self.dt
            time_arr.append(time)

            ############# --------------------- ##############

            ### Write data to files
            file_dt.write('%.15f' % self.dt + '\n')
            file_dt_trad.write('%.15f' % dt_trad_est + '\n')

            ############# --------------------- ##############

            # ### Test plots
            # plt.imshow(self.u.reshape(self.N_y, self.N_x), cmap = cm.plasma, origin = 'lower', extent = [0, 1, 0, 1])
            # plt.pause(self.dt)

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

        ### Write final data to files
        file_final = open(path + "Final_data.txt", 'w+')
        np.savetxt(file_final, u_ref.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file_final.close()
        
        ### Close files
        file_dt.close()

        ############# --------------------- ##############

        # ### Plot dt vs time
        # advCFL = np.ones(counter) * self.adv_cfl
        # difCFL = np.ones(counter) * self.dif_cfl

        # plt.figure()
        # plt.loglog(time_arr, dt_history, 'b.:', label = 'dt used')
        # plt.loglog(time_arr, advCFL, 'r', label = 'Adv. CFL')
        # plt.loglog(time_arr, difCFL, 'g', label = 'Diff. CFL')
        # plt.legend()
        # plt.show()

    ##############################################################################
    ##############################################################################

    def constant_h(self):

        time = 0                                    # Time
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products

        ############## --------------------- ##############

        ### Create directory
        # n_val = '{:3.0f}'.format(self.N)
        # eta_val = '{:2.0f}'.format(self.eta)
        # direc_cost_control = os.path.expanduser("~/PrJD/Cost Controller Data Sets/" + str(Process) + "/Constant")
        # path = os.path.expanduser(direc_cost_control + "/eta_" + str(eta_val) + "/N_" + str(n_val) + "/EXPRB43/")

        # if os.path.exists(path):
        #     shutil.rmtree(path)                     # remove previous directory with same name
        # os.makedirs(path, 0o777)                    # create directory with access rights
        
        ## Reshape into 1D
        self.u = self.u.reshape(self.N_x * self.N_y)

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

            # u_ref, num_mv_sol = RK2(self.A_adv, 2, self.u, self.dt)

            ### Update variables
            count_mv = count_mv + num_mv_sol
            counter = counter + 1

            self.u = u_sol.copy()
            time = time + self.dt

            ############# --------------------- ##############

            ### Test plots
            plt.imshow(self.u.reshape(self.N_y, self.N_x), cmap = cm.plasma, origin = 'lower', extent = [0, 1, 0, 1])
            plt.pause(self.dt)

            ############## --------------------- ##############

        print('Number of time steps = ', counter)
        print('Total number of matrix-vector products = ', count_mv)

        # ### Write final data to files
        # file_final = open(path + "Final_data.txt", 'w+')
        # np.savetxt(file_final, u_sol.reshape(self.N_y, self.N_x), fmt = '%.25f')
        # file_final.close()

##############################################################################