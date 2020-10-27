"""
Created on Mon Oct 26 18:24:14 2020

@author: Pranab JD

Description: 
    Runs the code using designated integrators
    and step size controllers for the  different
    equations under consideration.
"""

import os
import shutil
import numpy as np
from Porous_Medium_1D import *
from Adaptive_Step_Size import *
import matplotlib.pyplot as plt

from datetime import datetime

startTime = datetime.now()

##############################################################################

System_1 = Porous_Medium_1D
System_2 = Inviscid_Burgers_1D
System_3 = Viscous_Burgers_1D
System_4 = Diffusion_Advection_1D

class Run_Cost_Controller(System_1):
    
    def run(self):
    
        # ### Create directory
        # emax = '{:5.1e}'.format(self.error_tol)
        # n_val = '{:3.0f}'.format(self.N)
        # path = os.path.expanduser("~/PrJD/Burgers' Equation/1D/Viscous/Adaptive/B - 10/N_" + str(n_val) + "/Traditional/RE/tol " + str(emax) + "/EXPRB42/")
        # path_sim = os.path.expanduser("~/PrJD/Burgers' Equation/1D/Viscous/Adaptive/B - 10/N_" + str(n_val))
        #
        # if os.path.exists(path):
        #     shutil.rmtree(path)                     # remove previous directory with same name
        # os.makedirs(path, 0o777)                    # create directory with access rights
        #
        # ### Write simulation parameters to a file
        # file_param = open(path_sim + '/Simulation_Parameters.txt', 'w+')
        # file_param.write('N = %d' % self.N + '\n')
        # file_param.write('eta = %f' % self.eta + '\n')
        # file_param.write('CFL time = %.5e' % self.adv_cfl + '\n')
        # file_param.write('Simulation time = %e' % self.tmax + '\n')
        # file_param.write('Max. error = %e' % self.error_tol)
        # file_param.close()
        #
        # ## Create files
        # file_1 = open(path + "u.txt", 'w+')
        # file_2 = open(path + "u_ref.txt", 'w+')
        # file_3 = open(path + "dt.txt", 'w+')
        #
        # ## Write initial value of u to files
        # file_1.write(' '.join(map(str, self.u)) % self.u + '\n')
        # file_2.write(' '.join(map(str, self.u)) % self.u + '\n')

        time = 0                                    # Time
        counter = 0                                 # Counter for # of time steps
        count_mv = 0                                # Counter for matrix-vector products

        mat_vec_prod = []
        dt_temp = []

        cost_iter = 0
        trad_iter = 0
        dt_trad = self.dt
        Method_order = 2

        ############## --------------------- ##############

        ### Time loop ###
        while (time < self.tmax):

            print('Counter =', counter)
            
            if counter < 2:

                ### Traditional Controller
                u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt_trad)

                error = np.mean(abs(u_sol - u_ref))

                if error > self.error_tol:
                    u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Trad_Controller(self.Solution, Method_order, \
                                                                                            error, u, dt_trad, self.error_tol)
                else:
                    dt_used = dt_trad
                    num_mv_trad = 0

                    ### Estimate of dt for next time step if error < tol in the 1st try
                    new_dt = dt_used * (self.error_tol/error)**(1/(Method_order + 1))
                    dt_trad = 0.875 * new_dt          # Safety factor

                cost_trad = num_mv_sol + num_mv_trad
                cost_cont = 0
                trad_iter = trad_iter + 1

            ############## --------------------- ##############

            ### dt for final time step
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
                    u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Trad_Controller(self.Solution, Method_order, \
                                                                                            error, u, dt_final, self.error_tol)

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

                ### Traditional Controller
                u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt_trad)
                
                # print('dt initial', dt_trad)
                
                error = np.mean(abs(u_ref - u_sol))
                
                if error > self.error_tol:
                    u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Trad_Controller(self.Solution, Method_order, \
                                                                                            error, u, dt_trad, self.error_tol)
                
                else:
                    dt_used = dt_trad
                    num_mv_trad = 0
                    
                    print('Step size accepted')
                    print('----------------------------------------------------------------------------------------')
                
                    ### Estimate of dt for next time step if error < tol in the 1st try
                    new_dt = dt_used * (self.error_tol/error)**(1/(Method_order + 1))
                    dt_trad = 0.875 * new_dt          # Safety factor
                
                cost_trad = num_mv_sol + num_mv_trad
                cost_cont = 0
                trad_iter = trad_iter + 1
                
                # print('dt used', dt_used)

                ############## --------------------- ##############

                # ### Cost controller
                # mat_vec_prod_n = mat_vec_prod[counter - 1]; mat_vec_prod_n_1 = mat_vec_prod[counter - 2]
                # dt_temp_n = dt_temp[counter - 1]; dt_temp_n_1 = dt_temp[counter - 2]
                # 
                # dt_controller = Step_Size_Controller(mat_vec_prod_n, dt_temp_n, mat_vec_prod_n_1, dt_temp_n_1)
                # 
                # # if dt_trad <= 8e-8:
                # #     dt_trad = 1.25 * min(self.adv_cfl, self.dif_cfl)
                # 
                # dt = min(dt_controller, dt_trad)
                # 
                # # print('Initial approx for dt', dt)
                # 
                # ############## --------------------- ##############
                # 
                # ### Solve with dt
                # u_sol, u_ref, u, num_mv_sol = self.Solution(self.u, dt)
                # 
                # ### Error
                # error = np.mean(abs(u_sol - u_ref))
                # 
                # ############## --------------------- ##############
                # 
                # if error > self.error_tol:
                # 
                #     # print('Error = ', error)
                # 
                #     ### Traditional controller
                #     u_sol, u_ref, dt_inp, dt_used, dt_trad, num_mv_trad = Trad_Controller(self.Solution, Method_order, \
                #                                                                             error, u, dt_trad, self.error_tol)
                # 
                #     cost_trad = num_mv_sol + num_mv_trad
                #     cost_cont = 0
                #     trad_iter = trad_iter + 1
                # 
                # else:
                # 
                #     if dt == dt_trad:
                # 
                #         ### dt from traditional controller used; error < tolerance
                #         cost_trad = num_mv_sol
                #         cost_cont = 0
                #         trad_iter = trad_iter + 1
                # 
                #     elif dt == dt_controller:
                # 
                #         ### dt from cost controller used
                #         cost_cont = num_mv_sol
                #         cost_trad = 0
                #         cost_iter = cost_iter + 1
                # 
                #     else:
                # 
                #         print('Error in selecting dt!! Unknown dt used!!!')
                # 
                #     ############## --------------------- ##############
                # 
                #     ## dt used in this time step
                #     dt_used = dt
                # 
                #     ### Estimate of dt for next time step using traditional controller ###
                #     new_dt = dt * (self.error_tol/error)**(1/(Method_order + 1))
                #     dt_trad = 0.875 * new_dt          # Safety factor

                # print('Actual dt used', dt_used)

            ############## --------------------- ##############

            ### Update variables
            counter = counter + 1
            count_mv_iter = max(cost_cont, cost_trad)           # No. of matrix-vector products at each time step
            count_mv = count_mv + count_mv_iter                 # Total no. of matrix-vector products
            mat_vec_prod.append(count_mv_iter)                  # List of no. of matrix-vector products at each time step
            dt_temp.append(dt_used)                             # List of no. of dt at each time step

            self.u = u_ref.copy()
            self.dt = dt_used
            time = time + self.dt
            
            error = np.mean(abs(u_ref - u_sol))
            print('Error incurred = ', error)
            print('dt = ', self.dt, dt_trad)
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

            ############# --------------------- ##############

            # ### Write data to files
            # file_1.write(' '.join(map(str, self.u)) % self.u + '\n')
            # file_2.write(' '.join(map(str, u_ref)) % u_ref + '\n')
            # file_3.write('%.15f' % self.dt + '\n')

            ############# --------------------- ##############

            ## Test plots
            plt.plot(self.X, u_ref, 'rd', label = 'Reference')
            plt.plot(self.X, u_sol, 'b.', label = 'Data')
            plt.legend()
            plt.pause(self.dt/4)
            plt.clf()

            ############## --------------------- ##############

        print('Cost controller used in ', cost_iter, 'time steps')
        print('Traditional controller used in ', trad_iter, 'time steps')
        print('Number of time steps = ', counter)
        print('Total number of matrix-vector products = ', count_mv)

        ############## --------------------- ##############

        # ### Write simulation results to file
        # file_res = open(path + 'Results.txt', 'w+')
        # file_res.write('Number of time steps = %d' % counter + '\n')
        # file_res.write('Number of matrix-vector products = %d' % count_mv + '\n')
        # file_res.write('Cost controller used in %d' % cost_iter + ' time steps')
        # file_res.close()
        #
        # ### Close files
        # file_1.close()
        # file_2.close()
        # file_3.close()


        x_cnt = np.arange(0, counter, 1)
        advCFL = np.ones(counter) * self.adv_cfl
        difCFL = np.ones(counter) * self.dif_cfl

        plt.loglog(x_cnt, dt_temp, 'b')
        plt.loglog(x_cnt, advCFL, 'r')
        plt.loglog(x_cnt, difCFL, 'g')

##############################################################################

error_list_1 = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]
error_list_2 = [1e-5]

## Assign values for N, tmax, tol, and eta
for ii in error_list_2:

    loopTime = datetime.now()

    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')

    N = 500
    t_max = 1e-3
    eta = 100
    error_tol = ii

    def main():
        sim = Run_Cost_Controller(N, t_max, eta, error_tol)
        sim.run()
        plt.show()

    if __name__ == "__main__":
        main()

    print('Time for given tolerance = ', datetime.now() - loopTime)

##################################################################################

print('Total Time Elapsed = ', datetime.now() - startTime)