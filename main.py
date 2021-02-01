"""
Created on Tue Nov 10 15:29:57 2020

@author: Pranab JD

Description: 
    Runs the program with selected values/parameters of:
    (1) Equations (Systems)
    (2) Integrators
    (3) Dimensionality
"""

import os
import shutil
import Systems_1D
import Systems_2D

from datetime import datetime

startTime = datetime.now()

##############################################################################

### Tolerances ###
error_list_1 = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
error_list_2 = [1e-4, 5e-5]
error_list_3 = [1e-4]

## Assign values for N, tmax, tol, and eta
for ii in error_list_1:

    tolTime = datetime.now()

    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')

    ### ------------------------------------------------------------------- ###

    ### 1D ###

    N = 300
    t_max = 0.01
    eta = 50
    error_tol = ii

    run_1D = Systems_1D.Run_1D_Systems(N, t_max, eta, error_tol)

    ### ------------------------------------------------------------------- ###

    ### 2D ###

    ## 1/N_x * eta_x = 1/N_y * eta_y for equal numerical diffusion along X and Y
    ## for 1st order upwind scheme

    # N_x = 150
    # N_y = 150
    # t_max = 0.0325
    # eta_x = 100
    # eta_y = 100
    # error_tol = ii

    # run_2D = Systems_2D.Run_2D_Systems(N_x, N_y, t_max, eta_x, eta_y, error_tol)

    ### ------------------------------------------------------------------- ###

    def main():
        # run_1D.constant_h()
        # run_2D.constant_h()
        
        run_1D.adaptive_h()
        # run_2D.adaptive_h()

    if __name__ == "__main__":
        main()

    print('Time for given tolerance = ', datetime.now() - tolTime)

##############################################################################

print('Total Time Elapsed = ', datetime.now() - startTime)