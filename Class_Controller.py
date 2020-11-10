"""
Created on Mon Oct 26 18:33:28 2020

@author: Pranab JD

Description:
    Initialization of parameters and matrices for
    adaptive step size controller using Leja
    polynomial interpolation for
    exponential integrators.
"""


import numpy as np
from scipy.sparse import csr_matrix

##############################################################################

class Cost_Controller:

    def __init__(self, N, tmax, eta, error_tol):
        self.N = int(N)                 # Number of points along X
        self.xmin = 0                   # Min value of X
        self.xmax = 1                   # Max value of X
        self.tmax = tmax                # Maximum time
        self.error_tol = error_tol      # Maximum error permitted
        self.eta = eta                  # Peclet number
        self.initialize_spatial_domain()
        self.initialize_parameters()
        self.initialize_matrices()

	### Discretize the spatial domain
    def initialize_spatial_domain(self):
        self.dx = (self.xmax - self.xmin)/self.N
        self.X = np.linspace(self.xmin, self.xmax, self.N, endpoint = False)

    ### Parameters
    def initialize_parameters(self):
        self.adv_cfl = self.dx/self.eta
        self.dif_cfl = self.dx**2/2
        print('Advection CFL: ', self.adv_cfl)
        print('Diffusion CFL: ', self.dif_cfl)
        print('Tolerance:', self.error_tol)
        self.dt = 2 * min(self.adv_cfl, self.dif_cfl)    # N * CFL condition
        self.R = self.eta/self.dx
        self.F = 1/self.dx**2                            # Fourier mesh number

    ### Operator matrices
    def initialize_matrices(self):
        self.A_adv = np.zeros((self.N, self.N))
        self.A_dif = np.zeros((self.N, self.N))

        for ij in range(self.N):
            self.A_adv[ij, int(ij + 2) % self.N] = -1/6
            self.A_adv[ij, int(ij + 1) % self.N] =  6/6
            self.A_adv[ij, ij % self.N]          = -3/6
            self.A_adv[ij, int(ij - 1) % self.N] = -2/6

            self.A_dif[ij, int(ij + 1) % self.N] =  1
            self.A_dif[ij, ij % self.N]          = -2
            self.A_dif[ij, int(ij - 1) % self.N] =  1

        self.A_adv = csr_matrix(self.R * self.A_adv)
        self.A_dif = csr_matrix(self.F * self.A_dif)

##############################################################################
