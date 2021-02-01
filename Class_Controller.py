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
from scipy.sparse import csr_matrix, kron, identity

##############################################################################

class Cost_Controller_1D:

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
        self.dt = 0.9 * min(self.adv_cfl, self.dif_cfl)     # N * CFL condition
        self.R = self.eta/self.dx
        self.F = 1/self.dx**2                               # Fourier mesh number

    ### Operator matrices
    def initialize_matrices(self):
        self.A_adv = np.zeros((self.N, self.N))
        self.A_dif = np.zeros((self.N, self.N))

        for ij in range(self.N):
            ## 3rd order upwind 
            self.A_adv[ij, int(ij + 2) % self.N] = -1/6
            self.A_adv[ij, int(ij + 1) % self.N] =  6/6
            self.A_adv[ij, ij % self.N]          = -3/6
            self.A_adv[ij, int(ij - 1) % self.N] = -2/6

            ## 2nd order centered difference
            self.A_dif[ij, int(ij + 1) % self.N] =  1
            self.A_dif[ij, ij % self.N]          = -2
            self.A_dif[ij, int(ij - 1) % self.N] =  1

        self.A_adv = csr_matrix(self.R * self.A_adv)
        self.A_dif = csr_matrix(self.F * self.A_dif)

##############################################################################

class Cost_Controller_2D:

    def __init__(self, N_x, N_y, tmax, eta_x, eta_y, error_tol):
        self.N_x = int(N_x)                     # Number of points along X
        self.N_y = int(N_y)                     # Number of points along Y
        self.xmin = 0                           # Min value of X
        self.xmax = 1                           # Max value of X
        self.ymin = 0                           # Min value of Y
        self.ymax = 1                           # Max value of Y
        self.eta_x = eta_x                      # Peclet number along X
        self.eta_y = eta_y                      # Peclet number along Y
        self.tmax = tmax                        # Maximum time
        self.error_tol = error_tol              # Maximum error permitted
        self.initialize_spatial_domain()
        self.initialize_parameters()
        self.initialize_matrices()
        
    ### Discretize the spatial domain
    def initialize_spatial_domain(self):
        self.dx = (self.xmax - self.xmin)/self.N_x
        self.dy = (self.ymax - self.ymin)/self.N_y
        self.X = np.linspace(self.xmin, self.xmax, self.N_x, endpoint = False)
        self.Y = np.linspace(self.ymin, self.ymax, self.N_y, endpoint = False)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
              
    ### Parameters  
    def initialize_parameters(self):
        self.adv_cfl = (self.dx * self.dy)/(self.eta_y*self.dx + self.eta_x*self.dy)
        self.dif_cfl = (self.dx**2 * self.dy**2)/(2 *(self.dx**2 + self.dy**2))
        print('Advection CFL: ', self.adv_cfl)
        print('Diffusion CFL: ', self.dif_cfl)
        print('Tolerance:', self.error_tol)
        self.dt = 0.9 * min(self.adv_cfl, self.dif_cfl)     # N * CFL condition
        self.Rx = self.eta_x/self.dx
        self.Ry = self.eta_y/self.dy
        self.Fx = 1/self.dx**2                            # Fourier mesh number
        self.Fy = 1/self.dy**2                            # Fourier mesh number

    ### Operator matrices
    def initialize_matrices(self):
        self.Adv_x = np.zeros((self.N_x, self.N_x))       # Advection (X)
        self.Adv_y = np.zeros((self.N_y, self.N_y))       # Advection (Y)
        self.Dif_x = np.zeros((self.N_x, self.N_x))       # Diffusion (X)
        self.Dif_y = np.zeros((self.N_y, self.N_y))       # Diffusion (Y)

        for ij in range(self.N_x):
            ## 3rd order upwind 
            self.Adv_x[ij, int(ij + 2) % self.N_x] = -1/6
            self.Adv_x[ij, int(ij + 1) % self.N_x] =  6/6
            self.Adv_x[ij, ij % self.N_x]          = -3/6
            self.Adv_x[ij, int(ij - 1) % self.N_x] = -2/6
            
            ## 2nd order centered difference
            self.Dif_x[ij, int(ij + 1) % self.N_x] =  1
            self.Dif_x[ij, ij % self.N_x]          = -2
            self.Dif_x[ij, int(ij - 1) % self.N_x] =  1
        
        for ij in range(self.N_y):  
            ## 3rd order upwind   
            self.Adv_y[ij, int(ij + 2) % self.N_y] = -1/6
            self.Adv_y[ij, int(ij + 1) % self.N_y] =  6/6
            self.Adv_y[ij, ij % self.N_y]          = -3/6
            self.Adv_y[ij, int(ij - 1) % self.N_y] = -2/6
            
            ## 2nd order centered difference
            self.Dif_y[ij, int(ij + 1) % self.N_y] =  1
            self.Dif_y[ij, ij % self.N_y]          = -2
            self.Dif_y[ij, int(ij - 1) % self.N_y] =  1

        ### Merge X and Y to get a single matrix for advection and diffusion    
        self.A_adv = kron(identity(self.N_y), self.Rx * self.Adv_x) \
                   + kron(self.Ry * self.Adv_y, identity(self.N_x))      
        self.A_dif = kron(identity(self.N_y), self.Fx * self.Dif_x) \
                   + kron(self.Fy * self.Dif_y, identity(self.N_x))

##############################################################################