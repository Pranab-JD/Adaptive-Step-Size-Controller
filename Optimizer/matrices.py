from numpy import *
from Leja_Interpolation import *
from scipy.sparse import csr_matrix

def diffusion_periodic(n):
    A=n**2*(-2*diag(ones(n))+diag(ones(n-1),1)+diag(ones(n-1),-1));
    A[0,-1]=n**2
    A[-1,0]=n**2
    return A

def diffusion_reaction_periodic(n):
    return diffusion_periodic(n)+(2*pi+0.1)**2*identity(n)

def advection_periodic(n):
    A=0.5*n*(diag(ones(n-1),1)-diag(ones(n-1),-1));
    A[0,-1]=-0.5*n
    A[-1,0]=0.5*n
    return A

def diffusion_advection_reaction(n,x):
    sp=100.0*exp(-500.0*(x-0.5)**2)
    # sp=exp(0.2)*exp(-0.2/(1-100.0*(x-0.5)**2))*(1.0-sign(abs(x-0.5)-0.1));
    return diffusion_periodic(n)+15.0*advection_periodic(n)-sp*identity(n)

def x_periodic(n):
    return linspace(0,1,n,endpoint=False)

### ----------------------------------------------------------------------- ###

### Burgers's equation ###

"""
    Periodic boundary conditions are implemented in all matrices.
""" 


def initialize_params(n, eta):

    X = x_periodic(n)

    dx = X[1] - X[0]
    R = 1./6. * eta/dx      # Advection parameter
    F = 1/dx**2             # Diffusion parameter

    return X, dx, R, F

def Viscous_Burgers(n, x):

    X, dx, R, F = initialize_params(n, eta)

    A_dif = zeros((n, n))           # Diffusion Matrix
    A_adv = zeros((n, n))           # Advection Matrix

    for ij in range(n):
        A_adv[ij, int(ij + 2) % n] = - R/2
        A_adv[ij, int(ij + 1) % n] = 6 * R/2
        A_adv[ij, ij % n] = -3 * R/2
        A_adv[ij, int(ij - 1) % n] = -2 * R/2

        A_dif[ij, int(ij + 1) % n] = F
        A_dif[ij, ij % n] = -2 * F
        A_dif[ij, int(ij - 1) % n] = F

    A_adv = csr_matrix(A_adv)
    A_dif = csr_matrix(A_dif)

    return A_adv, A_dif


def Inviscid_Burgers(n, x):

    X, dx, R, F = initialize_params(n, eta)

    A_adv = zeros((n, n))           # Advection Matrix

    for ij in range(n):
        A_adv[ij, int(ij + 2) % n] = - R/2
        A_adv[ij, int(ij + 1) % n] = 6 * R/2
        A_adv[ij, ij % n] = -3 * R/2
        A_adv[ij, int(ij - 1) % n] = -2 * R/2

    A_adv = csr_matrix(A_adv)

    return A_adv

### ----------------------------------------------------------------------- ###
