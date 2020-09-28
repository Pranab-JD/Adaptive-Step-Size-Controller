"""
Created on Thu Jul 23 15:54:52 2020

@author: Pranab JD

Description: -
        Header file for Leja point interpolation
"""

import numpy as np

################################################################################################

def phi_1(z):
    return ((np.exp(z) - 1)/z)

def phi_2(z):
    return ((np.exp(z) - z - 1)/z**2)

def phi_3(z):
    return ((np.exp(z) - z**2/2 - z - 1)/z**3)

def phi_4(z):
    return ((np.exp(z) - z**3/6 - z**2/2 - z - 1)/z**4)

################################################################################################

def Leja_Points():
    """
    Load Leja points from binary file

    """
    dt = np.dtype("f8")
    return np.fromfile('real_leja_d.bin', dtype = dt)

def Divided_Difference(X, func):
    """
    Parameters
    ----------
    X       : Leja points
    func    : func(X)

    Returns
    -------
    div_diff : Polynomial coefficients

    """

    div_diff = func(X)

    for ii in range(1, int(len(X)/10)):
        for jj in range(ii):
            div_diff[ii] = (div_diff[jj] - div_diff[ii])/(X[jj] - X[ii])

    return div_diff

def divdiff(x, func):
    d = [func(x[0])]
    for m in range(1,int(len(x)/10)):
        d.append(func(x[m]))
        for i in range(m):
            d[m] = (d[m]-d[i])/(x[m]-x[i])

    return d


def Gershgorin(A):
    """
    Parameters
    ----------
    A       : N x N matrix A

    Returns
    -------
    eig_real : Largest real eigen value
    eig_imag : Largest imaginary eigen value

    """

    A_Herm = (A + A.T.conj())/2
    A_SkewHerm = (A - A.T.conj())/2

    row_sum_real = np.zeros(np.shape(A)[0])
    row_sum_imag = np.zeros(np.shape(A)[0])

    for ii in range(len(row_sum_real)):
        row_sum_real[ii] = np.sum(abs(A_Herm[ii, :]))
        row_sum_imag[ii] = np.sum(abs(A_SkewHerm[ii, :]))

    eig_real = np.max(row_sum_real)
    eig_imag = np.max(row_sum_imag)

    return eig_real, eig_imag

def Power_iteration(A, u, m):
    """
    Parameters
    ----------
    A       : N x N matrix A
    u       : Vector u
    m       : Index of u (u^m)

    Returns
    -------
    eigen_val[ii] : Largest eigen value (within 2.5% accuracy)
    (ii + 1) * 2  : Number of matrix-vector products
                    (ii + 1): No. of iterations (starts from 0)

    """

    A_Herm = (A + A.T.conj())/2
    A_SkewHerm = (A - A.T.conj())/2

    def eigens(A):

        tol = 0.1
        niters = 1000
        epsilon = 1e-7
        eigen_val = np.zeros(niters)
        vector = np.zeros(len(u))
        vector[0] = 1

        for ii in range(niters):

            eigen_vector = (A.dot((u + (epsilon * vector))**m) - (A.dot(u**m)))/epsilon

            eigen_val[ii] = np.max(abs(eigen_vector))

            if (abs(eigen_val[ii] - eigen_val[ii - 1]) <= tol * eigen_val[ii]):
                break
            else:
                eigen_vector = eigen_vector/eigen_val[ii]
                vector = eigen_vector.copy()

        return eigen_val[ii], ii

    eigen_real, i1 = eigens(A_Herm)
    eigen_imag, i2 = eigens(A_SkewHerm)

    return eigen_real, eigen_imag, ((2 * (i1 + 1)) + (2 * (i2 + 1)))

################################################################################################

def real_Leja_exp(A_adv, u, m_adv, u_func, dt, Leja_X, c_real, Gamma_real):
    """
    Parameters
    ----------
    A               : N x N matrix A
    u               : Vector u
    m               : Index of u (u^m)
    dt              : self.dt
    Leja_X          : Leja points
    c_real          : Shifting factor
    Gamma_real      : Scaling factor

    Returns
    ----------
    np.real(u_real) : Polynomial interpolation of u
                      at real Leja points
    ii * 2          : No. of matrix-vector products

    """

    def func(xx):
        return np.exp(dt * (c_real + Gamma_real*xx))

    ## Polynomial coefficients
    coeffs = Divided_Difference(Leja_X, func)

    ## a_0 term
    poly = u_func.copy()
    poly = coeffs[0] * poly

    ## a_1, a_2 .... a_n terms
    max_Leja_pts = 50
    y = u_func.copy()
    poly_tol = 1e-8
    epsilon = 1e-7
    scale_fact = 1/Gamma_real                                    # Re-scaling factor

    for ii in range(1, max_Leja_pts):

        shift_fact = -c_real * scale_fact - Leja_X[ii - 1]       # Re-shifting factor

        ## function: function to be multiplied to the matrix exponential of the Jacobian
        function = y.copy()
        Jacobian_function = (A_adv.dot((u + (epsilon * function))**m_adv) - A_adv.dot(u**m_adv))/epsilon

        y = y * shift_fact
        y = y + scale_fact * Jacobian_function
        
        poly = poly + coeffs[ii] * y

        ## If new number (next order) to be added < tol, ignore it
        if (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii]) < poly_tol:
            # print('No. of Leja points used (real exp) = ', ii)
            break

        if ii >= max_Leja_pts - 1:
            print('ERROR: max number of Leja iterations reached (real exp)')

    ## Solution
    u_real = poly.copy()

    return np.real(u_real), ii * 2

### ---------------------------------------------------------------------------------------- ###

def imag_Leja_exp(A, u, dt, c_imag, Gamma_imag):
    """
    Parameters
    ----------
    A	            : N x N matrix
    u               : Vector u and function to be 
                      multiplied to matrix exponential
    dt              : self.dt
    c_imag          : Shifting factor
    Gamma_imag      : Scaling factor

    Returns
    ----------
    np.real(u_imag) : Polynomial interpolation of u
                      at imaginary Leja points
    ii * 1          : Total no. of matrix-vector products

    """


    def func(xx):
        return np.exp(1j * dt * (c_imag + Gamma_imag*xx))

    ## Polynomial coefficients
    Leja_X = Leja_Points()
    coeffs = Divided_Difference(Leja_X, func)

    ## a_0 term
    poly = u.copy() + 0 * 1j
    poly = coeffs[0] * poly

    ## a_1, a_2 .... a_n terms
    max_Leja_pts = 50
    y = u.copy() + 0 * 1j
    poly_vals = np.zeros(max_Leja_pts)
    scale_fact = 1/Gamma_imag                                   # Re-scaling factor
    
    for ii in range(1, max_Leja_pts):

        shift_fact = -c_imag * scale_fact - Leja_X[ii - 1]      # Re-shifting factor

        u_temp = y.copy()                                       # u function with which the marix exponential is to be multiplied
        y = y * shift_fact
        y = y + scale_fact * A.dot(u_temp) * (-1j) 

        poly_vals[ii] = (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, ignore it
        if poly_vals[ii] - poly_vals[ii - 1] > 0 and ii > 1:
            # print(poly_vals[ii], poly_vals[ii - 1], poly_vals[ii - 2])
            # print('No. of Leja points used (imag exp) = ', ii)
            # print('-------------------------------------------------------------------------')
            break
        elif ii == 1:
            poly = poly + coeffs[ii] * y
        else:
            poly = poly + coeffs[ii] * y

        if ii >= max_Leja_pts - 1:
            print('ERROR: max number of Leja iterations reached (imag exp)')
            print('-------------------------------------------------------------------------')

    ## Solution
    u_imag = poly.copy()

    return np.real(u_imag), ii * 1

################################################################################################

def real_Leja_phi(A, u, m, nonlin_matrix_vector, dt, Leja_X, c_real, Gamma_real, phi_func):
    """
    Parameters
    ----------
    phi_func                : phi function
    nonlin_matrix_vector    : Nonlinear part of the equation
    dt                      : self.dt
    Leja_X                  : Leja points
    c_real                  : Shifting factor
    Gamma_real              : Scaling factor

    Returns
    ----------
    np.real(u_real)         : Polynomial interpolation of
                              nonlinear part using the phi_1
                              function at real Leja points

    """

    def func(xx):
    
        np.seterr(divide = 'ignore', invalid = 'ignore')
    
        zz = dt * (c_real + Gamma_real*xx)
        var = phi_func(zz)
    
        if phi_func == phi_1:
    
            for ii in range(len(Leja_X)):
                if zz[ii] <= 1e-9:
                    var[ii] = 1 + zz[ii] * (1./2. + zz[ii] * (1./6. + zz[ii] * (1./24. + 1./120.*zz[ii])))
    
        elif phi_func == phi_2:
    
            for ii in range(len(Leja_X)):
                if zz[ii] <= 1e-9:
                    var[ii] = 1./2. + zz[ii] * (1./6. + zz[ii] * (1./24. + zz[ii] * (1./120. + 1./720.*zz[ii])))
    
    
        elif phi_func == phi_3:
    
            for ii in range(len(Leja_X)):
                if zz[ii] <= 1e-9:
                    var[ii] = 1./6. + zz[ii] * (1./24. + zz[ii] * (1./120. + zz[ii] * (1./720. + 1./5040.*zz[ii])))
                    
        elif phi_func == phi_4:
    
            for ii in range(len(Leja_X)):
                if zz[ii] <= 1e-9:
                    var[ii] = 1./24. + zz[ii] * (1./120. + zz[ii] * (1./720. + zz[ii] * (1./5040. + 1./40320.*zz[ii])))
    
        else:
            print('Error: Phi function not defined!!')
    
        return var

    ## Polynomial coefficients
    coeffs = Divided_Difference(Leja_X, func)

    # a_0 term
    poly = nonlin_matrix_vector.copy()
    poly = coeffs[0] * poly

    # a_1, a_2 .... a_n terms
    max_Leja_pts = 50
    y = nonlin_matrix_vector.copy()
    poly_tol = 1e-8
    epsilon = 1e-12
    scale_fact = 1/Gamma_real                                    # Re-scaling factor

    for ii in range(1, max_Leja_pts):

        shift_fact = -c_real * scale_fact - Leja_X[ii - 1]       # Re-shifting factor

        ## function: function to be multiplied to the phi function applied to Jacobian
        function = y.copy()
        Jacobian_function = (A.dot((u + (epsilon * function))**m) - A.dot(u**m))/epsilon

        y = y * shift_fact
        y = y + scale_fact * Jacobian_function
        
        poly = poly + coeffs[ii] * y

        # If new number (next order) to be added < tol, ignore it
        if (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii]) < poly_tol:
            # print('No. of Leja points used (real phi) = ', ii)
            break

        if ii >= max_Leja_pts - 1:
            print('ERROR: max number of Leja iterations reached (real phi)')

    # Solution
    u_real = poly.copy()

    return u_real, ii * 2

### ---------------------------------------------------------------------------------------- ###

def imag_Leja_phi(u, nonlin_matrix_vector, dt, c_imag, Gamma_imag, phi_func, *A):
    """
    Parameters
    ----------
    u                       : Vector u
    nonlin_matrix_vector    : function to be multiplied to phi function
    dt                      : self.dt
    Leja_X                  : Leja points
    c_imag                  : Shifting factor
    Gamma_imag              : Scaling factor
    phi_func                : phi function
    *A						: N x N matrix A, power to which u is raised

    Returns
    ----------
    np.real(u_imag)         : Polynomial interpolation of
                              nonlinear part using the phi_1
                              function at imaginary Leja points
    ii * len(A)             : No. of matrix-vector products

    """
    
    
    def func(xx):
    
        np.seterr(divide = 'ignore', invalid = 'ignore')
    
        zz = (1j * dt * (c_imag + Gamma_imag*xx))
        
        var = phi_func(zz)
        cnt = 0
        if phi_func == phi_1:

            for ii in range(len(zz)):
                if abs(np.imag(zz[ii])) <= 1e-7:
                    print('Taylor exp')
                    var[ii] = 1 + zz[ii] * (1./2. + zz[ii] * (1./6. + zz[ii] * (1./24. + 1./120.*zz[ii])))
                    # print(var)
                    cnt = cnt + 1
    
        elif phi_func == phi_2:
    
            for ii in range(len(Leja_X)):
                if zz[ii] <= 1e-7:
                    var[ii] = 1./2. + zz[ii] * (1./6. + zz[ii] * (1./24. + zz[ii] * (1./120. + 1./720.*zz[ii])))
    
    
        elif phi_func == phi_3:
    
            for ii in range(len(zz)):
                if abs(np.imag(zz[ii])) <= 1e-7:
                    print('Taylor exp')
                    var[ii] = 1./6. + zz[ii] * (1./24. + zz[ii] * (1./120. + zz[ii] * (1./720. + 1./5040.*zz[ii])))
                    # print(var)
                    cnt = cnt + 1
                    
        elif phi_func == phi_4:
    
            for ii in range(len(Leja_X)):
                if zz[ii] <= 1e-7:
                    var[ii] = 1./24. + zz[ii] * (1./120. + zz[ii] * (1./720. + zz[ii] * (1./5040. + 1./40320.*zz[ii])))
    
        else:
            print('Error: Phi function not defined!!')
        
        if cnt>0:
            print(cnt)
    
        return var

    ## Polynomial coefficients
    Leja_X = Leja_Points()
    coeffs = Divided_Difference(Leja_X, func)
    
    ## Define matrices
    if len(A) == 2:
        A_adv = A[0]; m_adv = A[1]
    elif len(A) == 4:
        A_adv = A[0]; m_adv = A[1]
        A_dif = A[2]; m_dif = A[3]
    else:
        print("Error! Check number of matrices!!")
              

    ## a_0 term
    poly = nonlin_matrix_vector.copy() + 0 * 1j
    poly = coeffs[0] * poly
    
    ## a_1, a_2 .... a_n terms
    y = nonlin_matrix_vector.copy() + 0 * 1j
    max_Leja_pts = 50
    poly_vals = np.zeros(max_Leja_pts)
    poly_tol = 1e-4                                            ## for EXPRB42
    epsilon = 1e-7
    
    scale_fact = 1/Gamma_imag                                   # Re-scaling factor

    for ii in range(1, max_Leja_pts):

        shift_fact = -c_imag * scale_fact - Leja_X[ii - 1]      # Re-shifting factor
        
        ## function: function to be multiplied to the phi function applied to Jacobian
        function = y.copy()
        
        # if np.linalg.norm(function)/10 > 1.75 or np.linalg.norm(function)/10 < 0.02:
        #     epsilon = 1e-7/np.linalg.norm(function)
        # else:
        #     epsilon = 1e-7

        if len(A) == 2:
            Jacobian_function = (A_adv.dot((u + (epsilon * function))**m_adv) - A_adv.dot(u**m_adv))/epsilon
            
        if len(A) == 4:
            Jacobian_function = (A_adv.dot((u + (epsilon * function))**m_adv) + A_dif.dot((u + (epsilon * function))**m_dif) \
                                - A_adv.dot(u**m_adv) - A_dif.dot(u**m_dif))/epsilon
            
        y = y * shift_fact
        y = y + scale_fact * Jacobian_function * (-1j)
        
        poly_vals[ii] = (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, ignore it
        if  ii > 1 and (poly_vals[ii] - poly_vals[ii - 1] > 0 or poly_vals[ii] < poly_tol):
            # print(poly_vals[0:ii])
            # print('No. of Leja points used (imag phi) = ', ii)
            # print('-------------------------------------------------------------------------')
            poly = poly + coeffs[ii] * y
            break
        elif ii == 1:
            poly = poly + coeffs[ii] * y
        else:
            poly = poly + coeffs[ii] * y

        if ii >= max_Leja_pts - 1:
            print('ERROR: max number of Leja iterations reached (imag phi)')
            print('-------------------------------------------------------------------------')

    ## Solution
    u_imag = poly.copy()

    return np.real(u_imag), ii * len(A)


################################################################################################
