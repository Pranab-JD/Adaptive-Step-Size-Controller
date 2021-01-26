"""
Created on Sun Jan 10 22:51:52 2021

@author: Pranab JD

Description: Initial Conditions
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

###################################################################################################

X = np.linspace(0, 1, 1000)
Y = np.linspace(0, 1, 1000)

X2, Y2 = np.meshgrid(X, Y)

### ------------------------------------------------------------------------------------------- ###

### Viscous Burgers' Equation

sigma = 0.02; x_0 = 0.9; y_0 = 0.9

np.seterr(divide = 'ignore')
u_vis_1d = 1 + (np.exp(1 - (1/(1 - (2 * X - 1)**2)))) + 1./2. * np.exp(-(X - x_0)**2/(2 * sigma**2))
u_vis_2d = 1 + (np.exp(1 - (1/(1 - (2 * X2 - 1)**2)) - (1/(1 - (2 * Y2 - 1)**2)))) \
             + 1./2. * np.exp(-((X2 - x_0)**2 + (Y2 - y_0)**2)/(2 * sigma**2))

### ------------------------------------------------------------------------------------------- ###

### Inviscid Burgers' Equation
			 
epsilon_1 = 0.01; epsilon_2 = 0.01

u_inv_1d = 2 + epsilon_1 * np.sin(2 * np.pi * X) + epsilon_2 * np.sin(8 * np.pi * X + 0.3)
u_inv_2d = 2 + epsilon_1 * (np.sin(2 * np.pi * X2) + np.sin(2 * np.pi * Y2)) \
     	     + epsilon_2 * (np.sin(8 * np.pi * X2 + 0.3) + np.sin(8 * np.pi * Y2 + 0.3))

### ------------------------------------------------------------------------------------------- ###

### Porous Medium Equation
			
x_1 = 0.25; x_2 = 0.6; y_1 = 0.25; y_2 = 0.6

u_por_1d = 1 + np.heaviside(x_1 - X, x_1) + np.heaviside(X - x_2, x_2)
u_por_2d = 1 + np.heaviside(x_1 - X2, x_1) + np.heaviside(X2 - x_2, x_2) \
             + np.heaviside(y_1 - Y2, y_1) + np.heaviside(Y2 - y_2, y_2)
			
###################################################################################################
			
### 1D plot ###
plt.figure(figsize = (8, 6), dpi = 200)
plt.plot(X, u_por_1d, 'b')

plt.tick_params(axis = 'both', which = 'major', labelsize = 14)

plt.xlabel('$X$', fontsize = 16)
plt.ylabel('$U(x, t)$', fontsize = 16)
plt.xlim(0, 1)

plt.tight_layout()
#plt.savefig('./Initial Conditions/Por_1D.eps')


### 2D plot ###
plt.figure(figsize = (8, 6), dpi = 300)
        
plt.imshow(u_por_2d, origin = 'lower', cmap = cm.jet, extent = [0, 1, 0, 1], aspect = 'equal')
plt.colorbar()

plt.xlabel('$X$', fontsize = 14)
plt.ylabel('$Y$', fontsize = 14)
        
plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.tick_params(axis = 'both', which = 'minor', labelsize = 12)
        
plt.tight_layout()
#plt.savefig('./Initial Conditions/Por_2D.eps')

###################################################################################################
