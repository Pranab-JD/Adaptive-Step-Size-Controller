# Cost Controller for Exponential Integrators

Important points to be noted: -




## Technical Aspects

1. Order of the method is defined in Systems_1D.py and System_2D.py. Needs to changed in accordance with the method used. 
By default, it is set to 3.

2. If eigen values from the linear matrix operator is used, 'its_power' is not needed (= 0) in 'self.Solution'.

3. For embedded methods, 'its_sol' is already included in 'its_ref'.
