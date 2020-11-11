Important points to be noted: -

1. Order of the method is defined in Run_Code.py. Needs to be changed in accordance with the method used.

2. If eigen values from the linear matrix operator is used, 'its_power' is not needed (= 0) in 'self.Solution'.

3. For inviscid Burgers' equation, choose initial value of dt = 2 * self.adv_cfl.

4. For embedded methods, 'its_sol' is already included in 'its_ref'.
