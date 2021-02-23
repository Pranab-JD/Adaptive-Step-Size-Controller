# Cost Controller for Exponential Integrators

This code tests an adaptive step size controller, first proposed in [Einkemmer (2017), "An adaptive step size controller for iterative implicit methods"](https://www.sciencedirect.com/science/article/pii/S0168927418301387?via%3Dihub) for implicit integrators, for exponential integrators. 

This code implements polynomial interpolation at Leja points, the convergence of which is check iteratively.

This step size controller outperforms the traditional controller by 1 - 4 times for EXPRB43, for details, see [Deka & Einkemmer (2021), Efficient adaptive step size control for exponential integrators](https://arxiv.org/abs/2102.02524).




## Technical Aspects

1. Order of the method is defined in Systems_1D.py and System_2D.py. Needs to changed in accordance with the method used. 
By default, it is set to 3.

2. If eigen values from the linear matrix operator is used, 'its_power' is not needed (= 0) in 'self.Solution'.

3. For embedded methods, 'its_sol' is already included in 'its_ref'.
