# Cost Controller for Exponential Integrators

This code considers an adaptive step size controller, first proposed in [Einkemmer (2017), "An adaptive step size controller for iterative implicit methods"](https://www.sciencedirect.com/science/article/pii/S0168927418301387?via%3Dihub) for implicit integrators, for exponential integrators. 

This code computes the matrix exponential and the <img src="https://render.githubusercontent.com/render/math?math=\varphi"> function by means of polynomial interpolation at Leja points, the convergence of which is checked iteratively.

The step size controller has been engineered to minimize the computational cost (for iterative methods). This step size controller outperforms the traditional controller by 1 - 4 times for the EXPRB43 scheme in terms of computational cost. For details, see [Deka & Einkemmer (2021), Efficient adaptive step size control for exponential integrators](https://arxiv.org/abs/2102.02524).


## Code Structure
1. Class_controller.py - Initializes the domain, matrices (advection and diffusion), boundary conditions, etc.
2. 

## Technical Aspects
1. Order of the method is defined in Systems_1D.py (and System_2D.py). This needs to changed in accordance with the method used. By default, it is set to 3 (for EXPRB43).
2. If the eigen values from the linear matrix operator is used (usign the Gershgorin's disk theorem), then 'its_power' has to be set to 0 (in 'self.Solution').
3. For embedded schemes, 'its_sol' is already included in 'its_ref'.
