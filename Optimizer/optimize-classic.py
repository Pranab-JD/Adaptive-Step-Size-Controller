from sys import argv, exit
from numpy import *
from linimp import *
from matrices import x_periodic
from integration import integrate, integrate_new
from scipy.sparse import csr_matrix
from scipy.optimize import differential_evolution
from time import time
from scipy.interpolate import interp1d

def diffusion_periodic(n):
    A=n**2*(-2*diag(ones(n))+diag(ones(n-1),1)+diag(ones(n-1),-1));
    A[0,-1]=n**2; A[-1,0]=n**2
    return A

def advection_periodic(n):
    A=0.5*n*(diag(ones(n-1),1)-diag(ones(n-1),-1));
    A[0,-1]=-0.5*n; A[-1,0]=0.5*n
    return A

def cost_ctrler(cost,tau,cost_old,tau_old,alpha,beta,gamma,delta):
    cost_norm=cost/tau; cost_old_norm=cost_old/tau_old
    grad_c = (log(cost_norm)-log(cost_old_norm))/(log(tau)-log(tau_old));
    direc  = abs(grad_c)
    frac = exp(-alpha*tanh(beta*grad_c))
    if frac>=1.0 and frac<gamma:
        frac = gamma
    elif frac<=1.0 and frac>delta:
        frac = delta
    return frac


import signal
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def solve(n,peclet,tol,alpha,is_ref):
    x=x_periodic(n)
    method = sdirk45; order=4; T=0.2; tau0=1e-4;
    sigma = 0.0014; u0=exp(-(x-0.5)**2/(2*sigma**2))

    def integrate_new_gd(*args):
        # the ugly named arguments are appearently necessary in python <= 3.4
        return integrate_new(*args,ctrl=lambda *args: cost_ctrler(*args,alpha=alpha[0],beta=alpha[1],gamma=alpha[2],delta=alpha[3]))

    A = csr_matrix(diffusion_periodic(n)+peclet*advection_periodic(n))
    if is_ref == False:
        res=integrate_new_gd(A,u0,T,tol,tau0,method,order) # alpha, beta, gamma params
    else:
        print('running the standard step size controller')
        res=integrate(A,u0,T,tol,tau0,method,order)
    print('its: ', res[1])
    return res[1]


g_ref = []

def fitness(alpha,ref=False):
    global g_ref
    try:
        n_pecl = [(100,10),(300,100),(500,0),(500,100)]
        tols   = [1e-2, 1e-3, 1e-4, 1e-5,1e-7]
        fit=0.0
        i=0
        for n,pecl in n_pecl:
            for tol in tols:
                cost = solve(n,pecl,tol,alpha,ref)
                if ref==True: g_ref.append(cost)
                fit += cost/g_ref[i]
                i+=1

        rel_fit = fit/(len(n_pecl)*len(tols))
        print('fit: ', rel_fit, alpha)

        return rel_fit
    except TimeoutError:
        print('timeout reached without solution. aborting.')
        return 1e10

# This is just to get a baseline for comparison
print("----------- Baseline (P controller) -----------")
start = time()
fitness([],ref=True)  # 1.6,0.3,1.15,0.85
wall = time()-start
print('reference computed: ',g_ref)
print('estimated time per generation (min): ', wall*15*4*2/60)

## optimization for alpha,beta,gamma,delta params
print("----------- Optimization -----------")
print(fitness([0.4,0.3,1.15,0.85],ref=False))
def clbk(xk,convergence):
    print('clbk: ',xk)
    return convergence
bounds = [(0.5,2.0),(0.1,1.0),(1.05,1.4),(0.6,0.95)]
res = differential_evolution(fitness, bounds, disp=True, callback=clbk, popsize=15, maxiter=100)
print(res.x, res.fun)
