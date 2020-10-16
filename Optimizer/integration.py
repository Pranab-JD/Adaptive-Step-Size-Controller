from numpy import *
from time import time

def run_accuracy_cost(A,u0,tols,integrator,T,tau0,method,order,uT):
    it_list = []; acc_list = []
    h_lists=[]; c_lists=[]
    for tol in tols:
        res=integrator(A,u0,T,tol,tau0,method,order); print(res[1:5])
        h_lists.append(res[5]); c_lists.append(res[6])
        it_list.append([tol,res[1]])
        acc_list.append([max(abs(res[0]-uT)),res[1]])
    return array(it_list),array(acc_list),array(h_lists),array(c_lists)

#
# step size controllers
#

def nextstep_P(h_old,err,tol,order):
    safety=0.9
    return (safety*tol/err)**(1.0/float(order))*h_old

#
# time stepping
#

def integrate(A,u00,T,tol,tau0,method,order=1,tol_fac=0.1):
    u0=u00.copy()
    t=0.0; tau=tau0; tol_ls=tol_fac*tol;
    its=0; accepted=0; rejected=0; h_list=[]; c_list=[]
    uprev = u0.copy(); tauprev = 1.0; errest=0.0
    t0 = time()
    while t<T:
        t1 = time()
        u_est = u0 + tau*(u0-uprev)/tauprev
        # print('a',its)
        u1,_its  = method(A,u0,tau,tol_ls,u_est); its+=_its
        # print('b', its)
        errest += max(abs(u_est-u1));
        u1e,_   = method(A,method(A,u0,0.5*tau,tol_ls)[0],0.5*tau,tol_ls) # use Richardson extrapolation
        err_est = max(abs(u1-u1e))
        # print('full step',time()-t1,its)
        if err_est <= tol: # accept step
            uprev = u0.copy(); tauprev = tau
            u0 = u1; t += tau
            accepted+=1
            h_list.append(tau)
            c_list.append(_its/tau)
        else: # reject step
            rejected+=1
        tau = min(nextstep_P(tau,err_est,tol,order),T-t+1e-14)
        # progress indicator
        print("{:.0f}% {}".format(100.0*t/T,its),end='\r')
    # print('avg error in predictor: ', errest)
    # print('total runtime: ', time()-t0, its)
    return u1,its,accepted+rejected,accepted,rejected,h_list,c_list

def simple_cost_controller(cost,tau,cost_old,tau_old):
    alpha = 0.15
    delta = 1.0 - alpha*sign((cost_old/tau_old-cost/tau)*(tau_old-tau))
    print(cost,tau,cost_old,tau_old)
    if abs(delta-1.0) < 1e-15: # if the cost is zero we still try to increase the step
        delta = 1+alpha
    return delta

# this is almost no difference
def simple_cost_controller_log(cost,tau,cost_old,tau_old):
    delta = exp(-0.15*sign((cost_old/tau_old-cost/tau)*(tau_old-tau)))
    if abs(delta-1.0) < 1e-15: # if the cost is zero we still try to increase the step
        delta = exp(0.15)
    return delta

# gradient descent based cost controller
def gd_cost_controller(cost,tau,cost_old,tau_old,gamma):
    cost_norm=cost/tau; cost_old_norm=cost_old/tau_old
    # alpha=0.4; beta=0.3; gamma=1.15; delta=0.85
    # alpha=1.6; beta=0.3; gamma=1.15; delta=0.85
    # alpha=0.78255584;  beta=0.65890214;  gamma=1.31089235;  delta=0.64190744
    # alpha=0.58531103;  beta=0.52136619;  gamma=1.36816871;  delta=0.80581838;
    # penalized 
    alpha=1.19735982;  beta=0.44611854;  gamma=1.38440318;  delta=0.73715227
    # non-penalized
    # alpha=0.65241444;  beta=0.26862269;  gamma=1.37412002;  delta=0.64446017;
    grad_c = (log(cost_norm)-log(cost_old_norm))/(log(tau)-log(tau_old));
    direc  = abs(grad_c)
    frac = exp(-alpha*tanh(beta*grad_c))
    if frac>=1.0 and frac<gamma:
        frac = gamma
    elif frac<=1.0 and frac>delta:
        frac = delta
    return frac


def integrate_new(A,u00,T,tol,tau0,method,order=1,ctrl=simple_cost_controller,tol_fac=0.1):
    u0=u00.copy()
    t=0.0; tau=tau0; tol_ls=tol_fac*tol;
    its=0; accepted=0; rejected=0; h_list=[]; c_list=[]
    cost_old=0
    while t<T:
        u1,_its  = method(A,u0,tau,tol_ls)
        cost=_its+1 # +1 is for safety
        its+=_its
        u1e,_ = method(A,method(A,u0,0.5*tau,tol_ls)[0],0.5*tau,tol_ls)
        err_est = max(abs(u1-u1e))
        if err_est <= tol: # accept step
            u0 = u1; t += tau
            accepted+=1
            h_list.append(tau)
            c_list.append(_its/tau)
            # after accepted step optimize for cost under a step size constraint
            tau_max = nextstep_P(tau,err_est,tol,order)
            if cost_old != 0:
                delta=ctrl(cost,tau,cost_old,tau_old)
            else:
                delta=1e15
            cost_old = cost; tau_old = tau
            tau = min(min(tau*delta,tau_max),T-t+1e-14)
        else: # reject step
            rejected+=1
            tau = min(nextstep_P(tau,err_est,tol,order),T-t+1e-14); cost=0; cost_old=0
        # progress indicator
        print("{:.0f}% {}".format(100.0*t/T,its),end='\r')
    return u1,its,accepted+rejected,accepted,rejected,h_list,c_list

