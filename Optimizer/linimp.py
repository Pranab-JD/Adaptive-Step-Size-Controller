from numpy import *
from scipy.sparse import issparse, eye
from scipy.sparse.linalg import cg, gmres, bicgstab

#
# utility
#

class counter:
    def __init__(self):
        self.cnt=0
    def incr(self,x):
        self.cnt+=1

def GMRES(A,b,x0,tol):
    c=counter()
    return gmres(A,b,x0=x0,callback=c.incr,tol=tol)[0],c.cnt

#
# numerical methods
#

def euler(A,u0,tau,eps):
    n=len(u0)
    return GMRES(identity(n)-tau*A,u0,u0,eps)

def euler_est(A,u0,tau,eps):
    n=len(u0)
    u_est1 = u0 + tau*A.dot(u0)
    return GMRES(identity(n)-tau*A,u0,u_est1,eps)

def sdirk22(A,u0,tau,eps,u_est=None):
    gamma = 1.0-1.0/sqrt(2.0)
    n=len(u0)
    k1,c1 = GMRES(identity(n)-gamma*tau*A,tau*A.dot(u0),zeros(n),eps)
    k2,c2 = GMRES(identity(n)-gamma*tau*A,tau*A.dot(u0)+(1-2*gamma)*tau*A.dot(k1),zeros(n),eps);
    return (u0 + 0.5*k1 + 0.5*k2),c1+c2
    # return k1,c1

def sdirk22_est(A,u0,tau,eps):
    gamma = 1.0-1.0/sqrt(2.0)
    n=len(u0)
    k1,c1 = GMRES(eye(n)-gamma*tau*A,tau*A.dot(u0),tau*A.dot(u0),eps)
    k2,c2 = GMRES(eye(n)-gamma*tau*A,tau*A.dot(u0)+(1-2*gamma)*tau*A.dot(k1),tau*A.dot(u0),eps);
    return (u0 + 0.5*k1 + 0.5*k2),c1+c2
    # return k1,c1

def sdirk23(A,u0,tau,eps,u_est=None):
    gamma = (3.0+sqrt(3.0))/6.0
    n=len(u0)
    k1,c1 = GMRES(eye(n)-gamma*tau*A,tau*A.dot(u0),zeros(n),eps)
    k2,c2 = GMRES(eye(n)-gamma*tau*A,tau*A.dot(u0)+(1-2*gamma)*tau*A.dot(k1),zeros(n),eps);
    return (u0 + 0.5*k1 + 0.5*k2),c1+c2

# page 100 in the second book of Hairer and Wanner
def sdirk45(A,u0,tau,eps,u_est=None):
    n=len(u0)
    k1,c1 = GMRES(identity(n)-0.25*tau*A,tau*A.dot(u0),zeros(n),eps)
    k2,c2 = GMRES(identity(n)-0.25*tau*A,tau*A.dot(u0)+0.5*tau*A.dot(k1),zeros(n),eps);
    k3,c3 = GMRES(identity(n)-0.25*tau*A,tau*A.dot(u0)+17.0/50.0*tau*A.dot(k1)-1.0/25.0*tau*A.dot(k2),zeros(n),eps);
    k4,c4 = GMRES(identity(n)-0.25*tau*A,tau*A.dot(u0)+371.0/1360.0*tau*A.dot(k1)-137.0/2720.0*tau*A.dot(k2)+15.0/544.0*tau*A.dot(k3),zeros(n),eps);
    k5,c5 = GMRES(identity(n)-0.25*tau*A,tau*A.dot(u0)+25.0/24.0*tau*A.dot(k1)-49.0/48.0*tau*A.dot(k2)+125.0/16.0*tau*A.dot(k3)-85.0/12.0*tau*A.dot(k4),zeros(n),eps);
    return (u0 + 25.0/24.0*k1 - 49.0/48.0*k2 + 125.0/16.0*k3 - 85.0/12.0*k4 + 0.25*k5),c1+c2+c3+c4+c5

def RK4(A,u0,tau,eps,u_est=None):
    k1=A.dot(u0)
    k2=A.dot(u0+0.5*tau*k1)
    k3=A.dot(u0+0.5*tau*k2)
    k4=A.dot(u0+tau*k3)
    return u0+tau/6.0*(k1+2*k2+2*k3+k4),4

def ssprk33(A,u0,tau,eps,u_est=None):
    k1 = A.dot(u0)
    inp = u0+tau*k1
    k2 = A.dot(inp)
    inp = 0.75*u0 + 0.25*inp + 0.25*tau*k2
    k3 = A.dot(inp);
    return 1./3.*u0 + 2./3.*inp + 2./3.*tau*k3,3;

def cn(A,u0,tau,eps,u_est=None):
    n=len(u0)
    return GMRES(identity(n)-0.5*tau*A,u0+0.5*tau*A.dot(u0),u0,eps)

def cn_est(A,u0,tau,eps,u_est=None):
    n=len(u0)
    # u_est1 = u0 + tau*A.dot(u0) + 0.5*tau**2*A.dot(A.dot(u0))
    # a,b = GMRES(identity(n)-0.5*tau*A,u0+0.5*tau*A.dot(u0),u_est,eps)
    # u_est2 = linalg.inv(identity(n)-tau*A).dot(u0)
    # if u_est == None:
        # u_est = u_est2
    # a,b = GMRES(identity(n)-0.5*tau*A,u0+0.5*tau*A.dot(u0),u_est,eps)
    # print(max(abs(a-u0)), max(abs(a-u_est1)), max(abs(a-u_est2)))
    if u_est is None:
        u_est = u0.copy()
    return GMRES(identity(n)-0.5*tau*A,u0+0.5*tau*A.dot(u0),u_est,eps)


def radau3(A,u0,tau,eps,u_est=None):
    n=len(u0)
    # construct matrix
    Aradau=bmat([[identity(n)-5.0/12.0*tau*A,1.0/12.0*tau*A],[-3.0/4.0*tau*A,(identity(n)-1.0/4.0*tau*A)]]);
    k0=array([tau*A.dot(u0),tau*A.dot(u0)]).flatten()
    # solve system
    c=counter()
    k,its=GMRES(Aradau,k0,array([u0,u0]).flatten(),eps)
    # get result
    k1=k[0:n]; k2=k[n:(2*n)]
    return (u0 + 3.0/4.0*k1 + 1.0/4.0*k2).transpose(),its

# radau3 with an explicit estimate that is used in the iterative scheme
def radau3_est(A,u0,tau,eps):
    n=len(u0)
    # construct matrix
    Aradau=bmat([[identity(n)-5.0/12.0*tau*A,1.0/12.0*tau*A],[-3.0/4.0*tau*A,(identity(n)-1.0/4.0*tau*A)]]);
    k0=array([tau*A.dot(u0),tau*A.dot(u0)]).flatten()
    # estimate the solution by an explicit method
    u_est_k2 = u0 + tau*A.dot(u0)# + 0.5*tau**2*A.dot(A.dot(u0))
    # solve system
    c=counter()
    k,its=GMRES(Aradau,k0,array([u0,u_est_k2]).flatten(),eps)
    # get result
    k1=k[0:n]; k2=k[n:(2*n)]
    return (u0 + 3.0/4.0*k1 + 1.0/4.0*k2).transpose(),its
