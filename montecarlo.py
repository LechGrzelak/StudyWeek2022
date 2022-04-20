import numpy as np
import matplotlib.pyplot as plt


def SABR_S(S0: float,v0: float,beta: float,dW: float):
    return S0+gamma*v0*np.power(S0,beta)*dW

def SABR_v(v0: float,gamma: float,dW: float):
    return v0+gamma*v0*dW

def SABR_step(S0: float, v0: float, gamma: float, beta: float,rho: float, dt: float):
    dW1 = np.random.standard_normal(len(S0))*np.sqrt(dt)
    dW2 = rho*dW1+np.sqrt(1-rho*rho)*np.random.standard_normal(len(S0))*np.sqrt(dt)
    S_t = SABR_S(S0,v0,beta,dW1)
    v_t = SABR_v(v0,gamma,dW2)
    return (v_t,S_t)

def SABR(S0: float,alpha: float,beta: float,gamma: float,rho: float,T: float,n_p: int,n_t: int):
    dt = 1./float(n_t)
    S = np.ones(n_p)*S0
    v = np.ones(n_p)*alpha
    for i in range(n_t):
        S,v = SABR_step(S,v,gamma,beta,rho,dt)
        print(S)

    return S 


S0 = 1
alpha = 1
beta = 1
gamma = 1
rho = 0
T = 1
n_p = 1000
n_t = 100

S = SABR(S0,alpha,beta,gamma,rho,T,n_p,n_t)
print(np.mean(S))
