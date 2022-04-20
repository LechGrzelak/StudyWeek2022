import numpy as np
import matplotlib.pyplot as plt
import py_vollib.black_scholes.implied_volatility as bs


def call_price(S,S0,K,T,r):
    C = np.maximum(S-K,0).mean()
    # return C
    return bs.implied_volatility(C,S0,K,T,r,'c')

def SABR_S(S0: float,v0: float,beta: float,dW: float):
    # print(S0)
    return np.maximum(S0+gamma*v0*np.power(S0,beta)*dW,0)

def SABR_v(v0: float,gamma: float,dW: float):
    return np.maximum(v0*(1+gamma*dW),0)

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
        # print(S)
    S -= S.mean() - S0

    return S 


S0 = 1
K  = S0
r  = 0.1
alpha = 0.1
beta = 1
gamma = 0.5
rho = 0
T = 1
n_p = 1000
n_t = 100

S = SABR(S0,alpha,beta,gamma,rho,T,n_p,n_t)
print(np.mean(S))
print(call_price(S,S0,K,T,r))
