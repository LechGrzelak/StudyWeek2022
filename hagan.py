#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:53:17 2022

Hagan formula for SABR model
"""

import numpy as np

# Not ATM
def hagan_iv(S,K,T,a,b,g,rho):
    # S: Asset price
    # K: Strike
    # T: Maturity
    # a: Alpha (initial variance)
    # b: Beta (exponent)
    # g: Gamma (volofvol)
    # rho: Rho (correlation)
    
    X0 = np.log(S/K)
    Sb = (S*K)**(0.5*(1-b))
    Xb = ((1-b)*X0/24)**2
    z = g/a*Sb*X0
    x = np.log((np.sqrt(1-2*rho*z+z*z)+z-rho)/(1-rho))
    A = a/(Sb*(1+Xb+Xb**2/80))
    B = (1+(((1-b)*a/Sb)**2/24+0.25*rho*b*g*a/Sb+(2-3*rho**2)/24*g**2))*T
    
    return A*z/x*B

#ATM
def hagan_iv_atm(S,T,a,b,g,rho):
    Sb = (S)**((1-b))
    A = a/(Sb)
    B = (1+(((1-b)*a/Sb)**2/24+0.25*rho*b*g*a/Sb+(2-3*rho**2)/24*g**2))*T
    
    return A*B

