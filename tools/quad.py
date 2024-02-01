import numpy as np
import scipy as sp

def gauss_legendre(deg):
    # quadpy.c1.gauss_legendre
    return np.polynomial.legendre.leggauss(deg)

def gauss_laguerre(deg):
    # quadpy.e1r.gauss_laguerre
    return np.polynomial.laguerre.laggauss(deg)

def gauss_generalized_laguerre(deg, alpha):
    # quadpy.e1r.gauss_laguerre
    return sp.special.roots_genlaguerre(deg, alpha)