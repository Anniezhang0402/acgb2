"""
Standard GB2 distribution and its link to extreme-value distributions.
"""
from __future__ import annotations
import numpy as np 
from scipy.special import beta as beta_fn
from scipy.special import gamma as gamma_fn

def gb2_pdf(x, a, b, p, q):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    xp = x[pos]
    z = (xp / b) ** a
    out[pos] = (
        np.abs(a) / (beta_fn(p, q) * xp) * z**p / (1.0 + z) ** (p + q)
    )
    return out 

def gb2_frechet_limit_pdf(x, a, beta_scale, p):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    xp = x[pos]
    out[pos] = (
        np.abs(a)
        * xp ** (a * p - 1)
        * np.exp(-((xp / beta_scale) ** a))
        / (beta_scale ** (a * p) * gamma_fn(p))
    )
    return out 

def gb2_rvs(a, b, p, q, size, random_state=None):
    rng = np.random.default_rng(random_state)
    u1 = rng.chisquare(2 * p, size=size)
    u2 = rng.chisquare(2 * q, size=size)
    z = (u1 / u2) ** (1.0 / a)
    return b * z

