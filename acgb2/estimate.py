from __future__ import annotations
from dataclasses import dataclass 
import numpy as np 
from scipy.optimize import minimize
from .models import AcFModel, AcGB2Model

ACGB2_DEFAULT_LB = np.array([
    -3.0,  # a0 下界
    0.55,  # a1 下界
    8.0,   # a2 下界
    1.0,   # a3 下界
    -2.0,  # b0 下界
    0.05,  # b1 下界
    -8.0,  # b2 下界
    2.0,   # b3 下界
    0.5,   # mu 下界
    0.5,   # p 下界
    0.05,  # q 下界
])

ACGB2_DEFAULT_UB = np.array([
    -0.1,  # a0 上界
    0.95,  # a1 上界
    18.0,  # a2 上界
    3.0,   # a3 上界
    -0.8,  # b0 上界
    0.70,  # b1 上界
    -3.0,  # b2 上界
    6.5,   # b3 上界
    0.95,  # mu 上界
    0.98,  # p 上界
    0.5,   # q 上界
])

@dataclass 
class FitResult:
    model: object 
    theta: np.ndarray 
    loglik: float 
    success: bool 
    n_starts: int 
    n_success: int 

    def __repr__(self):
        return (
            f"FitResult(loglik={self.loglik:.4f}, success={self.success}, "
            f"starts={self.n_success}/{self.n_starts})"
        )

def _reorder_r_box_to_package(box_r):
    a0, a1, a2, a3, b0, b1, b2, b3, mu, p, q = box_r
    return np.array([mu, a0, a1, a2, a3, b0, b1, b2, b3, p, q])

def fit_acgb2(
    M,
    lb=None,
    ub=None,
    n_starts=50,
    burnin=100,
    random_state=None,
    verbose=False,
):
    M = np.asarray(M, dtype=float)
    if lb is None:
        lb = _reorder_r_box_to_package(ACGB2_DEFAULT_LB)
    if ub is None:
        ub = _reorder_r_box_to_package(ACGB2_DEFAULT_UB)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    rng = np.random.default_rng(random_state)

    def neg_ll(theta):
        val = AcGB2Model.from_vector(theta).loglik(M, burnin=burnin)
        return -val if np.isfinite(val) else 1e10
    return _multistart(neg_ll, lb, ub, n_starts, rng, AcGB2Model, verbose)

ACF_DEFAULT_LB = np.array([
    0.7,   # mu 下界
    -0.5,  # g0 下界
    0.60,  # g1 下界
    2.0,   # g2 下界
    1.0,   # g3 下界
    -6.0,  # b0 下界
    0.10,  # b1 下界
    3.0,   # b2 下界
    1.0,   # b3 下界
])

ACF_DEFAULT_UB = np.array([
    0.95,  # mu 上界
    0.5,   # g0 上界
    0.95,  # g1 上界
    10.0,  # g2 上界
    4.0,   # g3 上界
    -4.0,  # b0 上界
    0.60,  # b1 上界
    6.0,   # b2 上界
    4.0,   # b3 上界
])

def fit_acf(
    Q,
    lb=None,
    ub=None,
    n_starts=50,
    burnin=100,
    random_state=None,
    verbose=False,
):
    Q = np.asarray(Q, dtype=float)
    if lb is None:
        lb = ACF_DEFAULT_LB
    if ub is None:
        ub = ACF_DEFAULT_UB
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    rng = np.random.default_rng(random_state)

    def neg_ll(theta):
        val = AcFModel.from_vector(theta).loglik(Q, burnin=burnin)
        return -val if np.isfinite(val) else 1e10
    return _multistart(neg_ll, lb, ub, n_starts, rng, AcFModel, verbose)

def _multistart(neg_ll, lb, ub, n_starts, rng, model_cls, verbose):
    bounds = list(zip(lb, ub))
    best = None
    n_success = 0

    for s in range(n_starts):
        w = rng.uniform(size=lb.shape)
        x0 = w * lb + (1 - w) * ub
        res = minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-10},
        )

        if res.success and np.isfinite(res.fun) and res.fun < 1e9:
            n_success += 1
            if best is None or res.fun < best.fun:
                best = res
        if verbose and (s + 1) % 10 == 0:
            cur = -best.fun if best is not None else float("nan")
            print(f"  start {s+1}/{n_starts}  best loglik={cur:.4f}")

    if best is None:
        raise RuntimeError("All optimisation starts failed; check bounds/data.")

    return FitResult(
        model=model_cls.from_vector(best.x),  
        theta=best.x,                         
        loglik=-best.fun,                  
        success=True,                        
        n_starts=n_starts,                  
        n_success=n_success,                
    )




    