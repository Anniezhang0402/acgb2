from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import betaln
from .gb2 import gb2_rvs

def _stationary_log_init(c0, c1, c2, sign):
    return (c0 + sign * c2 / 2.0) / (1.0 - c1)

@dataclass 
class AcGB2Model:
    mu: float 

    a0: float
    a1: float
    a2: float
    a3: float

    b0: float
    b1: float
    b2: float
    b3: float

    p: float
    q: float

    @classmethod
    def from_vector(cls, theta):
        return cls(*[float(v) for v in theta])

    def to_vector(self):
        return np.array(
            [
                self.mu,
                self.a0,
                self.a1,
                self.a2,
                self.a3,
                self.b0,
                self.b1,
                self.b2,
                self.b3,
                self.p,
                self.q,
            ],
            dtype=float,
        )

    def recover_states(self, M):
        M = np.asarray(M, dtype=float)
        n = M.shape[0]
        log_a = np.empty(n)
        log_b = np.empty(n)
        log_a[0] = _stationary_log_init(self.a0, self.a1, self.a2, +1.0)
        log_b[0] = _stationary_log_init(self.b0, self.b1, self.b2, -1.0)

        for t in range(1, n):
            log_a[t] = (
                self.a0
                + self.a1 * log_a[t - 1]
                + self.a2 * np.exp(-self.a3 * M[t - 1])
            )
            log_b[t] = (
                self.b0
                + self.b1 * log_b[t - 1]
                - self.b2 * np.exp(-self.b3 * M[t - 1])
            )
        return np.exp(log_a), np.exp(log_b)

    def loglik(self, M, burnin=1):
        M = np.asarray(M, dtype=float)
        n = M.shape[0]
        A, B = self.recover_states(M)
        sl = slice(burnin - 1 if burnin >= 1 else 0, n)
        m = M[sl] - self.mu

        if np.any(m <= 0):
            return -np.inf
        A_, B_, p, q = A[sl], B[sl], self.p, self.q

        with np.errstate(over="ignore", invalid="ignore"):
            terms = (
                -betaln(p, q)
                + np.log(A_)
                - A_ * p * np.log(B_)
                + (A_ * p - 1.0) * np.log(m)
                - (p + q) * np.log1p((m / B_) ** A_)
            )
            val = np.mean(terms)
        return val if np.isfinite(val) else -np.inf

    def simulate(self, n, burnin=10000, random_state=None):
        rng = np.random.default_rng(random_state)
        total = n + burnin
        y = gb2_rvs(1.0, 1.0, self.p, self.q, size=total, random_state=rng)
        log_a = np.empty(total)
        log_b = np.empty(total)
        M = np.empty(total)
        log_a[0] = _stationary_log_init(self.a0, self.a1, self.a2, +1.0)
        log_b[0] = _stationary_log_init(self.b0, self.b1, self.b2, -1.0)
        M[0] = self.mu + np.exp(log_b[0]) * y[0] ** (1.0 / np.exp(log_a[0]))
        for t in range(1, total):
            log_a[t] = (
                self.a0
                + self.a1 * log_a[t - 1]
                + self.a2 * np.exp(-self.a3 * M[t - 1])
            )

            log_b[t] = (
                self.b0
                + self.b1 * log_b[t - 1]
                - self.b2 * np.exp(-self.b3 * M[t - 1])
            )
            A_t, B_t = np.exp(log_a[t]), np.exp(log_b[t])
            M[t] = self.mu + B_t * y[t] ** (1.0 / A_t)
        return M[burnin:], np.exp(log_a[burnin:]), np.exp(log_b[burnin:])

@dataclass
class AcFModel:
    mu: float

    g0: float
    g1: float
    g2: float
    g3: float

    b0: float
    b1: float
    b2: float
    b3: float

    @classmethod
    def from_vector(cls, theta):
        return cls(*[float(v) for v in theta])
    def to_vector(self):
        return np.array(
            [
                self.mu,
                self.g0,
                self.g1,
                self.g2,
                self.g3,
                self.b0,
                self.b1,
                self.b2,
                self.b3,
            ],
            dtype=float,
        )

    def recover_states(self, Q):
        Q = np.asarray(Q, dtype=float)
        n = Q.shape[0]
        log_al = np.empty(n)
        log_si = np.empty(n)
        log_al[0] = _stationary_log_init(self.g0, self.g1, self.g2, +1.0)
        log_si[0] = _stationary_log_init(self.b0, self.b1, self.b2, -1.0)
        for t in range(1, n):
            log_al[t] = (
                self.g0
                + self.g1 * log_al[t - 1]
                + self.g2 * np.exp(-self.g3 * Q[t - 1])
            )
            log_si[t] = (
                self.b0
                + self.b1 * log_si[t - 1]
                - self.b2 * np.exp(-self.b3 * Q[t - 1])
            )
        return np.exp(log_al), np.exp(log_si)

    def loglik(self, Q, burnin=1):
        Q = np.asarray(Q, dtype=float)
        n = Q.shape[0]
        alpha, sigma = self.recover_states(Q)
        sl = slice(burnin - 1 if burnin >= 1 else 0, n)
        z = Q[sl] - self.mu
        if np.any(z <= 0):
            return -np.inf
        al, si = alpha[sl], sigma[sl]
        terms = (
            np.log(al)
            + al * np.log(si)
            - (al + 1.0) * np.log(z)
            - (si / z) ** al
        )
        val = np.mean(terms)
        return val if np.isfinite(val) else -np.inf

    def simulate(self, n, burnin=10000, random_state=None):
        rng = np.random.default_rng(random_state)
        total = n + burnin
        y = 1.0 / rng.exponential(1.0, size=total)  # unit-Frechet
        log_al = np.empty(total)
        log_si = np.empty(total)
        Q = np.empty(total)
        log_al[0] = _stationary_log_init(self.g0, self.g1, self.g2, +1.0)
        log_si[0] = _stationary_log_init(self.b0, self.b1, self.b2, -1.0)
        Q[0] = self.mu + np.exp(log_si[0]) * y[0] ** (1.0 / np.exp(log_al[0]))
        for t in range(1, total):
            log_al[t] = (self.g0 + self.g1 * log_al[t - 1]
                         + self.g2 * np.exp(-self.g3 * Q[t - 1]))
            log_si[t] = (self.b0 + self.b1 * log_si[t - 1]
                         - self.b2 * np.exp(-self.b3 * Q[t - 1]))
            al_t, si_t = np.exp(log_al[t]), np.exp(log_si[t])
            Q[t] = self.mu + si_t * y[t] ** (1.0 / al_t)
        return Q[burnin:], np.exp(log_al[burnin:]), np.exp(log_si[burnin:])









