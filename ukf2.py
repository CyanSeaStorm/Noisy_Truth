import numpy as np
from scipy.linalg import cholesky

class UKF:
    def __init__(self, n, m, fx, hx, Q, R,
                 alpha=1e-3, beta=2.0, kappa=0.0):

        self.n = n
        self.m = m
        self.fx = fx
        self.hx = hx

        self.Q = Q
        self.R = R

        self.x = np.zeros(n)
        self.P = np.eye(n) * 1e-3

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = alpha**2 * (n + kappa) - n
        self.gamma = np.sqrt(n + self.lambda_)

        self.Wm = np.full(2*n + 1, 1.0 / (2*(n + self.lambda_)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    # ---------------- utilities ---------------- #

    def _sanitize_P(self):
        if not np.isfinite(self.P).all():
            self.P = np.eye(self.n) * 1e-3
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(self.n) * 1e-9

    def _sigma_points(self):
        self._sanitize_P()
        S = cholesky(self.P, lower=True)

        sigmas = np.zeros((2*self.n + 1, self.n))
        sigmas[0] = self.x

        for i in range(self.n):
            sigmas[i+1]        = self.x + self.gamma * S[:, i]
            sigmas[self.n+i+1] = self.x - self.gamma * S[:, i]

        return sigmas

    # ---------------- prediction ---------------- #

    def predict(self, dt, u=None):
        if dt <= 1e-6:
            return

        dt = np.clip(dt, 1e-3, 0.1)

        sigmas = self._sigma_points()
        sigmas_f = np.array([self.fx(s, dt, u) for s in sigmas])

        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)

        self.P = self.Q.copy()
        for i in range(len(sigmas_f)):
            d = sigmas_f[i] - self.x
            self.P += self.Wc[i] * np.outer(d, d)

    # ---------------- update ---------------- #

    def update(self, z):
        if z is None or not np.isfinite(z).all():
            return

        sigmas = self._sigma_points()
        Z = np.array([self.hx(s) for s in sigmas])

        z_pred = np.sum(self.Wm[:, None] * Z, axis=0)

        S = self.R.copy()
        for i in range(len(Z)):
            dz = Z[i] - z_pred
            S += self.Wc[i] * np.outer(dz, dz)

        Pxz = np.zeros((self.n, self.m))
        for i in range(len(sigmas)):
            dx = sigmas[i] - self.x
            dz = Z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(S)

        self.x += K @ (z - z_pred)
        self.P -= K @ S @ K.T
        self._sanitize_P()

