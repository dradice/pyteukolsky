"""
TeukolskyRHS: precomputed coefficients and right-hand side for the
Teukolsky mode equation in Kerr-Schild ingoing coordinates.

First-order-in-time reduction: evolves the pair (psi, v) where v = d_t psi.

Time derivatives (README §1.1, check_equations.py MODE / CHECK 3):
    d_t psi = v
    d_t v   = invA * (L[psi] + B * dr(v) + Cv * v)

where
    L[psi] = Delta * drr(psi) + Cr * dr(psi) + angular(psi) - V * psi
    A  = Sigma + 2*M*r,   Sigma = r^2 + a^2*mu^2
    Cv = 4r + 4i*a*mu + 6M
    Cr = 2i*a*m + 6r - 6M
    B  = 4*M*r
    V  = (2*mu - m)^2 / (1 - mu^2) - 2
"""

import numpy as np


class TeukolskyRHS:
    def __init__(self, grid, M, a, m, dissipation=0.0):
        """
        Parameters
        ----------
        grid        : Grid instance
        M           : black-hole mass
        a           : Kerr spin parameter (|a| < M)
        m           : azimuthal mode number (integer)
        dissipation : Kreiss-Oliger epsilon (default 0 = off)
        """
        self.grid = grid
        self.M = M
        self.a = a
        self.m = m
        self.dissipation = dissipation
        self.parity = (-1) ** m

        R  = grid.R.astype(complex)
        MU = grid.MU.astype(complex)

        self.Sigma = R**2 + a**2 * MU**2
        self.Delta = R**2 - 2*M*R + a**2
        self.A     = self.Sigma + 2*M*R
        self.invA  = 1.0 / self.A
        self.Cv    = 4*R + 4j*a*MU + 6*M
        self.Cr    = 2j*a*m + 6*R - 6*M
        self.B     = 4*M*R
        # V = (2*mu - m)^2 / (1 - mu^2) - 2; ghost mu lie outside (-1,1) so
        # denominator is nonzero everywhere (no division by zero)
        self.V = (2*MU - m)**2 / (1 - MU**2) - 2

    def rhs(self, psi, v):
        """Return (dpsi_dt, dv_dt) for state (psi, v).

        Parameters
        ----------
        psi, v : complex128 arrays of shape grid.shape

        Returns
        -------
        (dpsi_dt, dv_dt) : complex128 arrays of shape grid.shape
        """
        g = self.grid

        psi = psi.copy()
        v   = v.copy()

        g.fill_ghosts_r(psi)
        g.fill_ghosts_mu(psi, self.parity)
        g.fill_ghosts_r(v)
        g.fill_ghosts_mu(v, self.parity)

        psi_rr  = g.drr(psi)
        psi_r   = g.dr(psi)
        ang_psi = g.angular(psi)
        v_r     = g.dr(v)

        L = self.Delta * psi_rr + self.Cr * psi_r + ang_psi - self.V * psi

        dpsi_dt = v.copy()
        dv_dt   = self.invA * (L + self.B * v_r + self.Cv * v)

        if self.dissipation > 0:
            eps = self.dissipation
            dpsi_dt += g.ko_dissipation_r(psi, eps) + g.ko_dissipation_mu(psi, eps)
            dv_dt   += g.ko_dissipation_r(v,   eps) + g.ko_dissipation_mu(v,   eps)

        return dpsi_dt, dv_dt
