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
    def __init__(self, grid, M, a, m, dissipation=0.0, one_sided_horizon=False):
        """
        Parameters
        ----------
        grid        : Grid instance
        M           : black-hole mass
        a           : Kerr spin parameter (|a| < M)
        m           : azimuthal mode number (integer)
        dissipation : Kreiss-Oliger epsilon (default 0 = off)
        one_sided_horizon : if True, use a one-sided (outward) FD stencil at
            the first interior column outside r_+, so the exterior domain
            never reads ghost or inside-horizon interior data (real excision;
            see README "Inner (excision)"). Default False (opt-in).
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

        self.one_sided_horizon = one_sided_horizon
        self.r_plus = M + np.sqrt(max(M**2 - a**2, 0.0))

        # Delta is mu-independent (r-only); Delta = (r-r_+)(r-r_-) is also
        # positive for r < r_- (inside the inner Cauchy horizon), so i_exc
        # is taken as the start of the *last* contiguous Delta>0 run (the
        # r > r_+ region), not the first Delta>0 column -- robust even if
        # rmin is set below r_-. Consistent with evolve.py's cfl_dt
        # Delta_eff masking.
        Delta_col = self.Delta[0, grid.ghost:grid.ghost + grid.Nr].real
        non_positive = np.flatnonzero(Delta_col <= 0)
        i0 = int(non_positive[-1]) + 1 if non_positive.size > 0 else 0
        n_good = grid.Nr - i0
        MIN_GOOD_COLUMNS = 4   # drr_onesided needs i..i+3 in-bounds
        if n_good < MIN_GOOD_COLUMNS:
            raise ValueError(
                f"Only {n_good} interior column(s) have r > r_+ = "
                f"{self.r_plus:.4f}M (need >= {MIN_GOOD_COLUMNS}); "
                f"increase rmax/Nr or raise rmin."
            )
        self.i_exc = grid.ghost + i0

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

        if self.one_sided_horizon:
            i = self.i_exc
            psi_r[:, i]  = g.dr_onesided(psi, i)
            psi_rr[:, i] = g.drr_onesided(psi, i)
            v_r[:, i]    = g.dr_onesided(v, i)

        L = self.Delta * psi_rr + self.Cr * psi_r + ang_psi - self.V * psi

        dpsi_dt = v.copy()
        dv_dt   = self.invA * (L + self.B * v_r + self.Cv * v)

        # Sommerfeld outgoing BC at outer radial boundary:
        #   d_t psi = -d_r psi - psi/r  and  d_t v = -d_r v - v/r
        n_out = g.ghost + g.Nr - 1
        r_out = g.r[n_out]
        dpsi_dt[:, n_out] = -(psi_r[:, n_out] + psi[:, n_out] / r_out)
        dv_dt[:, n_out]   = -(v_r[:, n_out]   + v[:, n_out]   / r_out)

        if self.dissipation > 0:
            eps = self.dissipation
            Qp = g.ko_dissipation_r(psi, eps)
            Qv = g.ko_dissipation_r(v,   eps)
            if self.one_sided_horizon:
                # ko_dissipation_r has +/-2 reach; mask the two columns whose
                # stencil would otherwise read inward of the excision boundary.
                i = self.i_exc
                Qp[:, i:i + 2] = 0.0
                Qv[:, i:i + 2] = 0.0
            dpsi_dt += Qp + g.ko_dissipation_mu(psi, eps)
            dv_dt   += Qv + g.ko_dissipation_mu(v,   eps)

        return dpsi_dt, dv_dt
