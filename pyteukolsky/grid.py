"""
Grid: coordinates, finite-difference operators, and ghost-cell fills.

The radial grid is defined by interior cell-centre positions stored in
``self.r``.  By default they are uniformly spaced (``rmin``, ``rmax``,
``Nr`` required); any monotonically increasing sequence may be supplied
instead via the ``r_array`` keyword.  Ghost cells are appended on both
sides using the local boundary spacing.

All FD operators (``dr``, ``drr``) use 2nd-order formulas for general
non-uniform r-spacing; they reduce to the standard uniform-spacing formulas
when the grid happens to be uniform.

Angular coordinate: staggered uniform mu = cos(theta) in (-1, 1),
  mu_j = -1 + (j - 0.5) * dmu, j = 1..Nmu.

Ghost width is 2 cells per side in each direction.
"""

import numpy as np


class Grid:
    def __init__(self, rmin=None, rmax=None, Nmu=None, Nr=None, ghost=2,
                 M=1.0, r_array=None):
        """
        Parameters
        ----------
        rmin, rmax : float, optional
            Physical radial extent.  Required when ``r_array`` is None.
        Nmu        : int
            Number of interior angular cells (required).
        Nr         : int, optional
            Number of interior radial cells.  Required when ``r_array`` is None.
        ghost      : int
            Ghost-cell width (default 2).
        M          : float
            Black-hole mass (used in physics coefficients, not grid spacing).
        r_array    : array_like, optional
            Interior radial cell-centre positions.  When provided, rmin/rmax/Nr
            are inferred from it and the grid may be non-uniform.
            When None (default) a uniform grid is built from rmin/rmax/Nr.
        """
        if Nmu is None:
            raise ValueError("Nmu is required")
        self.ghost = ghost
        self.Nmu   = Nmu
        self.M     = M

        g = ghost

        # --- radial coordinate ---
        if r_array is not None:
            r_int   = np.asarray(r_array, dtype=float)
            self.Nr = len(r_int)
        else:
            if rmin is None or rmax is None or Nr is None:
                raise ValueError("rmin, rmax, Nr are required when r_array is not given")
            self.Nr = Nr
            dr_u    = (rmax - rmin) / Nr
            r_int   = np.linspace(rmin + 0.5 * dr_u, rmax - 0.5 * dr_u, Nr)

        # Ghost cells: extend using local boundary spacing so the FD stencils
        # near the boundary remain well-conditioned.
        dr_lo = r_int[1] - r_int[0]        # spacing at left boundary
        dr_hi = r_int[-1] - r_int[-2]      # spacing at right boundary
        r_lo  = r_int[0]  - np.arange(g, 0, -1) * dr_lo
        r_hi  = r_int[-1] + np.arange(1, g + 1) * dr_hi
        self.r = np.concatenate([r_lo, r_int, r_hi])

        # Interior cell widths (h+ + h-)/2, used by the CFL condition.
        # For a uniform grid this equals the cell spacing dr.
        self.dr_cell = 0.5 * (self.r[g + 1 : g + self.Nr + 1]
                             - self.r[g - 1 : g + self.Nr - 1])  # shape (Nr,)

        # Effective local spacing for the radial KO 5-point stencil:
        # 0.25*(r[i+2] - r[i-2]) → equals dr on a uniform grid.
        self._ko_h_r = 0.25 * (self.r[4:] - self.r[:-4])  # shape (N-4,)

        # --- angular coordinate (staggered, uniform in mu) ---
        self.dmu   = 2.0 / Nmu
        mu_int     = -1.0 + (np.arange(1, Nmu + 1) - 0.5) * self.dmu
        mu_lo      = mu_int[0]  - np.arange(g, 0, -1) * self.dmu
        mu_hi      = mu_int[-1] + np.arange(1, g + 1) * self.dmu
        self._mu   = np.concatenate([mu_lo, mu_int, mu_hi])

        # 2D meshes (angular index along axis 0, radial along axis 1)
        self.MU, self.R = np.meshgrid(self._mu, self.r, indexing='ij')

        # Slice selecting interior (non-ghost) cells
        self.interior = (slice(g, g + Nmu), slice(g, g + self.Nr))

    @property
    def shape(self):
        return (self.Nmu + 2 * self.ghost, self.Nr + 2 * self.ghost)

    # ------------------------------------------------------------------
    # Finite-difference operators — non-uniform r-spacing
    # ------------------------------------------------------------------

    def dr(self, f):
        """First radial derivative d/dr f (2nd-order, non-uniform spacing).

        Centered formula at all internal column positions; 3-point one-sided
        Lagrange formula at the two outermost columns (ghost cells only).
        """
        r = self.r
        h_plus  = r[2:] - r[1:-1]    # r[i+1] - r[i], shape (N-2,)
        h_minus = r[1:-1] - r[:-2]   # r[i]   - r[i-1]

        df = np.empty_like(f)

        df[:, 1:-1] = (
            h_minus**2 * f[:, 2:]
            - (h_plus**2 - h_minus**2) * f[:, 1:-1]
            - h_plus**2 * f[:, :-2]
        ) / (h_plus * h_minus * (h_plus + h_minus))

        # Left edge: one-sided using columns 0, 1, 2
        h1 = r[1] - r[0];  h2 = r[2] - r[0]
        df[:, 0] = (
            -f[:, 0] * (h1 + h2) / (h1 * h2)
            + f[:, 1] * h2 / (h1 * (h2 - h1))
            - f[:, 2] * h1 / (h2 * (h2 - h1))
        )

        # Right edge: one-sided using columns -3, -2, -1
        hm1 = r[-2] - r[-3];  hm2 = r[-1] - r[-3]
        df[:, -1] = (
            f[:, -3] * (hm2 - hm1) / (hm1 * hm2)
            - f[:, -2] * hm2 / (hm1 * (hm2 - hm1))
            + f[:, -1] * (2 * hm2 - hm1) / (hm2 * (hm2 - hm1))
        )

        return df

    def drr(self, f):
        """Second radial derivative d^2/dr^2 f (2nd-order, non-uniform spacing).

        Centered formula at all internal column positions; 3-point Lagrange
        formula at the two outermost columns (ghost cells only).
        """
        r = self.r
        h_plus  = r[2:] - r[1:-1]
        h_minus = r[1:-1] - r[:-2]

        d2f = np.empty_like(f)

        d2f[:, 1:-1] = (
            2.0 * (h_minus * f[:, 2:]
                   - (h_plus + h_minus) * f[:, 1:-1]
                   + h_plus * f[:, :-2])
            / (h_plus * h_minus * (h_plus + h_minus))
        )

        # Left edge: Lagrange 2nd derivative using columns 0, 1, 2
        h1 = r[1] - r[0];  h2 = r[2] - r[0]
        d2f[:, 0] = 2.0 * (
            f[:, 0] / (h1 * h2)
            - f[:, 1] / (h1 * (h2 - h1))
            + f[:, 2] / (h2 * (h2 - h1))
        )

        # Right edge: Lagrange 2nd derivative using columns -3, -2, -1
        hm1 = r[-2] - r[-3];  hm2 = r[-1] - r[-3]
        d2f[:, -1] = 2.0 * (
            f[:, -3] / (hm1 * hm2)
            - f[:, -2] / (hm1 * (hm2 - hm1))
            + f[:, -1] / (hm2 * (hm2 - hm1))
        )

        return d2f

    def angular(self, f):
        """Legendre operator d/dmu[(1-mu^2) d/dmu f], flux form.

        (1-mu^2) is evaluated at cell faces mu_{j+1/2} = mu_j + dmu/2.
        """
        mu  = self._mu
        dmu = self.dmu
        # Face values of (1 - mu^2) between cells j and j+1
        mu_face = mu[:-1] + 0.5 * dmu          # shape (Nmu+2g-1,)
        w_face  = 1.0 - mu_face**2

        # Flux at each face: F_{j+1/2} = (1-mu_{j+1/2}^2) * (f_{j+1} - f_j)/dmu
        F = w_face[:, np.newaxis] * (f[1:, :] - f[:-1, :]) / dmu

        # Divergence: (F_{j+1/2} - F_{j-1/2}) / dmu
        result = np.empty_like(f)
        result[1:-1, :] = (F[1:, :] - F[:-1, :]) / dmu
        # Boundary rows: one-sided (only used inside ghost region, filled separately)
        result[0, :]  = result[1, :]
        result[-1, :] = result[-2, :]
        return result

    # ------------------------------------------------------------------
    # Ghost-cell fills
    # ------------------------------------------------------------------

    def fill_ghosts_r(self, f, outer="sommerfeld", dt=None):
        """Fill radial ghost cells by 2nd-order polynomial extrapolation.

        Inner (excision) and outer ghosts are filled from the nearest 3
        interior cells.  Extrapolation is exact for functions quadratic in
        the cell index (i.e. quadratic in r for a uniform grid).
        """
        g = self.ghost
        n = self.Nr + g   # index of last interior cell

        # Inner ghosts
        f[:, g - 1] = 3 * f[:, g]     - 3 * f[:, g + 1] + f[:, g + 2]
        if g >= 2:
            f[:, g - 2] = 3 * f[:, g - 1] - 3 * f[:, g]     + f[:, g + 1]

        # Outer ghosts
        f[:, n]     = 3 * f[:, n - 1] - 3 * f[:, n - 2] + f[:, n - 3]
        if g >= 2:
            f[:, n + 1] = 3 * f[:, n]     - 3 * f[:, n - 1] + f[:, n - 2]

        return f

    def fill_ghosts_mu(self, f, parity):
        """Fill angular ghost cells by reflection across the poles.

        parity: sign factor for the reflected value (±1, typically (-1)**m).
        South pole (mu = -1): ghost cells at indices 0..g-1 mirror g..2g-1.
        North pole (mu = +1): ghost cells at n..n+g-1 mirror n-g..n-1.
        """
        g = self.ghost
        n = self.Nmu + g   # first northern ghost index

        for k in range(g):
            f[g - 1 - k, :] = parity * f[g + k, :]

        for k in range(g):
            f[n + k, :] = parity * f[n - 1 - k, :]

        return f

    # ------------------------------------------------------------------
    # Kreiss-Oliger dissipation (4th-difference, 2nd-order scheme)
    # ------------------------------------------------------------------

    def ko_dissipation_r(self, f, epsilon):
        """KO dissipation in radial direction.

        Uses the local effective spacing (precomputed at construction) so the
        formula is consistent for non-uniform grids.
        """
        Q = np.zeros_like(f)
        Q[:, 2:-2] = -(epsilon / 16.0) * (
            f[:, :-4] - 4 * f[:, 1:-3] + 6 * f[:, 2:-2]
            - 4 * f[:, 3:-1] + f[:, 4:]
        ) / self._ko_h_r
        return Q

    def ko_dissipation_mu(self, f, epsilon):
        """KO dissipation in angular direction: -eps/16 * fourth difference."""
        Q = np.zeros_like(f)
        Q[2:-2, :] = -(epsilon / 16.0) * (
            f[:-4, :] - 4 * f[1:-3, :] + 6 * f[2:-2, :]
            - 4 * f[3:-1, :] + f[4:, :]
        ) / self.dmu
        return Q
