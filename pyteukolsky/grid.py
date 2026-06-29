"""
Grid: coordinates, finite-difference operators, and ghost-cell fills.

Radial coordinate: uniform x with the logarithmic map r = M * exp(x), so
  drdx = r, d2rdx2 = r.
This stretches the grid geometrically in r, giving fine resolution near the
horizon and a coarse far zone.
Angular coordinate: staggered uniform mu = cos(theta) in (-1, 1),
  mu_j = -1 + (j - 0.5) * dmu, j = 1..Nmu.

Ghost width is 2 cells per side in each direction.
"""

import numpy as np


class Grid:
    def __init__(self, rmin, rmax, Nmu, Nr, ghost=2, M=1.0):
        """
        Parameters
        ----------
        rmin, rmax : physical radial extent (rmin < r_horizon for excision)
        Nmu        : number of interior angular cells
        Nr         : number of interior radial cells
        ghost      : ghost-cell width (default 2)
        M          : mass parameter for the log map r = M * exp(x)
        """
        self.ghost = ghost
        self.Nr = Nr
        self.Nmu = Nmu
        self.M = M

        # --- radial coordinate (log map r = M exp(x)) ---
        xmin = np.log(rmin / M)
        xmax = np.log(rmax / M)
        self.dx = (xmax - xmin) / Nr

        # x array including ghosts: Nr + 2*ghost cells
        g = ghost
        x_int = np.linspace(xmin + 0.5 * self.dx, xmax - 0.5 * self.dx, Nr)
        x_lo = x_int[0] - np.arange(g, 0, -1) * self.dx
        x_hi = x_int[-1] + np.arange(1, g + 1) * self.dx
        self._x = np.concatenate([x_lo, x_int, x_hi])

        self.r = M * np.exp(self._x)      # shape (Nr + 2g,)
        self.drdx = self.r                 # dr/dx = r for log map
        self.d2rdx2 = self.r               # d^2r/dx^2 = r for log map

        # Physical radial cell width at interior cells, used by the CFL
        # condition: dr_local = drdx * dx = r * dx for the log map.
        self.dr_cell = self.r[g : g + Nr] * self.dx   # shape (Nr,)

        # --- angular coordinate (staggered, uniform in mu) ---
        self.dmu = 2.0 / Nmu
        mu_int = -1.0 + (np.arange(1, Nmu + 1) - 0.5) * self.dmu
        mu_lo = mu_int[0] - np.arange(g, 0, -1) * self.dmu
        mu_hi = mu_int[-1] + np.arange(1, g + 1) * self.dmu
        self._mu = np.concatenate([mu_lo, mu_int, mu_hi])

        # 2D meshes (angular index along axis 0, radial along axis 1)
        self.MU, self.R = np.meshgrid(self._mu, self.r, indexing='ij')

        # Slice selecting interior (non-ghost) cells
        self.interior = (slice(g, g + Nmu), slice(g, g + Nr))

    @property
    def shape(self):
        return (self.Nmu + 2 * self.ghost, self.Nr + 2 * self.ghost)

    # ------------------------------------------------------------------
    # Finite-difference operators
    # ------------------------------------------------------------------

    def dr(self, f):
        """First radial derivative d/dr f, using d/dr = (1/drdx) d/dx."""
        fx = np.empty_like(f)
        # Second-order centered difference in x for interior + 1 ghost band
        fx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * self.dx)
        # One-sided at boundaries (needed for outermost ghost only)
        fx[:, 0] = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2.0 * self.dx)
        fx[:, -1] = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2.0 * self.dx)
        return fx / self.drdx[np.newaxis, :]

    def drr(self, f):
        """Second radial derivative d^2/dr^2 f.

        d^2f/dr^2 = (f_xx - (d2rdx2/drdx) f_x) / drdx^2
        For the log map drdx = d2rdx2 = r, so (d2rdx2/drdx) = 1.
        """
        fxx = np.empty_like(f)
        fxx[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / self.dx**2
        fxx[:, 0] = (2 * f[:, 0] - 5 * f[:, 1] + 4 * f[:, 2] - f[:, 3]) / self.dx**2
        fxx[:, -1] = (2 * f[:, -1] - 5 * f[:, -2] + 4 * f[:, -3] - f[:, -4]) / self.dx**2
        fx = self.dr(f) * self.drdx[np.newaxis, :]   # recover f_x = drdx * f_r
        ratio = self.d2rdx2 / self.drdx               # = 1 for log map
        return (fxx - ratio[np.newaxis, :] * fx) / self.drdx[np.newaxis, :]**2

    def angular(self, f):
        """Legendre operator d/dmu[(1-mu^2) d/dmu f], flux form.

        (1-mu^2) is evaluated at cell faces mu_{j+1/2} = mu_j + dmu/2.
        """
        mu = self._mu
        dmu = self.dmu
        # Face values of (1 - mu^2) between cells j and j+1
        mu_face = mu[:-1] + 0.5 * dmu          # shape (Nmu+2g-1,)
        w_face = 1.0 - mu_face**2

        # Flux at each face: F_{j+1/2} = (1-mu_{j+1/2}^2) * (f_{j+1} - f_j)/dmu
        F = w_face[:, np.newaxis] * (f[1:, :] - f[:-1, :]) / dmu

        # Divergence: (F_{j+1/2} - F_{j-1/2}) / dmu
        result = np.empty_like(f)
        result[1:-1, :] = (F[1:, :] - F[:-1, :]) / dmu
        # Boundary rows: one-sided (only needed inside ghost region, filled separately)
        result[0, :] = result[1, :]
        result[-1, :] = result[-2, :]
        return result

    # ------------------------------------------------------------------
    # Ghost-cell fills
    # ------------------------------------------------------------------

    def fill_ghosts_r(self, f, outer="sommerfeld", dt=None):
        """Fill radial ghost cells.

        Inner (excision): 2nd-order extrapolation from interior.
        Outer (Sommerfeld): d_t psi ~ -d_r psi - psi/r (approximate outgoing).
          When outer='extrapolate' or dt is None, uses 2nd-order extrapolation.
        """
        g = self.ghost

        # Inner ghosts: extrapolate from interior
        # ghost at g-1: use interior[g], g+1, g+2
        f[:, g - 1] = 3 * f[:, g] - 3 * f[:, g + 1] + f[:, g + 2]
        if g >= 2:
            f[:, g - 2] = 3 * f[:, g - 1] - 3 * f[:, g] + f[:, g + 1]

        # Outer ghosts: extrapolate (Sommerfeld applied in RHS, not here)
        n = self.Nr + g   # index of last interior cell
        f[:, n] = 3 * f[:, n - 1] - 3 * f[:, n - 2] + f[:, n - 3]
        if g >= 2:
            f[:, n + 1] = 3 * f[:, n] - 3 * f[:, n - 1] + f[:, n - 2]

        return f

    def fill_ghosts_mu(self, f, parity):
        """Fill angular ghost cells by reflection across the poles.

        parity: sign factor for the reflected value (±1, typically (-1)**m).
        South pole (mu = -1): ghost cells at indices 0..g-1 mirror g..2g-1.
        North pole (mu = +1): ghost cells at n..n+g-1 mirror n-g..n-1.
        """
        g = self.ghost
        n = self.Nmu + g   # first northern ghost index

        # South-pole ghosts: mu < -1
        for k in range(g):
            f[g - 1 - k, :] = parity * f[g + k, :]

        # North-pole ghosts: mu > +1
        for k in range(g):
            f[n + k, :] = parity * f[n - 1 - k, :]

        return f

    # ------------------------------------------------------------------
    # Kreiss-Oliger dissipation (4th-difference, 2nd-order scheme)
    # ------------------------------------------------------------------

    def ko_dissipation_r(self, f, epsilon):
        """KO dissipation in radial direction: -eps/16 * fourth difference."""
        Q = np.zeros_like(f)
        Q[:, 2:-2] = -(epsilon / 16.0) * (
            f[:, :-4] - 4 * f[:, 1:-3] + 6 * f[:, 2:-2]
            - 4 * f[:, 3:-1] + f[:, 4:]
        ) / self.dx
        return Q

    def ko_dissipation_mu(self, f, epsilon):
        """KO dissipation in angular direction: -eps/16 * fourth difference."""
        Q = np.zeros_like(f)
        Q[2:-2, :] = -(epsilon / 16.0) * (
            f[:-4, :] - 4 * f[1:-3, :] + 6 * f[2:-2, :]
            - 4 * f[3:-1, :] + f[4:, :]
        ) / self.dmu
        return Q
