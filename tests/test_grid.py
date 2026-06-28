"""
Milestone 1 tests: Grid finite-difference operators and ghost fills.

All FD operators are expected to converge at 2nd order.
Convergence is measured by halving the grid spacing and checking
that the L-infinity error ratio is ≥ 3.5 (ideal: 4).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyteukolsky.grid import Grid


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_grid(Nr, Nmu, rmin=2.0, rmax=100.0, M=1.0):
    return Grid(rmin=rmin, rmax=rmax, Nmu=Nmu, Nr=Nr, ghost=2, M=M)


def interior_error(grid, computed, exact):
    sl = grid.interior
    return np.max(np.abs(computed[sl] - exact[sl]))


# -----------------------------------------------------------------------
# Test dr: first radial derivative
# -----------------------------------------------------------------------

def test_dr_convergence():
    """dr(f) should be 2nd-order accurate for a smooth f(r)."""
    errors = []
    for Nr in (50, 100, 200):
        g = make_grid(Nr, 32)
        R = g.R
        f = np.sin(R / 10.0)
        exact = np.cos(R / 10.0) / 10.0
        err = interior_error(g, g.dr(f), exact)
        errors.append(err)

    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]
    assert ratio1 > 3.5, f"dr convergence ratio (50->100) = {ratio1:.2f}, expected > 3.5"
    assert ratio2 > 3.5, f"dr convergence ratio (100->200) = {ratio2:.2f}, expected > 3.5"


# -----------------------------------------------------------------------
# Test drr: second radial derivative
# -----------------------------------------------------------------------

def test_drr_convergence():
    """drr(f) should be 2nd-order accurate."""
    errors = []
    for Nr in (50, 100, 200):
        g = make_grid(Nr, 32)
        R = g.R
        f = np.sin(R / 10.0)
        exact = -np.sin(R / 10.0) / 100.0
        err = interior_error(g, g.drr(f), exact)
        errors.append(err)

    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]
    assert ratio1 > 3.5, f"drr convergence ratio (50->100) = {ratio1:.2f}, expected > 3.5"
    assert ratio2 > 3.5, f"drr convergence ratio (100->200) = {ratio2:.2f}, expected > 3.5"


# -----------------------------------------------------------------------
# Test angular: Legendre operator d/dmu[(1-mu^2) d/dmu f]
# -----------------------------------------------------------------------

def test_angular_convergence():
    """angular(f) should be 2nd-order for a smooth f(mu) away from poles."""
    errors = []
    for Nmu in (32, 64, 128):
        g = make_grid(50, Nmu)
        MU = g.MU
        # f = mu^2 - 1/3  =>  d/dmu[(1-mu^2)*2mu] = 2 - 6mu^2
        f = MU**2 - 1.0 / 3.0
        exact = 2.0 - 6.0 * MU**2
        err = interior_error(g, g.angular(f), exact)
        errors.append(err)

    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]
    assert ratio1 > 3.5, f"angular convergence ratio (32->64) = {ratio1:.2f}, expected > 3.5"
    assert ratio2 > 3.5, f"angular convergence ratio (64->128) = {ratio2:.2f}, expected > 3.5"


def test_angular_legendre_eigenfunction():
    """angular(P2) = -6 P2 where P2(mu) = (3mu^2-1)/2 (l=2 eigenvalue -l(l+1)=-6).

    The flux-form stencil has an inherent O(dmu^2) truncation error even for
    polynomials; for Nmu=128 (dmu=1/64) this is ~2e-4.  We verify convergence
    instead of demanding near-machine precision.
    """
    errors = []
    for Nmu in (64, 128, 256):
        g = make_grid(50, Nmu)
        MU = g.MU
        P2 = (3 * MU**2 - 1) / 2.0
        result = g.angular(P2)
        sl = g.interior
        err = np.max(np.abs(result[sl] + 6.0 * P2[sl]))
        errors.append(err)

    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]
    assert ratio1 > 3.5, f"eigenfunction convergence (64->128) = {ratio1:.2f}"
    assert ratio2 > 3.5, f"eigenfunction convergence (128->256) = {ratio2:.2f}"


# -----------------------------------------------------------------------
# Test fill_ghosts_mu: pole reflection
# -----------------------------------------------------------------------

def test_fill_ghosts_mu_even_parity():
    """Even parity (p=+1): ghost cells equal mirror interior cells."""
    g = make_grid(50, 32)
    f = np.ones(g.shape, dtype=float)
    f[:, :] = g.MU  # antisymmetric in mu; use a constant to test parity
    # Fill with a known pattern
    f[:] = 1.0
    g.fill_ghosts_mu(f, parity=1)
    gh = g.ghost
    # South ghost 0 should mirror interior index gh+0 with parity +1
    assert np.allclose(f[gh - 1, :], f[gh, :])
    assert np.allclose(f[gh - 2, :], f[gh + 1, :])
    n = g.Nmu + gh
    assert np.allclose(f[n, :], f[n - 1, :])
    assert np.allclose(f[n + 1, :], f[n - 2, :])


def test_fill_ghosts_mu_odd_parity():
    """Odd parity (p=-1): ghost cells are the negative of mirror interior cells."""
    g = make_grid(50, 32)
    f = np.random.default_rng(42).random(g.shape)
    # Save interior values before fill
    gh = g.ghost
    n = g.Nmu + gh
    interior_south = f[gh:gh + 2, :].copy()
    interior_north = f[n - 2:n, :].copy()

    g.fill_ghosts_mu(f, parity=-1)

    assert np.allclose(f[gh - 1, :], -interior_south[0, :])
    assert np.allclose(f[gh - 2, :], -interior_south[1, :])
    assert np.allclose(f[n, :], -interior_north[1, :])
    assert np.allclose(f[n + 1, :], -interior_north[0, :])


# -----------------------------------------------------------------------
# Test fill_ghosts_r: inner extrapolation
# -----------------------------------------------------------------------

def test_fill_ghosts_r_inner_extrapolation():
    """Inner ghost cells should be filled by 2nd-order extrapolation.

    The stencil extrapolates quadratically in the cell index, which is exact
    for polynomials quadratic in r on a uniform grid.
    """
    g = make_grid(50, 32)
    gh = g.ghost
    f = g.R**2          # quadratic in r → extrapolation exact for uniform grid
    expected = f.copy()
    f[:, :gh] = 0.0
    g.fill_ghosts_r(f)
    err = np.max(np.abs(f[:, :gh] - expected[:, :gh]))
    assert err < 1e-10, f"Inner ghost extrapolation error = {err:.2e}"


# -----------------------------------------------------------------------
# Test grid shape and coordinate properties
# -----------------------------------------------------------------------

def test_grid_shape():
    Nr, Nmu, g = 40, 24, 2
    grid = make_grid(Nr, Nmu)
    assert grid.shape == (Nmu + 2 * g, Nr + 2 * g)


def test_staggered_mu_in_interior():
    """Interior mu values should be strictly inside (-1, 1)."""
    g = make_grid(50, 32)
    sl = g.interior
    mu_int = g.MU[sl[0], 0]
    assert np.all(mu_int > -1.0) and np.all(mu_int < 1.0)


def test_uniform_grid_r():
    """Default grid is uniform: interior r values span [rmin, rmax] uniformly."""
    Nr, Nmu = 50, 32
    rmin, rmax = 2.0, 100.0
    g = make_grid(Nr, Nmu, rmin=rmin, rmax=rmax)
    r_int = g.r[g.ghost : g.ghost + g.Nr]
    dr = (rmax - rmin) / Nr
    expected = np.linspace(rmin + 0.5 * dr, rmax - 0.5 * dr, Nr)
    assert np.allclose(r_int, expected)


def test_custom_r_array():
    """A user-supplied r_array is stored as the interior r values."""
    r_custom = np.geomspace(2.0, 100.0, 40)   # log-spaced (non-uniform)
    g = Grid(Nmu=16, ghost=2, M=1.0, r_array=r_custom)
    r_int = g.r[g.ghost : g.ghost + g.Nr]
    assert np.allclose(r_int, r_custom)
    assert g.Nr == len(r_custom)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
