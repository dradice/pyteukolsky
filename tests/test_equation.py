"""
Milestone 2 tests: TeukolskyRHS coefficient arrays and rhs() method.

Coefficient arrays are cross-checked against the analytical expressions from
scripts/check_equations.py at a sample interior grid point.  The rhs()
method is checked for shape, dtype, linearity, zero-field behaviour, and
non-modification of inputs.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyteukolsky.grid import Grid
from pyteukolsky.equation import TeukolskyRHS


M0, A0, M0_VAL = 1.0, 0.5, 2   # default (M, a, m) for most tests


def make_rhs(M=M0, a=A0, m=M0_VAL, Nr=40, Nmu=32, dissipation=0.0):
    g = Grid(rmin=2.0, rmax=50.0, Nmu=Nmu, Nr=Nr, ghost=2, M=M)
    return TeukolskyRHS(g, M, a, m, dissipation=dissipation)


def sample_point(rhs):
    """Return (mu, r, i_mu, i_r) near the centre of the interior."""
    g   = rhs.grid
    gh  = g.ghost
    i_mu = gh + g.Nmu // 2
    i_r  = gh + g.Nr  // 2
    return g.MU[i_mu, i_r], g.R[i_mu, i_r], i_mu, i_r


# -----------------------------------------------------------------------
# Coefficient cross-checks (vs check_equations.py symbols at a sample point)
# -----------------------------------------------------------------------

def test_Sigma():
    rhs = make_rhs()
    mu, r, i, j = sample_point(rhs)
    assert np.isclose(rhs.Sigma[i, j], r**2 + rhs.a**2 * mu**2)


def test_Delta():
    rhs = make_rhs()
    mu, r, i, j = sample_point(rhs)
    assert np.isclose(rhs.Delta[i, j], r**2 - 2*rhs.M*r + rhs.a**2)


def test_A():
    rhs = make_rhs()
    mu, r, i, j = sample_point(rhs)
    expected = r**2 + rhs.a**2 * mu**2 + 2*rhs.M*r
    assert np.isclose(rhs.A[i, j], expected)


def test_invA():
    rhs = make_rhs()
    mu, r, i, j = sample_point(rhs)
    assert np.isclose(rhs.invA[i, j] * rhs.A[i, j], 1.0)


def test_Cv():
    rhs = make_rhs()
    mu, r, i, j = sample_point(rhs)
    expected = 4*r + 4j*rhs.a*mu + 6*rhs.M
    assert np.isclose(rhs.Cv[i, j], expected)


def test_Cr():
    rhs = make_rhs()
    mu, r, i, j = sample_point(rhs)
    expected = 2j*rhs.a*rhs.m + 6*r - 6*rhs.M
    assert np.isclose(rhs.Cr[i, j], expected)


def test_B():
    rhs = make_rhs()
    mu, r, i, j = sample_point(rhs)
    assert np.isclose(rhs.B[i, j], 4*rhs.M*r)


def test_V():
    """V = (2*mu - m)^2/(1 - mu^2) - 2, from CHECK 3 of check_equations.py."""
    rhs = make_rhs()
    mu, r, i, j = sample_point(rhs)
    m = rhs.m
    expected = (2*mu - m)**2 / (1 - mu**2) - 2
    assert np.isclose(rhs.V[i, j], expected)


def test_V_Schwarzschild_equator():
    """For a=0 at the equator (mu=0, m=2): V = 4/1 - 2 = 2."""
    g = Grid(rmin=2.0, rmax=50.0, Nmu=64, Nr=40, ghost=2, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2)
    # Find interior index closest to mu=0
    gh = g.ghost
    idx_mu = gh + np.argmin(np.abs(g.MU[gh:gh+g.Nmu, 0]))
    idx_r  = gh + g.Nr // 2
    mu_val = g.MU[idx_mu, idx_r].real
    expected = (2*mu_val - 2)**2 / (1 - mu_val**2) - 2
    assert np.isclose(rhs.V[idx_mu, idx_r].real, expected)


def test_parity():
    """Parity factor should be (-1)**m for all m."""
    for m in (0, 1, 2, 3):
        rhs = make_rhs(m=m)
        assert rhs.parity == (-1)**m


# -----------------------------------------------------------------------
# rhs() basic properties
# -----------------------------------------------------------------------

def test_rhs_output_shape():
    rhs = make_rhs()
    g = rhs.grid
    psi = np.zeros(g.shape, dtype=complex)
    v   = np.zeros(g.shape, dtype=complex)
    dpsi, dv = rhs.rhs(psi, v)
    assert dpsi.shape == g.shape
    assert dv.shape   == g.shape


def test_rhs_output_dtype():
    rhs = make_rhs()
    g = rhs.grid
    psi = np.zeros(g.shape, dtype=complex)
    v   = np.zeros(g.shape, dtype=complex)
    dpsi, dv = rhs.rhs(psi, v)
    assert dpsi.dtype == np.complex128
    assert dv.dtype   == np.complex128


def test_rhs_zero_field():
    """Zero fields give zero time derivatives at interior points."""
    rhs = make_rhs()
    g = rhs.grid
    psi = np.zeros(g.shape, dtype=complex)
    v   = np.zeros(g.shape, dtype=complex)
    dpsi, dv = rhs.rhs(psi, v)
    assert np.allclose(dpsi[g.interior], 0.0)
    assert np.allclose(dv[g.interior],   0.0)


def test_rhs_linearity():
    """rhs(alpha*psi, alpha*v) == alpha * rhs(psi, v) at interior points."""
    rhs = make_rhs()
    g = rhs.grid
    rng = np.random.default_rng(7)
    psi = rng.standard_normal(g.shape) + 1j*rng.standard_normal(g.shape)
    v   = rng.standard_normal(g.shape) + 1j*rng.standard_normal(g.shape)

    alpha = 3.14 + 2.71j
    d1, dv1 = rhs.rhs(psi, v)
    d2, dv2 = rhs.rhs(alpha*psi, alpha*v)

    sl = g.interior
    assert np.allclose(d2[sl],  alpha * d1[sl],  rtol=1e-10)
    assert np.allclose(dv2[sl], alpha * dv1[sl], rtol=1e-10)


def test_rhs_does_not_modify_inputs():
    """rhs() must not modify the caller's arrays."""
    rhs = make_rhs()
    g = rhs.grid
    rng = np.random.default_rng(99)
    psi = rng.standard_normal(g.shape) + 1j*rng.standard_normal(g.shape)
    v   = rng.standard_normal(g.shape) + 1j*rng.standard_normal(g.shape)
    psi_orig = psi.copy()
    v_orig   = v.copy()
    rhs.rhs(psi, v)
    assert np.array_equal(psi, psi_orig)
    assert np.array_equal(v,   v_orig)


def test_dissipation_changes_rhs():
    """Non-zero dissipation changes the RHS for a non-trivial field."""
    g    = Grid(rmin=2.0, rmax=50.0, Nmu=32, Nr=40, ghost=2, M=1.0)
    rhs0 = TeukolskyRHS(g, M=1.0, a=0.5, m=2, dissipation=0.0)
    rhs1 = TeukolskyRHS(g, M=1.0, a=0.5, m=2, dissipation=0.5)
    rng  = np.random.default_rng(42)
    psi  = rng.standard_normal(g.shape) + 1j*rng.standard_normal(g.shape)
    v    = rng.standard_normal(g.shape) + 1j*rng.standard_normal(g.shape)

    d0, dv0 = rhs0.rhs(psi, v)
    d1, dv1 = rhs1.rhs(psi, v)
    sl = g.interior
    changed = (not np.allclose(d0[sl], d1[sl]) or
               not np.allclose(dv0[sl], dv1[sl]))
    assert changed, "dissipation=0.5 should produce a different RHS than dissipation=0"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
