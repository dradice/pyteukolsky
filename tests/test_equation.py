"""
Milestone 2 tests: TeukolskyRHS coefficient arrays and rhs() method.

Coefficient arrays are cross-checked against the analytical expressions from
scripts/check_equations.py at a sample interior grid point.  The rhs()
method is checked for shape, dtype, linearity, zero-field behaviour, and
non-modification of inputs.
"""

import numpy as np
import pytest
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


def test_i_exc_safe_rmin():
    """rmin=1.99 (Schwarzschild): first interior column is already outside
    r_+=2M, so i_exc should equal grid.ghost (no excised columns)."""
    g = Grid(rmin=1.99, rmax=100.0, Nmu=16, Nr=100, ghost=2, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2, one_sided_horizon=True)
    assert rhs.i_exc == g.ghost


def test_i_exc_aggressive_rmin():
    """rmin=1.5 (inside r_+=2M, run_example.py's own default): several
    interior columns are inside the horizon, so i_exc must be found
    dynamically and land just outside r_+."""
    g = Grid(rmin=1.5, rmax=100.0, Nmu=16, Nr=200, ghost=2, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2, one_sided_horizon=True)
    assert rhs.i_exc > g.ghost
    assert g.r[rhs.i_exc].real > rhs.r_plus
    assert g.r[rhs.i_exc - 1].real <= rhs.r_plus


def test_i_exc_raises_when_no_good_columns():
    """A grid entirely inside the horizon should raise a clear error rather
    than silently picking a bogus excision column."""
    g = Grid(rmin=1.0, rmax=1.9, Nmu=16, Nr=50, ghost=2, M=1.0)
    with pytest.raises(ValueError):
        TeukolskyRHS(g, M=1.0, a=0.0, m=2, one_sided_horizon=True)


def test_one_sided_horizon_default_off_does_not_raise():
    """A horizon-straddling grid must not raise when the feature is off
    (i_exc/r_plus are always computed, but only used when the flag is set)."""
    g = Grid(rmin=1.5, rmax=100.0, Nmu=16, Nr=200, ghost=2, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2)  # one_sided_horizon=False
    assert rhs.one_sided_horizon is False


def test_one_sided_horizon_decouples_from_excised_region():
    """With one_sided_horizon=True and dissipation>0, perturbing psi at the
    column just inside the excision boundary must not change rhs() at the
    excision column or its immediate neighbor -- this is exactly the
    Kreiss-Oliger masking fix: ko_dissipation_r has +/-2 reach, so without
    masking Qp[:, i_exc:i_exc+2] would still depend on i_exc-1."""
    g = Grid(rmin=1.5, rmax=100.0, Nmu=16, Nr=200, ghost=2, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2, dissipation=0.3,
                        one_sided_horizon=True)
    rng = np.random.default_rng(11)
    psi = rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape)
    v   = rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape)

    dpsi0, dv0 = rhs.rhs(psi, v)

    i = rhs.i_exc
    psi_pert = psi.copy()
    psi_pert[:, i - 1] += 1e3 * (1.0 + 1.0j)   # huge perturbation just inside
    dpsi1, dv1 = rhs.rhs(psi_pert, v)

    assert np.allclose(dpsi0[:, i:i + 2], dpsi1[:, i:i + 2])
    assert np.allclose(dv0[:, i:i + 2],   dv1[:, i:i + 2])


# ---------------------------------------------------------------------
# order=4 excision consequences
#
# order=2 behaviour above (i_exc, MIN_GOOD_COLUMNS=4, 1-column override,
# 2-column KO mask) is untouched; these tests exercise the order=4 widening.
# ---------------------------------------------------------------------

def test_min_good_columns_is_7_at_order4():
    """A grid with exactly 6 good (r > r_+) interior columns must raise at
    order=4 (needs >= 7): rmin=1.5, rmax=3.0, Nr=10 gives n_good=6, which
    would PASS the order=2 threshold (need >= 4) but must fail at order=4,
    proving MIN_GOOD_COLUMNS actually rose from 4 to 7 (not left at 4)."""
    g = Grid(rmin=1.5, rmax=3.0, Nmu=16, Nr=10, ghost=3, order=4, M=1.0)
    with pytest.raises(ValueError):
        TeukolskyRHS(g, M=1.0, a=0.0, m=2, one_sided_horizon=True)


def test_min_good_columns_6_is_sufficient_at_order2():
    """The same n_good=6 configuration (rmin=1.5, rmax=3.0, Nr=10) must NOT
    raise at order=2, confirming the order=2 threshold (4) is untouched."""
    g = Grid(rmin=1.5, rmax=3.0, Nmu=16, Nr=10, ghost=2, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2, one_sided_horizon=True)
    n_good = g.ghost + g.Nr - rhs.i_exc
    assert n_good == 6


def test_min_good_columns_7_is_sufficient_at_order4():
    """Exactly 7 good columns (rmin=1.5, rmax=3.0, Nr=12) must NOT raise at
    order=4."""
    g = Grid(rmin=1.5, rmax=3.0, Nmu=16, Nr=12, ghost=3, order=4, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2, one_sided_horizon=True)
    n_good = g.ghost + g.Nr - rhs.i_exc
    assert n_good == 7


def test_i_exc_excision_width_order4():
    """The one-sided excision override covers 2 columns at order=4 (vs 1 at
    order=2), derived from grid.order via TeukolskyRHS._n_excision_cols."""
    g2 = Grid(rmin=1.5, rmax=100.0, Nmu=16, Nr=200, ghost=2, M=1.0)
    rhs2 = TeukolskyRHS(g2, M=1.0, a=0.0, m=2, one_sided_horizon=True)
    assert rhs2._n_excision_cols == 1

    g4 = Grid(rmin=1.5, rmax=100.0, Nmu=16, Nr=200, ghost=3, order=4, M=1.0)
    rhs4 = TeukolskyRHS(g4, M=1.0, a=0.0, m=2, one_sided_horizon=True)
    assert rhs4._n_excision_cols == 2


def test_one_sided_horizon_decouples_from_excised_region_order4():
    """order=4 analogue of test_one_sided_horizon_decouples_from_excised_region:
    with one_sided_horizon=True and dissipation>0 at order=4, perturbing psi
    at the column just inside the excision boundary (i_exc - 1) must not
    change rhs() at i_exc or i_exc+1 (both overridden columns) -- this is the
    order=4 excision-override-width fix plus the widened (i_exc:i_exc+3) KO
    mask acting together. ko_dissipation_r has +/-3 reach at order=4, so
    without the wider mask Qp[:, i_exc:i_exc+3] would still depend on
    i_exc-1."""
    g = Grid(rmin=1.5, rmax=100.0, Nmu=16, Nr=200, ghost=3, order=4, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2, dissipation=0.3,
                        one_sided_horizon=True)
    rng = np.random.default_rng(11)
    psi = rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape)
    v   = rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape)

    dpsi0, dv0 = rhs.rhs(psi, v)

    i = rhs.i_exc
    psi_pert = psi.copy()
    psi_pert[:, i - 1] += 1e3 * (1.0 + 1.0j)   # huge perturbation just inside
    dpsi1, dv1 = rhs.rhs(psi_pert, v)

    # Both overridden columns (i_exc, i_exc+1) must be unaffected.
    assert np.allclose(dpsi0[:, i:i + 2], dpsi1[:, i:i + 2])
    assert np.allclose(dv0[:, i:i + 2],   dv1[:, i:i + 2])


def test_order4_excision_reads_no_column_left_of_i_exc():
    """Provable-by-construction check (mirrors the style of the test above,
    at the level of the raw Grid stencils rather than rhs()): the order=4
    one-sided stencils used at i_exc and i_exc+1 must never read column
    i_exc - 1 or lower. Verified by perturbing every column < i_exc one at a
    time and checking dr_onesided/drr_onesided at i_exc and i_exc+1 are
    unchanged."""
    g = Grid(rmin=1.5, rmax=100.0, Nmu=16, Nr=200, ghost=3, order=4, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2, one_sided_horizon=True)
    i = rhs.i_exc
    rng = np.random.default_rng(5)
    f = rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape)
    g.fill_ghosts_r(f)
    g.fill_ghosts_mu(f, rhs.parity)

    base_dr_i   = g.dr_onesided(f, i)
    base_drr_i  = g.drr_onesided(f, i)
    base_dr_i1  = g.dr_onesided(f, i + 1)
    base_drr_i1 = g.drr_onesided(f, i + 1)

    for j in range(0, i):   # every column strictly left of i_exc
        f_pert = f.copy()
        f_pert[:, j] += 1e6 * (1.0 + 1.0j)
        assert np.allclose(g.dr_onesided(f_pert, i),  base_dr_i)
        assert np.allclose(g.drr_onesided(f_pert, i), base_drr_i)
        assert np.allclose(g.dr_onesided(f_pert, i + 1),  base_dr_i1)
        assert np.allclose(g.drr_onesided(f_pert, i + 1), base_drr_i1)


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
