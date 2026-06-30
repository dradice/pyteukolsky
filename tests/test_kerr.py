"""
Milestone 5 tests: Kerr (a != 0) ℓ=m=2 QNM validation and pole parity.

The TeukolskyRHS coefficients (Sigma, Delta, A, Cv, Cr) already carry full
a-dependence, so the 2D (r, mu) evolution captures the spin-weighted *spheroidal*
angular structure automatically (the a^2 omega^2 mu^2 coupling enters through
A = Sigma + 2 M r with Sigma = r^2 + a^2 mu^2 acting on d_t v).  Projecting the
detector waveform onto the spherical _{-2}Y_{2m} still isolates the dominant
ell=2 spheroidal content, so a damped-sinusoid fit recovers the Kerr QNM.

For a != 0 the field is genuinely complex (the coefficients Cv, Cr have
imaginary parts), so fit_qnm_frequency uses its complex code path
(log-amplitude and unwrapped-phase linear fits).

Reference frequencies: s=-2, ell=m=2, n=0 fundamental Kerr QNM, M = 1.
Values from the gravitational (s=-2) ell=m=2 fundamental tables of

    E. Berti, V. Cardoso & C. M. Will, "On gravitational-wave spectroscopy of
    massive black holes with the space interferometer LISA", Phys. Rev. D 73,
    064030 (2006), arXiv:gr-qc/0512160.  Tabulated data:
    https://pages.jh.edu/eberti2/ringdown/

computed by Leaver's continued-fraction method:

    E. W. Leaver, "An analytic representation for the quasi-normal modes of
    Kerr black holes", Proc. R. Soc. Lond. A 402, 285 (1985).

    a/M    M*omega_R     M*omega_I
    0.0    0.37367      -0.08896
    0.5    0.46412      -0.08460
    0.9    0.67163      -0.06489

Grid note: rmin = 0.99 * r_+ with r_+ = M + sqrt(M^2 - a^2) keeps every interior
cell outside the horizon (same inside-horizon-instability avoidance as the
Schwarzschild runs in test_validation.py).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyteukolsky.grid import Grid
from pyteukolsky.equation import TeukolskyRHS
from pyteukolsky.evolve import Evolution
from pyteukolsky.initialdata import swsh, gaussian_pulse
from pyteukolsky.diagnostics import project_swsh, fit_qnm_frequency


M = 1.0

# Published s=-2, ell=m=2, n=0 fundamental Kerr QNM frequencies.
PUBLISHED = {
    0.0: (0.37367, -0.08896),
    0.5: (0.46412, -0.08460),
    0.9: (0.67163, -0.06489),
}

# Fit window: after the outgoing burst (peaks at r_ext=30M around t~60-80M) and
# before the first Sommerfeld reflection from rmax=120M (arrives ~t>180M).
FIT_START = 100.0
FIT_END = 150.0

# Cache (a, m, force_parity) -> (times, psi_lm) so tests can share runs.
_CACHE = {}


def r_plus(a):
    return M + np.sqrt(M * M - a * a)


def run_kerr(a, m=2, force_parity=None, Nr=120, Nmu=24, t_final=160.0,
             cfl=0.4, diss=0.1, r_extract=30.0, rmax=120.0):
    """Run a Kerr ell=2 Gaussian pulse; return (times, psi_{2m}) at r_extract.

    psi_{2m} is the projection onto the spherical _{-2}Y_{2m} (complex series).
    """
    key = (a, m, force_parity, Nr, Nmu, t_final)
    if key in _CACHE:
        return _CACHE[key]

    rp = r_plus(a)
    g = Grid(rmin=0.99 * rp, rmax=rmax, Nr=Nr, Nmu=Nmu, ghost=2, M=M)
    rhs = TeukolskyRHS(g, M=M, a=a, m=m, dissipation=diss)
    if force_parity is not None:
        rhs.parity = force_parity
    evo = Evolution(rhs)
    psi0 = gaussian_pulse(g, r0=10.0, sigma_r=2.0, ell=2, m=m, spin=-2)
    evo.set_initial_data(psi0, psi0, dt_init=1e-3)
    evo.add_detector(r_extract)
    evo.evolve(t_final, cfl=cfl)

    mu = g._mu[g.ghost:g.ghost + g.Nmu]
    sw = swsh(-2, 2, m, mu)
    psi_lm = project_swsh(evo.waveforms[r_extract], mu, sw)

    _CACHE[key] = (evo.times, psi_lm)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Sanity: interior cells are outside the horizon for the spins we test
# ---------------------------------------------------------------------------

def test_interior_cells_outside_horizon():
    """rmin = 0.99 r_+ must place the first interior cell at r > r_+."""
    for a in (0.5, 0.9):
        rp = r_plus(a)
        g = Grid(rmin=0.99 * rp, rmax=120.0, Nr=120, Nmu=24, ghost=2, M=M)
        r_first = g.r[g.ghost].real
        assert r_first > rp, \
            f"a={a}: first interior cell r={r_first:.4f} not outside r_+={rp:.4f}"


# ---------------------------------------------------------------------------
# Kerr QNM frequencies vs published tables
# ---------------------------------------------------------------------------

def test_kerr_qnm_a05():
    """Kerr a=0.5 ell=m=2: Mω ≈ 0.46412 - 0.08460i (complex-path fit)."""
    times, psi = run_kerr(0.5)
    omega_R, omega_I = fit_qnm_frequency(times, psi, FIT_START, FIT_END)
    pub_R, pub_I = PUBLISHED[0.5]
    assert abs(omega_R - pub_R) < 0.015, \
        f"Mω_R = {omega_R:.4f}, expected {pub_R:.4f} ± 0.015"
    assert abs(omega_I - pub_I) < 0.012, \
        f"Mω_I = {omega_I:.4f}, expected {pub_I:.4f} ± 0.012"


def test_kerr_qnm_a09():
    """Kerr a=0.9 ell=m=2: Mω ≈ 0.67163 - 0.06489i (complex-path fit)."""
    times, psi = run_kerr(0.9)
    omega_R, omega_I = fit_qnm_frequency(times, psi, FIT_START, FIT_END)
    pub_R, pub_I = PUBLISHED[0.9]
    assert abs(omega_R - pub_R) < 0.015, \
        f"Mω_R = {omega_R:.4f}, expected {pub_R:.4f} ± 0.015"
    assert abs(omega_I - pub_I) < 0.012, \
        f"Mω_I = {omega_I:.4f}, expected {pub_I:.4f} ± 0.012"


def test_kerr_prograde_spin_trend():
    """Prograde (m=2, a>0) ringing frequency increases, decay rate decreases."""
    wR = {}
    wI = {}
    for a in (0.5, 0.9):
        times, psi = run_kerr(a)
        wR[a], wI[a] = fit_qnm_frequency(times, psi, FIT_START, FIT_END)

    schwarz_R = PUBLISHED[0.0][0]  # 0.37367
    # omega_R increases monotonically with prograde spin
    assert schwarz_R < wR[0.5] < wR[0.9], \
        f"omega_R not increasing: {schwarz_R:.4f} < {wR[0.5]:.4f} < {wR[0.9]:.4f}"
    # |omega_I| decreases (longer-lived ringing) as a -> extremal
    assert abs(wI[0.9]) < abs(wI[0.5]), \
        f"|omega_I| not decreasing with spin: {abs(wI[0.5]):.4f}, {abs(wI[0.9]):.4f}"


# ---------------------------------------------------------------------------
# Pole parity validation
# ---------------------------------------------------------------------------

def test_kerr_pole_parity():
    """The correct pole parity (-1)^m recovers the published QNM; the wrong
    parity does not.

    For m=2 the correct angular-ghost parity is (+1).  Flipping it to (-1)
    fills the across-pole ghost cells with the wrong sign, corrupting the
    angular operator near mu=±1 and shifting the recovered frequency well
    outside the validation tolerance.
    """
    pub_R, pub_I = PUBLISHED[0.5]

    t_ok, psi_ok = run_kerr(0.5, force_parity=None)   # (-1)^2 = +1 (default)
    wR_ok, wI_ok = fit_qnm_frequency(t_ok, psi_ok, FIT_START, FIT_END)

    t_bad, psi_bad = run_kerr(0.5, force_parity=-1)   # deliberately wrong
    wR_bad, wI_bad = fit_qnm_frequency(t_bad, psi_bad, FIT_START, FIT_END)

    err_ok = abs(wR_ok - pub_R)
    err_bad = abs(wR_bad - pub_R)

    assert err_ok < 0.015, f"correct parity off: Mω_R={wR_ok:.4f}"
    # Wrong parity should be clearly worse (the observed shift is ~0.05).
    assert err_bad > 0.03, \
        f"wrong parity not distinguishable: Mω_R={wR_bad:.4f} (err {err_bad:.4f})"
    assert err_bad > 3.0 * err_ok, \
        f"wrong parity ({err_bad:.4f}) not >> correct ({err_ok:.4f})"


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
